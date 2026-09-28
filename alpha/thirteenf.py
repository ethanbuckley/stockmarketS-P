"""
thirteenf.py - institutional ownership from SEC Form 13F structured data sets.

Lou, Polk & Skouras (2019) argue the overnight/intraday tug of war is driven by
CLIENTELES: institutions trade near the close, individuals near the open. Their
Tables 6-8 show institutional ownership conditions the effect. This module
builds the ownership panel needed to test that on our data.

Source: https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets
Each quarterly ZIP holds SUBMISSION.tsv (accession -> CIK, period) and
INFOTABLE.tsv (accession -> issuer, CUSIP, shares, value). We stream each ZIP,
aggregate to (period, CUSIP), write a small parquet, and delete the ZIP, so
peak disk stays at one file rather than ~3.7 GB.

Two gotchas handled here:
  * VALUE units changed from THOUSANDS of dollars to whole dollars for filings
    from Jan 2023. We detect it per file from the implied price rather than
    hardcoding a cutover date.
  * Only common stock, long positions: SSHPRNAMTTYPE == 'SH' and PUTCALL blank.
    Options and debt positions are excluded.

SEC fair-access requires a User-Agent carrying real contact details, else every
request 403s. Set it yourself:

    export SEC_USER_AGENT="Your Name your.email@domain.com"
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "alpha" / "cache13f"
LISTING = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
PLACEHOLDER = "AlphaResearch research@example.com"


def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        print(
            "  ! SEC_USER_AGENT is unset; using a placeholder contact.\n"
            "    SEC asks you to declare traffic with a real address:\n"
            '      export SEC_USER_AGENT="Your Name you@domain.com"',
            file=sys.stderr,
        )
        return PLACEHOLDER
    return ua


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
    return s


def list_files(s: requests.Session, since_year: int = 2014) -> list[str]:
    html = s.get(LISTING, timeout=30).text
    urls = sorted(set(re.findall(r'href="([^"]*form13f\.zip)"', html)))
    keep = []
    for u in urls:
        yrs = re.findall(r"(20\d\d)", Path(u).name)
        if yrs and max(int(y) for y in yrs) >= since_year:
            keep.append(u)
    return keep


def _member(z: zipfile.ZipFile, name: str) -> str | None:
    """Resolve a TSV inside the ZIP by basename.

    Most quarterly archives store members at the root ('INFOTABLE.tsv'), but
    some nest them under a directory ('01JUN2025-31AUG2025_form13f/INFOTABLE.tsv').
    Matching on the exact name silently skipped a whole quarter.
    """
    for n in z.namelist():
        if n.rsplit("/", 1)[-1] == name:
            return n
    return None


def aggregate_zip(blob: bytes) -> pd.DataFrame | None:
    """(period, cusip) -> holders, shares, dollar value, from one quarterly ZIP."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    m_sub, m_info = _member(z, "SUBMISSION.tsv"), _member(z, "INFOTABLE.tsv")
    if not (m_sub and m_info):
        return None

    sub = pd.read_csv(
        z.open(m_sub),
        sep="\t",
        usecols=["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT", "SUBMISSIONTYPE"],
        dtype={"ACCESSION_NUMBER": str, "CIK": str},
    )
    sub = sub[sub["SUBMISSIONTYPE"].astype(str).str.startswith("13F-HR")]
    sub["period"] = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    sub = sub.dropna(subset=["period"])[["ACCESSION_NUMBER", "CIK", "period"]]

    info = pd.read_csv(
        z.open(m_info),
        sep="\t",
        usecols=[
            "ACCESSION_NUMBER",
            "NAMEOFISSUER",
            "TITLEOFCLASS",
            "CUSIP",
            "VALUE",
            "SSHPRNAMT",
            "SSHPRNAMTTYPE",
            "PUTCALL",
        ],
        dtype={"ACCESSION_NUMBER": str, "CUSIP": str, "PUTCALL": str},
        low_memory=False,
    )
    info = info[
        (info["SSHPRNAMTTYPE"].astype(str).str.upper() == "SH")
        & (info["PUTCALL"].isna() | (info["PUTCALL"].astype(str).str.strip() == ""))
    ]
    info = info[info["SSHPRNAMT"] > 0]
    if info.empty:
        return None

    # VALUE switched from thousands to dollars in 2023; infer from implied price
    implied = (info["VALUE"] / info["SSHPRNAMT"]).median()
    scale = 1000.0 if implied < 1.0 else 1.0
    info["dollars"] = info["VALUE"] * scale

    df = info.merge(sub, on="ACCESSION_NUMBER", how="inner")
    if df.empty:
        return None
    df["CUSIP"] = df["CUSIP"].str.strip().str.upper().str.zfill(9)

    out = (
        df.groupby(["period", "CUSIP"], observed=True)
        .agg(
            holders=("CIK", "nunique"),
            inst_shares=("SSHPRNAMT", "sum"),
            inst_dollars=("dollars", "sum"),
            issuer=("NAMEOFISSUER", "first"),
            cls=("TITLEOFCLASS", "first"),
        )
        .reset_index()
    )
    out["implied_scale"] = scale
    return out


def ingest(since_year: int = 2014, pause: float = 1.0) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    s = session()
    files = list_files(s, since_year)
    print(f"{len(files)} quarterly files from {since_year}", flush=True)
    for i, u in enumerate(files, 1):
        tag = Path(u).stem.replace("_form13f", "")
        dest = CACHE / f"{tag}.parquet"
        if dest.exists():
            print(f"  [{i}/{len(files)}] {tag} cached", flush=True)
            continue
        t0 = time.time()
        try:
            r = s.get(f"https://www.sec.gov{u}", timeout=180)
            r.raise_for_status()
            agg = aggregate_zip(r.content)
            if agg is None or agg.empty:
                print(f"  [{i}/{len(files)}] {tag} EMPTY", flush=True)
                continue
            agg.to_parquet(dest, index=False)
            print(
                f"  [{i}/{len(files)}] {tag}  {len(agg):>7,} cusip-rows  "
                f"scale={agg.implied_scale.iloc[0]:g}  {time.time() - t0:.0f}s",
                flush=True,
            )
        except Exception as e:
            print(f"  [{i}/{len(files)}] {tag} FAILED {type(e).__name__}: {e}", flush=True)
        time.sleep(pause)


def load() -> pd.DataFrame:
    parts = sorted(CACHE.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no 13F cache in {CACHE}; run `python3 alpha/thirteenf.py ingest`")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # a period can appear in overlapping filing windows; keep the fullest record
    df = (
        df.sort_values("holders", ascending=False)
        .drop_duplicates(["period", "CUSIP"], keep="first")
        .sort_values(["CUSIP", "period"])
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Pass 2: manager-level holdings for our own universe.
#
# The (period, CUSIP) aggregate above is fine for building the name map but is
# WRONG as an ownership measure, for two reasons:
#   * SEC files are keyed by FILING window, not report period, so one period's
#     filers are split across several files (40% of our rows). Deduping keeps
#     one file's filers and drops the rest; summing double-counts amendments.
#   * SUBMISSIONTYPE '13F-HR/A' amendments RESTATE a manager's holdings, so the
#     same manager legitimately appears twice for one period.
# Both are fixed by carrying the filer's CIK and keeping each manager's LATEST
# filing per (period, issuer). Restricting to our ~450 issuers keeps it small.
# ---------------------------------------------------------------------------

HOLDINGS = REPO / "alpha" / "cache13f_holdings"


def _universe_cusip6(wide: bool = False) -> set[str]:
    name = "cusip_map_wide.csv" if wide else "cusip_map.csv"
    mp = pd.read_csv(REPO / "alpha" / name, dtype={"cusip6": str})
    return set(mp["cusip6"])


def holdings_from_zip(blob: bytes, keep: set[str]) -> pd.DataFrame | None:
    z = zipfile.ZipFile(io.BytesIO(blob))
    m_sub, m_info = _member(z, "SUBMISSION.tsv"), _member(z, "INFOTABLE.tsv")
    if not (m_sub and m_info):
        return None

    sub = pd.read_csv(
        z.open(m_sub),
        sep="\t",
        usecols=["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT", "FILING_DATE", "SUBMISSIONTYPE"],
        dtype={"ACCESSION_NUMBER": str, "CIK": str},
    )
    sub = sub[sub["SUBMISSIONTYPE"].astype(str).str.startswith("13F-HR")]
    sub["period"] = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    sub = sub.dropna(subset=["period", "filed"])[["ACCESSION_NUMBER", "CIK", "period", "filed"]]

    info = pd.read_csv(
        z.open(m_info),
        sep="\t",
        usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"],
        dtype={"ACCESSION_NUMBER": str, "CUSIP": str, "PUTCALL": str},
        low_memory=False,
    )
    info = info[
        (info["SSHPRNAMTTYPE"].astype(str).str.upper() == "SH")
        & (info["PUTCALL"].isna() | (info["PUTCALL"].astype(str).str.strip() == ""))
        & (info["SSHPRNAMT"] > 0)
    ]
    if info.empty:
        return None
    scale = 1000.0 if (info["VALUE"] / info["SSHPRNAMT"]).median() < 1.0 else 1.0
    info["cusip6"] = info["CUSIP"].str.strip().str.upper().str.zfill(9).str[:6]
    info = info[info["cusip6"].isin(keep)]
    if info.empty:
        return None
    info["dollars"] = info["VALUE"] * scale

    df = info.merge(sub, on="ACCESSION_NUMBER", how="inner")
    if df.empty:
        return None
    # one filing can list an issuer on several rows (share classes, co-managers)
    return (
        df.groupby(["period", "cusip6", "CIK", "filed"], observed=True)
        .agg(shares=("SSHPRNAMT", "sum"), dollars=("dollars", "sum"))
        .reset_index()
    )


def ingest_holdings(since_year: int = 2014, pause: float = 1.0, wide: bool = False) -> None:
    out_dir = REPO / "alpha" / ("cache13f_wide" if wide else "cache13f_holdings")
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = _universe_cusip6(wide)
    s = session()
    files = list_files(s, since_year)
    print(f"{len(files)} files; tracking {len(keep)} issuers", flush=True)
    for i, u in enumerate(files, 1):
        tag = Path(u).stem.replace("_form13f", "")
        dest = out_dir / f"{tag}.parquet"
        if dest.exists():
            print(f"  [{i}/{len(files)}] {tag} cached", flush=True)
            continue
        t0 = time.time()
        try:
            r = s.get(f"https://www.sec.gov{u}", timeout=180)
            r.raise_for_status()
            h = holdings_from_zip(r.content, keep)
            if h is None or h.empty:
                print(f"  [{i}/{len(files)}] {tag} EMPTY", flush=True)
                continue
            h.to_parquet(dest, index=False)
            print(f"  [{i}/{len(files)}] {tag}  {len(h):>7,} manager-rows  {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(files)}] {tag} FAILED {type(e).__name__}: {e}", flush=True)
        time.sleep(pause)


def load_holdings(wide: bool = False) -> pd.DataFrame:
    """Per (period, issuer): breadth and shares, each manager counted once at
    their LATEST filing for that period."""
    src = REPO / "alpha" / ("cache13f_wide" if wide else "cache13f_holdings")
    parts = sorted(src.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no holdings in {src}; run `thirteenf.py holdings`")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values("filed").drop_duplicates(["period", "cusip6", "CIK"], keep="last")
    return (
        df.groupby(["period", "cusip6"], observed=True)
        .agg(
            breadth=("CIK", "nunique"),
            inst_shares=("shares", "sum"),
            inst_dollars=("dollars", "sum"),
            last_filed=("filed", "max"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "holdings", "summary"])
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--since", type=int, default=2014)
    a = ap.parse_args()
    if a.cmd == "ingest":
        ingest(a.since)
    elif a.cmd == "holdings":
        ingest_holdings(a.since, wide=a.wide)
    else:
        d = load()
        print(
            f"rows={len(d):,}  cusips={d.CUSIP.nunique():,}  "
            f"periods={d.period.nunique()}  {d.period.min().date()} -> {d.period.max().date()}"
        )
        print(d.groupby("period").agg(cusips=("CUSIP", "size"), holders=("holders", "median")).tail(8).to_string())
