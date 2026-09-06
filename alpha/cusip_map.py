"""
cusip_map.py - map our tickers to 13F CUSIPs by issuer name.

CUSIP is a licensed identifier, so there is no free CUSIP->ticker table. We
build one: SEC's company_tickers.json gives ticker -> official company name,
and 13F INFOTABLE gives CUSIP -> NAMEOFISSUER. Normalise both and match.

We key on CUSIP6, the six-character issuer prefix, rather than the full
nine-character CUSIP. Share classes (GOOGL/GOOG, BRK.A/BRK.B) share a CUSIP6,
which is what we want: institutional ownership is a property of the company.

Each candidate match is scored by aggregate institutional dollars, so a typo'd
or stale name loses to the real one. The mapping is checked against six CUSIPs
that are publicly documented, and any ticker whose winning candidate holds less
than 80% of that ticker's matched dollars is dropped as ambiguous.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alpha.thirteenf import load, user_agent  # noqa: E402

TICKER_JSON = "https://www.sec.gov/files/company_tickers.json"
MAP_PATH = REPO / "alpha" / "cusip_map.csv"

# publicly documented issuer prefixes, used as a correctness check
KNOWN = {"AAPL": "037833", "MSFT": "594918", "ABT": "002824",
         "JNJ": "478160", "XOM": "30231G", "JPM": "46625H"}

SUFFIXES = r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LP|LLC|HLDGS?|HOLDINGS?|" \
           r"GROUP|GRP|THE|CLASS|CL|COM|NEW|SA|NV|AG|TRUST|REIT|INTL|INTERNATIONAL)\b"


def norm(s: str) -> str:
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(SUFFIXES, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sec_titles() -> pd.DataFrame:
    r = requests.get(TICKER_JSON, headers={"User-Agent": user_agent()}, timeout=30)
    r.raise_for_status()
    d = json.loads(r.text)
    t = pd.DataFrame(d.values())
    t["key"] = t["title"].map(norm)
    return t[["ticker", "title", "key"]]


def build(tickers: list[str] | None = None, min_breadth: int = 0) -> pd.DataFrame:
    """Map tickers to CUSIP6. tickers=None maps every ticker SEC lists, which is
    how the small-cap universe is drawn."""
    own = load()
    own["cusip6"] = own["CUSIP"].str[:6]
    if min_breadth:
        b = own.groupby("cusip6")["holders"].max()
        own = own[own["cusip6"].isin(b[b >= min_breadth].index)]
    # one row per (cusip6, issuer name) with its total institutional dollars
    cand = (own.groupby(["cusip6", "issuer"], observed=True)["inst_dollars"]
               .sum().reset_index())
    cand["key"] = cand["issuer"].map(norm)

    titles = sec_titles()
    if tickers is not None:
        titles = titles[titles["ticker"].isin(tickers)]
    m = titles.merge(cand, on="key", how="inner")
    if m.empty:
        raise RuntimeError("no name matches; check normalisation")

    # Collapse name variants FIRST. Filers spell the same issuer many ways
    # ("APPLE INC", "Apple, Inc.", "APPLE INC COM"), all against one CUSIP6.
    # Scoring variants separately splits the real issuer's dollars and can push
    # a correct match below the ambiguity threshold.
    m = (m.groupby(["ticker", "title", "cusip6"], observed=True)
           .agg(inst_dollars=("inst_dollars", "sum"),
                issuer=("issuer", "first"))
           .reset_index())
    tot = m.groupby("ticker")["inst_dollars"].transform("sum")
    m["share"] = m["inst_dollars"] / tot
    best = m.sort_values("inst_dollars", ascending=False).drop_duplicates("ticker")
    best = best[best["share"] >= 0.80]          # drop ambiguous name collisions
    return best[["ticker", "cusip6", "issuer", "title", "inst_dollars", "share"]]


if __name__ == "__main__":
    import pickle
    with open(REPO / "master_cache.pkl", "rb") as f:
        _, meta = pickle.load(f)
    tk = sorted(meta["tickers"])
    mp = build(tk)
    mp.to_csv(MAP_PATH, index=False)

    print(f"mapped {len(mp)}/{len(tk)} tickers ({len(mp)/len(tk):.0%})")
    print("\ncheck against publicly documented CUSIP6:")
    ok = 0
    for t, c6 in KNOWN.items():
        got = mp.loc[mp.ticker == t, "cusip6"]
        got = got.iloc[0] if len(got) else "MISSING"
        flag = "ok" if got == c6 else "MISMATCH"
        ok += got == c6
        print(f"  {t:<6} expected {c6}  got {got:<8} {flag}")
    print(f"  {ok}/{len(KNOWN)} correct")
    print("\nlargest by institutional dollars:")
    print(mp.head(5)[["ticker", "cusip6", "issuer", "share"]].to_string(index=False))
    print(f"\nwrote {MAP_PATH}")
