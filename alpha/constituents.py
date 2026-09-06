"""
constituents.py - point-in-time S&P 500 membership.

The root README names survivorship bias as an uncorrected limitation:

    The universe is the current S&P 500 membership scraped from Wikipedia and
    applied back to 2015. Companies that were removed from the index along the
    way are missing, which inflates measured hit rates.

Wikipedia's own page no longer carries a parseable change log, but its REVISION
HISTORY does the job better: fetch the article as it stood on a past date and
read the constituent table exactly as it was then. The 2016 snapshot contains
ATVI, the 2020 snapshot contains ABMD; both were later acquired and are absent
from today's list, which is precisely the population that causes the bias.

Note this fixes only half of it. Point-in-time membership stops us using a
company's history from BEFORE it joined the index (a real and upward bias,
since companies are added after they have already grown). It cannot resurrect
price history for companies that were removed and delisted, because Yahoo drops
those series entirely. See README for what that leaves.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "alpha" / "sp500_membership.parquet"
API = "https://en.wikipedia.org/w/api.php"
PAGE = "List of S&P 500 companies"
UA = {"User-Agent": "alpha-research/0.1 (academic research; contact via GitHub)"}


def revision_at(ts: pd.Timestamp) -> tuple[int, str] | None:
    r = requests.get(API, headers=UA, timeout=30, params={
        "action": "query", "prop": "revisions", "titles": PAGE,
        "rvlimit": 1, "rvdir": "older",
        "rvstart": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rvprop": "ids|timestamp", "format": "json"})
    pages = r.json().get("query", {}).get("pages", {})
    for pg in pages.values():
        revs = pg.get("revisions")
        if revs:
            return revs[0]["revid"], revs[0]["timestamp"]
    return None


def constituents_of(revid: int) -> list[str] | None:
    html = requests.get(f"https://en.wikipedia.org/w/index.php?oldid={revid}",
                        headers=UA, timeout=30).text
    for d in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in d.columns]
        hit = next((c for c in cols if "ymbol" in c or c.strip() in ("Ticker", "Ticker symbol")), None)
        if hit and len(d) > 400:
            s = (d[hit].astype(str).str.strip().str.upper()
                 .str.replace(".", "-", regex=False)          # BRK.B -> BRK-B
                 .str.replace(r"\[.*\]", "", regex=True))
            s = s[s.str.fullmatch(r"[A-Z][A-Z0-9\-]{0,6}")]
            return sorted(set(s))
    return None


def build(start="2014-12-31", end=None, freq="QE", pause=0.8) -> pd.DataFrame:
    end = end or pd.Timestamp.today().normalize()
    dates = pd.date_range(start, end, freq=freq)
    rows, seen = [], {}
    for d in dates:
        rv = revision_at(d)
        if rv is None:
            print(f"  {d.date()}: no revision", flush=True)
            continue
        revid, stamp = rv
        if revid not in seen:
            tk = constituents_of(revid)
            seen[revid] = tk
            time.sleep(pause)
        tk = seen[revid]
        if not tk:
            print(f"  {d.date()}: table not parsed (rev {revid})", flush=True)
            continue
        rows += [{"asof": d, "revid": revid, "rev_time": stamp[:10], "Ticker": t} for t in tk]
        print(f"  {d.date()}: {len(tk)} members (rev {stamp[:10]})", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-12-31")
    a = ap.parse_args()
    df = build(a.start)
    df.to_parquet(OUT, index=False)
    n = df.groupby("asof")["Ticker"].size()
    print(f"\nsnapshots={n.size}  members/snapshot {n.min()}-{n.max()}  "
          f"distinct tickers ever={df.Ticker.nunique()}")
    print(f"wrote {OUT}")
