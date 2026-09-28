"""
shares_outstanding.py - historical share counts, the denominator for
institutional ownership.

13F reports SSHPRNAMT as the raw share count on the report date, so the
denominator must be the raw contemporaneous count too. yfinance's
get_shares_full() returns exactly that (not split-adjusted), which makes the
ratio inst_shares / shares_outstanding split-invariant as long as both come
from the same date. Our price history is split-adjusted, but that does not
touch this ratio.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "alpha" / "shares_outstanding.parquet"


def fetch(tickers: list[str], pause: float = 0.4) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            s = yf.Ticker(t).get_shares_full(start="2014-01-01")
            if s is not None and len(s):
                d = s.rename("shares").to_frame().reset_index()
                d.columns = ["Date", "shares"]
                d["Ticker"] = t
                rows.append(d)
        except Exception as e:
            print(f"  {t}: {type(e).__name__}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}", flush=True)
        time.sleep(pause)
    df = pd.concat(rows, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"], keep="last").reset_index(drop=True)


if __name__ == "__main__":
    with open(REPO / "master_cache.pkl", "rb") as f:
        _, meta = pickle.load(f)
    tk = sorted(meta["tickers"])
    print(f"fetching share counts for {len(tk)} tickers ...", flush=True)
    t0 = time.time()
    df = fetch(tk)
    df.to_parquet(OUT, index=False)
    print(
        f"rows={len(df):,}  tickers={df.Ticker.nunique()}  "
        f"{df.Date.min().date()} -> {df.Date.max().date()}  ({time.time() - t0:.0f}s)"
    )
    print(f"wrote {OUT}")
