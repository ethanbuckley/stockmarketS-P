"""download_wide.py - OHLC for the small/mid-cap extension universe.

SURVIVORSHIP WARNING: Yahoo serves no history for delisted tickers (11/11 of
our removed S&P 500 names returned nothing), so this universe is
currently-listed names only. Small-cap delisting rates are several times those
of large caps, so this file is MORE survivorship-biased than the S&P 500 panel,
not less. Treat anything computed from it as suggestive.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alpha.overnight import decompose, download  # noqa: E402

OUT = REPO / "alpha" / "ohlc_wide.parquet"

if __name__ == "__main__":
    mp = pd.read_csv(REPO / "alpha" / "cusip_map_wide.csv")
    have = set(pd.read_parquet(REPO / "alpha" / "ohlc_open.parquet")["Ticker"].unique())
    tk = sorted(set(mp["ticker"]) - have)
    print(f"{len(tk)} additional tickers (excluding {len(have)} already downloaded)", flush=True)
    t0 = time.time()
    raw = download(tk, chunk=40, pause=0.8)
    out = decompose(raw)
    print(f"rows={len(out):,} tickers={out.Ticker.nunique()} "
          f"max decomposition error={out.attrs['max_decomposition_error']:.1e}")
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")
