"""
overnight.py - decompose close-to-close returns into overnight and intraday
components, following Lou, Polk & Skouras (2019, JFE 134:192-213),
"A tug of war: Overnight versus intraday expected returns".

    r_intraday,t  = Close_t / Open_t      - 1
    r_overnight,t = Open_t  / Close_{t-1} - 1
    (1 + r_overnight,t)(1 + r_intraday,t) = (1 + r_close_to_close,t)

yfinance's auto_adjust scales Open/High/Low/Close by the same factor on a
given day, so the intraday leg is adjustment-invariant and the overnight leg
carries the dividend/split adjustment. That is exactly the convention the
paper adopts (p.196 n.9: corporate events are assumed to move prices overnight).
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "alpha" / "ohlc_open.parquet"


def tickers_from_cache() -> list[str]:
    with open(REPO / "master_cache.pkl", "rb") as f:
        _, meta = pickle.load(f)
    return sorted(meta["tickers"])


def download(tickers: list[str], start="2014-12-01", end=None, chunk=40, pause=1.5) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(4):
            try:
                d = yf.download(part, start=start, end=end, auto_adjust=True,
                                progress=False, threads=False, timeout=30)
                if d is not None and len(d):
                    frames.append(d)
                break
            except Exception as e:
                print(f"   retry {attempt+1} on chunk {i//chunk}: {type(e).__name__}", flush=True)
                time.sleep(5 * (attempt + 1))
        print(f"  {min(i+chunk, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(pause)
    wide = pd.concat(frames, axis=1)
    long = (wide.stack(level="Ticker", future_stack=True)
                .rename_axis(["Date", "Ticker"]).reset_index())
    return long.dropna(subset=["Open", "Close"])


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()
    g = df.groupby("Ticker", observed=True)
    prev_close = g["Close"].shift(1)

    # only across consecutive trading days on the panel calendar
    cal = pd.Index(sorted(df["Date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    dpos = df["Date"].map(pos)
    contiguous = (dpos - g["Date"].transform(lambda s: s.map(pos)).shift(1)) == 1

    df["r_cc"] = df["Close"] / prev_close - 1
    df["r_intraday"] = df["Close"] / df["Open"] - 1
    df["r_overnight"] = df["Open"] / prev_close - 1
    df.loc[~contiguous, ["r_cc", "r_overnight"]] = np.nan

    # sanity: the two legs must compound to the close-to-close return
    chk = (1 + df["r_overnight"]) * (1 + df["r_intraday"]) - 1 - df["r_cc"]
    df.attrs["max_decomposition_error"] = float(np.nanmax(np.abs(chk)))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    tk = tickers_from_cache()
    if a.limit:
        tk = tk[:a.limit]
    print(f"downloading {len(tk)} tickers ...", flush=True)
    t0 = time.time()
    raw = download(tk)
    out = decompose(raw)
    print(f"rows={len(out):,}  tickers={out.Ticker.nunique()}  "
          f"{out.Date.min().date()} -> {out.Date.max().date()}")
    print(f"max |(1+on)(1+id)-1 - cc| = {out.attrs['max_decomposition_error']:.2e}")
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}  ({time.time()-t0:.0f}s)")
