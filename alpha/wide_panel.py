"""
wide_panel.py - lean daily panel for the small/mid-cap extension.

The engineered features in master_cache.pkl exist only for the 503 S&P names,
so the extension needs features computed from OHLC alone. That is not a
compromise here: diagnose_overnight.py found the overnight model IMPROVES when
close-to-close return features are dropped (IC 0.0293 -> 0.0298), and the
signal's real content is the overnight/intraday history, all of which comes
straight from the decomposition.

SURVIVORSHIP: currently-listed names only. See download_wide.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FEATS = ["on_lag1", "id_lag1", "on_5", "id_5", "on_21", "id_21",
         "on_ewma", "id_ewma", "tug", "tug_21",
         "ret_lag1", "ret_lag2", "ret_5", "vol_21", "vol_surge", "dv_rank"]


def build(min_dollar_volume: float = 1e6, min_price: float = 5.0) -> pd.DataFrame:
    from alpha.source import price_files
    df = pd.concat([pd.read_parquet(p) for p in price_files()], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    df = (df.sort_values(["Ticker", "Date"])
            .drop_duplicates(["Ticker", "Date"], keep="first")
            .reset_index(drop=True))

    df["dv"] = df["Close"] * df["Volume"]
    df = df[(df["Close"] >= min_price) & (df["Volume"] > 0)]
    med = df.groupby("Ticker", observed=True)["dv"].transform("median")
    df = df[med >= min_dollar_volume]

    g = df.groupby("Ticker", observed=True)
    for col, tag in (("r_overnight", "on"), ("r_intraday", "id")):
        df[f"{tag}_lag1"] = g[col].shift(1)
        df[f"{tag}_5"] = g[col].transform(lambda s: s.shift(1).rolling(5).sum())
        df[f"{tag}_21"] = g[col].transform(lambda s: s.shift(1).rolling(21).sum())
        df[f"{tag}_ewma"] = g[col].transform(
            lambda s: s.shift(1).ewm(halflife=60, min_periods=21).mean())
    df["tug"] = df["on_ewma"] - df["id_ewma"]
    df["tug_21"] = df["on_21"] - df["id_21"]
    df["ret_lag1"] = g["r_cc"].shift(1)
    df["ret_lag2"] = g["r_cc"].shift(2)
    df["ret_5"] = g["r_cc"].transform(lambda s: s.shift(1).rolling(5).sum())
    df["vol_21"] = g["r_cc"].transform(lambda s: s.shift(1).rolling(21).std())
    df["vol_surge"] = df["Volume"] / g["Volume"].transform(
        lambda s: s.shift(1).rolling(21).mean())
    df["dv_rank"] = df.groupby("Date", observed=True)["dv"].rank(pct=True)

    cal = pd.Index(sorted(df["Date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    df["dpos"] = df["Date"].map(pos)
    ok = (g["dpos"].shift(-1) - df["dpos"]) == 1
    for col, tag in (("r_overnight", "y_on"), ("r_intraday", "y_id"), ("r_cc", "y_cc")):
        df[tag] = g[col].shift(-1).where(ok)

    df = df.dropna(subset=FEATS + ["y_on", "y_cc"])
    for t in ("y_on", "y_id", "y_cc"):
        df[t] = df.groupby("Date", observed=True)[t].transform(
            lambda s: s.clip(s.quantile(0.005), s.quantile(0.995)))
        df[t] = df[t] - df.groupby("Date", observed=True)[t].transform("mean")
    df = df[df.groupby("Date", observed=True)["Ticker"].transform("size") >= 200]
    return df.reset_index(drop=True)


def rank_x(df: pd.DataFrame) -> np.ndarray:
    r = df.groupby("Date", observed=True)[FEATS].rank(pct=True) - 0.5
    return r.values


if __name__ == "__main__":
    d = build()
    print(f"rows={len(d):,} tickers={d.Ticker.nunique():,} "
          f"{d.Date.min().date()} -> {d.Date.max().date()}")
    print(f"names/day: median {d.groupby('Date').size().median():.0f}")
