"""
alpha_lab.py - cross-sectional next-day return prediction, scored as a
tradeable dollar-neutral book rather than as a classifier.

Reads the OHLCV+feature panel from stockmarketS-P-main/master_cache.pkl.
Read-only: nothing here writes to that repo.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CACHE = Path.home() / "Documents/GitHub/stockmarketS-P-main/master_cache.pkl"
ANN = np.sqrt(252.0)

MARKET_WIDE = ["Day_Of_Week","SPY_Return","QQQ_Return","SMH_Return","VIX_Change",
               "TNX_Change","QQQ_Lag_1","QQQ_Lag_2","QQQ_Lag_3"]
STOCK_SPECIFIC = ["RSI","MACD","BB_Position","Price_to_VWAP","ATR_Ratio","Return",
                  "Volume_Surge","Rel_SPY","Rel_QQQ","Rel_SMH",
                  "Return_Lag_1","Return_Lag_2","Return_Lag_3"]


# ------------------------------------------------------------------ data
def load_panel() -> tuple[pd.DataFrame, dict]:
    with open(CACHE, "rb") as f:
        df, meta = pickle.load(f)
    df = df.reset_index()
    df.columns = ["Date"] + list(df.columns[1:])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # TARGET: next trading day's close-to-close return, per ticker.
    df["fwd_ret"] = df.groupby("Ticker", observed=True)["Return"].shift(-1)
    # Only keep the label when the ticker's next row really is the next
    # trading day on the panel calendar (no jumping a suspension/gap).
    cal = pd.Index(sorted(df["Date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    df["dpos"] = df["Date"].map(pos)
    nxt = df.groupby("Ticker", observed=True)["dpos"].shift(-1)
    df.loc[(nxt - df["dpos"]) != 1, "fwd_ret"] = np.nan
    df = df.drop(columns=["dpos", "Target"])

    df = df[(df["Close"] > 1.0) & (df["Volume"] > 0)]
    df = df.dropna(subset=["fwd_ret"] + STOCK_SPECIFIC + MARKET_WIDE)

    # winsorise the label at the daily 0.5/99.5 pct to stop a single halted
    # name dominating the sum; keeps sign, caps leverage of one print.
    def _wins(s):
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        return s.clip(lo, hi)
    df["fwd_ret"] = df.groupby("Date", observed=True)["fwd_ret"].transform(_wins)

    # cross-sectional demean -> the alpha target (market factor removed)
    df["fwd_ret_cs"] = df["fwd_ret"] - df.groupby("Date", observed=True)["fwd_ret"].transform("mean")

    n = df.groupby("Date", observed=True)["Ticker"].transform("size")
    df = df[n >= 100]
    return df.reset_index(drop=True), meta


def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional percentile rank centred on 0 for stock-specific
    columns. Market-wide columns stay raw: their cross-sectional rank is
    constant by construction, so they act only as conditioning variables."""
    out = df[MARKET_WIDE].copy()
    r = df.groupby(df["Date"], observed=True)[STOCK_SPECIFIC].rank(pct=True) - 0.5
    for c in STOCK_SPECIFIC:
        out[c + "_r"] = r[c].values
    return out


# ------------------------------------------------------------- evaluation
def weights_from_scores(scores: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Dollar-neutral, unit-gross weights. This makes the user's
    sum(pred * actual) scale-free: daily PnL becomes a return on $1 gross."""
    s = pd.Series(scores, index=dates)
    s = s - s.groupby(level=0).transform("mean")
    gross = s.abs().groupby(level=0).transform("sum").replace(0.0, np.nan)
    return (s / gross).values


def turnover_series(w: np.ndarray, dates: np.ndarray, tickers: np.ndarray) -> pd.Series:
    wide = pd.DataFrame({"d": dates, "t": tickers, "w": w}).pivot_table(
        index="d", columns="t", values="w").fillna(0.0).sort_index()
    return wide.diff().abs().sum(axis=1)


def evaluate(pred, actual, dates, tickers, name, cost_bps=(0.0, 1.0, 2.0, 5.0)):
    d = pd.DataFrame({"p": pred, "a": actual, "d": dates, "t": tickers}).dropna()
    gb = d.groupby("d")
    ic  = gb.apply(lambda x: x["p"].corr(x["a"]))
    ric = gb.apply(lambda x: x["p"].corr(x["a"], method="spearman"))

    w = weights_from_scores(d["p"].values, d["d"].values)
    pnl = pd.DataFrame({"d": d["d"].values, "x": w * d["a"].values}).groupby("d")["x"].sum()
    turn = turnover_series(w, d["d"].values, d["t"].values).reindex(pnl.index).fillna(0.0)

    row = {
        "model": name, "days": int(len(pnl)),
        "mean_IC": ic.mean(), "IC_t": ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))),
        "rankIC": ric.mean(),
        "gross_bps": pnl.mean() * 1e4,
        "gross_SR": pnl.mean() / pnl.std(ddof=1) * ANN,
        "turnover": turn.mean(),
    }
    for c in cost_bps:
        net = pnl - c * 1e-4 * turn
        row[f"SR@{c:g}bp"] = net.mean() / net.std(ddof=1) * ANN
    row["raw_sum_pXa"] = float((d["p"] * d["a"]).sum())
    return row, pnl, turn, ic
