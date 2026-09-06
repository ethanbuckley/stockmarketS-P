"""
decomposed.py - the actionable test from Lou, Polk & Skouras (2019).

The paper's core claim, transposed to our daily pipeline: close-to-close return
is the SUM of two components whose predictability has OPPOSITE sign, so a model
trained on close-to-close is fitting the residual of two partly-cancelling
signals. Predict the legs separately instead.

Targets:  next-day overnight (close_t -> open_t+1)
          next-day intraday  (open_t+1 -> close_t+1)
          next-day close-to-close (the existing benchmark)

Every feature is known by the close of day t, so all three are executable.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alpha.lab import ANN, evaluate  # noqa: E402

BASE_FEATURES = ["RSI","MACD","BB_Position","Price_to_VWAP","ATR_Ratio","Return",
                 "Volume_Surge","Rel_SPY","Rel_QQQ","Rel_SMH",
                 "Return_Lag_1","Return_Lag_2","Return_Lag_3"]
MARKET_WIDE = ["Day_Of_Week","SPY_Return","QQQ_Return","SMH_Return","VIX_Change",
               "TNX_Change","QQQ_Lag_1","QQQ_Lag_2","QQQ_Lag_3"]


def build_panel() -> pd.DataFrame:
    on = pd.read_parquet(REPO / "alpha" / "ohlc_open.parquet")
    on["Date"] = pd.to_datetime(on["Date"])
    with open(REPO / "master_cache.pkl", "rb") as f:
        cache, _ = pickle.load(f)
    cache = cache.reset_index()
    cache.columns = ["Date"] + list(cache.columns[1:])
    cache["Date"] = pd.to_datetime(cache["Date"])

    df = on.merge(cache[["Date","Ticker"] + BASE_FEATURES + MARKET_WIDE],
                  on=["Date","Ticker"], how="inner")
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)
    g = df.groupby("Ticker", observed=True)

    # ---- overnight/intraday history features (all backward-looking) --------
    for col, tag in (("r_overnight","on"), ("r_intraday","id")):
        df[f"{tag}_lag1"] = g[col].shift(1)
        df[f"{tag}_5"]    = g[col].transform(lambda s: s.shift(1).rolling(5).sum())
        df[f"{tag}_21"]   = g[col].transform(lambda s: s.shift(1).rolling(21).sum())
        df[f"{tag}_ewma"] = g[col].transform(lambda s: s.shift(1).ewm(halflife=60, min_periods=21).mean())
    # the paper's firm-level tug of war
    df["tug"] = df["on_ewma"] - df["id_ewma"]
    df["tug_21"] = df["on_21"] - df["id_21"]

    # ---- targets: next day's two legs and their compound -------------------
    cal = pd.Index(sorted(df["Date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    df["dpos"] = df["Date"].map(pos)
    nxt = g["dpos"].shift(-1)
    ok = (nxt - df["dpos"]) == 1
    for col, tag in (("r_overnight","y_on"), ("r_intraday","y_id"), ("r_cc","y_cc")):
        df[tag] = g[col].shift(-1).where(ok)

    df = df[(df["Close"] > 5.0) & (df["Volume"] > 0)]
    feats = BASE_FEATURES + MARKET_WIDE + [
        "on_lag1","id_lag1","on_5","id_5","on_21","id_21","on_ewma","id_ewma","tug","tug_21"]
    df = df.dropna(subset=feats + ["y_on","y_id","y_cc"])

    # winsorise + cross-sectionally demean each target
    for t in ("y_on","y_id","y_cc"):
        df[t] = df.groupby("Date", observed=True)[t].transform(
            lambda s: s.clip(s.quantile(0.005), s.quantile(0.995)))
        df[t] = df[t] - df.groupby("Date", observed=True)[t].transform("mean")

    df = df[df.groupby("Date", observed=True)["Ticker"].transform("size") >= 100]
    return df.reset_index(drop=True), feats


def rank_x(df, feats):
    cs = [f for f in feats if f not in MARKET_WIDE]
    out = df[MARKET_WIDE].copy()
    r = df.groupby(df["Date"], observed=True)[cs].rank(pct=True) - 0.5
    for c in cs:
        out[c + "_r"] = r[c].values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=100.0)
    a = ap.parse_args()
    from sklearn.linear_model import Ridge

    t0 = time.time()
    df, feats = build_panel()
    print(f"panel {len(df):,} rows | {df.Ticker.nunique()} tickers | "
          f"{df.Date.min().date()} -> {df.Date.max().date()} ({time.time()-t0:.0f}s)", flush=True)
    X = rank_x(df, feats)
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    TEST = [y for y in range(2020, 2027) if (years == y).sum() > 0]

    rows = []
    preds_store = {}
    for target in ("y_on", "y_id", "y_cc"):
        y = df[target].values
        pred = np.full(len(df), np.nan)
        for yr in TEST:
            te, tr = years == yr, years < yr
            if tr.sum() == 0:
                continue
            trd = np.unique(dates[tr])
            tr = tr & (dates < trd[-1])
            m = Ridge(alpha=a.alpha).fit(X.values[tr], y[tr])
            pred[te] = m.predict(X.values[te])
        mask = np.isin(years, TEST)
        preds_store[target] = (pred, mask)
        r, pnl, turn, ic = evaluate(pred[mask], y[mask], dates[mask], tick[mask], target)
        rows.append(r)

    # the composite book: trade the overnight leg on its own signal and the
    # intraday leg on its own, i.e. two books instead of one close-to-close book
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== next-day prediction, by return component (OOS 2020-2026) ===")
    print(res[["model","days","mean_IC","IC_t","gross_bps","gross_SR","turnover",
               "SR@1bp","SR@2bp","SR@5bp"]].to_string(index=False))
    res.to_csv(REPO / "alpha" / "results" / "decomposed.csv", index=False)

    # combined: sum of the two component books' daily pnl
    from alpha.lab import weights_from_scores
    mask = preds_store["y_cc"][1]
    tot = None
    for t in ("y_on", "y_id"):
        p = preds_store[t][0][mask]
        w = weights_from_scores(p, dates[mask])
        pn = pd.DataFrame({"d": dates[mask], "x": w * df[t].values[mask]}).groupby("d")["x"].sum()
        tot = pn if tot is None else tot + pn
    print(f"\ncombined two-leg book: {tot.mean()*1e4:.3f} bps/day  "
          f"gross SR = {tot.mean()/tot.std(ddof=1)*ANN:.2f}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
