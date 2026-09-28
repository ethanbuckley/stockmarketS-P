"""
tradeable.py - the two questions that decide whether LPS is useful to us.

A) Do the overnight/intraday features improve the CLOSE-TO-CLOSE book we can
   already trade cheaply?  (same panel, with and without them)

B) Is an OVERNIGHT-ONLY book tradeable?  This needs different cost accounting
   from a close-to-close book, and getting it wrong flatters the strategy
   enormously:

     close-to-close book: hold the position through the day, trade only the
       change in weights -> cost = c * turnover, turnover ~0.5-0.8/day.
     overnight-only book: must be FLAT during the day to avoid the intraday
       leg, so it buys 1.0 gross at the close and sells 1.0 gross at the open,
       every day -> cost = (c_close + c_open) * 1.0, i.e. ~2.0 gross traded/day
       regardless of how slowly the signal moves.

   Smoothing therefore cannot rescue an overnight book the way it rescues a
   close-to-close one: smoothing cuts rebalancing turnover, not the daily
   in-and-out.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alpha.decomposed import build_panel, rank_x  # noqa: E402
from alpha.lab import ANN, turnover_series, weights_from_scores  # noqa: E402

TEST = list(range(2020, 2027))
LPS = ["on_lag1", "id_lag1", "on_5", "id_5", "on_21", "id_21", "on_ewma", "id_ewma", "tug", "tug_21"]


def oos_pred(X, y, years, dates):
    p = np.full(len(y), np.nan)
    for yr in TEST:
        te, tr = years == yr, years < yr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        trd = np.unique(dates[tr])
        tr = tr & (dates < trd[-1])
        p[te] = Ridge(alpha=100.0).fit(X[tr], y[tr]).predict(X[te])
    return p


def book(pred, actual, dates, tickers, smooth=1):
    d = pd.DataFrame({"p": pred, "a": actual, "d": dates, "t": tickers}).dropna()
    if smooth > 1:
        d = d.sort_values(["t", "d"])
        d["p"] = d.groupby("t", observed=True)["p"].transform(lambda x: x.rolling(smooth, min_periods=1).mean())
    w = weights_from_scores(d["p"].values, d["d"].values)
    pnl = pd.DataFrame({"d": d["d"].values, "x": w * d["a"].values}).groupby("d")["x"].sum()
    turn = turnover_series(w, d["d"].values, d["t"].values).reindex(pnl.index).fillna(0)
    ic = d.groupby("d").apply(lambda x: x["p"].corr(x["a"]))
    return pnl, turn, ic


def sr(x):
    return x.mean() / x.std(ddof=1) * ANN


def main():
    t0 = time.time()
    df, feats = build_panel()
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    base = [f for f in feats if f not in LPS]

    Xall = rank_x(df, feats)
    Xbase = rank_x(df, base)
    print(f"panel {len(df):,} rows | base feats {Xbase.shape[1]} | +LPS {Xall.shape[1]}", flush=True)

    # ---- A) does LPS help the close-to-close book? ------------------------
    print("\n=== A) close-to-close book, same panel, with vs without LPS features ===")
    print(
        f"{'features':<26}{'IC':>9}{'IC t':>8}{'gross bps':>11}{'turn':>7}"
        f"{'SR 0bp':>9}{'SR 1bp':>9}{'SR 2bp':>9}{'SR 5bp':>9}"
    )
    ycc = df["y_cc"].values
    cc_preds = {}
    for name, X in (("base only", Xbase), ("base + LPS", Xall)):
        p = oos_pred(X.values, ycc, years, dates)
        cc_preds[name] = p
        for sm in (1, 20):
            m = np.isin(years, TEST)
            pnl, turn, ic = book(p[m], ycc[m], dates[m], tick[m], smooth=sm)
            tag = f"{name}{'' if sm == 1 else f' (smooth {sm}d)'}"
            row = (
                f"{tag:<26}{ic.mean():>9.4f}{ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))):>8.2f}"
                f"{pnl.mean() * 1e4:>11.3f}{turn.mean():>7.3f}"
            )
            for c in (0, 1, 2, 5):
                row += f"{sr(pnl - c * 1e-4 * turn):>9.2f}"
            print(row)

    # ---- B) overnight-only book, correct cost accounting -------------------
    print("\n=== B) overnight-only book: buy at close, sell at open, every day ===")
    yon = df["y_on"].values
    p_on = oos_pred(Xall.values, yon, years, dates)
    m = np.isin(years, TEST)
    pnl, turn, ic = book(p_on[m], yon[m], dates[m], tick[m])
    print(
        f"  IC = {ic.mean():.4f} (t={ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))):.2f})"
        f"   gross = {pnl.mean() * 1e4:.3f} bps/day   gross SR = {sr(pnl):.2f}"
    )
    print(f"  rebalancing turnover would be {turn.mean():.3f}/day, but the book must go")
    print("  flat each morning, so it trades 1.0 gross in + 1.0 gross out = 2.0/day.\n")
    print(f"  {'cost per side (bps)':<22}{'daily cost':>12}{'net bps/day':>14}{'net SR':>9}")
    for c in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        cost = 2.0 * c * 1e-4
        net = pnl - cost
        print(f"  {c:<22.2f}{cost * 1e4:>11.2f}b{net.mean() * 1e4:>14.3f}{sr(net):>9.2f}")
    print(f"\n  break-even one-way cost = {pnl.mean() * 1e4 / 2:.2f} bps per side")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
