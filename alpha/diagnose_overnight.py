"""
diagnose_overnight.py - is the overnight alpha real, or bid-ask bounce?

The worry: y_on = Open_{t+1}/Close_t - 1. If Close_t prints on a downtick, the
next open mechanically bounces up. A reversal feature built from Close_t would
then "predict" that bounce without any tradeable content. LPS guard against this
by using a first-half-hour VWAP instead of the raw open (p.195)
we only have
the raw open, so we test the artefact directly.

Three tests:
  1. EXTRA LAG  - push every feature back one more day. Bounce is a one-day
                  effect, so an artefact dies
                  a real effect mostly survives.
  2. LIQUIDITY  - split by dollar volume. Bounce is largest in the least liquid
                  names
                  real alpha should not be concentrated there.
  3. DROP CLOSE - remove today's close-to-close return and its lags, the
                  features most exposed to a closing-print artefact.
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
from alpha.lab import evaluate  # noqa: E402

TEST = list(range(2020, 2027))


def walk(X, y, years, dates, tick, label, extra_lag=False, tickers=None):
    pred = np.full(len(y), np.nan)
    Xv = X.values
    if extra_lag:  # shift each ticker's feature row forward 1 day
        Xdf = X.copy()
        Xdf["_t"] = tickers
        Xv = Xdf.groupby("_t", observed=True)[list(X.columns)].shift(1).values
    ok = ~np.isnan(Xv).any(axis=1)
    for yr in TEST:
        te = (years == yr) & ok
        tr = (years < yr) & ok
        if tr.sum() == 0 or te.sum() == 0:
            continue
        trd = np.unique(dates[tr])
        tr = tr & (dates < trd[-1])
        m = Ridge(alpha=100.0).fit(Xv[tr], y[tr])
        pred[te] = m.predict(Xv[te])
    mask = np.isin(years, TEST) & ok
    r, *_ = evaluate(pred[mask], y[mask], dates[mask], tick[mask], label)
    return r


def main():
    t0 = time.time()
    df, feats = build_panel()
    X = rank_x(df, feats)
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    y = df["y_on"].values
    rows = [walk(X, y, years, dates, tick, "1. baseline overnight")]

    # --- test 1: one extra day of lag on every feature ---------------------
    rows.append(walk(X, y, years, dates, tick, "2. all features +1 day lag", extra_lag=True, tickers=tick))

    # --- test 3: drop close-to-close return features ------------------------
    drop = [c for c in X.columns if c.startswith(("Return_r", "Return_Lag"))]
    rows.append(walk(X.drop(columns=drop), y, years, dates, tick, f"3. drop {len(drop)} close-return feats"))

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== is the overnight signal an artefact? ===")
    print(res[["model", "mean_IC", "IC_t", "gross_SR", "turnover", "SR@2bp"]].to_string(index=False))

    # --- test 2: liquidity tertiles ----------------------------------------
    dv = df["Close"] * df["Volume"]
    ter = dv.groupby(df["Date"]).transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    pred = np.full(len(y), np.nan)
    Xv = X.values
    for yr in TEST:
        te, tr = years == yr, years < yr
        if tr.sum() == 0:
            continue
        trd = np.unique(dates[tr])
        tr = tr & (dates < trd[-1])
        pred[te] = Ridge(alpha=100.0).fit(Xv[tr], y[tr]).predict(Xv[te])
    mask = np.isin(years, TEST)
    print("\n=== overnight IC by dollar-volume tertile ===")
    lr = []
    for t in ("low", "mid", "high"):
        m = mask & (ter.values == t)
        r, *_ = evaluate(pred[m], y[m], dates[m], tick[m], f"tertile {t}")
        lr.append(r)
    print(pd.DataFrame(lr)[["model", "mean_IC", "IC_t", "gross_SR"]].to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
