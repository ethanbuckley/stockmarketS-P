"""
execution_timing.py - the LPS application that costs nothing.

From the paper's conclusion: institutions trading anomalies at low frequency
"can nonetheless benefit from our results by using them to optimally time their
orders - at the open vs. close of trading".

We already run a close-to-close book and it has to rebalance. Each rebalance can
execute in the closing auction of day t or the opening auction of day t+1. The
difference in fill price between those two choices IS the overnight return. So a
model that forecasts the cross-section of overnight returns can pick the cheaper
side, on trades we were making anyway.

  want to BUY  and overnight return is positive -> buy at the CLOSE (before the rise)
  want to BUY  and overnight return is negative -> buy at the OPEN  (after the fall)
  want to SELL -> the opposite

Gain per unit traded = -sign(trade) * r_overnight when we choose the open.
We only get the predictable part of that, so the realised gain scales with the
overnight model's skill, not with |r_overnight|.
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
from alpha.lab import weights_from_scores  # noqa: E402

TEST = list(range(2020, 2027))
LPS = ["on_lag1","id_lag1","on_5","id_5","on_21","id_21","on_ewma","id_ewma","tug","tug_21"]


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


def main():
    t0 = time.time()
    df, feats = build_panel()
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    base = [f for f in feats if f not in LPS]

    # the book we actually trade: close-to-close, base features, 20-day smoothed
    p_cc = oos_pred(rank_x(df, base).values, df["y_cc"].values, years, dates)
    # the timing model: forecast of tomorrow's overnight return
    p_on = oos_pred(rank_x(df, feats).values, df["y_on"].values, years, dates)

    m = np.isin(years, TEST)
    d = pd.DataFrame({"d": dates[m], "t": tick[m], "p": p_cc[m],
                      "pon": p_on[m], "ron": df["y_on"].values[m]}).dropna()
    d = d.sort_values(["t", "d"])
    d["p"] = d.groupby("t", observed=True)["p"].transform(
        lambda x: x.rolling(20, min_periods=1).mean())

    d["w"] = weights_from_scores(d["p"].values, d["d"].values)
    d = d.sort_values(["t", "d"])
    d["dw"] = d.groupby("t", observed=True)["w"].diff().fillna(d["w"])
    d = d[d["dw"].abs() > 1e-9]

    traded = d["dw"].abs().groupby(d["d"]).sum()
    print(f"book rebalances {traded.mean():.4f} gross/day over {traded.size} days")

    # Choose the open only when the expected gain from waiting is positive.
    # Gain from waiting = -sign(trade) * r_overnight, so route to the open when
    # -sign(trade) * forecast > 0, i.e. sign(trade) * forecast < 0:
    #   buying  and the stock is forecast to fall overnight -> buy at the open
    #   selling and the stock is forecast to rise overnight -> sell at the open
    use_open = np.sign(d["dw"].values) * d["pon"].values < 0
    gain = np.where(use_open, -np.sign(d["dw"].values) * d["ron"].values, 0.0) * d["dw"].abs().values
    daily = pd.Series(gain, index=d["d"].values).groupby(level=0).sum()

    # benchmarks: always close (0 by construction), always open, perfect foresight
    always_open = pd.Series(-np.sign(d["dw"].values) * d["ron"].values * d["dw"].abs().values,
                            index=d["d"].values).groupby(level=0).sum()
    perfect = pd.Series(np.abs(d["ron"].values) * d["dw"].abs().values,
                        index=d["d"].values).groupby(level=0).sum()

    print(f"\n{'execution rule':<34}{'bps/day':>10}{'bps/yr':>10}")
    for name, s in (("always trade at the close", pd.Series(0.0, index=daily.index)),
                    ("always trade at the open", always_open),
                    ("model-timed (open vs close)", daily),
                    ("perfect foresight (ceiling)", perfect)):
        print(f"{name:<34}{s.mean()*1e4:>10.3f}{s.mean()*252*1e4:>10.1f}")

    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
    print(f"\nmodel-timed gain t-stat = {t:.2f}   "
          f"({100*use_open.mean():.0f}% of trades routed to the open)")
    print(f"share of the perfect-foresight ceiling captured: "
          f"{daily.mean()/perfect.mean()*100:.1f}%")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
