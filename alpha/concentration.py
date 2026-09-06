"""
concentration.py - the overnight book lives or dies on its break-even cost
(1.22 bps/side with score-proportional weights). Two ways to move it:

  1. WEIGHTING. Gross traded is fixed at 1.0 in + 1.0 out per day whatever the
     weights, so any weighting that raises alpha per unit of gross raises the
     break-even directly. Tested: score-proportional vs equal-weight extremes.

  2. THE FLIP. The tug of war suggests holding the position overnight and
     REVERSING it during the day, earning both legs. It earns more gross, but
     flipping +w to -w trades 2|w| rather than |w|, so the book trades ~4.0
     gross/day instead of 2.0. Tested rather than assumed.
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
from alpha.lab import ANN  # noqa: E402

TEST = list(range(2020, 2027))


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


def weight(scores, dates, scheme):
    s = pd.Series(scores, index=dates)
    if scheme == "proportional":
        w = s - s.groupby(level=0).transform("mean")
    else:
        q = int(scheme.split("_")[1])
        r = s.groupby(level=0).rank(pct=True)
        w = pd.Series(0.0, index=s.index)
        w[r >= 1 - 1 / q] = 1.0
        w[r <= 1 / q] = -1.0
    g = w.abs().groupby(level=0).transform("sum").replace(0, np.nan)
    return (w / g).values


def main():
    t0 = time.time()
    df, feats = build_panel()
    years = df["Date"].dt.year.values
    dates = df["Date"].values
    X = rank_x(df, feats).values
    m = np.isin(years, TEST)

    p_on = oos_pred(X, df["y_on"].values, years, dates)
    p_id = oos_pred(X, df["y_id"].values, years, dates)
    d = pd.DataFrame({"d": dates[m], "pon": p_on[m], "pid": p_id[m],
                      "ron": df["y_on"].values[m], "rid": df["y_id"].values[m]}).dropna()

    print("=== 1. weighting scheme, overnight-only book (gross traded = 2.0/day) ===")
    print(f"{'scheme':<20}{'gross bps/day':>15}{'gross SR':>10}{'break-even bps/side':>22}")
    best = None
    for scheme in ("proportional", "extreme_10", "extreme_5", "extreme_3"):
        w = weight(d["pon"].values, d["d"].values, scheme)
        pnl = pd.DataFrame({"d": d["d"].values, "x": w * d["ron"].values}).groupby("d")["x"].sum()
        be = pnl.mean() * 1e4 / 2.0
        print(f"{scheme:<20}{pnl.mean()*1e4:>15.3f}{pnl.mean()/pnl.std(ddof=1)*ANN:>10.2f}{be:>22.2f}")
        if best is None or be > best[1]:
            best = (scheme, be)

    print("\n=== 2. the flip: long overnight, reversed intraday ===")
    w_on = weight(d["pon"].values, d["d"].values, "proportional")
    w_id = weight(d["pid"].values, d["d"].values, "proportional")
    pnl_on = pd.DataFrame({"d": d["d"].values, "x": w_on * d["ron"].values}).groupby("d")["x"].sum()
    pnl_id = pd.DataFrame({"d": d["d"].values, "x": w_id * d["rid"].values}).groupby("d")["x"].sum()
    both = pnl_on + pnl_id
    # gross traded: overnight-only swaps 0<->w twice (2.0); the flip swaps
    # w_id <-> w_on twice, and those are near-independent books
    gross_flip = pd.Series(np.abs(w_on - w_id), index=d["d"].values).groupby(level=0).sum().mean() * 2
    print(f"  overnight leg only : {pnl_on.mean()*1e4:6.3f} bps/day  gross traded 2.00/day"
          f"  -> break-even {pnl_on.mean()*1e4/2.0:.2f} bps/side")
    print(f"  flip (both legs)   : {both.mean()*1e4:6.3f} bps/day  gross traded {gross_flip:.2f}/day"
          f"  -> break-even {both.mean()*1e4/gross_flip:.2f} bps/side")
    verdict = "BETTER" if both.mean() * 1e4 / gross_flip > pnl_on.mean() * 1e4 / 2 else "WORSE"
    print(f"\n  the flip earns {both.mean()/pnl_on.mean()-1:+.0%} more gross but trades "
          f"{gross_flip/2.0-1:+.0%} more, so it is {verdict} on break-even cost.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
