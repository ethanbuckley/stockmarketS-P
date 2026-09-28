"""
diagnose_wide.py - is the small-cap overnight "alpha" tradeable, or spread?

mechanism_wide.py finds the overnight effect is 2.4x STRONGER in the least
institutionally-held names (IC 0.063, t=22, gross SR 6.8). Two reasons to
distrust that before believing it:

  1. On the S&P 500 the same signal was strongest in the MOST liquid tertile,
     which is what cleared it of bid-ask bounce. Here the pattern inverts:
     strongest where least liquid. That is the microstructure signature.
  2. Overnight return is Open/PrevClose - 1. In a wide-spread name the close
     can print on the bid and the open on the ask, manufacturing a reversal
     that no one can trade.

Two tests:
  A. EXTRA LAG. Bounce is a one-day effect; push every feature back a day.
  B. SPREAD vs BREAK-EVEN. Estimate the effective spread per bucket with the
     Corwin-Schultz (2012) high-low estimator, which needs only daily highs and
     lows, and compare it to the break-even cost the strategy can pay. If the
     spread exceeds the break-even, the alpha is not reachable.
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
from alpha.lab import evaluate  # noqa: E402
from alpha.mechanism_wide import attach_breadth, oos_pred  # noqa: E402
from alpha.wide_panel import FEATS, build, rank_x  # noqa: E402

TEST = list(range(2020, 2027))


def corwin_schultz(df: pd.DataFrame) -> pd.Series:
    """Proportional effective spread from daily high/low pairs (CS 2012).

    beta uses two consecutive single-day log ranges; gamma uses the two-day
    range. Negative estimates are set to zero, per the paper's own convention.
    """
    g = df.groupby("Ticker", observed=True)
    hl = np.log(df["High"] / df["Low"]) ** 2
    hl2 = hl + g[hl.name if hl.name else "tmp"].shift(1) if False else hl + hl.groupby(df["Ticker"]).shift(1)
    h2 = np.maximum(df["High"], g["High"].shift(1))
    l2 = np.minimum(df["Low"], g["Low"].shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * hl2) - np.sqrt(hl2)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


def decile_book(scores, actual, dates):
    s = pd.Series(scores, index=dates)
    r = s.groupby(level=0).rank(pct=True)
    w = pd.Series(0.0, index=s.index)
    w[r >= 0.9] = 1.0
    w[r <= 0.1] = -1.0
    gr = w.abs().groupby(level=0).transform("sum").replace(0, np.nan)
    return pd.DataFrame({"d": dates, "x": (w / gr).values * actual}).groupby("d")["x"].sum()


def main():
    t0 = time.time()
    df = build()
    df = attach_breadth(df)
    df = df[df["breadth"].notna()].reset_index(drop=True)
    df["spread"] = corwin_schultz(df)

    X = rank_x(df)
    y = df["y_on"].values
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    p = oos_pred(X, y, years, dates)
    m = np.isin(years, TEST)
    tb = df.groupby("Date", observed=True)["breadth"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])
    )

    # ---- A) extra day of lag -------------------------------------------
    Xl = pd.DataFrame(X, columns=FEATS)
    Xl["_t"] = tick
    Xlag = Xl.groupby("_t", observed=True)[FEATS].shift(1).values
    ok = ~np.isnan(Xlag).any(axis=1)
    plag = np.full(len(y), np.nan)
    for yr in TEST:
        te, tr = (years == yr) & ok, (years < yr) & ok
        if tr.sum() == 0 or te.sum() == 0:
            continue
        trd = np.unique(dates[tr])
        tr = tr & (dates < trd[-1])
        plag[te] = Ridge(alpha=100.0).fit(Xlag[tr], y[tr]).predict(Xlag[te])

    print("=== A) does an extra day of lag kill it? (bounce is a 1-day effect) ===")
    print(f"{'breadth bucket':<18}{'IC base':>10}{'IC +1d lag':>13}{'retained':>11}")
    for t in ("low", "mid", "high"):
        sel = m & (tb.values == t)
        r0, *_ = evaluate(p[sel], y[sel], dates[sel], tick[sel], "b")
        s2 = sel & ok
        r1, *_ = evaluate(plag[s2], y[s2], dates[s2], tick[s2], "l")
        print(f"{t:<18}{r0['mean_IC']:>10.4f}{r1['mean_IC']:>13.4f}{r1['mean_IC'] / r0['mean_IC']:>10.0%}")

    # ---- B) spread vs break-even ---------------------------------------
    print("\n=== B) can the strategy pay the spread it must cross? ===")
    print(f"{'breadth bucket':<18}{'gross bps/day':>15}{'break-even/side':>17}{'est. spread/side':>18}{'verdict':>12}")
    for t in ("low", "mid", "high"):
        sel = m & (tb.values == t)
        pnl = decile_book(p[sel], y[sel], dates[sel])
        be = pnl.mean() * 1e4 / 2.0
        # traded names are the extreme deciles; charge their own spreads
        sc = pd.Series(p[sel], index=dates[sel])
        rk = sc.groupby(level=0).rank(pct=True)
        traded = (rk >= 0.9) | (rk <= 0.1)
        half = df.loc[sel, "spread"].values[traded.values] * 1e4 / 2.0
        med = np.nanmedian(half)
        print(
            f"{t:<18}{pnl.mean() * 1e4:>15.3f}{be:>17.2f}{med:>18.1f}{('TRADEABLE' if be > med else 'not viable'):>12}"
        )
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
