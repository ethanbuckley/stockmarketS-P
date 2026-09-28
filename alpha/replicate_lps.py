"""
replicate_lps.py - replicate Table 1 of Lou, Polk & Skouras (2019) on our
S&P 500 sample (2015-2026), i.e. OUT of their 1993-2013 sample period.

Panel A: sort on last month's overnight return -> measure next month's
         overnight and intraday returns of the decile 10-1 spread.
Panel B: same, sorting on last month's intraday return.

Their finding (value-weighted, CRSP, 1993-2013):
    Panel A  overnight 10-1 = +3.47%/mo (t=16.57),  intraday 10-1 = -3.24% (t=-9.34)
    Panel B  intraday  10-1 = +2.19%/mo (t= 6.72),  overnight 10-1 = -1.81% (t=-8.44)

Differences to keep in mind when comparing: our portfolios are equal-weighted
(no market-cap data), the universe is today's S&P 500 applied back (survivorship
bias, and all large-cap), and the sample is 11 years not 21.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def monthly_components(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["r_overnight", "r_intraday"]).copy()
    d = d[d["Close"] > 5.0]  # paper's $5 screen
    d["ym"] = d["Date"].values.astype("datetime64[M]")
    g = d.groupby(["Ticker", "ym"], observed=True)
    m = g.agg(
        on=("r_overnight", lambda s: np.prod(1 + s) - 1),
        id=("r_intraday", lambda s: np.prod(1 + s) - 1),
        n=("r_overnight", "size"),
    ).reset_index()
    return m[m["n"] >= 15]  # a near-complete month


def nw_tstat(x: np.ndarray, lags: int = 12) -> float:
    """Newey-West t-stat on the mean, matching the paper's 12-lag correction."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    T = len(x)
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / T
    v = gamma0
    for L in range(1, min(lags, T - 1) + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * (e[L:] @ e[:-L]) / T
    return mu / np.sqrt(v / T)


def spread(m: pd.DataFrame, sort_on: str) -> pd.DataFrame:
    m = m.sort_values(["Ticker", "ym"]).copy()
    g = m.groupby("Ticker", observed=True)
    m["rank_var"] = g[sort_on].shift(1)  # last month's component
    m = m.dropna(subset=["rank_var"])
    m = m[m.groupby("ym", observed=True)["Ticker"].transform("size") >= 100]
    m["dec"] = m.groupby("ym", observed=True)["rank_var"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1
    )

    out = {}
    for leg in ("on", "id"):
        p = m.pivot_table(index="ym", columns="dec", values=leg, aggfunc="mean")
        out[leg] = pd.DataFrame({"d1": p[1], "d10": p[10], "spread": p[10] - p[1]})
    return out


def report(m: pd.DataFrame):
    for panel, sort_on, label in (("A", "on", "overnight"), ("B", "id", "intraday")):
        res = spread(m, sort_on)
        print(f"\n--- Panel {panel}: portfolios sorted on last month's {label} return ---")
        print(f"{'':<22}{'decile 1':>11}{'decile 10':>11}{'10-1':>11}{'t(10-1)':>10}")
        for leg, name in (("on", "Overnight"), ("id", "Intraday")):
            r = res[leg] * 100
            print(
                f"  next-month {name:<10}{r['d1'].mean():>10.2f}%{r['d10'].mean():>10.2f}%"
                f"{r['spread'].mean():>10.2f}%{nw_tstat(res[leg]['spread'].values):>10.2f}"
            )
        n = len(res["on"])
        print(f"  months = {n}")


if __name__ == "__main__":
    df = pd.read_parquet(REPO / "alpha" / "ohlc_open.parquet")
    m = monthly_components(df)
    print(f"panel: {len(m):,} ticker-months | {m.Ticker.nunique()} tickers | {m.ym.min()} -> {m.ym.max()}")
    report(m)
