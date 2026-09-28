"""
pit_analysis.py - how much did survivorship bias inflate our results?

Applies point-in-time S&P 500 membership to the daily panel and re-runs the
headline tests. Membership snapshots are quarterly; a stock counts as a member
from the snapshot that first lists it until the snapshot that drops it, matched
as-of so no future membership information leaks backwards.

This corrects the PRE-INCLUSION half of the bias: without it, a company that
joined the index in 2022 contributes its 2015-2021 history, which it earned
while growing into the index. It does NOT correct the DELISTED half, because
Yahoo serves no history for removed tickers. The coverage report below bounds
what is left uncorrected.
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
from alpha.source import membership_file  # noqa: E402


def membership_coverage(panel_tickers: set[str]) -> pd.DataFrame:
    m = pd.read_parquet(membership_file())
    m = m[m["asof"] >= "2015-01-01"]
    ever = set(m["Ticker"])
    have = ever & panel_tickers
    miss = ever - panel_tickers
    print(f"PIT members ever, 2015-2026 : {len(ever)}")
    print(f"  with price history        : {len(have)} ({len(have) / len(ever):.0%})")
    print(f"  MISSING (delisted/renamed): {len(miss)} ({len(miss) / len(ever):.0%})")
    by = m.assign(has=m["Ticker"].isin(panel_tickers)).groupby(m["asof"].dt.year)["has"].agg(["size", "mean"])
    by.columns = ["member-quarters", "price coverage"]
    print("\ncoverage of point-in-time members by year:")
    print(by.to_string(float_format=lambda v: f"{v:.1%}"))
    return m


def attach_membership(df: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """as-of join: each day uses the most recent snapshot at or before it."""
    m = m[["asof", "Ticker"]].copy()
    m["asof"] = m["asof"].astype("datetime64[ns]")
    m["is_member"] = True
    df = df.copy()
    df["Date"] = df["Date"].astype("datetime64[ns]")
    out = pd.merge_asof(
        df.sort_values("Date"),
        m.sort_values("asof"),
        left_on="Date",
        right_on="asof",
        by="Ticker",
        direction="backward",
        tolerance=pd.Timedelta("200D"),
    )
    out["is_member"] = out["is_member"].fillna(False).astype(bool)
    return out


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


def decile_book(scores, actual, dates):
    s = pd.Series(scores, index=dates)
    r = s.groupby(level=0).rank(pct=True)
    w = pd.Series(0.0, index=s.index)
    w[r >= 0.9] = 1.0
    w[r <= 0.1] = -1.0
    g = w.abs().groupby(level=0).transform("sum").replace(0, np.nan)
    return pd.DataFrame({"d": dates, "x": (w / g).values * actual}).groupby("d")["x"].sum()


def main():
    t0 = time.time()
    df, feats = build_panel()
    m = membership_coverage(set(df["Ticker"].unique()))
    df = attach_membership(df, m)
    share = df["is_member"].mean()
    print(f"\npanel rows kept as point-in-time members: {share:.1%} ({df['is_member'].sum():,} of {len(df):,})")

    X = rank_x(df, feats).values
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values

    print("\n=== headline tests, all names vs point-in-time members ===")
    print(
        f"{'target':<10}{'universe':<16}{'names/day':>10}{'IC':>9}{'IC t':>8}"
        f"{'gross bps':>11}{'gross SR':>10}{'break-even':>12}"
    )
    for target in ("y_cc", "y_on"):
        y = df[target].values
        for label, keep in (("all names", np.ones(len(df), bool)), ("PIT members", df["is_member"].values)):
            sub = keep & np.isin(years, TEST)
            p = oos_pred(X[keep], y[keep], years[keep], dates[keep])
            full = np.full(len(df), np.nan)
            full[keep] = p
            r, *_ = evaluate(full[sub], y[sub], dates[sub], tick[sub], target)
            pnl = decile_book(full[sub], y[sub], dates[sub])
            n = pd.Series(dates[sub]).value_counts().mean()
            print(
                f"{target:<10}{label:<16}{n:>10.0f}{r['mean_IC']:>9.4f}{r['IC_t']:>8.2f}"
                f"{pnl.mean() * 1e4:>11.3f}{r['gross_SR']:>10.2f}"
                f"{pnl.mean() * 1e4 / 2:>12.2f}"
            )
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
