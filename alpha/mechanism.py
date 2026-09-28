"""
mechanism.py - does institutional ownership condition the overnight effect?

Lou, Polk & Skouras argue the overnight/intraday tug of war is a CLIENTELE
phenomenon: institutions trade at and near the close, individuals near the
open. Their Table 8 shows the momentum tug of war is strongest where
institutional active weight is high (overnight spread 1.15%/mo, t=5.39).

If that mechanism is right on our data, the overnight signal should be stronger
in stocks institutions dominate. That is not just a science question: if the
effect concentrates, we can trade only the names where it lives, which raises
alpha per unit of gross traded and therefore raises the break-even cost that
decides whether any of this is usable.

Three tests:
  A. overnight IC within institutional-ownership terciles
  B. do ownership features improve the overnight model?
  C. does restricting the book to high-ownership names raise break-even cost?
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
from alpha.ownership import attach  # noqa: E402

TEST = list(range(2020, 2027))
OWN_FEATS = ["io", "breadth", "d_io", "d_breadth"]


def oos_pred(X, y, years, dates):
    p = np.full(len(y), np.nan)
    ok = ~np.isnan(X).any(axis=1)
    for yr in TEST:
        te = (years == yr) & ok
        tr = (years < yr) & ok
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
    w = w / g
    return pd.DataFrame({"d": dates, "x": w.values * actual}).groupby("d")["x"].sum()


def main():
    t0 = time.time()
    df, feats = build_panel()
    df = attach(df)
    cov = df["io"].notna().mean()
    print(
        f"panel {len(df):,} rows | ownership coverage {cov:.1%} | "
        f"median staleness {df['ownership_age_days'].median():.0f} days",
        flush=True,
    )
    df = df[df["io"].notna()].reset_index(drop=True)

    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    y = df["y_on"].values
    Xbase = rank_x(df, feats).values
    p = oos_pred(Xbase, y, years, dates)
    m = np.isin(years, TEST)

    # ---- A) is the effect concentrated where institutions dominate? -------
    # Two independent measures. io needs shares outstanding as a denominator
    # and is distorted by 13F co-manager double counting; breadth (number of
    # filers) needs no denominator at all. If both agree, the finding does not
    # rest on the denominator.
    terciles = {}
    for var in ("io", "breadth"):
        terciles[var] = df.groupby("Date", observed=True)[var].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])
        )
    print(f"\n(io capped at 1.0 for {df['io_capped'].mean():.1%} of rows; breadth is the denominator-free check)")
    for var in ("io", "breadth"):
        print(f"\n=== A) overnight IC by {var} tercile ===")
        rows = []
        for t in ("low", "mid", "high"):
            sel = m & (terciles[var].values == t)
            r, *_ = evaluate(p[sel], y[sel], dates[sel], tick[sel], f"{var} {t}")
            r["mean_" + var] = df.loc[sel, var].mean()
            rows.append(r)
        print(
            pd.DataFrame(rows)[["model", "mean_" + var, "mean_IC", "IC_t", "gross_SR"]].to_string(
                index=False, float_format=lambda v: f"{v:,.4f}"
            )
        )
    ter = terciles["io"]

    # ---- B) do ownership features improve the model? -----------------------
    print("\n=== B) overnight model, with and without ownership features ===")
    Xown = rank_x(df, feats + OWN_FEATS).values
    rows = []
    for name, X in (("base", Xbase), ("base + ownership", Xown)):
        pp = oos_pred(X, y, years, dates)
        r, *_ = evaluate(pp[m], y[m], dates[m], tick[m], name)
        rows.append(r)
    print(
        pd.DataFrame(rows)[["model", "mean_IC", "IC_t", "gross_bps", "gross_SR"]].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"
        )
    )

    # ---- C) does restricting to high-ownership names pay? ------------------
    print("\n=== C) overnight book, decile weights, gross traded 2.0/day ===")
    print(f"{'universe':<28}{'names/day':>11}{'gross bps/day':>15}{'break-even bps/side':>22}")
    for label, sel in (
        ("all names", m),
        ("io high tercile", m & (ter.values == "high")),
        ("io mid+high", m & (ter.values != "low")),
        ("io low tercile", m & (ter.values == "low")),
    ):
        if sel.sum() == 0:
            continue
        pnl = decile_book(p[sel], y[sel], dates[sel])
        n = pd.Series(dates[sel]).value_counts().mean()
        print(f"{label:<28}{n:>11.0f}{pnl.mean() * 1e4:>15.3f}{pnl.mean() * 1e4 / 2:>22.2f}")

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
