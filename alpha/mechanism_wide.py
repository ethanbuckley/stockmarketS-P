"""
mechanism_wide.py - re-test the LPS clientele mechanism where it can actually
be tested.

The S&P 500 test was underpowered: institutional breadth spans only 5.9x from
its 5th to 95th percentile there, against 83x across all 13F issuers, and 88%
of the tradeable 13F universe sits below the S&P 500's 5th percentile. This
re-runs it on the extended universe.

Conditioning variable is BREADTH (number of 13F filers), not the ownership
fraction, for two reasons: it needs no shares-outstanding denominator, and it
is therefore immune to the co-manager double counting that forced a cap on io.
Breadth is mechanically size-correlated, so the headline test DOUBLE SORTS on
dollar volume to separate institutional presence from market cap.

SURVIVORSHIP: currently-listed names only, and small-cap delisting rates are
several times large-cap ones. Read every number here as suggestive.
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
from alpha.thirteenf import load_holdings  # noqa: E402
from alpha.wide_panel import build, rank_x  # noqa: E402

TEST = list(range(2020, 2027))
AVAIL_LAG_DAYS = 60


def attach_breadth(df: pd.DataFrame) -> pd.DataFrame:
    h = load_holdings(wide=True)
    mp = pd.read_csv(REPO / "alpha" / "cusip_map_wide.csv", dtype={"cusip6": str})
    q = h.merge(mp[["ticker", "cusip6"]], on="cusip6", how="inner").rename(columns={"ticker": "Ticker"})
    q = (
        q.groupby(["Ticker", "period"], observed=True)
        .agg(breadth=("breadth", "sum"), inst_shares=("inst_shares", "sum"))
        .reset_index()
    )
    q["available"] = (q["period"] + pd.Timedelta(days=AVAIL_LAG_DAYS)).astype("datetime64[ns]")
    q = q.sort_values("available")
    out = pd.merge_asof(
        df.sort_values("Date"),
        q[["Ticker", "available", "breadth"]],
        left_on="Date",
        right_on="available",
        by="Ticker",
        direction="backward",
        tolerance=pd.Timedelta("200D"),
    )
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


def main():
    t0 = time.time()
    df = build()
    print(
        f"panel {len(df):,} rows | {df.Ticker.nunique():,} tickers | "
        f"median {df.groupby('Date').size().median():.0f} names/day",
        flush=True,
    )
    df = attach_breadth(df)
    cov = df["breadth"].notna().mean()
    df = df[df["breadth"].notna()].reset_index(drop=True)
    print(
        f"13F breadth coverage {cov:.1%}; "
        f"breadth p5={df.breadth.quantile(0.05):.0f} p95={df.breadth.quantile(0.95):.0f} "
        f"({df.breadth.quantile(0.95) / max(df.breadth.quantile(0.05), 1):.0f}x range)",
        flush=True,
    )

    X = rank_x(df)
    y = df["y_on"].values
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    p = oos_pred(X, y, years, dates)
    m = np.isin(years, TEST)

    def ter(col):
        return df.groupby("Date", observed=True)[col].transform(
            lambda s: (
                pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])
                if s.notna().sum() >= 30
                else pd.Series(np.nan, index=s.index)
            )
        )

    tb, tv = ter("breadth"), ter("dv")

    print("\n=== overnight IC by institutional breadth (single sort) ===")
    rows = []
    for t in ("low", "mid", "high"):
        sel = m & (tb.values == t)
        r, *_ = evaluate(p[sel], y[sel], dates[sel], tick[sel], f"breadth {t}")
        r["mean_breadth"] = df.loc[sel, "breadth"].mean()
        rows.append(r)
    print(
        pd.DataFrame(rows)[["model", "mean_breadth", "mean_IC", "IC_t", "gross_SR"]].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"
        )
    )

    print("\n=== overnight IC, breadth WITHIN dollar-volume tercile (double sort) ===")
    print("   (isolates institutional presence from size)")
    print(f"{'size tercile':<16}{'breadth low':>14}{'breadth mid':>14}{'breadth high':>14}")
    for v in ("low", "mid", "high"):
        cells = []
        for b in ("low", "mid", "high"):
            sel = m & (tv.values == v) & (tb.values == b)
            if sel.sum() < 5000:
                cells.append("     n/a")
                continue
            r, *_ = evaluate(p[sel], y[sel], dates[sel], tick[sel], "x")
            cells.append(f"{r['mean_IC']:>7.4f} ({r['IC_t']:>4.1f})")
        print(f"{'dv ' + v:<16}" + "".join(f"{c:>14}" for c in cells))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
