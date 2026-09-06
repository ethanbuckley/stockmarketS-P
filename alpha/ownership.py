"""
ownership.py - assemble the institutional ownership panel and attach it to the
daily price panel without lookahead.

TIMING IS THE WHOLE GAME HERE. A 13F for quarter-end Q is not public until up
to 45 days later, so using it on any date before that is lookahead bias and
would manufacture a signal out of nothing. We use Q + 60 days as the
availability date, which is strictly more conservative than the statutory
deadline, and merge as-of on that date rather than on the report date.

Measures built per ticker-quarter:
  io          institutional shares / shares outstanding  (the standard level)
  breadth     number of distinct 13F filers holding the name (Chen/Hong/Stein)
  d_io        quarter-on-quarter change in io
  d_breadth   quarter-on-quarter change in breadth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alpha.thirteenf import load_holdings  # noqa: E402

AVAIL_LAG_DAYS = 60


def quarterly_panel() -> pd.DataFrame:
    # load_holdings() counts each manager once per (period, issuer) at their
    # LATEST filing, which is what makes breadth and inst_shares correct across
    # SEC's filing-window files and across 13F-HR/A amendments.
    own = load_holdings()
    mp = pd.read_csv(REPO / "alpha" / "cusip_map.csv", dtype={"cusip6": str})

    q = (own.merge(mp[["ticker", "cusip6"]], on="cusip6", how="inner")
            .groupby(["ticker", "period"], observed=True)
            .agg(inst_shares=("inst_shares", "sum"),
                 breadth=("breadth", "sum"),
                 inst_dollars=("inst_dollars", "sum"))
            .reset_index()
            .rename(columns={"ticker": "Ticker"}))

    so = pd.read_parquet(REPO / "alpha" / "shares_outstanding.parquet")
    # merge_asof refuses mismatched datetime resolutions (us vs ms)
    q["period"] = q["period"].astype("datetime64[ns]")
    so["Date"] = so["Date"].astype("datetime64[ns]")
    q = q.sort_values(["Ticker", "period"])
    so = so.sort_values(["Ticker", "Date"])
    q = pd.merge_asof(q.sort_values("period"),
                      so.rename(columns={"Date": "period"}).sort_values("period"),
                      on="period", by="Ticker", direction="nearest",
                      tolerance=pd.Timedelta("120D"))

    # 13F CO-MANAGER DOUBLE COUNTING. When several managers share investment
    # discretion over a block, each may report it, so summed institutional
    # shares can exceed shares outstanding. It is systematic per name (ON
    # Semiconductor reads ~1.2-1.35 in every quarter), not a date misalignment.
    # INFOTABLE carries INVESTMENTDISCRETION/OTHERMANAGER to resolve this
    # properly; we do not keep those fields, so we cap instead and rely on
    # RANK-based conditioning downstream, which tolerates a monotonic
    # distortion. `breadth` needs no denominator and is the robustness check.
    q["io_raw"] = q["inst_shares"] / q["shares"]
    q["io_capped"] = q["io_raw"] > 1.0
    q["io"] = q["io_raw"].clip(upper=1.0)
    q.loc[q["io_raw"] <= 0.01, "io"] = np.nan   # failed denominator or no match
    g = q.groupby("Ticker", observed=True)
    q["d_io"] = g["io"].diff()
    q["d_breadth"] = g["breadth"].pct_change()
    q["available"] = q["period"] + pd.Timedelta(days=AVAIL_LAG_DAYS)
    return q.dropna(subset=["io"]).sort_values("available").reset_index(drop=True)


def attach(daily: pd.DataFrame) -> pd.DataFrame:
    """as-of merge: each day sees only 13F data already public on that day."""
    q = quarterly_panel()
    cols = ["Ticker", "available", "io", "io_raw", "io_capped", "breadth", "d_io", "d_breadth"]
    daily = daily.copy()
    daily["Date"] = daily["Date"].astype("datetime64[ns]")
    q["available"] = q["available"].astype("datetime64[ns]")
    left = daily.sort_values("Date")
    out = pd.merge_asof(left, q[cols].sort_values("available"),
                        left_on="Date", right_on="available", by="Ticker",
                        direction="backward", tolerance=pd.Timedelta("200D"))
    out["ownership_age_days"] = (out["Date"] - out["available"]).dt.days
    return out


if __name__ == "__main__":
    q = quarterly_panel()
    print(f"ticker-quarters={len(q):,}  tickers={q.Ticker.nunique()}  "
          f"{q.period.min().date()} -> {q.period.max().date()}")
    print("\ninstitutional ownership (io) distribution:")
    print(q["io"].describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
    print("\nbreadth (number of 13F filers):")
    print(q["breadth"].describe(percentiles=[.05, .5, .95]).round(0).to_string())
    print("\nhighest and lowest io, latest quarter:")
    last = q[q.period == q.period.max()].sort_values("io")
    print("  low :", ", ".join(f"{r.Ticker}={r.io:.2f}" for r in last.head(4).itertuples()))
    print("  high:", ", ".join(f"{r.Ticker}={r.io:.2f}" for r in last.tail(4).itertuples()))
