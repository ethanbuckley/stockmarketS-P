"""
pit_screener.py - what survivorship bias does to the SCREENER's headline numbers.

The root README's universe is point-in-time by symbol, rebuilt from Wikipedia's
change history, but Yahoo serves no prices for about half of the removed
companies, so they are missing. CRSP has exact daily S&P 500 membership and
prices for every member, delisted or not, so the full bias can be measured.

The experiment isolates the bias and nothing else. Both arms use the same CRSP
prices, the same feature and label code imported from screener.py, the same
walk-forward evaluation from evaluate.py, and the same date range. Only the
universe differs:

  current-members : today's index membership applied to every past date, the
                    universe the README reported before it went point-in-time
  point-in-time   : the index as it actually stood on each date, including
                    members that were later removed or delisted

Why this matters more here than for the alpha book: Test B found the delisted
half worth ~nothing for a dollar-neutral decile book. Precision@15 is a
concentrated LONG selection, which is exactly the case where survivorship bias
is expected to bite hardest, so that result does not transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluate import run_evaluation  # noqa: E402
from screener import process_ticker  # noqa: E402

CRSP_PRICES = REPO / "alpha" / "ohlc_compustat.parquet"  # holds CRSP data
CRSP_MEMB = REPO / "alpha" / "sp500_membership_compustat.parquet"
MACRO = ["SPY", "QQQ", "SMH", "^VIX", "^TNX"]
OUT = REPO / "alpha" / "results" / "pit_screener.json"


def macro_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Macro series in the wide shape build_macro_features expects. ETFs and
    indices do not delist, so yfinance is fine for these."""
    import yfinance as yf

    d = yf.download(
        MACRO, start="2014-11-01", end=str(index.max().date()), auto_adjust=True, progress=False, threads=False
    )
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d.reindex(index).ffill()


def wide_from_crsp(px: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """(field, ticker) MultiIndex columns, the shape process_ticker reads."""
    parts = {}
    for field in ("Close", "High", "Low", "Volume"):
        parts[field] = px.pivot_table(index="Date", columns="Ticker", values=field)
    wide = pd.concat(parts, axis=1)
    wide.index = pd.to_datetime(wide.index).tz_localize(None)
    tickers = sorted(parts["Close"].columns)

    mac = macro_frame(wide.index)
    for field in ("Close", "High", "Low", "Volume"):
        if field in mac.columns.get_level_values(0):
            for m in MACRO:
                if m in mac[field].columns:
                    wide[(field, m)] = mac[field][m].values
    return wide.sort_index(axis=1), tickers


def build_master(wide: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frames = []
    for i, t in enumerate(tickers, 1):
        try:
            frames.append(process_ticker(t, wide))
        except Exception as e:
            print(f"  {t}: {type(e).__name__}", flush=True)
        if i % 200 == 0:
            print(f"  features {i}/{len(tickers)}", flush=True)
    return pd.concat(frames)


def universes(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    memb = pd.read_parquet(CRSP_MEMB)
    memb["asof"] = pd.to_datetime(memb["asof"]).astype("datetime64[ns]")

    # today's membership applied to every past date = the published bias
    current = set(memb.loc[memb["asof"] == memb["asof"].max(), "Ticker"])
    biased = master[master["Ticker"].isin(current)]

    # the index as it actually stood each day
    memb = memb.rename(columns={"asof": "Date"})
    memb["is_member"] = True
    m = master.reset_index().rename(columns={"index": "Date"})
    m["Date"] = pd.to_datetime(m["Date"]).astype("datetime64[ns]")
    m = m.merge(memb[["Date", "Ticker", "is_member"]], on=["Date", "Ticker"], how="left")
    pit = m[m["is_member"].fillna(False).astype(bool)].drop(columns=["is_member"])
    pit = pit.set_index("Date").sort_index()
    return {"current-members": biased, "point-in-time": pit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test-year", type=int, default=2020)
    a = ap.parse_args()
    t0 = time.time()

    px = pd.read_parquet(CRSP_PRICES)
    print(
        f"CRSP prices: {len(px):,} rows, {px.Ticker.nunique()} names, {px.Date.min().date()} -> {px.Date.max().date()}",
        flush=True,
    )
    wide, tickers = wide_from_crsp(px)
    print(f"building features for {len(tickers)} names ...", flush=True)
    master = build_master(wide, tickers)
    print(f"master: {len(master):,} rows ({time.time() - t0:.0f}s)", flush=True)

    out = {}
    for name, frame in universes(master).items():
        print(
            f"\n=== {name}: {len(frame):,} rows, "
            f"{frame['Ticker'].nunique()} names, "
            f"{frame.groupby(level=0).size().median():.0f}/day ===",
            flush=True,
        )
        res, daily, _, _ = run_evaluation(frame, first_test_year=a.first_test_year, causality_check=False)
        d = res["overall_daily"]
        out[name] = {
            "rows": int(len(frame)),
            "names": int(frame["Ticker"].nunique()),
            "auc_pooled": res["pooled"]["roc_auc"],
            "precision_top15": d["precision_top15_mean"],
            "base_rate": d["base_rate_daily_mean"],
            "excess": d["excess_precision_top15_mean"],
            "excess_ci95": d["excess_precision_top15_ci95"],
            "frac_days_beats_base": d["frac_days_top15_beats_base"],
            "top_decile_lift": d.get("top_decile_lift_mean"),
            "daily_auc": d["daily_auc_mean"],
            "daily_auc_ci95": d["daily_auc_ci95"],
            "n_days": int(len(daily)),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    b, p = out["current-members"], out["point-in-time"]
    print("\n" + "=" * 72)
    print("WHAT SURVIVORSHIP BIAS IS WORTH TO THE SCREENER'S HEADLINE NUMBERS")
    print("=" * 72)
    print(f"{'metric':<28}{'current-members':>18}{'point-in-time':>16}{'change':>10}")
    for label, key, fmt in (
        ("ROC AUC (pooled)", "auc_pooled", "{:.3f}"),
        ("Per-day AUC", "daily_auc", "{:.3f}"),
        ("Precision@15", "precision_top15", "{:.1%}"),
        ("Base rate", "base_rate", "{:.1%}"),
        ("Excess over base", "excess", "{:+.1%}"),
        ("Days beating base", "frac_days_beats_base", "{:.1%}"),
    ):
        bv, pv = b[key], p[key]
        print(f"{label:<28}{fmt.format(bv):>18}{fmt.format(pv):>16}{(pv - bv):>+10.3f}")
    print(f"\nexcess 95% CI  current-members {b['excess_ci95']}")
    print(f"               point-in-time   {p['excess_ci95']}")
    print(f"\nwrote {OUT}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
