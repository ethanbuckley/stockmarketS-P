"""Expanding-window walk-forward over calendar-year test blocks."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from alpha.lab import evaluate, load_panel, rank_features

TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def build_models(seed=0):
    import xgboost as xgb
    from sklearn.linear_model import Ridge

    return {
        "ridge": lambda: Ridge(alpha=100.0),
        "xgb": lambda: xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="fwd_ret_cs", choices=["fwd_ret_cs", "fwd_ret"])
    ap.add_argument("--models", default="reversal,ridge,xgb,shuffle")
    ap.add_argument("--quick", action="store_true", help="2024-2025 only, small xgb")
    ap.add_argument("--out", default="wf_results")
    args = ap.parse_args()

    t0 = time.time()
    panel, meta = load_panel()
    print(
        f"panel: {len(panel):,} rows | {panel.Ticker.nunique()} tickers | "
        f"{panel.Date.min().date()} -> {panel.Date.max().date()} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    X = rank_features(panel)
    y = panel[args.target].values
    dates = panel["Date"].values
    tick = panel["Ticker"].values
    years = panel["Date"].dt.year.values

    want = args.models.split(",")
    years_run = [2024, 2025] if args.quick else TEST_YEARS
    factories = build_models()
    if args.quick:
        import xgboost as xgb

        factories["xgb"] = lambda: xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=50,
            tree_method="hist",
            n_jobs=-1,
            random_state=0,
            verbosity=0,
        )

    preds = {m: np.full(len(panel), np.nan) for m in want}
    rng = np.random.default_rng(7)

    for yr in years_run:
        te = years == yr
        tr = years < yr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        # purge: label is next-day, so drop the final training day
        tr_dates = np.unique(dates[tr])
        tr = tr & (dates < tr_dates[-1])
        Xtr, ytr = X.values[tr], y[tr]
        Xte = X.values[te]
        print(f"  fold {yr}: train {tr.sum():,} / test {te.sum():,}", flush=True)

        if "reversal" in want:  # no fitting: pure baseline
            preds["reversal"][te] = -X["Return_r"].values[te]
        if "shuffle" in want:  # null control
            preds["shuffle"][te] = rng.permutation(X["Return_r"].values[te])
        for m in want:
            if m in ("reversal", "shuffle"):
                continue
            t1 = time.time()
            mdl = factories[m]()
            mdl.fit(Xtr, ytr)
            preds[m][te] = mdl.predict(Xte)
            print(f"     {m}: {time.time() - t1:.0f}s", flush=True)

    mask = np.isin(years, years_run)
    # dump OOS predictions for downstream turnover/smoothing analysis
    dump = []
    for m in want:
        dump.append(
            pd.DataFrame(
                {
                    "Date": dates[mask],
                    "Ticker": tick[mask],
                    "model": m,
                    "pred": preds[m][mask],
                    "actual": panel[args.target].values[mask],
                }
            )
        )
    pd.concat(dump).dropna().to_pickle("oos_predictions.pkl")

    # per-year stability
    yr_rows = []
    for yr in years_run:
        ym = mask & (years == yr)
        for m in want:
            r, *_ = evaluate(preds[m][ym], panel[args.target].values[ym], dates[ym], tick[ym], m)
            r["year"] = yr
            yr_rows.append(r)
    pd.DataFrame(yr_rows).to_csv(f"{args.out}_{args.target}_byyear.csv", index=False)

    rows, pnls = [], {}
    for m in want:
        r, pnl, turn, ic = evaluate(preds[m][mask], panel[args.target].values[mask], dates[mask], tick[mask], m)
        rows.append(r)
        pnls[m] = pnl
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== OUT-OF-SAMPLE, target =", args.target, "===")
    print(res.to_string(index=False))
    res.to_csv(f"{args.out}_{args.target}.csv", index=False)
    pd.DataFrame(pnls).to_csv(f"{args.out}_{args.target}_pnl.csv")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
