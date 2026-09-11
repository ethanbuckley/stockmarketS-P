"""
evaluate.py
===========
Walk-forward validation of the screener's XGBoost classifier.

Expanding-window folds with annual test blocks: train on everything up to
five trading days before each test year (the purge removes training rows
whose 5-day label windows would overlap the test period), refit the
production model per fold, then score every test day cross-sectionally,
exactly as the screener is used: rank all stocks, take the top 15.

Outputs three small committed artefacts consumed by the dashboard and README:
    data/validation_metrics.json       headline + per-fold metrics, fold table,
                                       data snapshot, and the caveats that must
                                       accompany any quoted number
    data/validation_daily.csv          one row per test day
    data/validation_calibration.csv    reliability-curve bins per fold + pooled

Runs offline like generate_signals.py (needs the full pipeline deps plus
scikit-learn); nothing here executes in the deployed app.

Usage:
    python evaluate.py [--first-test-year 2020] [--skip-partial]
                       [--cache master_cache.pkl]
                       [--shuffled-target-check] [--purge-ablation]

Author: Ethan Buckley
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BOTTOM_N_CANDIDATES,
    DATA_DIR,
    DATA_START_DATE,
    EARLY_STOPPING_ROUNDS,
    EARLY_STOPPING_VALIDATION_FRACTION,
    FEATURE_COLUMNS,
    FORWARD_WINDOW_DAYS,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TOP_N_CANDIDATES,
    VALIDATION_CALIBRATION_PATH,
    VALIDATION_DAILY_PATH,
    VALIDATION_METRICS_PATH,
    XGB_PARAMS,
)
from screener import build_master_dataframe, build_technical_features, train_model

METRICS_PATH = VALIDATION_METRICS_PATH
DAILY_PATH = VALIDATION_DAILY_PATH
CALIBRATION_PATH = VALIDATION_CALIBRATION_PATH

# Bumped when the artefact layout or the meaning of a recorded number
# changes. 2: join-date truncation of the universe, early-stopped tree
# count per fold, volume no longer forward-filled.
SCHEMA_VERSION = 2


def _roc_auc(y, p) -> float:
    """ROC AUC via scikit-learn, imported lazily so importing this module
    (tests, the dashboard's docs) does not require sklearn."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


# Features computed by build_technical_features from a single ticker's OHLCV.
# The remaining FEATURE_COLUMNS are macro pct_changes/shifts, causal by
# construction; rolling/ewm windows checked here are the only nontrivial
# windowing in the pipeline.
TECHNICAL_FEATURES = [
    "RSI",
    "MACD",
    "BB_Position",
    "Price_to_VWAP",
    "ATR_Ratio",
    "Return",
    "Volume_Surge",
    "Day_Of_Week",
]

# Wording is shipped inside the same artefact as the numbers, so the app and
# README cannot quote a metric without its caveats travelling with it.
CAVEATS = [
    "Survivorship bias: the universe is today's S&P 500 constituents. Each "
    "ticker enters the panel only from the date it joined the index (taken "
    "from the Wikipedia constituents table), so no pre-membership history is "
    "used; but companies removed from the index since 2015 are absent, and "
    "that half of the bias still inflates measured hit rates.",
    "Prices are a single yfinance snapshot with auto-adjustment applied at "
    "download time; adjusted history can differ slightly from what was "
    "observable in real time.",
    "Consecutive test days share overlapping 5-day label windows and are not "
    "independent; confidence intervals use a moving-block bootstrap with "
    "block length 5.",
    "A low predicted probability is not a symmetric short signal: label 0 "
    "mixes 'stop-loss hit first' with 'no barrier hit within the window'.",
    "These are classifier-quality metrics only. No portfolio construction, "
    "transaction costs, slippage or capacity effects are modelled; nothing "
    "here is a tradeable return.",
]


# =============================================================================
# FOLDS AND LEAKAGE CONTROL
# =============================================================================


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_end: pd.Timestamp  # last usable training date, after the purge
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    partial: bool


def make_folds(
    cal: pd.DatetimeIndex,
    first_test_year: int = 2020,
    purge_days: int = FORWARD_WINDOW_DAYS,
    include_partial_final: bool = True,
) -> list[Fold]:
    """
    Expanding-window folds with calendar-year test blocks.

    A training row at calendar position p uses bars p+1 ... p+purge_days for
    its label, so it leaks iff p + purge_days >= position(test_start). The
    last safe training date is therefore cal[position(test_start) - purge_days - 1]:
    exactly the last `purge_days` trading days before each test year are purged.
    """
    last_year = int(cal[-1].year)
    folds: list[Fold] = []
    fold_id = 0
    for year in range(first_test_year, last_year + 1):
        test_dates = cal[cal.year == year]
        if len(test_dates) == 0:
            continue
        partial = year == last_year
        if partial and not include_partial_final:
            continue
        test_start, test_end = test_dates[0], test_dates[-1]
        pos = int(cal.searchsorted(test_start))
        if pos - purge_days - 1 < 0:
            raise ValueError(f"Not enough history before {year} to train after a {purge_days}-day purge.")
        fold_id += 1
        folds.append(
            Fold(
                fold_id=fold_id,
                train_end=cal[pos - purge_days - 1],
                test_start=test_start,
                test_end=test_end,
                partial=partial,
            )
        )
    if not folds:
        raise ValueError(f"No test data found from {first_test_year} onwards.")
    return folds


def split_fold(valid_df: pd.DataFrame, cal: pd.DatetimeIndex, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the filtered (complete features + valid label) frame into the
    fold's train and test sets, and asserts the purge gap held.
    """
    train = valid_df[valid_df.index <= fold.train_end]
    test = valid_df[(valid_df.index >= fold.test_start) & (valid_df.index <= fold.test_end)]

    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Fold {fold.fold_id}: empty train or test set.")

    # Hard leakage guarantee: the last training bar plus the label window
    # must end strictly before the first test bar.
    train_max_pos = int(cal.searchsorted(train.index.max()))
    test_min_pos = int(cal.searchsorted(test.index.min()))
    assert train_max_pos + FORWARD_WINDOW_DAYS < test_min_pos, (
        f"Fold {fold.fold_id}: purge violated (train ends at position {train_max_pos}, test starts at {test_min_pos})."
    )
    return train, test


def reconcile_rows(valid_df: pd.DataFrame, fold: Fold, n_train: int, n_test: int) -> None:
    """Accounts for every filtered row: train + purged + test + outside."""
    n_total = len(valid_df)
    n_purged = int(((valid_df.index > fold.train_end) & (valid_df.index < fold.test_start)).sum())
    n_outside = int((valid_df.index > fold.test_end).sum())
    assert n_train + n_purged + n_test + n_outside == n_total, (
        f"Fold {fold.fold_id}: row reconciliation failed "
        f"({n_train} + {n_purged} + {n_test} + {n_outside} != {n_total})."
    )


def feature_causality_check(master_df: pd.DataFrame, n_tickers: int = 3, seed: int = 42) -> None:
    """
    Proves the features are backward-looking: rebuild the technical features
    from data truncated at a random cutoff and assert the cutoff row equals
    the full-sample row. Any future-dependent feature added later fails here
    immediately. Macro features are single-lag pct_changes/shifts and are
    causal by construction.
    """
    rng = np.random.default_rng(seed)
    counts = master_df.dropna(subset=["Close"]).groupby("Ticker").size()
    eligible = counts[counts >= 60].index.to_numpy()
    if len(eligible) == 0:
        raise ValueError("No ticker has enough history for the causality check.")
    chosen = rng.choice(eligible, size=min(n_tickers, len(eligible)), replace=False)

    for ticker in chosen:
        sub = (
            master_df[master_df["Ticker"] == ticker][["Close", "High", "Low", "Volume"]].dropna(subset=["Close"]).copy()
        )
        cut_pos = int(rng.integers(40, len(sub)))
        cutoff = sub.index[cut_pos]

        full = build_technical_features(sub.copy())
        truncated = build_technical_features(sub.iloc[: cut_pos + 1].copy())

        full_row = full.loc[cutoff, TECHNICAL_FEATURES].astype(float).to_numpy()
        trunc_row = truncated.loc[cutoff, TECHNICAL_FEATURES].astype(float).to_numpy()
        assert np.allclose(full_row, trunc_row, equal_nan=True), (
            f"Feature causality violated for {ticker} at {cutoff.date()}: "
            "a feature value changed when future data was removed."
        )
    print(f"Feature-causality check passed for {len(chosen)} ticker(s).")


# =============================================================================
# METRICS
# =============================================================================


def daily_cross_sectional_metrics(
    day_df: pd.DataFrame,
    top_n: int = TOP_N_CANDIDATES,
    bottom_n: int = BOTTOM_N_CANDIDATES,
) -> dict:
    """
    Deployment-faithful metrics for one test day: rank the cross-section by
    predicted probability (ties broken by ticker for determinism), then ask
    how the top-N and bottom-N actually resolved.
    """
    n = len(day_df)
    base_rate = float(day_df["Target"].mean())
    row = {
        "n_candidates": n,
        "base_rate": base_rate,
        "precision_top15": np.nan,
        "excess_top15": np.nan,
        "bottom5_pos_rate": np.nan,
        "excess_bottom5": np.nan,
        "top_decile_lift": np.nan,
    }

    ranked = day_df.sort_values(["p_hat", "Ticker"], ascending=[False, True])

    if n >= top_n:
        precision = float(ranked.head(top_n)["Target"].mean())
        row["precision_top15"] = precision
        row["excess_top15"] = precision - base_rate

    if n >= bottom_n:
        bottom_rate = float(ranked.tail(bottom_n)["Target"].mean())
        row["bottom5_pos_rate"] = bottom_rate
        row["excess_bottom5"] = base_rate - bottom_rate

    if base_rate > 0:
        k = max(1, n // 10)
        row["top_decile_lift"] = float(ranked.head(k)["Target"].mean()) / base_rate

    return row


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability curve data (quantiles avoid empty bins)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    rows = [
        {
            "bin": int(b),
            "p_lo": float(g["p"].min()),
            "p_hi": float(g["p"].max()),
            "mean_predicted": float(g["p"].mean()),
            "observed_rate": float(g["y"].mean()),
            "count": int(len(g)),
        }
        for b, g in df.groupby("bin")
    ]
    return pd.DataFrame(rows)


def block_bootstrap_ci(
    series: np.ndarray,
    block: int = FORWARD_WINDOW_DAYS,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> list[float]:
    """
    Moving-block bootstrap CI for the mean of a daily series. Consecutive
    days share overlapping label windows, so iid resampling would understate
    the interval; blocks of length FORWARD_WINDOW_DAYS preserve the local
    dependence structure.
    """
    n = len(series)
    if n == 0:
        return [float("nan"), float("nan")]
    if n <= block:
        return [float(np.min(series)), float(np.max(series))]

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts_max = n - block
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([series[s : s + block] for s in starts])[:n]
        means[i] = sample.mean()
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def daily_aggregates(daily: pd.DataFrame, seed: int = 42) -> dict:
    """Distributional summary of the per-day metrics (fold-level or pooled)."""
    p15 = daily["precision_top15"].dropna()
    excess = daily["excess_top15"].dropna()
    b5_excess = daily["excess_bottom5"].dropna()
    lift = daily["top_decile_lift"].dropna()

    return {
        "n_days": int(len(daily)),
        "base_rate_daily_mean": float(daily["base_rate"].mean()),
        "precision_top15_mean": float(p15.mean()),
        "precision_top15_median": float(p15.median()),
        "precision_top15_p10": float(p15.quantile(0.10)),
        "precision_top15_p90": float(p15.quantile(0.90)),
        "excess_precision_top15_mean": float(excess.mean()),
        "excess_precision_top15_ci95": block_bootstrap_ci(excess.to_numpy(), seed=seed),
        "frac_days_top15_beats_base": float((excess > 0).mean()),
        "bottom5_pos_rate_mean": float(daily["bottom5_pos_rate"].dropna().mean()),
        "excess_bottom5_mean": float(b5_excess.mean()),
        "frac_days_bottom5_beats_base": float((b5_excess > 0).mean()),
        "top_decile_lift_mean": float(lift.mean()),
        "top_decile_lift_median": float(lift.median()),
        "n_days_skipped_lt_top_n": int(daily["precision_top15"].isna().sum()),
        "n_days_zero_positives": int((daily["base_rate"] == 0).sum()),
    }


def evaluate_fold(model, test_df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, dict, pd.DataFrame, np.ndarray]:
    """Scores one fold's test set.

    Returns (daily metrics, fold summary, calibration bins, p_hat), where
    p_hat is the test-row prediction vector in test_df order so the caller
    can pool it without predicting a second time.
    """
    test_df = test_df.copy()
    test_df["p_hat"] = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    daily_rows = []
    for date, day_df in test_df.groupby(level=0):
        row = daily_cross_sectional_metrics(day_df)
        row["date"] = date
        row["fold_id"] = fold.fold_id
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    y = test_df["Target"].to_numpy()
    p = test_df["p_hat"].to_numpy()
    summary = {
        "fold_id": fold.fold_id,
        "roc_auc": _roc_auc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "base_rate": float(y.mean()),
        "n_test_rows": int(len(y)),
        **daily_aggregates(daily),
    }

    calib = reliability_bins(y, p)
    calib.insert(0, "fold_id", str(fold.fold_id))
    return daily, summary, calib, p


# =============================================================================
# EVALUATION DRIVER
# =============================================================================


def run_evaluation(
    master_df: pd.DataFrame,
    first_test_year: int = 2020,
    include_partial_final: bool = True,
    shuffled_target_check: bool = False,
    purge_ablation: bool = False,
    causality_check: bool = True,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full walk-forward evaluation on a labelled master panel.
    Returns (results, daily_df, calibration_df); results holds the fold
    table plus per-fold, pooled and overall-daily metrics.
    """
    cal = master_df.index.unique().sort_values()
    if causality_check:
        feature_causality_check(master_df, seed=seed)

    valid_df = master_df.dropna(subset=FEATURE_COLUMNS + ["Target"])
    folds = make_folds(cal, first_test_year, include_partial_final=include_partial_final)

    fold_table = []
    per_fold = []
    all_daily: list[pd.DataFrame] = []
    all_calib: list[pd.DataFrame] = []
    pooled_y: list[np.ndarray] = []
    pooled_p: list[np.ndarray] = []

    for fold in folds:
        t0 = time.perf_counter()
        train, test = split_fold(valid_df, cal, fold)
        reconcile_rows(valid_df, fold, len(train), len(test))

        model = train_model(train)

        # Train-vs-test generalisation gap (console evidence only): a
        # near-zero gap would suggest test rows contaminated the pool.
        train_auc = _roc_auc(train["Target"], model.predict_proba(train[FEATURE_COLUMNS])[:, 1])

        daily, summary, calib, p_hat = evaluate_fold(model, test, fold)

        fold_table.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(train.index.min().date()),
                "train_end_after_purge": str(train.index.max().date()),
                "test_start": str(test.index.min().date()),
                "test_end": str(fold.test_end.date()),
                "test_end_effective": str(test.index.max().date()),
                "n_train_rows": int(len(train)),
                "n_test_rows": int(len(test)),
                "n_test_days": int(daily.shape[0]),
                "n_tickers_in_test": int(test["Ticker"].nunique()),
                "n_estimators": int(model.get_params()["n_estimators"]),
                "partial": fold.partial,
            }
        )
        per_fold.append(summary)
        all_daily.append(daily)
        all_calib.append(calib)
        pooled_y.append(test["Target"].to_numpy())
        pooled_p.append(p_hat)

        elapsed = time.perf_counter() - t0
        print(
            f"Fold {fold.fold_id} ({fold.test_start.year}"
            f"{', partial' if fold.partial else ''}): "
            f"train {len(train):,} rows to {train.index.max().date()}, "
            f"test {len(test):,} rows, "
            f"{model.get_params()['n_estimators']} trees, "
            f"AUC {summary['roc_auc']:.4f} (train {train_auc:.4f}), "
            f"P@15 {summary['precision_top15_mean']:.3f} "
            f"vs base {summary['base_rate_daily_mean']:.3f} "
            f"[{elapsed:.0f}s]"
        )

        if fold.fold_id == 1 and shuffled_target_check:
            _shuffled_target_test(train, test, seed=seed)
        if fold.fold_id == 1 and purge_ablation:
            _purge_ablation_test(valid_df, cal, fold, test, model)

    daily_all = pd.concat(all_daily, ignore_index=True)
    overall_daily = daily_aggregates(daily_all, seed=seed)
    # Lean CSV schema; excess_bottom5 is derivable as base_rate - bottom5_pos_rate
    daily_df = daily_all[
        [
            "date",
            "fold_id",
            "n_candidates",
            "base_rate",
            "precision_top15",
            "excess_top15",
            "bottom5_pos_rate",
            "top_decile_lift",
        ]
    ]

    y_pooled = np.concatenate(pooled_y)
    p_pooled = np.concatenate(pooled_p)
    pooled_calib = reliability_bins(y_pooled, p_pooled)
    pooled_calib.insert(0, "fold_id", "pooled")
    calib_df = pd.concat(all_calib + [pooled_calib], ignore_index=True)

    results = {
        "folds": fold_table,
        "per_fold": per_fold,
        "pooled": {
            "roc_auc": _roc_auc(y_pooled, p_pooled),
            "brier": float(np.mean((p_pooled - y_pooled) ** 2)),
            "base_rate": float(y_pooled.mean()),
            "n_rows": int(len(y_pooled)),
            "note": "Each test row is scored by its own fold's model (standard walk-forward pooling).",
        },
        "overall_daily": overall_daily,
    }
    return results, daily_df, calib_df


def _shuffled_target_test(
    train: pd.DataFrame, test: pd.DataFrame, seed: int = 42, n_permutations: int = 5
) -> list[float]:
    """
    Leakage canary: refit fold 1 with permuted training labels and score the
    real test labels. Information reaching the test set through anything
    other than the labels would push these AUCs *above* 0.5.

    The null distribution is wider than the row count suggests. Macro
    features (SPY/VIX/TNX changes, day of week) are identical for every
    ticker on a given day, so a noise-fit model's predictions move together
    across the whole cross-section and the effective sample is the number of
    test days, not test rows. A single permutation can land at 0.46 or 0.53
    by chance, so several are run and only the upper tail is a warning.
    """
    print(f"Running shuffled-target leakage check ({n_permutations} permutations, refits fold 1)...")
    aucs = []
    for i in range(n_permutations):
        rng = np.random.default_rng(seed + i)
        shuffled = train.copy()
        shuffled["Target"] = rng.permutation(shuffled["Target"].to_numpy())
        model = train_model(shuffled)
        auc = _roc_auc(test["Target"], model.predict_proba(test[FEATURE_COLUMNS])[:, 1])
        aucs.append(auc)
        print(f"  permutation {i + 1}: {model.get_params()['n_estimators']} trees, test AUC {auc:.4f}")
    verdict = "OK" if max(aucs) <= 0.55 else "WARNING: investigate before publishing"
    print(f"  Shuffled-target AUC range {min(aucs):.4f} to {max(aucs):.4f} (real labels ~0.65): {verdict}")
    return aucs


def _purge_ablation_test(
    valid_df: pd.DataFrame,
    cal: pd.DatetimeIndex,
    fold: Fold,
    test: pd.DataFrame,
    purged_model,
) -> None:
    """
    Quantifies what the purge removes: train fold 1 again without the 5-day
    purge and compare AUC on the first 10 test days. Console evidence only,
    never written to the committed artefacts.
    """
    print("Running purge-ablation check (refits fold 1 without the purge)...")
    pos = int(cal.searchsorted(fold.test_start))
    unpurged_train = valid_df[valid_df.index <= cal[pos - 1]]
    unpurged_model = train_model(unpurged_train)

    first_days = test.index.unique().sort_values()[:10]
    early = test[test.index.isin(first_days)]
    y = early["Target"].to_numpy()
    auc_unpurged = _roc_auc(y, unpurged_model.predict_proba(early[FEATURE_COLUMNS])[:, 1])
    auc_purged = _roc_auc(y, purged_model.predict_proba(early[FEATURE_COLUMNS])[:, 1])
    print(
        f"  First 10 test days: purged AUC {auc_purged:.4f}, "
        f"unpurged AUC {auc_unpurged:.4f} "
        f"(unpurged advantage = boundary leakage the purge removes)"
    )


# =============================================================================
# ARTEFACT WRITING
# =============================================================================


def _json_safe(obj):
    """Rounds floats and converts NaN to null so the JSON is strict and stable."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, 6)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _library_versions() -> dict:
    import sklearn
    import xgboost
    import yfinance

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
        "yfinance": yfinance.__version__,
        "sklearn": sklearn.__version__,
    }


def write_artefacts(
    results: dict,
    daily_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    metadata: dict,
    cache_used: bool,
    first_test_year: int,
) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "data_snapshot": {
            "download_timestamp_utc": metadata.get("download_timestamp_utc"),
            "data_start": DATA_START_DATE,
            "last_price_date": metadata.get("last_price_date"),
            "n_tickers": len(metadata.get("tickers", [])),
            "ticker_source": "Wikipedia S&P 500 constituents as of the download date",
            "join_date_truncation": metadata.get("join_date_truncation", {"applied": False}),
            "cache_used": cache_used,
            "versions": _library_versions(),
        },
        "label_definition": {
            "type": "triple_barrier",
            "forward_window_days": FORWARD_WINDOW_DAYS,
            "take_profit_pct": TAKE_PROFIT_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "positive_iff": (
                "take-profit barrier hit strictly before stop-loss within the "
                "window; ties and no-hit windows are labelled 0"
            ),
        },
        "model": {
            "type": "XGBClassifier",
            "params": XGB_PARAMS,
            "n_estimators_note": (
                "params.n_estimators is a ceiling. Per fold, the tree count is "
                "chosen by early stopping on the most recent "
                f"{EARLY_STOPPING_VALIDATION_FRACTION:.0%} of training dates "
                f"(purged by {FORWARD_WINDOW_DAYS} days, patience "
                f"{EARLY_STOPPING_ROUNDS} rounds), then the model is refit on "
                "all training rows; see validation_scheme.folds[*].n_estimators."
            ),
            "feature_columns": FEATURE_COLUMNS,
        },
        "validation_scheme": {
            "type": "expanding_window_walk_forward",
            "first_test_year": first_test_year,
            "test_block": "calendar_year",
            "purge_trading_days": FORWARD_WINDOW_DAYS,
            "purge_note": (
                "Training rows in the last 5 trading days before each test "
                "block are dropped because their labels look into the test "
                "period. Features are strictly backward-looking, so the test "
                "side needs no purge; walk-forward order means no embargo is "
                "required either."
            ),
            "folds": results["folds"],
        },
        "caveats": CAVEATS,
        "results": {
            "pooled": results["pooled"],
            "per_fold": results["per_fold"],
            "overall_daily": results["overall_daily"],
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
        f.write("\n")

    daily_out = daily_df.copy()
    daily_out["date"] = pd.to_datetime(daily_out["date"]).dt.strftime("%Y-%m-%d")
    for col in daily_out.columns:
        if daily_out[col].dtype == float:
            daily_out[col] = daily_out[col].round(6)
    daily_out.to_csv(DAILY_PATH, index=False)

    calib_out = calib_df.copy()
    for col in ["p_lo", "p_hi", "mean_predicted", "observed_rate"]:
        calib_out[col] = calib_out[col].round(6)
    calib_out.to_csv(CALIBRATION_PATH, index=False)

    print(f"\nArtefacts written:\n  {METRICS_PATH}\n  {DAILY_PATH}\n  {CALIBRATION_PATH}")


# =============================================================================
# MAIN
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation of the screener's XGBoost classifier.")
    parser.add_argument(
        "--first-test-year",
        type=int,
        default=2020,
        help="first calendar year used as a test block (default 2020)",
    )
    parser.add_argument(
        "--skip-partial",
        action="store_true",
        help="exclude the final, incomplete calendar year from the folds",
    )
    parser.add_argument(
        "--cache",
        metavar="PATH",
        default=None,
        help="pickle the labelled master panel here and reuse it on later runs (e.g. master_cache.pkl; gitignored)",
    )
    parser.add_argument(
        "--shuffled-target-check",
        action="store_true",
        help="leakage canary: refit fold 1 on permuted labels, expect test AUC ~0.5",
    )
    parser.add_argument(
        "--purge-ablation",
        action="store_true",
        help="refit fold 1 without the purge and report the boundary-leakage AUC gap",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    cache_used = False
    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached master panel from {args.cache}...")
        master_df, metadata = pd.read_pickle(args.cache)
        cache_used = True
    else:
        master_df, metadata = build_master_dataframe()
        if args.cache:
            pd.to_pickle((master_df, metadata), args.cache)
            print(f"Master panel cached to {args.cache}.")

    results, daily_df, calib_df = run_evaluation(
        master_df,
        first_test_year=args.first_test_year,
        include_partial_final=not args.skip_partial,
        shuffled_target_check=args.shuffled_target_check,
        purge_ablation=args.purge_ablation,
    )

    write_artefacts(results, daily_df, calib_df, metadata, cache_used, args.first_test_year)

    pooled = results["pooled"]
    overall = results["overall_daily"]
    print(
        f"\nPooled out-of-sample: AUC {pooled['roc_auc']:.4f}, "
        f"Brier {pooled['brier']:.4f}, base rate {pooled['base_rate']:.3f} "
        f"({pooled['n_rows']:,} rows)"
    )
    print(
        f"Daily precision@{TOP_N_CANDIDATES}: mean {overall['precision_top15_mean']:.3f} "
        f"vs daily base rate {overall['base_rate_daily_mean']:.3f}; "
        f"beats base on {overall['frac_days_top15_beats_base']:.1%} of days"
    )
    print(f"Total runtime: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
