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
    data/backtest_daily.csv            daily returns of the long-only backtest
                                       and its benchmarks

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
    BACKTEST_COST_BPS,
    BACKTEST_HOLD_DAYS,
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
    TRADING_DAYS_PER_YEAR,
    VALIDATION_CALIBRATION_PATH,
    VALIDATION_DAILY_PATH,
    VALIDATION_METRICS_PATH,
    XGB_PARAMS,
)
from screener import build_master_dataframe, build_technical_features, train_model

BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_daily.csv")

# Subset of the daily aggregates reported for model-free baselines
BASELINE_KEYS = (
    "daily_auc_mean",
    "daily_auc_ci95",
    "precision_top15_mean",
    "excess_precision_top15_mean",
    "excess_precision_top15_ci95",
    "frac_days_top15_beats_base",
    "top_decile_lift_mean",
)

METRICS_PATH = VALIDATION_METRICS_PATH
DAILY_PATH = VALIDATION_DAILY_PATH
CALIBRATION_PATH = VALIDATION_CALIBRATION_PATH

# Bumped when the artefact layout or the meaning of a recorded number
# changes. 3: point-in-time universe (former members included up to their
# removal date where prices exist), per-day cross-sectional AUC, portfolio
# backtest block and data/backtest_daily.csv.
SCHEMA_VERSION = 3


def _roc_auc(y, p) -> float:
    """ROC AUC via scikit-learn, imported lazily so importing this module
    (tests, the dashboard's docs) does not require sklearn."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def rank_auc(y, p) -> float:
    """ROC AUC by the rank (Mann-Whitney) formula; NaN if one class is absent.
    Pure pandas/numpy so the per-day metrics need no sklearn."""
    y = np.asarray(y, dtype=float)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(p, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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
def caveats(metadata: dict) -> list[str]:
    """Wording shipped inside the same artefact as the numbers, so the app and
    README cannot quote a metric without its caveats travelling with it."""
    u = metadata.get("universe", {})
    n_removed = u.get("n_removed_since_start", 0)
    n_recovered = u.get("n_removed_with_price_data", 0)
    return [
        "Survivorship bias: the universe is point-in-time by symbol. Current "
        "members enter from their S&P 500 join date (Wikipedia constituents "
        "table) and companies removed since 2015 are included up to their "
        f"removal date where Yahoo still has prices ({n_recovered} of {n_removed} "
        "removed tickers). The remaining removed companies, mostly acquired "
        "or delisted, are absent, which still biases measured hit rates upward.",
        "Prices are a single yfinance snapshot with auto-adjustment applied at "
        "download time; adjusted history can differ slightly from what was "
        "observable in real time.",
        "Consecutive test days share overlapping 5-day label windows and are not "
        "independent; confidence intervals use a moving-block bootstrap with "
        "block length 5.",
        "A low predicted probability is not a symmetric short signal: label 0 "
        "mixes 'stop-loss hit first' with 'no barrier hit within the window'.",
        "Pooled ROC AUC mixes two things: knowing which days have high hit "
        "rates (macro features) and ranking stocks within a day. Only the "
        "within-day part is usable by a screener; see the mean per-day AUC.",
        "The backtest is a stylised long-only book: each day's top picks, "
        f"equal-weighted, held {BACKTEST_HOLD_DAYS} trading days close-to-close "
        f"as overlapping tranches, charged {BACKTEST_COST_BPS:.0f} bps per side. "
        "No barrier exits, no slippage model, no capacity or borrow "
        "constraints, and the same survivorship bias as above.",
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
        "daily_auc": rank_auc(day_df["Target"], day_df["p_hat"]),
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


def mean_per_day_auc(scored: pd.DataFrame) -> float:
    """Average of the within-day rank AUCs of p_hat against Target (date index)."""
    return float(np.nanmean([rank_auc(g["Target"], g["p_hat"]) for _, g in scored.groupby(level=0)]))


def score_baseline(test_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """A model-free ranking: p_hat is the raw feature value. Shares the metric
    code with the real model so the comparison is like for like."""
    return test_df[["Ticker", "Target"]].assign(p_hat=test_df[feature].to_numpy(dtype=float))


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
    auc = daily["daily_auc"].dropna()

    return {
        "n_days": int(len(daily)),
        "daily_auc_mean": float(auc.mean()),
        "daily_auc_ci95": block_bootstrap_ci(auc.to_numpy(), seed=seed),
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


def evaluate_fold(model, test_df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Scores one fold's test set.

    Returns (daily metrics, fold summary, calibration bins, scored), where
    scored holds Ticker, Target and p_hat for every test row (date index) so
    the caller can pool predictions and run the backtest without predicting
    a second time.
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
    return daily, summary, calib, test_df[["Ticker", "Target", "p_hat"]]


# =============================================================================
# PORTFOLIO BACKTEST
# =============================================================================


def run_backtest(
    scored: pd.DataFrame,
    returns_wide: pd.DataFrame,
    spy_returns: pd.Series,
    top_n: int = TOP_N_CANDIDATES,
    hold_days: int = BACKTEST_HOLD_DAYS,
    cost_bps: float = BACKTEST_COST_BPS,
) -> pd.DataFrame:
    """
    Daily returns of a long-only book built from the walk-forward predictions.

    At each test day's close the top_n stocks by p_hat form a tranche that is
    held for the next hold_days sessions, equal-weighted (rebalanced daily
    within the tranche, a small approximation to buy-and-hold). The book is
    the equal-weighted average of the active tranches, so 1/hold_days of it
    rolls every day. Each tranche pays cost_bps on its first day (entry) and
    on its last day (exit). Positions whose return is missing on a day
    (ticker left the panel) are dropped from that tranche's mean.

    Benchmarks: the equal-weighted average return of the previous day's whole
    eligible cross-section (what the ranking is chosen from, cost-free), and
    SPY. Returns a frame with one row per session after the first pick.
    """
    dates = returns_wide.index
    values = returns_wide.to_numpy(dtype=float)
    col_of = {t: i for i, t in enumerate(returns_wide.columns)}
    cost = cost_bps / 1e4

    ranked = scored.sort_values(["p_hat", "Ticker"], ascending=[False, True])
    picks = ranked.groupby(level=0).head(top_n).groupby(level=0)["Ticker"].apply(list).to_dict()
    eligible = scored.groupby(level=0)["Ticker"].apply(list).to_dict()

    def mean_return(day_idx: int, tickers: list[str]) -> float:
        cols = [col_of[t] for t in tickers if t in col_of]
        r = values[day_idx, cols]
        r = r[~np.isnan(r)]
        return float(r.mean()) if len(r) else float("nan")

    pick_dates = sorted(picks)
    first_idx = int(dates.searchsorted(pick_dates[0])) + 1
    last_idx = min(int(dates.searchsorted(pick_dates[-1])) + hold_days, len(dates) - 1)
    rows = []
    for t_idx in range(first_idx, last_idx + 1):
        t = dates[t_idx]
        tranche_ret, tranche_cost = [], []
        for k in range(1, hold_days + 1):
            s_idx = t_idx - k
            if s_idx < 0 or dates[s_idx] not in picks:
                continue
            r = mean_return(t_idx, picks[dates[s_idx]])
            if np.isnan(r):
                continue
            tranche_ret.append(r)
            tranche_cost.append((cost if k == 1 else 0.0) + (cost if k == hold_days else 0.0))
        if not tranche_ret:
            continue
        prev = dates[t_idx - 1]
        rows.append(
            {
                "date": t,
                "strategy_gross": float(np.mean(tranche_ret)),
                "strategy_net": float(np.mean(tranche_ret) - np.mean(tranche_cost)),
                "universe_ew": mean_return(t_idx, eligible[prev]) if prev in eligible else float("nan"),
                "spy": float(spy_returns.get(t, np.nan)),
                "n_tranches": len(tranche_ret),
            }
        )
    return pd.DataFrame(rows)


def performance_summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Annualised return/vol, Sharpe (rf = 0), max drawdown and total return of a daily series."""
    r = returns.dropna()
    if len(r) < 2:
        return {k: float("nan") for k in ("total_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "n_days")}
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    # A daily loss of 100% or more wipes the book out; the geometric mean is then -100%.
    ann_return = float(equity.iloc[-1] ** (periods_per_year / len(r)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    ann_vol = float(r.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else float("nan")
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": total,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "n_days": int(len(r)),
    }


def backtest_results(bt: pd.DataFrame) -> dict:
    """JSON block: assumptions, per-series summaries and calendar-year returns."""
    series = ["strategy_net", "strategy_gross", "universe_ew", "spy"]
    by_year = []
    for year, g in bt.groupby(bt["date"].dt.year):
        row = {"year": int(year), "n_days": int(len(g))}
        for col in series:
            row[col] = float((1.0 + g[col].dropna()).prod() - 1.0)
        by_year.append(row)
    return {
        "assumptions": {
            "side": "long_only",
            "top_n": TOP_N_CANDIDATES,
            "hold_days": BACKTEST_HOLD_DAYS,
            "weighting": "equal, overlapping tranches (1/hold_days rolls per day)",
            "cost_bps_per_side": BACKTEST_COST_BPS,
            "execution": "close-to-close; no barrier exits, slippage or capacity model",
            "benchmark_universe_ew": "equal-weighted previous-day eligible cross-section, no costs",
        },
        "series": {col: performance_summary(bt[col]) for col in series},
        "by_year": by_year,
    }


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
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full walk-forward evaluation on a labelled master panel.
    Returns (results, daily_df, calibration_df, backtest_df); results holds
    the fold table plus per-fold, pooled, overall-daily and backtest metrics.
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
    all_scored: list[pd.DataFrame] = []

    for fold in folds:
        t0 = time.perf_counter()
        train, test = split_fold(valid_df, cal, fold)
        reconcile_rows(valid_df, fold, len(train), len(test))

        model = train_model(train)

        # Train-vs-test generalisation gap (console evidence only): a
        # near-zero gap would suggest test rows contaminated the pool.
        train_auc = _roc_auc(train["Target"], model.predict_proba(train[FEATURE_COLUMNS])[:, 1])

        daily, summary, calib, scored = evaluate_fold(model, test, fold)

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
        pooled_p.append(scored["p_hat"].to_numpy())
        all_scored.append(scored)

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
            "daily_auc",
        ]
    ]

    # Portfolio backtest from the same out-of-sample predictions. Daily
    # returns come from the unfiltered panel so that a stock that leaves the
    # index mid-hold still has its return on the days it is held.
    scored_all = pd.concat(all_scored)
    returns_wide = master_df.pivot_table(index=master_df.index, columns="Ticker", values="Return")
    spy_returns = master_df["SPY_Return"].groupby(level=0).first()
    backtest_df = run_backtest(scored_all, returns_wide, spy_returns)

    # Volatility-only baseline: rank each day by ATR_Ratio, no model. Volatile
    # stocks hit either barrier more often, so this is the bar the classifier
    # has to clear, and its backtest is the fair comparison for the book.
    test_rows = valid_df.loc[scored_all.index.unique()]
    atr_scored = score_baseline(test_rows, "ATR_Ratio")
    atr_daily = pd.DataFrame(
        [daily_cross_sectional_metrics(day_df) | {"date": date} for date, day_df in atr_scored.groupby(level=0)]
    )
    atr_backtest = run_backtest(atr_scored, returns_wide, spy_returns)
    baselines = {
        "atr_rank": {
            "description": "Each test day ranked by ATR_Ratio (14-day ATR / close) alone; no model.",
            **{k: v for k, v in daily_aggregates(atr_daily, seed=seed).items() if k in BASELINE_KEYS},
            "backtest": {col: performance_summary(atr_backtest[col]) for col in ("strategy_net", "strategy_gross")},
        }
    }

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
        "backtest": backtest_results(backtest_df),
        "baselines": baselines,
    }
    return results, daily_df, calib_df, backtest_df


def _shuffled_target_test(
    train: pd.DataFrame, test: pd.DataFrame, seed: int = 42, n_permutations: int = 5
) -> list[float]:
    """
    Leakage canary: refit fold 1 with fully permuted training labels and
    score the real test labels. Judged on the mean per-day (within-day) AUC
    averaged over permutations, which has a clean null at 0.5: on this data
    five permutations gave 0.490 to 0.504. Information reaching the test set
    through anything other than the labels would push it above 0.5.

    Two tempting alternatives are not nulls and are deliberately not used:
    - Pooled AUC of a permuted model wobbles between about 0.46 and 0.53
      because macro features are shared by every ticker on a day, so its
      effective sample is the number of test days, not rows.
    - Permuting labels *within* each date preserves daily base rates, and a
      model fit to that still reaches per-day AUC ~0.55: it learns that
      high-volatility days have high hit rates, ranks volatile stocks first,
      and volatile stocks genuinely do hit barriers more often within a day.
      That is the volatility confound measured by the ATR baseline, not a leak.
    """
    print(f"Running shuffled-target leakage check ({n_permutations} full permutations, refits fold 1)...")
    daily_aucs = []
    for i in range(n_permutations):
        rng = np.random.default_rng(seed + i)
        shuffled = train.copy()
        shuffled["Target"] = rng.permutation(shuffled["Target"].to_numpy())
        model = train_model(shuffled)
        p = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        pooled = _roc_auc(test["Target"], p)
        per_day = mean_per_day_auc(test[["Ticker", "Target"]].assign(p_hat=p))
        daily_aucs.append(per_day)
        print(
            f"  permutation {i + 1}: {model.get_params()['n_estimators']} trees, "
            f"pooled AUC {pooled:.4f}, mean per-day AUC {per_day:.4f}"
        )
    mean_auc = float(np.mean(daily_aucs))
    verdict = "OK" if mean_auc <= 0.52 and max(daily_aucs) <= 0.55 else "WARNING: investigate before publishing"
    print(
        f"  Per-day AUC under permutation: mean {mean_auc:.4f}, range {min(daily_aucs):.4f} to "
        f"{max(daily_aucs):.4f} (expected 0.5): {verdict}"
    )
    return daily_aucs


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
    backtest_df: pd.DataFrame,
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
            "universe": metadata.get("universe", {"type": "unknown"}),
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
        "caveats": caveats(metadata),
        "results": {
            "pooled": results["pooled"],
            "per_fold": results["per_fold"],
            "overall_daily": results["overall_daily"],
            "backtest": results["backtest"],
            "baselines": results["baselines"],
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

    bt_out = backtest_df.copy()
    bt_out["date"] = pd.to_datetime(bt_out["date"]).dt.strftime("%Y-%m-%d")
    for col in ("strategy_gross", "strategy_net", "universe_ew", "spy"):
        bt_out[col] = bt_out[col].round(8)
    bt_out.to_csv(BACKTEST_PATH, index=False)

    print(f"\nArtefacts written:\n  {METRICS_PATH}\n  {DAILY_PATH}\n  {CALIBRATION_PATH}\n  {BACKTEST_PATH}")


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

    results, daily_df, calib_df, backtest_df = run_evaluation(
        master_df,
        first_test_year=args.first_test_year,
        include_partial_final=not args.skip_partial,
        shuffled_target_check=args.shuffled_target_check,
        purge_ablation=args.purge_ablation,
    )

    write_artefacts(results, daily_df, calib_df, backtest_df, metadata, cache_used, args.first_test_year)

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
    bt = results["backtest"]["series"]
    atr = results["baselines"]["atr_rank"]
    print(
        f"ATR-rank baseline (no model): per-day AUC {atr['daily_auc_mean']:.4f}, "
        f"P@15 {atr['precision_top15_mean']:.3f}, backtest net {atr['backtest']['strategy_net']['ann_return']:+.1%}/yr"
    )
    print(
        f"Mean per-day AUC {overall['daily_auc_mean']:.4f}; backtest (net of {BACKTEST_COST_BPS:.0f} bps/side): "
        f"{bt['strategy_net']['ann_return']:+.1%}/yr, Sharpe {bt['strategy_net']['sharpe']:.2f}, "
        f"max DD {bt['strategy_net']['max_drawdown']:.1%} vs universe EW {bt['universe_ew']['ann_return']:+.1%}/yr, "
        f"SPY {bt['spy']['ann_return']:+.1%}/yr"
    )
    print(f"Total runtime: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
