"""Tests for the walk-forward evaluation machinery, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

import evaluate
from evaluate import (
    block_bootstrap_ci,
    daily_cross_sectional_metrics,
    feature_causality_check,
    make_folds,
    reliability_bins,
    split_fold,
)
from screener import FEATURE_COLUMNS, FORWARD_WINDOW_DAYS


def calendar(start="2018-01-01", end="2021-06-30") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(start, end))


def test_make_folds_boundaries_and_purge():
    cal = calendar()
    folds = make_folds(cal, first_test_year=2020)
    assert [f.fold_id for f in folds] == [1, 2]

    f2020, f2021 = folds
    assert f2020.test_start == cal[cal.year == 2020][0]
    assert f2020.test_end == cal[cal.year == 2020][-1]
    assert not f2020.partial
    assert f2021.partial  # data ends mid-2021

    # The purge: train_end sits exactly W+1 calendar positions before test_start
    pos = cal.searchsorted(f2020.test_start)
    assert f2020.train_end == cal[pos - FORWARD_WINDOW_DAYS - 1]


def test_make_folds_can_skip_partial_final_year():
    folds = make_folds(calendar(), first_test_year=2020, include_partial_final=False)
    assert [f.fold_id for f in folds] == [1]
    assert not folds[0].partial


def test_make_folds_insufficient_history_raises():
    cal = calendar("2020-01-01", "2020-12-31")
    with pytest.raises(ValueError):
        make_folds(cal, first_test_year=2020)


def test_split_fold_enforces_purge_gap():
    cal = calendar()
    df = pd.DataFrame({"x": range(len(cal))}, index=cal)
    fold = make_folds(cal, first_test_year=2020)[0]

    train, test = split_fold(df, cal, fold)

    pos_train = cal.searchsorted(train.index.max())
    pos_test = cal.searchsorted(test.index.min())
    assert pos_test - pos_train == FORWARD_WINDOW_DAYS + 1
    assert set(train.index).isdisjoint(test.index)


def test_daily_cross_sectional_metrics_hand_computed():
    n = 20
    day = pd.DataFrame(
        {
            "Ticker": [f"T{i:02d}" for i in range(n)],
            "p_hat": np.linspace(1.0, 0.0, n),       # T00 ranked highest
            "Target": [1.0] * 8 + [0.0] * 12,        # positives = top 8 by p_hat
        }
    )
    row = daily_cross_sectional_metrics(day, top_n=15, bottom_n=5)

    assert row["n_candidates"] == 20
    assert row["base_rate"] == pytest.approx(0.4)
    assert row["precision_top15"] == pytest.approx(8 / 15)
    assert row["excess_top15"] == pytest.approx(8 / 15 - 0.4)
    assert row["bottom5_pos_rate"] == 0.0
    assert row["excess_bottom5"] == pytest.approx(0.4)
    assert row["top_decile_lift"] == pytest.approx(2.5)  # k=2, both positive


def test_daily_metrics_skip_small_days():
    day = pd.DataFrame({"Ticker": ["A", "B"], "p_hat": [0.9, 0.1], "Target": [1.0, 0.0]})
    row = daily_cross_sectional_metrics(day, top_n=15, bottom_n=5)
    assert np.isnan(row["precision_top15"])
    assert np.isnan(row["bottom5_pos_rate"])
    assert row["top_decile_lift"] == pytest.approx(2.0)  # k=1, top row positive


def test_block_bootstrap_ci_deterministic_and_sane():
    x = np.random.default_rng(0).normal(0.05, 0.02, 300)
    ci_a = block_bootstrap_ci(x, seed=42)
    ci_b = block_bootstrap_ci(x, seed=42)
    assert ci_a == ci_b
    assert ci_a[0] < x.mean() < ci_a[1]


def test_reliability_bins_partition_all_rows():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(float)
    bins = reliability_bins(y, p, n_bins=10)
    assert bins["count"].sum() == 500
    assert bins["observed_rate"].between(0, 1).all()
    assert bins["mean_predicted"].between(0, 1).all()


def test_feature_causality_check_passes_on_causal_features(make_ohlcv):
    closes = 100.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 80)))
    df = make_ohlcv(closes)
    df["Ticker"] = "AAA"
    feature_causality_check(df, n_tickers=1)  # raises on violation


def synthetic_master(n_tickers: int = 20) -> pd.DataFrame:
    """A labelled master panel with an informative signal in feature 0."""
    cal = calendar("2018-01-01", "2021-03-31")
    rng = np.random.default_rng(5)
    frames = []
    for i in range(n_tickers):
        df = pd.DataFrame(index=cal)
        for col in FEATURE_COLUMNS:
            df[col] = rng.normal(size=len(cal))
        df["Ticker"] = f"T{i:02d}"
        df["Close"] = 100.0
        signal = df[FEATURE_COLUMNS[0]] + rng.normal(0, 1.0, len(cal))
        df["Target"] = (signal > 0).astype(float)
        # Mimic the label horizon: no labels for the final W bars
        df.iloc[-FORWARD_WINDOW_DAYS:, df.columns.get_loc("Target")] = np.nan
        frames.append(df)
    return pd.concat(frames)


def test_run_evaluation_end_to_end_mini():
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")

    results, daily, calib = evaluate.run_evaluation(
        synthetic_master(), first_test_year=2020, causality_check=False
    )

    assert len(results["folds"]) == 2  # 2020 full, 2021 partial
    assert all(1 <= f["n_estimators"] <= 1000 for f in results["folds"])
    assert results["folds"][0]["partial"] is False
    assert results["folds"][1]["partial"] is True

    # The synthetic signal is learnable, so pooled AUC must be well above chance
    assert results["pooled"]["roc_auc"] > 0.55

    n_test_days = sum(f["n_test_days"] for f in results["folds"])
    assert len(daily) == n_test_days
    assert (calib["fold_id"] == "pooled").any()

    for summary in results["per_fold"]:
        assert 0.0 <= summary["roc_auc"] <= 1.0
        assert 0.0 <= summary["brier"] <= 1.0
        assert summary["n_test_rows"] > 0
        assert summary["n_days_skipped_lt_top_n"] == 0  # 20 candidates every day
