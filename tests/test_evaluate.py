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
    performance_summary,
    rank_auc,
    reliability_bins,
    run_backtest,
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
            "p_hat": np.linspace(1.0, 0.0, n),  # T00 ranked highest
            "Target": [1.0] * 8 + [0.0] * 12,  # positives = top 8 by p_hat
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
    assert row["daily_auc"] == pytest.approx(1.0)  # positives are exactly the top-ranked rows


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

    results, daily, calib, backtest = evaluate.run_evaluation(
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
    assert 0.5 < results["overall_daily"]["daily_auc_mean"] <= 1.0

    # Backtest: one row per session after the first pick, benchmarks present
    assert len(backtest) >= n_test_days - 1
    assert backtest["n_tranches"].max() == evaluate.BACKTEST_HOLD_DAYS
    assert (backtest["strategy_net"] <= backtest["strategy_gross"]).all()
    bt = results["backtest"]
    assert set(bt["series"]) == {"strategy_net", "strategy_gross", "universe_ew", "spy"}
    atr = results["baselines"]["atr_rank"]
    assert 0.0 <= atr["daily_auc_mean"] <= 1.0 and "strategy_net" in atr["backtest"]
    # The synthetic signal lives in feature 0, not ATR_Ratio, so the model must beat the baseline
    assert results["overall_daily"]["daily_auc_mean"] > atr["daily_auc_mean"]
    assert bt["series"]["strategy_net"]["n_days"] == len(backtest)
    assert sum(y["n_days"] for y in bt["by_year"]) == len(backtest)
    assert (calib["fold_id"] == "pooled").any()

    for summary in results["per_fold"]:
        assert 0.0 <= summary["roc_auc"] <= 1.0
        assert 0.0 <= summary["brier"] <= 1.0
        assert summary["n_test_rows"] > 0
        assert summary["n_days_skipped_lt_top_n"] == 0  # 20 candidates every day


def test_rank_auc_matches_definition():
    assert rank_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert rank_auc([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8]) == 0.0
    assert rank_auc([1, 0], [0.5, 0.5]) == 0.5  # ties count half
    assert np.isnan(rank_auc([1, 1], [0.3, 0.4]))  # one class only


def test_run_backtest_hand_computed():
    cal = pd.DatetimeIndex(pd.bdate_range("2021-01-01", periods=8))
    tickers = ["A", "B", "C"]
    # Constant daily returns so the book's arithmetic is checkable by hand
    returns = pd.DataFrame({"A": 0.01, "B": 0.02, "C": -0.01}, index=cal)
    spy = pd.Series(0.005, index=cal)
    # Picks on days 0 and 1: top-1 by p_hat is A both days
    scored = pd.DataFrame(
        {"Ticker": tickers * 2, "Target": [1, 0, 0] * 2, "p_hat": [0.9, 0.5, 0.1] * 2},
        index=cal[[0, 0, 0, 1, 1, 1]],
    )
    bt = run_backtest(scored, returns, spy, top_n=1, hold_days=2, cost_bps=10.0)
    bt = bt.set_index("date")

    # Day 1: one tranche (opened day 0), entry cost only
    assert bt.loc[cal[1], "strategy_gross"] == pytest.approx(0.01)
    assert bt.loc[cal[1], "strategy_net"] == pytest.approx(0.01 - 0.001)
    assert bt.loc[cal[1], "universe_ew"] == pytest.approx((0.01 + 0.02 - 0.01) / 3)
    # Day 2: tranche from day 0 exits (cost), tranche from day 1 enters (cost)
    assert bt.loc[cal[2], "n_tranches"] == 2
    assert bt.loc[cal[2], "strategy_net"] == pytest.approx(0.01 - 0.001)
    # Day 3: only the day-1 tranche remains, exiting
    assert bt.loc[cal[3], "n_tranches"] == 1
    assert bt.loc[cal[3], "strategy_net"] == pytest.approx(0.01 - 0.001)
    assert cal[4] not in bt.index  # nothing held after the last tranche exits
    assert (bt["spy"] == 0.005).all()


def test_performance_summary_constant_return():
    r = pd.Series([0.001] * 252)
    s = performance_summary(r)
    assert s["total_return"] == pytest.approx(1.001**252 - 1)
    assert s["ann_return"] == pytest.approx(1.001**252 - 1)
    assert s["max_drawdown"] == 0.0
    assert s["n_days"] == 252
