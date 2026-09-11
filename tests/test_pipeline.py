"""Tests for the dataset build/split logic and model training, run on a
synthetic market-data panel with all network access stubbed out."""

import numpy as np
import pandas as pd
import pytest

import screener
from screener import (
    FEATURE_COLUMNS,
    FORWARD_WINDOW_DAYS,
    MACRO_TICKERS,
    build_dataset,
    build_master_dataframe,
)

EQUITIES = ["AAA", "BBB"]
N_DAYS = 80


def constituents(tickers, date_added=None) -> pd.DataFrame:
    """Wikipedia-shaped constituent table; unknown join dates by default."""
    return pd.DataFrame(
        {"Ticker": list(tickers), "Date_Added": pd.to_datetime([date_added] * len(tickers))}
    )


def synthetic_panel() -> pd.DataFrame:
    """A yfinance-shaped panel: MultiIndex columns (field, ticker)."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2024-01-01", periods=N_DAYS)
    frames = {}
    for t in EQUITIES + MACRO_TICKERS:
        base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, N_DAYS)))
        frames[("Close", t)] = base
        frames[("High", t)] = base * 1.01
        frames[("Low", t)] = base * 0.99
        frames[("Volume", t)] = rng.integers(100_000, 1_000_000, N_DAYS).astype(float)
    panel = pd.DataFrame(frames, index=index)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    return panel


@pytest.fixture
def stubbed_market(monkeypatch):
    panel = synthetic_panel()
    monkeypatch.setattr(screener, "fetch_sp500_constituents", lambda: constituents(EQUITIES))
    monkeypatch.setattr(
        screener,
        "download_market_data",
        lambda tickers, start=screener.DATA_START_DATE: panel,
    )
    return panel


def test_join_date_truncates_pre_membership_rows(monkeypatch):
    panel = synthetic_panel()
    joined = panel.index[30]
    table = constituents(EQUITIES)
    table.loc[table["Ticker"] == "BBB", "Date_Added"] = joined
    monkeypatch.setattr(screener, "fetch_sp500_constituents", lambda: table)
    monkeypatch.setattr(screener, "download_market_data", lambda tickers, start=None: panel)

    master_df, metadata = build_master_dataframe()

    aaa = master_df[master_df["Ticker"] == "AAA"]
    bbb = master_df[master_df["Ticker"] == "BBB"]
    assert len(aaa) == N_DAYS                      # unknown join date: full history
    assert bbb.index.min() == joined               # member only from its join date
    assert len(bbb) == N_DAYS - 30
    # Features on the join day still use the pre-join price history: the
    # 20-day rolling windows are already warm, so the first row is complete.
    assert bbb.iloc[0][FEATURE_COLUMNS].notna().all()
    assert metadata["join_date_truncation"] == {
        "applied": True, "n_tickers_truncated": 1, "n_tickers_unknown_join_date": 1,
    }


def test_volume_is_not_forward_filled():
    panel = synthetic_panel()
    gap = panel.index[40]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        if (field, "AAA") in panel.columns:
            panel.loc[gap, (field, "AAA")] = np.nan
    panel[("Open", "AAA")] = panel[("Close", "AAA")]

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            return panel.copy()

    import screener as mod
    original = mod.yf
    mod.yf = FakeYF
    try:
        out = mod.download_market_data(EQUITIES)
    finally:
        mod.yf = original

    assert out.loc[gap, ("Close", "AAA")] == panel.loc[panel.index[39], ("Close", "AAA")]
    assert np.isnan(out.loc[gap, ("Volume", "AAA")])


def test_master_dataframe_is_unfiltered_and_carries_metadata(stubbed_market):
    master_df, metadata = build_master_dataframe()
    # Raw panel: one row per (date, ticker), no dropna applied
    assert len(master_df) == N_DAYS * len(EQUITIES)
    assert metadata["tickers"] == EQUITIES
    assert metadata["last_price_date"] == str(stubbed_market.index[-1].date())
    assert "download_timestamp_utc" in metadata


def test_tickers_with_no_prices_are_excluded(monkeypatch):
    # A symbol Yahoo knows but returns no rows for must not reach the panel
    # or the metadata; it would otherwise inflate n_tickers in the artefacts.
    panel = synthetic_panel()
    for field in ("Close", "High", "Low", "Volume"):
        panel[(field, "GHOST")] = np.nan
    monkeypatch.setattr(screener, "fetch_sp500_constituents", lambda: constituents([*EQUITIES, "GHOST"]))
    monkeypatch.setattr(screener, "download_market_data", lambda tickers, start=None: panel)

    master_df, metadata = build_master_dataframe()
    assert metadata["tickers"] == EQUITIES
    assert "GHOST" not in set(master_df["Ticker"])


def test_split_invariants(stubbed_market):
    train_df, latest_df, master_df = build_dataset()
    assert len(master_df) == N_DAYS * len(EQUITIES)

    # Exactly one prediction row per ticker, on the final calendar date
    assert len(latest_df) == len(EQUITIES)
    assert set(latest_df["Ticker"]) == set(EQUITIES)
    assert (latest_df.index == stubbed_market.index[-1]).all()

    # Training rows are fully populated in features and label
    assert train_df[FEATURE_COLUMNS + ["Target"]].notna().all().all()

    # The label horizon means no training row within the last W bars
    assert train_df.index.max() == stubbed_market.index[-(FORWARD_WINDOW_DAYS + 1)]


def test_run_pipeline_output_schema(stubbed_market):
    pytest.importorskip("xgboost")
    leaderboard, prices = screener.run_pipeline(skip_sentiment=True)

    assert list(leaderboard.columns) == ["Ticker", "Close", "Confidence", "Sentiment_Score", "Signal_Pool"]
    assert leaderboard["Sentiment_Score"].isna().all()  # sentiment skipped -> blank
    assert leaderboard["Confidence"].is_monotonic_decreasing
    assert leaderboard["Confidence"].between(0, 100).all()
    # Two tickers: both land in the long pool (top-N wins the overlap tie)
    assert set(leaderboard["Signal_Pool"]) == {screener.LONG_POOL}

    # Price history covers every candidate, most recent dates last
    assert list(prices.columns) == leaderboard["Ticker"].tolist()
    assert prices.index.name == "Date"
    assert prices.index.is_monotonic_increasing
    assert prices.index[-1] == stubbed_market.index[-1]
    assert len(prices) == min(screener.PRICE_HISTORY_DAYS, N_DAYS)


def test_score_and_rank_labels_both_pools():
    pytest.importorskip("xgboost")

    class ConstantModel:
        def predict_proba(self, X):
            p = np.linspace(0.9, 0.1, len(X))
            return np.column_stack([1 - p, p])

    n = screener.TOP_N_CANDIDATES + screener.BOTTOM_N_CANDIDATES + 5
    latest = pd.DataFrame(np.zeros((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    latest["Ticker"] = [f"T{i:02d}" for i in range(n)]

    focus = screener.score_and_rank(ConstantModel(), latest)
    assert (focus["Signal_Pool"] == screener.LONG_POOL).sum() == screener.TOP_N_CANDIDATES
    assert (focus["Signal_Pool"] == screener.SHORT_POOL).sum() == screener.BOTTOM_N_CANDIDATES
    assert focus.loc[focus["Signal_Pool"] == screener.LONG_POOL, "Probability"].min() > \
        focus.loc[focus["Signal_Pool"] == screener.SHORT_POOL, "Probability"].max()


def test_train_model_smoke_without_dates_uses_ceiling():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    n = 200
    train_df = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    train_df["Target"] = rng.integers(0, 2, n).astype(float)

    assert screener.choose_n_estimators(train_df) is None  # RangeIndex: no slice
    model = screener.train_model(train_df)
    proba = model.predict_proba(train_df[FEATURE_COLUMNS])

    assert model.get_params()["n_estimators"] == screener.XGB_PARAMS["n_estimators"]
    assert proba.shape == (n, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_early_stopping_picks_fewer_trees_than_ceiling():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(2)
    cal = pd.bdate_range("2019-01-01", periods=600)
    n_per_day = 5
    idx = cal.repeat(n_per_day)
    X = rng.normal(size=(len(idx), len(FEATURE_COLUMNS)))
    train_df = pd.DataFrame(X, columns=FEATURE_COLUMNS, index=idx)
    # Weak signal in feature 0 plus noise: more trees soon overfit
    train_df["Target"] = ((X[:, 0] + rng.normal(0, 2, len(idx))) > 0).astype(float)

    n_trees = screener.choose_n_estimators(train_df)
    assert n_trees is not None
    assert 1 <= n_trees < screener.XGB_PARAMS["n_estimators"]

    model = screener.train_model(train_df)
    assert model.get_params()["n_estimators"] == n_trees
