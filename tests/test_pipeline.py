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
    monkeypatch.setattr(screener, "fetch_sp500_tickers", lambda: list(EQUITIES))
    monkeypatch.setattr(
        screener,
        "download_market_data",
        lambda tickers, start=screener.DATA_START_DATE: panel,
    )
    return panel


def test_master_dataframe_is_unfiltered_and_carries_metadata(stubbed_market):
    master_df, metadata = build_master_dataframe()
    # Raw panel: one row per (date, ticker), no dropna applied
    assert len(master_df) == N_DAYS * len(EQUITIES)
    assert metadata["tickers"] == EQUITIES
    assert metadata["last_price_date"] == str(stubbed_market.index[-1].date())
    assert "download_timestamp_utc" in metadata


def test_split_invariants(stubbed_market):
    train_df, latest_df = build_dataset()

    # Exactly one prediction row per ticker, on the final calendar date
    assert len(latest_df) == len(EQUITIES)
    assert set(latest_df["Ticker"]) == set(EQUITIES)
    assert (latest_df.index == stubbed_market.index[-1]).all()

    # Training rows are fully populated in features and label
    assert train_df[FEATURE_COLUMNS + ["Target"]].notna().all().all()

    # The label horizon means no training row within the last W bars
    assert train_df.index.max() == stubbed_market.index[-(FORWARD_WINDOW_DAYS + 1)]


def test_run_pipeline_leaderboard_schema(stubbed_market):
    pytest.importorskip("xgboost")
    leaderboard = screener.run_pipeline(skip_sentiment=True)

    assert list(leaderboard.columns) == ["Ticker", "Close", "Confidence", "Sentiment_Score"]
    assert leaderboard["Sentiment_Score"].isna().all()  # sentiment skipped -> blank
    assert leaderboard["Confidence"].is_monotonic_decreasing
    assert leaderboard["Confidence"].between(0, 100).all()


def test_train_model_smoke():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    n = 200
    train_df = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    train_df["Target"] = rng.integers(0, 2, n).astype(float)

    model = screener.train_model(train_df)
    proba = model.predict_proba(train_df[FEATURE_COLUMNS])

    assert proba.shape == (n, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
