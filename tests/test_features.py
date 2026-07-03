"""Unit tests for the technical-indicator feature calculations."""

import numpy as np
import pandas as pd

from screener import build_technical_features


def test_rsi_limits_and_bounds(make_ohlcv):
    rising = build_technical_features(make_ohlcv(np.arange(100.0, 160.0)))
    assert np.isclose(rising["RSI"].iloc[-1], 100.0)

    falling = build_technical_features(make_ohlcv(np.arange(160.0, 100.0, -1.0)))
    assert np.isclose(falling["RSI"].iloc[-1], 0.0)

    rng = np.random.default_rng(42)
    walk = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    noisy = build_technical_features(make_ohlcv(walk))
    rsi = noisy["RSI"].dropna()
    assert ((rsi >= 0.0) & (rsi <= 100.0)).all()


def test_macd_is_zero_for_constant_price(make_ohlcv):
    out = build_technical_features(make_ohlcv([100.0] * 60))
    assert np.allclose(out["MACD"], 0.0)


def test_bb_position_matches_definition(make_ohlcv):
    # Pin the wiring (20-day window, 2 standard deviations) against an
    # independently written version of the formula.
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    df = make_ohlcv(closes)
    out = build_technical_features(df.copy())

    close = pd.Series(closes, index=df.index)
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    expected = (close - (sma - 2 * std)) / ((sma + 2 * std) - (sma - 2 * std))

    pd.testing.assert_series_equal(out["BB_Position"], expected, check_names=False)
    assert out["BB_Position"].iloc[:19].isna().all()


def test_price_to_vwap_is_one_for_constant_series(make_ohlcv):
    out = build_technical_features(make_ohlcv([100.0] * 30))
    assert np.allclose(out["Price_to_VWAP"].iloc[5:], 1.0)


def test_atr_ratio_is_zero_for_constant_series(make_ohlcv):
    out = build_technical_features(make_ohlcv([100.0] * 30))
    assert np.allclose(out["ATR_Ratio"].iloc[14:], 0.0)


def test_volume_surge_is_one_for_constant_volume(make_ohlcv):
    out = build_technical_features(make_ohlcv([100.0] * 40))
    assert np.allclose(out["Volume_Surge"].iloc[19:], 1.0)


def test_day_of_week_is_weekday_index(make_ohlcv):
    out = build_technical_features(make_ohlcv([100.0] * 10))
    assert out["Day_Of_Week"].isin(range(5)).all()  # business days only
