"""Shared fixtures for the screener test suite. All tests are network-free."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_ohlcv():
    """Builds a minimal OHLCV frame on a business-day index.

    Highs and lows default to the closes, so barrier crossings only happen
    where a test explicitly places them.
    """

    def _make(closes, highs=None, lows=None, volume=None) -> pd.DataFrame:
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        highs = np.asarray(highs, dtype=float) if highs is not None else closes.copy()
        lows = np.asarray(lows, dtype=float) if lows is not None else closes.copy()
        volume = np.asarray(volume, dtype=float) if volume is not None else np.full(n, 1_000_000.0)
        index = pd.bdate_range("2024-01-01", periods=n)
        return pd.DataFrame(
            {"Close": closes, "High": highs, "Low": lows, "Volume": volume},
            index=index,
        )

    return _make
