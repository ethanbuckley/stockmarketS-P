"""Unit tests for triple-barrier labelling.

The labelling window for entry day i covers the next FORWARD_WINDOW_DAYS
bars, i+1 ... i+W inclusive. The entry bar itself is excluded. Label 1
requires the take-profit barrier to be crossed strictly before the
stop-loss barrier; every other outcome (stop first, tie, neither) is 0.
"""

import numpy as np

from screener import (
    FORWARD_WINDOW_DAYS,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    apply_triple_barrier_labels,
)

W = FORWARD_WINDOW_DAYS
N = 20
TP_PRICE = 100.0 * (1 + TAKE_PROFIT_PCT)   # exact barrier value for close=100
SL_PRICE = 100.0 * (1 - STOP_LOSS_PCT)


def flat(make_ohlcv, n=N):
    return make_ohlcv([100.0] * n)


def test_upper_barrier_first_labels_one(make_ohlcv):
    df = flat(make_ohlcv)
    df.iloc[3, df.columns.get_loc("High")] = TP_PRICE + 0.1
    out = apply_triple_barrier_labels(df)
    # Entries 0, 1 and 2 see bar 3 inside their forward windows
    assert out["Target"].iloc[0] == 1.0
    assert out["Target"].iloc[2] == 1.0


def test_lower_barrier_first_labels_zero(make_ohlcv):
    df = flat(make_ohlcv)
    df.iloc[1, df.columns.get_loc("Low")] = SL_PRICE - 0.1   # stop hit at bar 1
    df.iloc[3, df.columns.get_loc("High")] = TP_PRICE + 0.1  # take-profit later
    out = apply_triple_barrier_labels(df)
    assert out["Target"].iloc[0] == 0.0


def test_both_barriers_same_bar_is_conservative_zero(make_ohlcv):
    # When both barriers are crossed on the same bar the daily data cannot
    # say which was hit first; the implementation resolves the tie as 0.
    df = flat(make_ohlcv)
    df.iloc[2, df.columns.get_loc("High")] = TP_PRICE + 0.1
    df.iloc[2, df.columns.get_loc("Low")] = SL_PRICE - 0.1
    out = apply_triple_barrier_labels(df)
    assert out["Target"].iloc[1] == 0.0


def test_neither_barrier_hit_labels_zero(make_ohlcv):
    out = apply_triple_barrier_labels(flat(make_ohlcv))
    labelled = out["Target"].iloc[: N - W]
    assert (labelled == 0.0).all()


def test_entry_bar_itself_is_excluded(make_ohlcv):
    # A spike on the entry bar must not label the entry bar: the window
    # starts one bar after entry. This is the classic alignment bug.
    df = flat(make_ohlcv)
    spike = 6
    df.iloc[spike, df.columns.get_loc("High")] = TP_PRICE + 0.1
    out = apply_triple_barrier_labels(df)
    assert out["Target"].iloc[spike] == 0.0     # own bar excluded
    assert out["Target"].iloc[spike - 1] == 1.0  # previous entry sees it


def test_window_boundaries(make_ohlcv):
    # Bar i+W is the last bar inside the window; bar i+W+1 is outside.
    df = flat(make_ohlcv)
    hit = 6
    df.iloc[hit, df.columns.get_loc("High")] = TP_PRICE + 0.1
    out = apply_triple_barrier_labels(df)
    assert out["Target"].iloc[hit - W] == 1.0       # hit exactly at i+W
    assert out["Target"].iloc[hit - W - 1] == 0.0   # hit at i+W+1: outside


def test_touch_counts_as_crossing(make_ohlcv):
    # >= comparison: touching the barrier exactly is a hit.
    df = flat(make_ohlcv)
    df.iloc[2, df.columns.get_loc("High")] = TP_PRICE
    out = apply_triple_barrier_labels(df)
    assert out["Target"].iloc[1] == 1.0


def test_tail_rows_are_nan_and_length_preserved(make_ohlcv):
    out = apply_triple_barrier_labels(flat(make_ohlcv))
    assert len(out) == N
    assert out["Target"].iloc[N - W:].isna().all()
    assert out["Target"].iloc[: N - W].notna().all()


def test_short_series_yields_all_nan(make_ohlcv):
    for n in (1, 3, W):
        out = apply_triple_barrier_labels(flat(make_ohlcv, n=n))
        assert len(out) == n
        assert out["Target"].isna().all()


def test_minimum_viable_series_labels_one_row(make_ohlcv):
    out = apply_triple_barrier_labels(flat(make_ohlcv, n=W + 1))
    assert out["Target"].notna().sum() == 1
    assert np.isnan(out["Target"].iloc[-1])
