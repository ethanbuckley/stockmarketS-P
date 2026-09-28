"""Tests for the Compustat->panel transformation.

The WRDS connection cannot be tested without credentials, so everything here
exercises the pure transformation against synthetic Compustat-shaped rows. When
access is approved the only unverified step left is the connection itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpha.wrds_source import membership_to_panel, to_panel  # noqa: E402


def secd(rows):
    """Build a comp.secd-shaped frame. rows: (date, tic, open, close, ajexdi, trfd)."""
    return pd.DataFrame(
        [
            {
                "gvkey": "001690",
                "iid": "01",
                "datadate": d,
                "tic": t,
                "cusip": "X",
                "prcod": o,
                "prccd": c,
                "prchd": max(o, c),
                "prcld": min(o, c),
                "cshtrd": 1e6,
                "ajexdi": aj,
                "trfd": tr,
            }
            for d, t, o, c, aj, tr in rows
        ]
    )


def test_decomposition_identity_is_exact():
    raw = secd(
        [
            ("2024-01-02", "AAA", 100.0, 101.0, 1.0, 1.0),
            ("2024-01-03", "AAA", 102.0, 99.5, 1.0, 1.0),
            ("2024-01-04", "AAA", 98.0, 103.25, 1.0, 1.0),
        ]
    )
    p = to_panel(raw)
    ok = p["r_cc"].notna()
    lhs = (1 + p.loc[ok, "r_overnight"]) * (1 + p.loc[ok, "r_intraday"]) - 1
    assert np.allclose(lhs, p.loc[ok, "r_cc"], atol=1e-12)
    assert p.attrs["max_decomposition_error"] < 1e-12


def test_split_is_neutral_and_lands_in_the_overnight_leg():
    """2-for-1 split: Compustat halves the raw price and drops ajexdi from 2 to
    1. Adjusted prices are continuous, so a flat stock must show a ZERO
    overnight return across the split, not -50%."""
    raw = secd(
        [
            ("2024-01-02", "AAA", 100.0, 100.0, 2.0, 1.0),  # pre-split
            ("2024-01-03", "AAA", 50.0, 50.0, 1.0, 1.0),
        ]
    )  # post-split
    p = to_panel(raw).sort_values("Date").reset_index(drop=True)
    assert p.loc[1, "r_overnight"] == pytest.approx(0.0, abs=1e-12)
    assert p.loc[1, "r_cc"] == pytest.approx(0.0, abs=1e-12)


def test_intraday_leg_is_adjustment_invariant():
    """The intraday leg shares the day's ajexdi/trfd, so it must equal the raw
    close/open ratio regardless of any corporate action that day."""
    raw = secd([("2024-01-02", "AAA", 48.0, 50.0, 7.5, 1.31)])
    p = to_panel(raw)
    assert p.loc[0, "r_intraday"] == pytest.approx(50.0 / 48.0 - 1, abs=1e-12)


def test_dividend_moves_the_overnight_leg_not_the_intraday_leg():
    """LPS assume corporate actions move prices overnight (p.196 n.9). A pure
    dividend (trfd step, flat raw price) must show up entirely overnight."""
    raw = secd([("2024-01-02", "AAA", 100.0, 100.0, 1.0, 1.00), ("2024-01-03", "AAA", 100.0, 100.0, 1.0, 1.02)])
    p = to_panel(raw).sort_values("Date").reset_index(drop=True)
    assert p.loc[1, "r_overnight"] == pytest.approx(0.02, abs=1e-12)
    assert p.loc[1, "r_intraday"] == pytest.approx(0.0, abs=1e-12)


def test_no_overnight_return_across_a_calendar_gap():
    """A ticker missing from the panel calendar must not have its overnight
    return computed across the hole."""
    raw = secd(
        [
            ("2024-01-02", "AAA", 100.0, 100.0, 1.0, 1.0),
            ("2024-01-03", "AAA", 100.0, 100.0, 1.0, 1.0),
            ("2024-01-05", "AAA", 100.0, 100.0, 1.0, 1.0),
            ("2024-01-02", "BBB", 10.0, 10.0, 1.0, 1.0),
            ("2024-01-03", "BBB", 10.0, 10.0, 1.0, 1.0),
            ("2024-01-04", "BBB", 10.0, 10.0, 1.0, 1.0),
            ("2024-01-05", "BBB", 10.0, 10.0, 1.0, 1.0),
        ]
    )
    p = to_panel(raw)
    gap = p[(p.Ticker == "AAA") & (p.Date == "2024-01-05")]
    assert gap["r_overnight"].isna().all()
    assert gap["r_intraday"].notna().all()  # intraday needs no prior day


def test_schema_matches_the_yfinance_source():
    """The rest of alpha/ reads ohlc_open.parquet; Compustat output must be a
    drop-in, or wide_panel.build() and decomposed.build_panel() break."""
    required = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "r_cc", "r_intraday", "r_overnight"}
    p = to_panel(secd([("2024-01-02", "AAA", 100.0, 101.0, 1.0, 1.0)]))
    assert required <= set(p.columns)
    assert p["Date"].dtype == np.dtype("datetime64[ns]")


def test_bad_rows_are_dropped_not_propagated():
    raw = secd(
        [
            ("2024-01-02", "AAA", 100.0, 101.0, 1.0, 1.0),
            ("2024-01-03", "AAA", 0.0, 101.0, 1.0, 1.0),  # zero open
            ("2024-01-04", "AAA", 100.0, -5.0, 1.0, 1.0),
        ]
    )  # negative close
    p = to_panel(raw)
    assert len(p) == 1
    assert np.isfinite(p[["Open", "Close"]].to_numpy()).all()


def test_membership_spans_expand_to_snapshots():
    """from/thru spans -> (asof, Ticker) snapshots; NULL thru means current."""
    raw = pd.DataFrame(
        [
            {"gvkey": "1", "iid": "01", "from": "2015-01-01", "thru": "2018-06-30", "Ticker": "OLD"},
            {"gvkey": "2", "iid": "01", "from": "2015-01-01", "thru": None, "Ticker": "STAY"},
            {"gvkey": "3", "iid": "01", "from": "2020-01-01", "thru": None, "Ticker": "NEW"},
        ]
    )
    m = membership_to_panel(raw, freq="YE")
    at2016 = set(m[m["asof"].dt.year == 2016]["Ticker"])
    at2021 = set(m[m["asof"].dt.year == 2021]["Ticker"])
    assert at2016 == {"OLD", "STAY"}  # OLD still in, NEW not yet
    assert at2021 == {"STAY", "NEW"}  # OLD has left
    assert "OLD" in set(m["Ticker"])  # the delisted name is retained
