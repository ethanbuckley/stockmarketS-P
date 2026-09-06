"""
source.py - which price and membership files the analysis reads.

Everything downstream (wide_panel, decomposed, pit_analysis) goes through here,
so switching from the yfinance/Wikipedia layer to Compustat is a flag rather
than an edit in five places.

    ALPHA_SOURCE=auto       use Compustat if its files exist, else yfinance
    ALPHA_SOURCE=compustat  require Compustat; fail loudly if absent
    ALPHA_SOURCE=yfinance   force the original layer (for A/B comparison)
"""
from __future__ import annotations

import os
from pathlib import Path

ALPHA = Path(__file__).resolve().parent

COMPUSTAT_PRICES = ALPHA / "ohlc_compustat.parquet"
COMPUSTAT_MEMB = ALPHA / "sp500_membership_compustat.parquet"
YF_PRICES = [ALPHA / "ohlc_open.parquet", ALPHA / "ohlc_wide.parquet"]
WIKI_MEMB = ALPHA / "sp500_membership.parquet"


def mode() -> str:
    m = os.environ.get("ALPHA_SOURCE", "auto").strip().lower()
    if m not in ("auto", "compustat", "yfinance"):
        raise ValueError(f"ALPHA_SOURCE must be auto|compustat|yfinance, got {m!r}")
    if m == "auto":
        return "compustat" if COMPUSTAT_PRICES.exists() else "yfinance"
    return m


def price_files() -> list[Path]:
    if mode() == "compustat":
        if not COMPUSTAT_PRICES.exists():
            raise FileNotFoundError(
                f"{COMPUSTAT_PRICES} missing; run `python3 alpha/wrds_source.py prices`")
        return [COMPUSTAT_PRICES]
    return [p for p in YF_PRICES if p.exists()]


def membership_file() -> Path:
    if mode() == "compustat" and COMPUSTAT_MEMB.exists():
        return COMPUSTAT_MEMB
    return WIKI_MEMB


def describe() -> str:
    return (f"source={mode()}  prices={[p.name for p in price_files()]}  "
            f"membership={membership_file().name}")
