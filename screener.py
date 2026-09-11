"""
S&P 500 stock screener
=========================
A two-stage stock screening pipeline that combines:
  1. XGBoost binary classifier trained on technical indicators (quantitative signal)
  2. FinBERT NLP sentiment analysis on live news headlines (qualitative signal)

The model predicts whether a stock will hit a +4% take-profit before a -4% stop-loss
within a 5-day forward window, a "triple-barrier labelling" approach from financial ML.

Usage:
    python screener.py [--tickers-limit N] [--skip-sentiment]

Tunable settings live in config.py.

Author: Ethan Buckley
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import warnings
from io import StringIO
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

from config import (  # noqa: F401  (re-exported for callers and tests)
    BOTTOM_N_CANDIDATES,
    DATA_START_DATE,
    EARLY_STOPPING_MIN_VALIDATION_DAYS,
    EARLY_STOPPING_ROUNDS,
    EARLY_STOPPING_VALIDATION_FRACTION,
    FEATURE_COLUMNS,
    FORWARD_WINDOW_DAYS,
    LONG_POOL,
    MACRO_TICKERS,
    NEWS_ARTICLES_PER_TICKER,
    NEWS_MAX_AGE_DAYS,
    PRICE_HISTORY_DAYS,
    SHORT_POOL,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TOP_N_CANDIDATES,
    XGB_PARAMS,
)

if TYPE_CHECKING:
    from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

PRICE_FIELDS = ["Open", "High", "Low", "Close"]


# =============================================================================
# STEP 1: DATA ACQUISITION
# =============================================================================


def fetch_sp500_constituents() -> pd.DataFrame:
    """
    Scrapes the current S&P 500 constituent table from Wikipedia.

    Returns a DataFrame with columns:
      - Ticker: symbol with dots replaced by hyphens (BRK.B -> BRK-B) to
        match the Yahoo Finance format
      - Date_Added: when the company joined the index (NaT if unparseable).
        build_master_dataframe drops each ticker's rows before this date so
        the training panel does not contain pre-membership history.

    Raises RuntimeError if the page cannot be fetched or the constituents
    table cannot be parsed: a silently wrong universe would corrupt
    everything downstream, so this fails loudly.
    """
    tables = _read_wikipedia_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if not tables or "Symbol" not in tables[0].columns:
        raise RuntimeError(
            "Could not parse the S&P 500 constituents table from Wikipedia (the page layout may have changed)."
        )
    table = tables[0]
    constituents = pd.DataFrame(
        {
            "Ticker": table["Symbol"].str.replace(".", "-", regex=False),
            "Date_Added": pd.to_datetime(table.get("Date added"), errors="coerce"),
        }
    )
    if len(constituents) < 400:
        raise RuntimeError(
            f"Parsed only {len(constituents)} tickers from Wikipedia; expected roughly 500. "
            "Refusing to continue with a partial universe."
        )
    return constituents


def fetch_sp500_tickers() -> list[str]:
    """Current S&P 500 symbols in Yahoo Finance format."""
    return fetch_sp500_constituents()["Ticker"].tolist()


def _read_wikipedia_tables(url: str) -> list[pd.DataFrame]:
    headers = {
        # Mimic a browser request; Wikipedia blocks the default requests user-agent
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/116.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def fetch_sp500_changes() -> pd.DataFrame:
    """
    Index additions and removals from Wikipedia's "Historical components of
    the S&P 500" article (one row per change, back to 1976).

    Returns columns Date, Added, Removed (tickers in Yahoo format; NaN where a
    row only added or only removed). Raises RuntimeError if the table is not
    found, for the same reason fetch_sp500_constituents does.
    """
    url = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
    tables = _read_wikipedia_tables(url)
    table = next((t for t in tables if any("Removed" in str(c) for c in t.columns)), None)
    if table is None or table.shape[1] < 5:
        raise RuntimeError("Could not find the S&P 500 change-history table on Wikipedia.")
    table = table.iloc[:, :5].copy()
    table.columns = ["Date", "Added", "Added_Security", "Removed", "Removed_Security"]

    def symbols(col: pd.Series) -> pd.Series:
        return col.astype("string").str.strip().str.replace(".", "-", regex=False)

    changes = pd.DataFrame(
        {
            "Date": pd.to_datetime(table["Date"], errors="coerce"),
            "Added": symbols(table["Added"]),
            "Removed": symbols(table["Removed"]),
        }
    ).dropna(subset=["Date"])
    if len(changes) < 100:
        raise RuntimeError(f"Parsed only {len(changes)} index changes from Wikipedia; expected hundreds.")
    return changes.reset_index(drop=True)


def build_membership(constituents: pd.DataFrame, changes: pd.DataFrame, start: str = DATA_START_DATE) -> pd.DataFrame:
    """
    Point-in-time S&P 500 membership intervals from the current constituent
    table and the change history.

    One row per (ticker, interval) with columns Ticker, Start, End,
    Is_Current. Current members get [Date_Added, NaT). A ticker removed on or
    after `start` and not currently in the index gets [last recorded
    addition before that removal, removal date); Start is NaT when no
    addition is on record, meaning "member since before the data starts".
    Removed symbols that coincide with a current symbol (ticker reuse) are
    skipped rather than risk attaching one company's history to another.

    Membership is by symbol, so a company whose ticker changed while in the
    index (FB -> META) is covered by its current symbol's history on Yahoo.
    """
    start_ts = pd.Timestamp(start)
    current = set(constituents["Ticker"])
    rows = [(r.Ticker, r.Date_Added, pd.NaT, True) for r in constituents.itertuples(index=False)]

    removals = changes.dropna(subset=["Removed"])
    removals = removals[(removals["Date"] >= start_ts) & ~removals["Removed"].isin(current)]
    additions = changes.dropna(subset=["Added"])
    for ticker, group in removals.groupby("Removed"):
        for end in sorted(group["Date"]):
            prior = additions[(additions["Added"] == ticker) & (additions["Date"] < end)]
            begin = prior["Date"].max() if len(prior) else pd.NaT
            rows.append((ticker, begin, end, False))

    membership = pd.DataFrame(rows, columns=["Ticker", "Start", "End", "Is_Current"])
    membership["Start"] = pd.to_datetime(membership["Start"])
    membership["End"] = pd.to_datetime(membership["End"])
    return membership.sort_values(["Ticker", "End"], na_position="last").reset_index(drop=True)


def fetch_sp500_membership() -> pd.DataFrame:
    """Live point-in-time membership table; see build_membership."""
    return build_membership(fetch_sp500_constituents(), fetch_sp500_changes())


def membership_mask(index: pd.DatetimeIndex, intervals: pd.DataFrame) -> np.ndarray:
    """True where a date falls inside any of a ticker's [Start, End) intervals."""
    mask = np.zeros(len(index), dtype=bool)
    for r in intervals.itertuples(index=False):
        inside = np.ones(len(index), dtype=bool)
        if pd.notna(r.Start):
            inside &= index >= r.Start
        if pd.notna(r.End):
            inside &= index < r.End
        mask |= inside
    return mask


def download_market_data(tickers: list[str], start: str = DATA_START_DATE) -> pd.DataFrame:
    """
    Downloads OHLCV data for all tickers from Yahoo Finance.

    Prices are forward-filled so that macro series with their own holiday
    calendars (^TNX closes on bond-market holidays) still align with equity
    rows. Volume is NOT filled: a day with no trade has no volume, and
    leaving it NaN means the volume features are NaN and the row is dropped
    rather than trained on as if it were a real session.

    auto_adjust is passed explicitly because yfinance changed its default
    between versions; adjusted prices are required so that splits and
    dividends do not appear as spurious barrier crossings.
    """
    all_tickers = tickers + MACRO_TICKERS
    data = yf.download(all_tickers, start=start, progress=False, threads=True, auto_adjust=True)
    if data is None or data.empty:
        raise RuntimeError("yfinance returned no market data.")

    missing = [t for t in tickers if t not in tickers_with_data(data)]
    if missing:
        shown = ", ".join(missing[:10]) + (", ..." if len(missing) > 10 else "")
        logger.warning("No price data for %d ticker(s): %s", len(missing), shown)

    # Downstream code joins per-ticker frames against the macro series on this
    # index, so normalise it to tz-naive once here rather than per ticker.
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    is_price = data.columns.get_level_values(0).isin(PRICE_FIELDS)
    data.loc[:, is_price] = data.loc[:, is_price].ffill()
    return data


def tickers_with_data(raw_data: pd.DataFrame) -> list[str]:
    """Tickers in a yfinance panel that returned at least one closing price."""
    close = raw_data["Close"]
    return [t for t in close.columns if close[t].notna().any()]


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================


def apply_triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies triple-barrier labelling to create the target variable.

    For each day i, we look at the next FORWARD_WINDOW_DAYS bars:
    - If the high crosses TAKE_PROFIT_PCT above the close -> label 1 (bullish)
    - If the low crosses STOP_LOSS_PCT below the close  -> label 0 (bearish)
    - If neither barrier is hit within the window       -> label 0 (bearish by default)

    This is preferable to simple N-day forward returns because it more closely
    models how a real trade with risk management would play out.

    Implementation uses sliding_window_view for a fully vectorized O(n*W) NumPy
    pass instead of nested Python loops, making it ~100-200x faster on long series.
    """
    W = FORWARD_WINDOW_DAYS
    closes = df["Close"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    n = len(closes)

    if n <= W:
        # Too short for any forward window; sliding_window_view would raise.
        df["Target"] = np.nan
        return df

    # sliding_window_view(arr, W)[k] == arr[k : k+W].
    # We want the window starting one bar *after* entry i, so we use index i+1:
    #   forward_highs[i] = highs[i+1 : i+1+W]
    # sliding_window_view gives (n-W+1) windows; slicing [1:] drops window 0
    # (which starts at bar 0) so row i of the result covers bars i+1...i+W.
    # Valid for i in 0 ... n-W-1, matching the original loop range.
    forward_highs = sliding_window_view(highs, W)[1:]  # shape (n-W, W)
    forward_lows = sliding_window_view(lows, W)[1:]  # shape (n-W, W)

    upper_barriers = (closes[: n - W] * (1 + TAKE_PROFIT_PCT))[:, None]  # (n-W, 1)
    lower_barriers = (closes[: n - W] * (1 - STOP_LOSS_PCT))[:, None]  # (n-W, 1)

    hit_upper = forward_highs >= upper_barriers  # (n-W, W) bool
    hit_lower = forward_lows <= lower_barriers  # (n-W, W) bool

    # Index of first crossing within the window; W means "never crossed"
    first_upper = np.where(hit_upper.any(axis=1), np.argmax(hit_upper, axis=1), W)
    first_lower = np.where(hit_lower.any(axis=1), np.argmax(hit_lower, axis=1), W)

    # Label 1 only when the upper barrier is crossed strictly before the lower one
    target = np.where(first_upper < first_lower, 1.0, 0.0)

    full_target = np.full(n, np.nan)
    full_target[: n - W] = target
    df["Target"] = full_target
    return df


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators that form the model's feature set.

    Features are grouped by what information they encode:
      - Momentum:   RSI, MACD, lagged returns
      - Volatility: Bollinger Band position, ATR ratio
      - Volume:     Volume surge, VWAP deviation
      - Market:     Macro returns (SPY, QQQ, SMH), VIX, TNX changes
      - Relative:   Stock return vs macro benchmarks (measures sector outperformance)
    """
    # --- Momentum: RSI (Exponential Weighted, 14-period equivalent) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # --- Momentum: MACD (12/26 EMA crossover) ---
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    # --- Volatility: Bollinger Band Position ---
    # 0 = at lower band, 1 = at upper band; values outside [0,1] mean a breakout
    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)
    df["BB_Position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower)

    # --- Volume: VWAP deviation (5-day rolling) ---
    # How far the current price is from its volume-weighted average, a proxy for fair value
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap_5d = (typical_price * df["Volume"]).rolling(5).sum() / df["Volume"].rolling(5).sum()
    df["Price_to_VWAP"] = df["Close"] / vwap_5d

    # --- Volatility: Average True Range ratio (normalised by price) ---
    # Measures recent volatility relative to the stock's price level
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift()).abs()
    low_prev_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR_Ratio"] = true_range.rolling(14).mean() / df["Close"]

    # --- Daily return and volume surge ---
    df["Return"] = df["Close"].pct_change()
    df["Volume_Surge"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- Calendar feature ---
    # Day-of-week can encode known seasonal patterns (e.g. Monday effect)
    df["Day_Of_Week"] = df.index.dayofweek

    return df


def build_macro_features(df: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds macro-market context features using benchmark ETFs and indices.

    These features tell the model whether a stock is moving with or against
    the broader market, which is critical for distinguishing alpha from beta.
    """
    df["SPY_Return"] = raw_data["Close"]["SPY"].pct_change()
    df["QQQ_Return"] = raw_data["Close"]["QQQ"].pct_change()
    df["SMH_Return"] = raw_data["Close"]["SMH"].pct_change()
    df["VIX_Change"] = raw_data["Close"]["^VIX"].pct_change()
    df["TNX_Change"] = raw_data["Close"]["^TNX"].pct_change()

    # Relative performance: stock return minus benchmark return
    df["Rel_SPY"] = df["Return"] - df["SPY_Return"]
    df["Rel_QQQ"] = df["Return"] - df["QQQ_Return"]
    df["Rel_SMH"] = df["Return"] - df["SMH_Return"]

    # Lagged returns: give the model access to recent price history
    for lag in range(1, 4):
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)
        df[f"QQQ_Lag_{lag}"] = df["QQQ_Return"].shift(lag)

    return df


def process_ticker(ticker: str, raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline for a single ticker.

    Returns a DataFrame with OHLCV data, labels, and all features.
    """
    df = pd.DataFrame(index=raw_data.index)
    df["Close"] = raw_data["Close"][ticker]
    df["High"] = raw_data["High"][ticker]
    df["Low"] = raw_data["Low"][ticker]
    df["Volume"] = raw_data["Volume"][ticker]
    df["Ticker"] = ticker

    df = apply_triple_barrier_labels(df)
    df = build_technical_features(df)
    df = build_macro_features(df, raw_data)

    return df


def build_master_dataframe(
    start: str = DATA_START_DATE,
    tickers_limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetches data and builds the labelled master feature panel for the
    point-in-time S&P 500 universe.

    Returns (master_df, metadata):
      - master_df: one row per (date, ticker) on the union yfinance calendar
        (tz-naive DatetimeIndex), with OHLCV, Ticker, all FEATURE_COLUMNS and
        Target. Features are computed on each ticker's full price history,
        but a row is kept only for dates on which the ticker was an index
        member (see build_membership): current members from their join date,
        and companies removed since `start` up to their removal date, where
        Yahoo still has their prices. No dropna is applied here; callers do
        their own filtering.
      - metadata: download timestamp, ticker lists, last price date and a
        universe summary (how many removed tickers could and could not be
        recovered), so downstream artefacts record exactly what was used.
    """
    print("Fetching S&P 500 constituents and change history from Wikipedia...")
    membership = fetch_sp500_membership()
    if tickers_limit is not None:
        # Smoke runs: first N current members only, no removed tickers.
        keep = membership.loc[membership["Is_Current"], "Ticker"].head(tickers_limit)
        membership = membership[membership["Ticker"].isin(keep)]
    tickers = membership["Ticker"].unique().tolist()
    current_set = set(membership.loc[membership["Is_Current"], "Ticker"])
    removed_set = set(tickers) - current_set

    print(f"Downloading market data for {len(tickers)} stocks ({len(removed_set)} former members) since {start}...")
    raw_data = download_market_data(tickers, start=start)

    # Tickers that returned no data at all cannot be processed
    with_data = set(tickers_with_data(raw_data))
    available = [t for t in tickers if t in with_data]

    print("Engineering features for all tickers...")
    frames = []
    for ticker in available:
        df = process_ticker(ticker, raw_data)
        intervals = membership[membership["Ticker"] == ticker]
        frames.append(df[membership_mask(df.index, intervals)])
    master_df = pd.concat(frames)
    master_df = master_df[master_df["Close"].notna()]

    metadata = {
        "download_timestamp_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tickers": available,
        "current_tickers": [t for t in available if t in current_set],
        "last_price_date": str(master_df.index.max().date()),
        "universe": {
            "type": "point_in_time",
            "n_current": len(current_set & with_data),
            "n_removed_since_start": len(removed_set),
            "n_removed_with_price_data": len(removed_set & with_data),
            "removed_without_price_data": sorted(removed_set - with_data),
        },
    }
    return master_df, metadata


class Dataset(NamedTuple):
    train: pd.DataFrame  # rows with complete features and a valid label
    latest: pd.DataFrame  # most recent complete-feature row per ticker (today's signal)
    master: pd.DataFrame  # the unfiltered panel (for price history export)


def build_dataset(tickers_limit: int | None = None) -> Dataset:
    """Builds the frames the screener needs from one download."""
    master_df, metadata = build_master_dataframe(tickers_limit=tickers_limit)

    train_df = master_df.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()
    # Only current members can be today's candidates; a former member's last
    # row sits at its removal date.
    latest_df = (
        master_df[master_df["Ticker"].isin(metadata["current_tickers"])]
        .dropna(subset=FEATURE_COLUMNS)
        .groupby("Ticker")
        .tail(1)
        .copy()
    )
    return Dataset(train_df, latest_df, master_df)


def candidate_price_history(
    master_df: pd.DataFrame, tickers: list[str], days: int = PRICE_HISTORY_DAYS
) -> pd.DataFrame:
    """
    Wide frame of adjusted closes (one column per ticker) over the last
    `days` trading dates in the panel. Written next to the signals so the
    dashboard's Monte Carlo tab needs no live price download.
    """
    closes = master_df.loc[master_df["Ticker"].isin(tickers), ["Ticker", "Close"]]
    wide = closes.pivot_table(index=closes.index, columns="Ticker", values="Close")
    wide = wide.reindex(columns=[t for t in tickers if t in wide.columns])
    wide.index.name = "Date"
    return wide.tail(days)


# =============================================================================
# STEP 3: MODEL TRAINING
# =============================================================================


def train_model(train_df: pd.DataFrame) -> XGBClassifier:
    """
    Trains a single XGBoost classifier across all S&P 500 stocks.

    Using one "universal" model (rather than one per stock) means the model
    learns patterns that generalise across the market, not just overfit to
    one ticker's history. The output probability (predict_proba) is used
    as the ranking score, not a hard buy/sell decision.

    The number of trees is chosen by early stopping (choose_n_estimators)
    and the final model is refit on all of train_df with that count, so the
    returned model has seen every training row. XGB_PARAMS["n_estimators"]
    is the ceiling.
    """
    # Imported lazily so that importing this module (tests, evaluation,
    # generate_signals) does not require xgboost to be installed.
    from xgboost import XGBClassifier

    n_estimators = choose_n_estimators(train_df) or XGB_PARAMS["n_estimators"]
    model = XGBClassifier(**{**XGB_PARAMS, "n_estimators": n_estimators})
    model.fit(train_df[FEATURE_COLUMNS], train_df["Target"])
    return model


def choose_n_estimators(train_df: pd.DataFrame) -> int | None:
    """
    Picks the boosting-round count by early stopping on the most recent
    EARLY_STOPPING_VALIDATION_FRACTION of training dates.

    The FORWARD_WINDOW_DAYS trading days before the validation slice are
    purged from the fit set, exactly as evaluate.py purges before a test
    block, so the validation labels cannot overlap the fit labels. Because
    the slice is carved from the training data, this never touches a
    walk-forward test period.

    Returns None when the frame has no DatetimeIndex or too little history
    for a meaningful slice; callers then fall back to the ceiling.
    """
    from xgboost import XGBClassifier

    if not isinstance(train_df.index, pd.DatetimeIndex):
        return None
    dates = train_df.index.unique().sort_values()
    n_val = int(len(dates) * EARLY_STOPPING_VALIDATION_FRACTION)
    if n_val < EARLY_STOPPING_MIN_VALIDATION_DAYS or len(dates) - n_val - FORWARD_WINDOW_DAYS - 1 < 1:
        return None

    val_start = dates[-n_val]
    fit_end = dates[-n_val - FORWARD_WINDOW_DAYS - 1]
    fit = train_df[train_df.index <= fit_end]
    val = train_df[train_df.index >= val_start]
    if fit["Target"].nunique() < 2 or val["Target"].nunique() < 2:
        return None

    probe = XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    probe.fit(
        fit[FEATURE_COLUMNS],
        fit["Target"],
        eval_set=[(val[FEATURE_COLUMNS], val["Target"])],
        verbose=False,
    )
    return int(probe.best_iteration) + 1


def score_and_rank(model: XGBClassifier, latest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores the latest row per ticker and returns the focus list for sentiment
    analysis: the TOP_N_CANDIDATES highest and BOTTOM_N_CANDIDATES lowest by
    predicted probability. Adds "Probability" and "Signal_Pool" (LONG_POOL
    or SHORT_POOL) columns; no sentiment yet. Membership of a pool is the
    model's signal; there is no absolute probability threshold, because a
    calibrated probability of ~50% is already well above the ~25-30% base
    rate and an absolute cut would rarely, if ever, be reached.
    """
    latest_df = latest_df.copy()
    latest_df["Probability"] = model.predict_proba(latest_df[FEATURE_COLUMNS])[:, 1]

    top_candidates = latest_df.nlargest(TOP_N_CANDIDATES, "Probability").assign(Signal_Pool=LONG_POOL)
    bottom_candidates = latest_df.nsmallest(BOTTOM_N_CANDIDATES, "Probability").assign(Signal_Pool=SHORT_POOL)
    # On a tiny universe (--tickers-limit) the two ends can overlap; the
    # long pool wins the tie.
    return pd.concat([top_candidates, bottom_candidates]).drop_duplicates(subset="Ticker").copy()


# =============================================================================
# STEP 4: SENTIMENT ANALYSIS (FINBERT)
# =============================================================================


def load_sentiment_model():
    """
    Loads the FinBERT model, a BERT variant fine-tuned on financial text.
    Returns a HuggingFace sentiment-analysis pipeline.
    """
    # Imported lazily: transformers (and its torch dependency) are only
    # needed when sentiment is actually requested. Quieten their logging so
    # only the screener's own progress output appears.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def _headline_and_age(article: dict, now: pd.Timestamp) -> tuple[str | None, float | None]:
    """
    Pulls (title, age in days) out of a yfinance news item. yfinance has used
    two payload shapes: a flat dict with "title"/"providerPublishTime" (epoch
    seconds) and a nested {"content": {"title", "pubDate" (ISO 8601)}}.
    Age is None when no publish time can be parsed.
    """
    content = article.get("content") if isinstance(article.get("content"), dict) else article
    title = content.get("title")
    raw = content.get("pubDate") or content.get("displayTime") or article.get("providerPublishTime")
    if raw in (None, ""):
        return title, None
    unit = "s" if isinstance(raw, (int, float)) else None
    published = pd.to_datetime(raw, utc=True, unit=unit, errors="coerce")
    if pd.isna(published):
        return title, None
    return title, (now - published).total_seconds() / 86400.0


def get_news_sentiment(ticker: str, sentiment_model, now: pd.Timestamp | None = None) -> float | None:
    """
    Fetches recent news headlines for a ticker and scores them with FinBERT.

    Only headlines published within NEWS_MAX_AGE_DAYS of `now` (default: the
    current UTC time) are scored; items with no parseable publish time are
    kept, since their age is unknown rather than known to be stale.

    Returns a float in roughly [-1, +1], or None when no verdict is possible:
      - Positive = bullish news sentiment
      - Negative = bearish news sentiment
      - 0.0 = genuinely neutral coverage
      - None = no news found, or the fetch/scoring failed

    None is deliberately distinct from 0.0: "we could not read the news" is
    not the same signal as "the news is neutral". Each headline is scored
    individually; the final score is the mean of the signed scores, where
    FinBERT's positive/negative/neutral labels map to +score/-score/0.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    try:
        news_items = yf.Ticker(ticker).news

        if not news_items:
            return None

        headlines = []
        for article in news_items:
            title, age_days = _headline_and_age(article, now)
            if not title:
                continue
            if age_days is not None and age_days > NEWS_MAX_AGE_DAYS:
                continue
            headlines.append(title)
            if len(headlines) == NEWS_ARTICLES_PER_TICKER:
                break

        if not headlines:
            return None

        results = sentiment_model(headlines)

        signed_scores = []
        for r in results:
            if r["label"] == "positive":
                signed_scores.append(r["score"])
            elif r["label"] == "negative":
                signed_scores.append(-r["score"])
            else:  # neutral headlines carry no directional signal
                signed_scores.append(0.0)
        return sum(signed_scores) / len(signed_scores)

    except Exception:
        logger.warning("News sentiment lookup failed for %s", ticker)
        return None


# =============================================================================
# STEP 5: PIPELINE ORCHESTRATION
# =============================================================================


class PipelineOutput(NamedTuple):
    leaderboard: pd.DataFrame  # Ticker, Close, Confidence, Sentiment_Score, Signal_Pool
    candidate_prices: pd.DataFrame  # PRICE_HISTORY_DAYS of adjusted closes per candidate


def run_pipeline(tickers_limit: int | None = None, skip_sentiment: bool = False) -> PipelineOutput:
    """
    Runs the full two-stage pipeline. The leaderboard has columns
    [Ticker, Close, Confidence, Sentiment_Score, Signal_Pool], sorted by
    Confidence descending. Missing sentiment is NaN (rendered as a blank
    field in the CSV), never silently coerced to 0.
    """
    train_df, latest_df, master_df = build_dataset(tickers_limit=tickers_limit)

    print("Training XGBoost model...")
    model = train_model(train_df)
    print(f"  {model.get_params()['n_estimators']} boosting rounds (chosen by early stopping)")

    focus_list = score_and_rank(model, latest_df)

    if skip_sentiment:
        focus_list["Sentiment_Score"] = np.nan
    else:
        print("\nLoading FinBERT sentiment model (may take a moment on first run)...")
        sentiment_model = load_sentiment_model()

        print(f"\nScanning news sentiment for {len(focus_list)} candidates...")
        sentiments = []
        for i, ticker in enumerate(focus_list["Ticker"], start=1):
            sys.stdout.write(f"\r  [{i}/{len(focus_list)}] Analysing: {ticker:<6}")
            sys.stdout.flush()
            sentiments.append(get_news_sentiment(ticker, sentiment_model))
        print()
        focus_list["Sentiment_Score"] = [s if s is not None else np.nan for s in sentiments]

    focus_list["Confidence"] = (focus_list["Probability"] * 100).round(2)

    leaderboard = (
        focus_list[["Ticker", "Close", "Confidence", "Sentiment_Score", "Signal_Pool"]]
        .sort_values("Confidence", ascending=False)
        .reset_index(drop=True)
    )
    prices = candidate_price_history(master_df, leaderboard["Ticker"].tolist())
    return PipelineOutput(leaderboard, prices)


# =============================================================================
# STEP 6: OUTPUT FORMATTING
# =============================================================================


def print_results(leaderboard: pd.DataFrame) -> None:
    """Prints the final screener results to the console in a readable format."""
    separator = "=" * 65
    row_divider = "-" * 65
    header = f"{'Rank':<6} | {'Ticker':<6} | {'Price':<10} | {'Sentiment':<15} | {'Confidence'}"

    def fmt_sentiment(value: float) -> str:
        # NaN means no news was available, not neutral sentiment
        return "n/a" if pd.isna(value) else f"{value:.2f}"

    print(f"\n\n{separator}")
    print("   S&P 500 AI Screener: latest predictions")
    print(separator)

    print("\nTop 5 long candidates (highest model confidence)")
    print(row_divider)
    print(header)
    print(row_divider)
    for rank, row in enumerate(leaderboard.head(5).itertuples(), start=1):
        print(
            f"#{rank:<5} | {row.Ticker:<6} | ${row.Close:<9.2f} | "
            f"{fmt_sentiment(row.Sentiment_Score):<15} | {row.Confidence:.2f}%"
        )

    print("\nBottom 5 short candidates (lowest model confidence)")
    print(row_divider)
    print(header)
    print(row_divider)
    bottom_start = max(len(leaderboard) - 4, 1)
    for rank, row in enumerate(leaderboard.tail(5).itertuples(), start=bottom_start):
        print(
            f"#{rank:<5} | {row.Ticker:<6} | ${row.Close:<9.2f} | "
            f"{fmt_sentiment(row.Sentiment_Score):<15} | {row.Confidence:.2f}%"
        )

    print(f"\n{separator}")
    print("Interpretation guide:")
    print(f"  LONG:  in the top-{TOP_N_CANDIDATES} pool  AND  Sentiment > 0  ->  bullish confluence")
    print(f"  SHORT: in the bottom-{BOTTOM_N_CANDIDATES} pool  AND  Sentiment < 0  ->  bearish confluence")
    print("  Confidence is the model's probability that +4% is hit before -4% within 5 days;")
    print("  compare it with the ~25-30% base rate, not with 50%.")
    print(separator)


# =============================================================================
# MAIN
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S&P 500 stock screener: XGBoost signals plus FinBERT news sentiment.")
    parser.add_argument(
        "--tickers-limit",
        type=int,
        default=None,
        metavar="N",
        help="only process the first N tickers (fast smoke run; not for real signals)",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="skip the FinBERT sentiment stage; Sentiment_Score is left blank",
    )
    return parser.parse_args(argv)


def quiet_third_party_warnings() -> None:
    """Hide yfinance/pandas FutureWarnings during a console run. Call from an
    entry point, never at import time, so tests and library users still see them."""
    warnings.filterwarnings("ignore")


def main() -> None:
    quiet_third_party_warnings()
    args = parse_args()
    output = run_pipeline(tickers_limit=args.tickers_limit, skip_sentiment=args.skip_sentiment)
    print_results(output.leaderboard)


if __name__ == "__main__":
    main()
