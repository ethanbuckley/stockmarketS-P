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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

if TYPE_CHECKING:
    from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION: all magic numbers in one place for easy adjustment
# =============================================================================

DATA_START_DATE = "2015-01-01"

# Triple-barrier labelling parameters
FORWARD_WINDOW_DAYS = 5   # How many trading days ahead to look
TAKE_PROFIT_PCT = 0.04    # +4% triggers a positive label (1)
STOP_LOSS_PCT = 0.04      # -4% triggers a negative label (0)

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

# How many candidates to deep-dive with FinBERT sentiment
TOP_N_CANDIDATES = 15   # Long candidates (highest predicted probability)
BOTTOM_N_CANDIDATES = 5  # Short candidates (lowest predicted probability)
NEWS_ARTICLES_PER_TICKER = 10

# Macro benchmark ETFs and indices used as market-context features
MACRO_TICKERS = ["SPY", "QQQ", "SMH", "^VIX", "^TNX"]

# The feature set fed to XGBoost; each feature is listed explicitly for clarity
FEATURE_COLUMNS = [
    "Return", "RSI", "MACD", "BB_Position",
    "Price_to_VWAP", "ATR_Ratio", "Volume_Surge", "Day_Of_Week",
    "SPY_Return", "QQQ_Return", "SMH_Return", "VIX_Change", "TNX_Change",
    "Rel_SPY", "Rel_QQQ", "Rel_SMH",
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
    "QQQ_Lag_1", "QQQ_Lag_2", "QQQ_Lag_3",
]

# Interpretation thresholds (percent confidence) used by the console output
# and the dashboard when flagging long/short confluence with sentiment.
LONG_CONFIDENCE_PCT = 55.0
SHORT_CONFIDENCE_PCT = 45.0


# =============================================================================
# STEP 1: DATA ACQUISITION
# =============================================================================

def fetch_sp500_tickers() -> list[str]:
    """
    Scrapes the current S&P 500 constituent list from Wikipedia.

    Returns a list of ticker symbols with dots replaced by hyphens
    (e.g. BRK.B -> BRK-B) to match the Yahoo Finance format.

    Raises RuntimeError if the page cannot be fetched or the constituents
    table cannot be parsed: a silently wrong universe would corrupt
    everything downstream, so this fails loudly.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
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

    tables = pd.read_html(StringIO(response.text))
    if not tables or "Symbol" not in tables[0].columns:
        raise RuntimeError(
            "Could not parse the S&P 500 constituents table from Wikipedia "
            "(the page layout may have changed)."
        )
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    if len(tickers) < 400:
        raise RuntimeError(
            f"Parsed only {len(tickers)} tickers from Wikipedia; expected roughly 500. "
            "Refusing to continue with a partial universe."
        )
    return tickers


def download_market_data(tickers: list[str], start: str = DATA_START_DATE) -> pd.DataFrame:
    """
    Downloads OHLCV data for all tickers from Yahoo Finance.
    Forward-fills missing values to handle non-trading days and data gaps.

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

    return data.ffill()


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
    highs  = df["High"].to_numpy(dtype=float)
    lows   = df["Low"].to_numpy(dtype=float)
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
    forward_highs = sliding_window_view(highs, W)[1:]   # shape (n-W, W)
    forward_lows  = sliding_window_view(lows,  W)[1:]   # shape (n-W, W)

    upper_barriers = (closes[: n - W] * (1 + TAKE_PROFIT_PCT))[:, None]  # (n-W, 1)
    lower_barriers = (closes[: n - W] * (1 - STOP_LOSS_PCT))[:, None]    # (n-W, 1)

    hit_upper = forward_highs >= upper_barriers  # (n-W, W) bool
    hit_lower = forward_lows  <= lower_barriers  # (n-W, W) bool

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
    Fetches data and builds the labelled master feature panel for the whole universe.

    Returns (master_df, metadata):
      - master_df: one row per (date, ticker) on the union yfinance calendar
        (tz-naive DatetimeIndex), with OHLCV, Ticker, all FEATURE_COLUMNS and
        Target. No dropna and no date filtering is applied here; callers
        (training, prediction, walk-forward evaluation) do their own filtering.
      - metadata: download timestamp, ticker list, and last price date, so
        downstream artefacts can record exactly which data snapshot they used.
    """
    print("Fetching live S&P 500 ticker list from Wikipedia...")
    tickers = fetch_sp500_tickers()
    if tickers_limit is not None:
        tickers = tickers[:tickers_limit]

    print(f"Downloading market data for {len(tickers)} stocks since {start}...")
    raw_data = download_market_data(tickers, start=start)

    # Tickers that returned no data at all cannot be processed
    with_data = set(tickers_with_data(raw_data))
    available = [t for t in tickers if t in with_data]

    print("Engineering features for all tickers...")
    master_df = pd.concat([process_ticker(t, raw_data) for t in available])

    metadata = {
        "download_timestamp_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tickers": available,
        "last_price_date": str(master_df.index.max().date()),
    }
    return master_df, metadata


def build_dataset(tickers_limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the two frames the screener needs:
      - train_df: all rows with complete features and a valid label
      - latest_df: the most recent complete-feature row per ticker (today's signal)
    """
    master_df, _ = build_master_dataframe(tickers_limit=tickers_limit)

    train_df = master_df.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()
    latest_df = (
        master_df
        .dropna(subset=FEATURE_COLUMNS)
        .groupby("Ticker")
        .tail(1)
        .copy()
    )
    return train_df, latest_df


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
    """
    # Imported lazily so that importing this module (tests, evaluation,
    # generate_signals) does not require xgboost to be installed.
    from xgboost import XGBClassifier

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(train_df[FEATURE_COLUMNS], train_df["Target"])
    return model


def score_and_rank(model: XGBClassifier, latest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores the latest row per ticker and returns the focus list for sentiment
    analysis: the TOP_N_CANDIDATES highest and BOTTOM_N_CANDIDATES lowest by
    predicted probability. Adds a "Probability" column; no sentiment yet.
    """
    latest_df = latest_df.copy()
    latest_df["Probability"] = model.predict_proba(latest_df[FEATURE_COLUMNS])[:, 1]

    top_candidates = latest_df.nlargest(TOP_N_CANDIDATES, "Probability")
    bottom_candidates = latest_df.nsmallest(BOTTOM_N_CANDIDATES, "Probability")
    # On a tiny universe (--tickers-limit) the two ends can overlap.
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


def get_news_sentiment(ticker: str, sentiment_model) -> float | None:
    """
    Fetches recent news headlines for a ticker and scores them with FinBERT.

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
    try:
        news_items = yf.Ticker(ticker).news

        if not news_items:
            return None

        headlines = []
        for article in news_items[:NEWS_ARTICLES_PER_TICKER]:
            # yfinance returns news in two possible formats depending on version
            if "content" in article and "title" in article.get("content", {}):
                headlines.append(article["content"]["title"])
            elif "title" in article:
                headlines.append(article["title"])

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

def run_pipeline(tickers_limit: int | None = None, skip_sentiment: bool = False) -> pd.DataFrame:
    """
    Runs the full two-stage pipeline and returns the leaderboard DataFrame
    with columns [Ticker, Close, Confidence, Sentiment_Score], sorted by
    Confidence descending. Missing sentiment is NaN (rendered as a blank
    field in the CSV), never silently coerced to 0.
    """
    train_df, latest_df = build_dataset(tickers_limit=tickers_limit)

    print("Training XGBoost model...")
    model = train_model(train_df)

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
        focus_list[["Ticker", "Close", "Confidence", "Sentiment_Score"]]
        .sort_values("Confidence", ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard


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
    print(f"  LONG:  Confidence > {LONG_CONFIDENCE_PCT:.0f}%  AND  Sentiment > 0  ->  bullish confluence")
    print(f"  SHORT: Confidence < {SHORT_CONFIDENCE_PCT:.0f}%  AND  Sentiment < 0  ->  bearish confluence")
    print(separator)


# =============================================================================
# MAIN
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S&P 500 stock screener: XGBoost signals plus FinBERT news sentiment."
    )
    parser.add_argument(
        "--tickers-limit", type=int, default=None, metavar="N",
        help="only process the first N tickers (fast smoke run; not for real signals)",
    )
    parser.add_argument(
        "--skip-sentiment", action="store_true",
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
    leaderboard = run_pipeline(tickers_limit=args.tickers_limit, skip_sentiment=args.skip_sentiment)
    print_results(leaderboard)


if __name__ == "__main__":
    main()
