"""
S&P 500 AI Stock Screener
=========================
A two-stage stock screening pipeline that combines:
  1. XGBoost binary classifier trained on technical indicators (quantitative signal)
  2. FinBERT NLP sentiment analysis on live news headlines (qualitative signal)

The model predicts whether a stock will hit a +4% take-profit before a -4% stop-loss
within a 5-day forward window — a "triple-barrier labelling" approach from financial ML.

Author: Ethan Buckley
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from io import StringIO
from xgboost import XGBClassifier
from transformers import pipeline

# Suppress noisy third-party logs so only our own print statements appear
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =============================================================================
# CONFIGURATION — all magic numbers in one place for easy adjustment
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


# =============================================================================
# STEP 1: DATA ACQUISITION
# =============================================================================

def fetch_sp500_tickers() -> list[str]:
    """
    Scrapes the current S&P 500 constituent list from Wikipedia.

    Returns a list of ticker symbols with dots replaced by hyphens
    (e.g. BRK.B → BRK-B) to match the Yahoo Finance format.
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
    response = requests.get(url, headers=headers)
    table = pd.read_html(StringIO(response.text))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def download_market_data(tickers: list[str]) -> pd.DataFrame:
    """
    Downloads OHLCV data for all tickers from Yahoo Finance.
    Forward-fills missing values to handle non-trading days and data gaps.
    """
    all_tickers = tickers + MACRO_TICKERS
    data = yf.download(all_tickers, start=DATA_START_DATE, progress=False, threads=True)
    data = data.ffill()
    return data


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def apply_triple_barrier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies triple-barrier labelling to create the target variable.

    For each day i, we look at the next FORWARD_WINDOW_DAYS bars:
    - If the high crosses TAKE_PROFIT_PCT above the close → label 1 (bullish)
    - If the low crosses STOP_LOSS_PCT below the close  → label 0 (bearish)
    - If neither barrier is hit within the window       → label 0 (bearish by default)

    This is preferable to simple N-day forward returns because it more closely
    models how a real trade with risk management would play out.

    Implementation uses sliding_window_view for a fully vectorized O(n·W) NumPy
    pass instead of nested Python loops, making it ~100–200× faster on long series.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    W = FORWARD_WINDOW_DAYS
    closes = df["Close"].to_numpy(dtype=float)
    highs  = df["High"].to_numpy(dtype=float)
    lows   = df["Low"].to_numpy(dtype=float)
    n = len(closes)

    # sliding_window_view(arr, W)[k] == arr[k : k+W].
    # We want the window starting one bar *after* entry i, so we use index i+1:
    #   forward_highs[i] = highs[i+1 : i+1+W]
    # sliding_window_view gives (n-W+1) windows; slicing [1:] drops window 0
    # (which starts at bar 0) so row i of the result covers bars i+1…i+W.
    # Valid for i in 0 … n-W-1, matching the original loop range.
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
    # How far the current price is from its volume-weighted average — a proxy for fair value
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
    df.index = pd.to_datetime(df.index).tz_localize(None)

    df["Close"] = raw_data["Close"][ticker]
    df["High"] = raw_data["High"][ticker]
    df["Low"] = raw_data["Low"][ticker]
    df["Volume"] = raw_data["Volume"][ticker]
    df["Ticker"] = ticker

    df = apply_triple_barrier_labels(df)
    df = build_technical_features(df)
    df = build_macro_features(df, raw_data)

    return df


# =============================================================================
# STEP 3: MODEL TRAINING
# =============================================================================

# The feature set fed to XGBoost — each feature is listed explicitly for clarity
FEATURE_COLUMNS = [
    "Return", "RSI", "MACD", "BB_Position",
    "Price_to_VWAP", "ATR_Ratio", "Volume_Surge", "Day_Of_Week",
    "SPY_Return", "QQQ_Return", "SMH_Return", "VIX_Change", "TNX_Change",
    "Rel_SPY", "Rel_QQQ", "Rel_SMH",
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
    "QQQ_Lag_1", "QQQ_Lag_2", "QQQ_Lag_3",
]


def train_model(train_df: pd.DataFrame) -> XGBClassifier:
    """
    Trains a single XGBoost classifier across all S&P 500 stocks.

    Using one "universal" model (rather than one per stock) means the model
    learns patterns that generalise across the market, not just overfit to
    one ticker's history. The output probability (predict_proba) is used
    as the ranking score — not a hard buy/sell decision.
    """
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(train_df[FEATURE_COLUMNS], train_df["Target"])
    return model


# =============================================================================
# STEP 4: SENTIMENT ANALYSIS (FINBERT)
# =============================================================================

def load_sentiment_model():
    """
    Loads the FinBERT model, a BERT variant fine-tuned on financial text.
    Returns a HuggingFace sentiment-analysis pipeline.
    """
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def get_news_sentiment(ticker: str, sentiment_model) -> float:
    """
    Fetches recent news headlines for a ticker and scores them with FinBERT.

    Returns a float in roughly [-1, +1]:
      - Positive = bullish news sentiment
      - Negative = bearish news sentiment
      - 0.0 = no news found or neutral

    Each headline is scored individually; the final score is the mean.
    """
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news

        if not news_items:
            return 0.0

        headlines = []
        for article in news_items[:NEWS_ARTICLES_PER_TICKER]:
            # yfinance returns news in two possible formats depending on version
            if "content" in article and "title" in article.get("content", {}):
                headlines.append(article["content"]["title"])
            elif "title" in article:
                headlines.append(article["title"])

        if not headlines:
            return 0.0

        results = sentiment_model(headlines)

        # Convert FinBERT labels to signed scores: positive → +score, negative → -score
        signed_scores = [
            r["score"] if r["label"] == "positive" else -r["score"]
            for r in results
        ]
        return sum(signed_scores) / len(signed_scores)

    except Exception:
        return 0.0


# =============================================================================
# STEP 5: OUTPUT FORMATTING
# =============================================================================

def print_results(leaderboard: pd.DataFrame) -> None:
    """Prints the final screener results to the console in a readable format."""
    separator = "=" * 65
    row_divider = "-" * 65
    header = f"{'Rank':<6} | {'Ticker':<6} | {'Price':<10} | {'AI Sentiment':<15} | {'Confidence'}"

    print(f"\n\n{separator}")
    print("   🚀  ELITE S&P 500 AI SCREENER  (Live Predictions)  🚀")
    print(separator)

    print("\n🟢  TOP 5 BREAKOUT CANDIDATES  (Strongest Long Signals)  🟢")
    print(row_divider)
    print(header)
    print(row_divider)
    for rank, row in enumerate(leaderboard.head(5).itertuples(), start=1):
        print(
            f"#{rank:<5} | {row.Ticker:<6} | ${row.Close:<9.2f} | "
            f"{row.Sentiment_Score:<15.2f} | {row.Confidence:.2f}%"
        )

    print("\n🔴  BOTTOM 5 BREAKDOWN CANDIDATES  (Strongest Short Signals)  🔴")
    print(row_divider)
    print(header)
    print(row_divider)
    for rank, row in enumerate(leaderboard.tail(5).itertuples(), start=16):
        print(
            f"#{rank:<5} | {row.Ticker:<6} | ${row.Close:<9.2f} | "
            f"{row.Sentiment_Score:<15.2f} | {row.Confidence:.2f}%"
        )

    print(f"\n{separator}")
    print("Interpretation guide:")
    print("  LONG:  Confidence > 55%  AND  Sentiment > 0  →  bullish confluence")
    print("  SHORT: Confidence < 45%  AND  Sentiment < 0  →  bearish confluence")
    print(separator)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # --- 1. Fetch data ---
    print("Fetching live S&P 500 ticker list from Wikipedia...")
    sp500_tickers = fetch_sp500_tickers()

    print(f"Downloading market data for {len(sp500_tickers)} stocks since {DATA_START_DATE}...")
    raw_data = download_market_data(sp500_tickers)

    # --- 2. Build feature panel ---
    print("Engineering features for all tickers...")
    all_ticker_dfs = [process_ticker(t, raw_data) for t in sp500_tickers]
    master_df = pd.concat(all_ticker_dfs)

    # Training set: all rows with complete features and a valid label
    train_df = master_df.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()

    # Prediction set: only the most recent row per ticker (today's signal)
    latest_df = (
        master_df
        .dropna(subset=FEATURE_COLUMNS)
        .groupby("Ticker")
        .tail(1)
        .copy()
    )

    # --- 3. Train model ---
    print("Training XGBoost model...")
    model = train_model(train_df)

    # --- 4. Score all stocks and select candidates for sentiment analysis ---
    latest_df["Probability"] = model.predict_proba(latest_df[FEATURE_COLUMNS])[:, 1]

    top_candidates = latest_df.nlargest(TOP_N_CANDIDATES, "Probability")
    bottom_candidates = latest_df.nsmallest(BOTTOM_N_CANDIDATES, "Probability")
    focus_list = pd.concat([top_candidates, bottom_candidates]).copy()

    # --- 5. Run FinBERT sentiment on the shortlisted candidates ---
    print("\nLoading FinBERT sentiment model (may take a moment on first run)...")
    sentiment_model = load_sentiment_model()

    print(f"\nScanning news sentiment for {len(focus_list)} candidates...")
    sentiments = []
    for i, ticker in enumerate(focus_list["Ticker"], start=1):
        sys.stdout.write(f"\r  [{i}/{len(focus_list)}] Analysing: {ticker:<6}")
        sys.stdout.flush()
        sentiments.append(get_news_sentiment(ticker, sentiment_model))

    focus_list["Sentiment_Score"] = sentiments
    focus_list["Confidence"] = (focus_list["Probability"] * 100).round(2)

    # --- 6. Build and display final leaderboard ---
    leaderboard = (
        focus_list[["Ticker", "Close", "Sentiment_Score", "Confidence"]]
        .sort_values("Confidence", ascending=False)
        .reset_index(drop=True)
    )

    print_results(leaderboard)


if __name__ == "__main__":
    main()
