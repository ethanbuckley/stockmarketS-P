"""
generate_signals.py
===================
Runs the full S&P 500 screening pipeline and writes results to
data/latest_signals.csv for consumption by the Streamlit dashboard.

Meant to be run locally on a regular cadence (e.g. weekly), NOT inside the
deployed app.  The output file is small (<5 MB) and can be committed to the
repo so the dashboard always has data to display even without a live run.

Usage:
    python generate_signals.py

Output columns in data/latest_signals.csv:
    Ticker, Close, Confidence, Sentiment_Score, generated_at (UTC ISO-8601)
"""

import os
import sys
import datetime

import pandas as pd

# Import shared pipeline functions from screener.py
from screener import (
    fetch_sp500_tickers,
    download_market_data,
    process_ticker,
    train_model,
    load_sentiment_model,
    get_news_sentiment,
    print_results,
    FEATURE_COLUMNS,
    TOP_N_CANDIDATES,
    BOTTOM_N_CANDIDATES,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "latest_signals.csv")


def run_pipeline() -> pd.DataFrame:
    """Runs the full pipeline and returns the leaderboard DataFrame."""

    # 1. Fetch data
    print("Fetching live S&P 500 ticker list from Wikipedia...")
    sp500_tickers = fetch_sp500_tickers()

    print(f"Downloading market data for {len(sp500_tickers)} stocks...")
    raw_data = download_market_data(sp500_tickers)

    # 2. Feature engineering across all tickers
    print("Engineering features for all tickers...")
    all_ticker_dfs = [process_ticker(t, raw_data) for t in sp500_tickers]
    master_df = pd.concat(all_ticker_dfs)

    train_df = master_df.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()
    latest_df = (
        master_df
        .dropna(subset=FEATURE_COLUMNS)
        .groupby("Ticker")
        .tail(1)
        .copy()
    )

    # 3. Train XGBoost and score all stocks
    print("Training XGBoost model...")
    model = train_model(train_df)

    latest_df["Probability"] = model.predict_proba(latest_df[FEATURE_COLUMNS])[:, 1]
    top_candidates    = latest_df.nlargest(TOP_N_CANDIDATES, "Probability")
    bottom_candidates = latest_df.nsmallest(BOTTOM_N_CANDIDATES, "Probability")
    focus_list = pd.concat([top_candidates, bottom_candidates]).copy()

    # 4. FinBERT sentiment on shortlisted candidates
    print("\nLoading FinBERT sentiment model (may take a moment on first run)...")
    sentiment_model = load_sentiment_model()

    print(f"\nScanning news sentiment for {len(focus_list)} candidates...")
    sentiments = []
    for i, ticker in enumerate(focus_list["Ticker"], start=1):
        sys.stdout.write(f"\r  [{i}/{len(focus_list)}] Analysing: {ticker:<6}")
        sys.stdout.flush()
        sentiments.append(get_news_sentiment(ticker, sentiment_model))
    print()

    focus_list["Sentiment_Score"] = sentiments
    focus_list["Confidence"] = (focus_list["Probability"] * 100).round(2)

    leaderboard = (
        focus_list[["Ticker", "Close", "Confidence", "Sentiment_Score"]]
        .sort_values("Confidence", ascending=False)
        .reset_index(drop=True)
    )

    return leaderboard


def save_signals(leaderboard: pd.DataFrame) -> None:
    """Writes the leaderboard to data/latest_signals.csv with a UTC timestamp."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    leaderboard = leaderboard.copy()
    leaderboard["generated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    leaderboard.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSignals written to {OUTPUT_PATH}  ({len(leaderboard)} rows)")


if __name__ == "__main__":
    leaderboard = run_pipeline()
    print_results(leaderboard)
    save_signals(leaderboard)
