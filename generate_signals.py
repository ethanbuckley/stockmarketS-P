"""
generate_signals.py
===================
Runs the full S&P 500 screening pipeline and writes results to
data/latest_signals.csv for consumption by the Streamlit dashboard.

Meant to be run locally on a regular cadence (e.g. weekly), NOT inside the
deployed app.  The output file is small (<5 MB) and can be committed to the
repo so the dashboard always has data to display even without a live run.

Usage:
    python generate_signals.py [--tickers-limit N] [--skip-sentiment]

Output columns in data/latest_signals.csv:
    Ticker, Close, Confidence, Sentiment_Score, generated_at (UTC ISO-8601)

A blank Sentiment_Score means no news was available for that ticker at
generation time (or the lookup failed); it does not mean neutral sentiment.
"""

import datetime
import os

import pandas as pd

from screener import parse_args, print_results, quiet_third_party_warnings, run_pipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "latest_signals.csv")


def save_signals(leaderboard: pd.DataFrame) -> None:
    """Writes the leaderboard to data/latest_signals.csv with a UTC timestamp."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    leaderboard = leaderboard.copy()
    leaderboard["generated_at"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    leaderboard.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSignals written to {OUTPUT_PATH}  ({len(leaderboard)} rows)")


def main() -> None:
    quiet_third_party_warnings()
    # Same flags as `python screener.py`; the only difference is the CSV write.
    args = parse_args()
    leaderboard = run_pipeline(tickers_limit=args.tickers_limit, skip_sentiment=args.skip_sentiment)
    print_results(leaderboard)
    save_signals(leaderboard)


if __name__ == "__main__":
    main()
