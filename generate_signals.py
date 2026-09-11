"""
generate_signals.py
===================
Runs the full S&P 500 screening pipeline and writes the dashboard's inputs:

    data/latest_signals.csv     Ticker, Close, Confidence, Sentiment_Score,
                                Signal_Pool, generated_at (UTC ISO-8601)
    data/candidate_prices.csv   PRICE_HISTORY_DAYS of adjusted closes for the
                                candidates (Monte Carlo tab input)

Runs weekly in GitHub Actions (.github/workflows/refresh-signals.yml) and
can be run locally; it never executes inside the deployed app.

Usage:
    python generate_signals.py [--tickers-limit N] [--skip-sentiment]

A blank Sentiment_Score means no recent news was available for that ticker
at generation time (or the lookup failed); it does not mean neutral sentiment.
"""

import datetime
import os

import pandas as pd

from config import CANDIDATE_PRICES_PATH, DATA_DIR, SIGNALS_PATH
from screener import parse_args, print_results, quiet_third_party_warnings, run_pipeline


def save_signals(leaderboard: pd.DataFrame, candidate_prices: pd.DataFrame) -> None:
    """Writes the leaderboard (with a UTC timestamp) and the price history."""
    os.makedirs(DATA_DIR, exist_ok=True)

    leaderboard = leaderboard.copy()
    leaderboard["generated_at"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    leaderboard.to_csv(SIGNALS_PATH, index=False)
    print(f"\nSignals written to {SIGNALS_PATH}  ({len(leaderboard)} rows)")

    candidate_prices.round(6).to_csv(CANDIDATE_PRICES_PATH, date_format="%Y-%m-%d")
    print(
        f"Price history written to {CANDIDATE_PRICES_PATH}  "
        f"({candidate_prices.shape[0]} days x {candidate_prices.shape[1]} tickers)"
    )


def main() -> None:
    quiet_third_party_warnings()
    # Same flags as `python screener.py`; the only difference is the file output.
    args = parse_args()
    output = run_pipeline(tickers_limit=args.tickers_limit, skip_sentiment=args.skip_sentiment)
    print_results(output.leaderboard)
    save_signals(output.leaderboard, output.candidate_prices)


if __name__ == "__main__":
    main()
