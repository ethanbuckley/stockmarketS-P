"""
config.py
=========
Shared, dependency-free settings for the pipeline (screener.py), the
walk-forward evaluation (evaluate.py) and the dashboard (app.py).

app.py imports this module and nothing else from the pipeline, so the
deployed dashboard never pulls in xgboost, torch or transformers.
"""

import os

# --- Paths -------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, "data")
SIGNALS_PATH = os.path.join(DATA_DIR, "latest_signals.csv")
CANDIDATE_PRICES_PATH = os.path.join(DATA_DIR, "candidate_prices.csv")
VALIDATION_METRICS_PATH = os.path.join(DATA_DIR, "validation_metrics.json")
VALIDATION_DAILY_PATH = os.path.join(DATA_DIR, "validation_daily.csv")
VALIDATION_CALIBRATION_PATH = os.path.join(DATA_DIR, "validation_calibration.csv")

# --- Data --------------------------------------------------------------------
DATA_START_DATE = "2015-01-01"

# Macro benchmark ETFs and indices used as market-context features
MACRO_TICKERS = ["SPY", "QQQ", "SMH", "^VIX", "^TNX"]

# Trading days of adjusted close history written alongside the signals for
# the dashboard's Monte Carlo tab (so the deployed app never calls Yahoo).
PRICE_HISTORY_DAYS = 252

# --- Triple-barrier labelling ------------------------------------------------
FORWARD_WINDOW_DAYS = 5   # How many trading days ahead to look
TAKE_PROFIT_PCT = 0.04    # +4% triggers a positive label (1)
STOP_LOSS_PCT = 0.04      # -4% triggers a negative label (0)

# --- Model -------------------------------------------------------------------
# The feature set fed to XGBoost; each feature is listed explicitly for clarity
FEATURE_COLUMNS = [
    "Return", "RSI", "MACD", "BB_Position",
    "Price_to_VWAP", "ATR_Ratio", "Volume_Surge", "Day_Of_Week",
    "SPY_Return", "QQQ_Return", "SMH_Return", "VIX_Change", "TNX_Change",
    "Rel_SPY", "Rel_QQQ", "Rel_SMH",
    "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
    "QQQ_Lag_1", "QQQ_Lag_2", "QQQ_Lag_3",
]

# n_estimators is a ceiling: train_model picks the tree count by early
# stopping on a purged validation slice of the training data, then refits.
XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
}
EARLY_STOPPING_ROUNDS = 30
EARLY_STOPPING_VALIDATION_FRACTION = 0.10   # of unique training dates
EARLY_STOPPING_MIN_VALIDATION_DAYS = 40     # below this, fit the ceiling instead

# --- Signals -----------------------------------------------------------------
# How many candidates to deep-dive with FinBERT sentiment. These ranks ARE the
# signal: a stock in the long pool with positive news sentiment is a long
# candidate, one in the short pool with negative sentiment a short candidate.
TOP_N_CANDIDATES = 15    # long pool (highest predicted probability)
BOTTOM_N_CANDIDATES = 5  # short pool (lowest predicted probability)
LONG_POOL = "long"
SHORT_POOL = "short"

# --- News sentiment ----------------------------------------------------------
NEWS_ARTICLES_PER_TICKER = 10
NEWS_MAX_AGE_DAYS = 7    # headlines older than this are not "today's sentiment"
