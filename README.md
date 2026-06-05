# S&P 500 AI Stock Screener

A two-stage stock screening pipeline that combines a machine learning classifier with NLP sentiment analysis to identify long and short candidates across the S&P 500.

Built as an independent project applying quantitative finance techniques alongside my Physics and Physical Chemistry degree at UCL.

---

## How It Works

The screener runs in two stages.

**Stage 1: XGBoost Classifier**

An XGBoost binary classifier is trained on 10+ years of daily OHLCV data across all S&P 500 constituents. The target variable uses triple-barrier labelling: for each trading day, the model checks whether the stock hits a +4% take-profit or a -4% stop-loss within the next 5 trading days. This is preferable to simple forward returns because it more closely reflects how a real trade with risk management plays out.

Features fall into five groups:
- Momentum: RSI, MACD, lagged returns
- Volatility: Bollinger Band position, ATR ratio
- Volume: volume surge, VWAP deviation
- Market context: SPY, QQQ, SMH benchmark returns, VIX, 10Y Treasury yield
- Relative performance: stock return vs macro benchmarks

One model is trained across all tickers rather than one per stock, so the classifier learns patterns that generalise across the market rather than fitting to individual ticker history.

**Stage 2: FinBERT Sentiment Analysis**

The top 15 and bottom 5 candidates by XGBoost confidence score are passed to FinBERT, a BERT model fine-tuned on financial text. Live news headlines are fetched for each candidate and scored. The final output combines both signals: a bullish confluence requires high model confidence and positive news sentiment.

---

## Output

The screener prints a ranked leaderboard to the console:

```
=================================================================
   S&P 500 AI SCREENER  (Live Predictions)
=================================================================

TOP 5 BREAKOUT CANDIDATES  (Strongest Long Signals)
-----------------------------------------------------------------
Rank   | Ticker | Price      | AI Sentiment    | Confidence
-----------------------------------------------------------------
#1     | NVDA   | $875.20    | 0.61            | 73.45%
#2     | CEG    | $214.80    | 0.48            | 71.22%
...

BOTTOM 5 BREAKDOWN CANDIDATES  (Strongest Short Signals)
```

Interpretation:
- Long signal: Confidence > 55% and Sentiment > 0
- Short signal: Confidence < 45% and Sentiment < 0

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Data acquisition | yfinance, requests |
| Feature engineering | pandas, NumPy |
| ML classifier | XGBoost |
| NLP sentiment | FinBERT (ProsusAI/finbert via HuggingFace Transformers) |
| Universe | S&P 500 (scraped live from Wikipedia) |
| Training data | ~1.3 million daily observations (2015 to present) |

---

## Installation

```bash
git clone https://github.com/ethanbuckley/stockmarketS-P.git
cd stockmarketS-P
pip install -r requirements.txt
```

**Requirements:**
```
yfinance
xgboost
transformers
torch
pandas
numpy
requests
```

---

## Usage

```bash
python screener.py
```

On first run, FinBERT downloads automatically (~400MB). Subsequent runs load from cache.

The full pipeline takes roughly 15 to 30 minutes depending on internet speed and hardware.

---

## Project Structure

```
stockmarketS-P/
├── screener.py          # Main pipeline
├── requirements.txt     # Dependencies
└── README.md
```

---

## Methodology Notes

Triple-barrier labelling is used rather than simple N-day forward returns because it captures the asymmetric nature of real trades: a +4% move on day 2 is a win regardless of what happens in days 3 to 5. The approach follows Marcos Lopez de Prado's framework for financial machine learning.

Training a single model across all tickers means the classifier learns cross-sectional patterns (which technical setups tend to precede breakouts across the market) rather than memorising one stock's history. Sector-specific nuance is captured through the relative performance features.

FinBERT was chosen over general-purpose sentiment models because it is fine-tuned on financial news, analyst reports, and earnings call transcripts, which makes it more reliable on financial headlines than models trained on general text.

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Past model performance does not guarantee future results.

---

## Author

Ethan Buckley, MSci Natural Sciences (Physics and Physical Chemistry), UCL
[ethan.buckley.24@ucl.ac.uk](mailto:ethan.buckley.24@ucl.ac.uk)
