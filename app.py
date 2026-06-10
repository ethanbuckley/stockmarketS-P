"""
app.py — S&P 500 AI Screener Dashboard
Reads data/latest_signals.csv produced by generate_signals.py.
Deploy to Streamlit Community Cloud; no heavy ML dependencies required.
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="S&P 500 AI Screener",
    page_icon="📈",
    layout="wide",
)

# =============================================================================
# DISCLAIMER + HEADER
# =============================================================================

st.title("S&P 500 AI Stock Screener")

st.error(
    "**Disclaimer:** This is a personal educational project built to demonstrate "
    "quantitative finance techniques. It does **not** constitute financial advice. "
    "Past model performance does not guarantee future results. Do not make investment "
    "decisions based on this tool.",
    icon="⚠️",
)

# =============================================================================
# LOAD DATA
# =============================================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "latest_signals.csv")


@st.cache_data(ttl=3600)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


if not os.path.exists(DATA_PATH):
    st.warning(
        "No signals file found at `data/latest_signals.csv`. "
        "Run `python generate_signals.py` locally to generate it, "
        "then commit the file to the repository."
    )
    st.stop()

df = load_data(DATA_PATH)

# =============================================================================
# LAST UPDATED TIMESTAMP
# =============================================================================

if "generated_at" in df.columns:
    generated_at = df["generated_at"].iloc[0]
    st.caption(f"Last updated: **{generated_at} UTC**")
else:
    st.caption("Last updated: timestamp not available")

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

st.sidebar.header("Filters")

min_conf, max_conf = float(df["Confidence"].min()), float(df["Confidence"].max())
conf_range = st.sidebar.slider(
    "Confidence (%)",
    min_value=round(min_conf, 1),
    max_value=round(max_conf, 1),
    value=(round(min_conf, 1), round(max_conf, 1)),
    step=0.1,
)

min_sent = float(df["Sentiment_Score"].min())
max_sent = float(df["Sentiment_Score"].max())
sent_range = st.sidebar.slider(
    "Sentiment Score",
    min_value=round(min_sent, 2),
    max_value=round(max_sent, 2),
    value=(round(min_sent, 2), round(max_sent, 2)),
    step=0.01,
)

signal_filter = st.sidebar.radio(
    "Signal type",
    options=["All", "Long candidates (Conf > 55%, Sent > 0)", "Short candidates (Conf < 45%, Sent < 0)"],
)

# Apply filters
filtered = df[
    (df["Confidence"] >= conf_range[0]) &
    (df["Confidence"] <= conf_range[1]) &
    (df["Sentiment_Score"] >= sent_range[0]) &
    (df["Sentiment_Score"] <= sent_range[1])
].copy()

if signal_filter.startswith("Long"):
    filtered = filtered[(filtered["Confidence"] > 55) & (filtered["Sentiment_Score"] > 0)]
elif signal_filter.startswith("Short"):
    filtered = filtered[(filtered["Confidence"] < 45) & (filtered["Sentiment_Score"] < 0)]

# =============================================================================
# SUMMARY METRICS
# =============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidates shown", len(df))
col2.metric("Showing", len(filtered))
col3.metric("Long signals", int(((df["Confidence"] > 55) & (df["Sentiment_Score"] > 0)).sum()))
col4.metric("Short signals", int(((df["Confidence"] < 45) & (df["Sentiment_Score"] < 0)).sum()))

st.divider()

# =============================================================================
# MAIN TABLE
# =============================================================================

st.subheader("Screener Leaderboard")

display_cols = ["Ticker", "Close", "Confidence", "Sentiment_Score"]
display_df = filtered[display_cols].sort_values("Confidence", ascending=False).reset_index(drop=True)
display_df.index += 1  # 1-based rank

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
)

st.divider()

# =============================================================================
# CHARTS
# =============================================================================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top 10 by Confidence")
    top10 = df.nlargest(10, "Confidence").sort_values("Confidence")
    fig_bar = px.bar(
        top10,
        x="Confidence",
        y="Ticker",
        orientation="h",
        color="Confidence",
        color_continuous_scale="RdYlGn",
        labels={"Confidence": "Confidence (%)"},
        title="Top 10 by Model Confidence",
    )
    fig_bar.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Confidence vs Sentiment")
    fig_scatter = px.scatter(
        df,
        x="Sentiment_Score",
        y="Confidence",
        text="Ticker",
        color="Confidence",
        color_continuous_scale="RdYlGn",
        labels={"Sentiment_Score": "FinBERT Sentiment Score", "Confidence": "XGBoost Confidence (%)"},
        title="Signal Map",
    )
    fig_scatter.update_traces(textposition="top center", marker_size=8)
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_hline(y=55, line_dash="dash", line_color="green", opacity=0.5,
                          annotation_text="Long threshold", annotation_position="right")
    fig_scatter.add_hline(y=45, line_dash="dash", line_color="red", opacity=0.5,
                          annotation_text="Short threshold", annotation_position="right")
    fig_scatter.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# =============================================================================
# METHODOLOGY EXPLAINER
# =============================================================================

with st.expander("How this works"):
    st.markdown(
        """
        This screener combines two complementary signals to identify long and short
        candidates across the S&P 500.

        **Stage 1 — XGBoost Classifier (quantitative signal)**

        An XGBoost gradient-boosting model is trained on 10+ years of daily price data
        across all S&P 500 constituents (~1.3 million observations). The target label uses
        the *triple-barrier method*: for each trading day, the model asks whether the stock
        will hit a +4% take-profit *before* it hits a −4% stop-loss within the next 5 trading
        days. This is more realistic than simple N-day forward returns because it mirrors
        how a real trade with risk management plays out.

        Features include momentum indicators (RSI, MACD, lagged returns), volatility
        measures (Bollinger Band position, ATR ratio), volume signals (VWAP deviation,
        volume surge), macro context (SPY, QQQ, SMH, VIX, 10Y Treasury), and
        relative performance vs benchmarks. One model is trained across all tickers so
        it learns cross-sectional patterns rather than fitting to any single stock's history.

        **Stage 2 — FinBERT Sentiment Analysis (qualitative signal)**

        The top 15 and bottom 5 candidates by XGBoost confidence are passed to
        [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on
        financial news, analyst reports, and earnings call transcripts. Live headlines
        are fetched for each candidate and scored. The final sentiment score is the
        mean signed score across up to 10 recent headlines.

        **Interpreting the output**

        | Signal | Condition |
        |--------|-----------|
        | Long candidate | Confidence > 55% **and** Sentiment > 0 |
        | Short candidate | Confidence < 45% **and** Sentiment < 0 |

        High confidence alone is not a buy signal — both the quantitative and qualitative
        signals should align. Mixed signals (high confidence, negative sentiment) warrant
        caution.
        """
    )
