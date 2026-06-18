"""
app.py — S&P 500 AI Screener Dashboard
Reads data/latest_signals.csv produced by generate_signals.py.
Deploy to Streamlit Community Cloud; no heavy ML dependencies required.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

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
# SIDEBAR FILTERS (apply to Screener tab only)
# =============================================================================

st.sidebar.header("Screener Filters")

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
# TABS
# =============================================================================

tab_screener, tab_mc = st.tabs(["Screener", "Monte Carlo Risk"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SCREENER (existing content, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

with tab_screener:

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates shown", len(df))
    col2.metric("Showing", len(filtered))
    col3.metric("Long signals", int(((df["Confidence"] > 55) & (df["Sentiment_Score"] > 0)).sum()))
    col4.metric("Short signals", int(((df["Confidence"] < 45) & (df["Sentiment_Score"] < 0)).sum()))

    st.divider()
    st.subheader("Screener Leaderboard")

    display_cols = ["Ticker", "Close", "Confidence", "Sentiment_Score"]
    display_df = filtered[display_cols].sort_values("Confidence", ascending=False).reset_index(drop=True)
    display_df.index += 1

    st.dataframe(display_df, use_container_width=True, height=400)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 by Confidence")
        top10 = df.nlargest(10, "Confidence").sort_values("Confidence")
        fig_bar = px.bar(
            top10,
            x="Confidence", y="Ticker", orientation="h",
            color="Confidence", color_continuous_scale="RdYlGn",
            labels={"Confidence": "Confidence (%)"},
            title="Top 10 by Model Confidence",
        )
        fig_bar.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("Confidence vs Sentiment")
        fig_scatter = px.scatter(
            df, x="Sentiment_Score", y="Confidence", text="Ticker",
            color="Confidence", color_continuous_scale="RdYlGn",
            labels={"Sentiment_Score": "FinBERT Sentiment Score",
                    "Confidence": "XGBoost Confidence (%)"},
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MONTE CARLO RISK
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    """Download adjusted close prices for a tuple of tickers."""
    raw = yf.download(list(tickers), period=period, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = list(tickers)
    return prices.dropna(how="all")


def run_monte_carlo(
    log_ret: pd.DataFrame,
    horizon: int,
    n_paths: int,
    initial_value: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate correlated GBM paths for an equal-weighted portfolio.

    Uses a Cholesky decomposition of the historical correlation matrix so that
    the co-movement structure between assets is preserved across simulated paths.

    Returns portfolio_values of shape (n_paths, horizon + 1).
    """
    rng = np.random.default_rng(seed)

    mu    = log_ret.mean().values          # daily mean log-return per asset
    sigma = log_ret.std().values           # daily vol per asset
    corr  = log_ret.corr().values          # correlation matrix
    n_assets = len(mu)

    # Cholesky factor so that L @ L.T == corr
    # Clip eigenvalues to handle near-singular matrices from short histories
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(corr_psd)

    # Draw iid standard normals, then correlate: (n_paths, horizon, n_assets)
    Z = rng.standard_normal((n_paths, horizon, n_assets))
    Z_corr = Z @ L.T

    # GBM daily log-returns: (μ - ½σ²)dt + σ√dt · Z
    daily_log_ret = (mu - 0.5 * sigma ** 2) + sigma * Z_corr  # dt = 1 day

    # Equal-weighted portfolio log-return each day
    w = np.ones(n_assets) / n_assets
    port_daily = daily_log_ret @ w                   # (n_paths, horizon)
    port_cum   = np.cumsum(port_daily, axis=1)       # (n_paths, horizon)

    portfolio_values = initial_value * np.exp(
        np.concatenate([np.zeros((n_paths, 1)), port_cum], axis=1)
    )  # (n_paths, horizon + 1)

    return portfolio_values


with tab_mc:
    st.subheader("Monte Carlo Portfolio Risk Simulation")
    st.markdown(
        "Simulates thousands of possible future portfolio paths using **Geometric Brownian Motion** "
        "calibrated to the historical return distribution of the screener's top candidates. "
        "Correlated asset paths are generated via Cholesky decomposition of the empirical "
        "correlation matrix."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    all_tickers = df["Ticker"].tolist()
    default_tickers = df.nlargest(5, "Confidence")["Ticker"].tolist()

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        chosen = st.multiselect(
            "Portfolio tickers (from screener candidates)",
            options=all_tickers,
            default=default_tickers,
            max_selections=10,
        )
    with c2:
        horizon_label = st.selectbox("Horizon", ["1 month (21 days)", "3 months (63 days)", "1 year (252 days)"])
        horizon_map   = {"1 month (21 days)": 21, "3 months (63 days)": 63, "1 year (252 days)": 252}
        horizon       = horizon_map[horizon_label]
    with c3:
        n_paths = st.selectbox("Simulated paths", [5_000, 10_000, 50_000], index=1)
    with c4:
        initial_value = st.number_input("Initial portfolio ($)", value=10_000, step=1_000)

    if not chosen:
        st.info("Select at least one ticker above to run the simulation.")
        st.stop()

    # ── Fetch & simulate ──────────────────────────────────────────────────────
    with st.spinner(f"Fetching 1 year of price history for {', '.join(chosen)} …"):
        prices = fetch_prices(tuple(chosen), period="1y")

    # Drop tickers that came back entirely empty
    prices = prices[[t for t in chosen if t in prices.columns]].dropna()
    valid_tickers = list(prices.columns)

    if len(valid_tickers) == 0:
        st.error("Could not fetch price data for any selected ticker.")
        st.stop()

    if len(valid_tickers) < len(chosen):
        missing = set(chosen) - set(valid_tickers)
        st.warning(f"No price data for: {', '.join(missing)}. Continuing with {', '.join(valid_tickers)}.")

    log_ret = np.log(prices / prices.shift(1)).dropna()

    with st.spinner(f"Running {n_paths:,} Monte Carlo paths …"):
        port_values = run_monte_carlo(log_ret, horizon, n_paths, float(initial_value))

    # ── Risk metrics ──────────────────────────────────────────────────────────
    final_values  = port_values[:, -1]
    final_returns = (final_values - initial_value) / initial_value

    var_95  = float(np.percentile(final_values, 5))
    cvar_95 = float(final_values[final_values <= var_95].mean())
    p_loss  = float((final_values < initial_value).mean())
    med_ret = float(np.median(final_returns) * 100)
    mean_ret = float(np.mean(final_returns) * 100)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("VaR (95%)",  f"${initial_value - var_95:,.0f}",
              help="Maximum loss in 95% of scenarios — you lose less than this amount 95% of the time")
    m2.metric("CVaR (95%)", f"${initial_value - cvar_95:,.0f}",
              help="Average loss in the worst 5% of scenarios (Expected Shortfall)")
    m3.metric("P(loss)",    f"{p_loss*100:.1f}%",
              help="Fraction of simulated paths that end below the initial investment")
    m4.metric("Median return", f"{med_ret:+.1f}%")
    m5.metric("Mean return",   f"{mean_ret:+.1f}%")

    st.divider()

    # ── Fan chart ─────────────────────────────────────────────────────────────
    t_axis = list(range(horizon + 1))
    pcts   = np.percentile(port_values, [5, 25, 50, 75, 95], axis=0)

    fig_fan = go.Figure()

    # Shaded bands
    fig_fan.add_trace(go.Scatter(
        x=t_axis + t_axis[::-1],
        y=pcts[4].tolist() + pcts[0].tolist()[::-1],
        fill="toself", fillcolor="rgba(99,102,241,0.10)",
        line=dict(width=0), name="5th–95th pct", showlegend=True,
    ))
    fig_fan.add_trace(go.Scatter(
        x=t_axis + t_axis[::-1],
        y=pcts[3].tolist() + pcts[1].tolist()[::-1],
        fill="toself", fillcolor="rgba(99,102,241,0.20)",
        line=dict(width=0), name="25th–75th pct", showlegend=True,
    ))

    # Median line
    fig_fan.add_trace(go.Scatter(
        x=t_axis, y=pcts[2], name="Median",
        line=dict(color="#6366F1", width=2.5),
    ))

    # Initial value reference
    fig_fan.add_hline(y=initial_value, line_dash="dot",
                      line_color="gray", opacity=0.6,
                      annotation_text="Initial value", annotation_position="right")

    fig_fan.update_layout(
        title=f"Simulated portfolio paths — {horizon}-day horizon  "
              f"({n_paths:,} paths, equal-weighted: {', '.join(valid_tickers)})",
        xaxis_title="Trading days",
        yaxis_title="Portfolio value ($)",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=60, b=0),
        height=420,
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # ── Final value distribution ───────────────────────────────────────────────
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_values, nbinsx=80,
        marker_color="#6366F1", opacity=0.75, name="Final value",
    ))
    fig_hist.add_vline(x=initial_value, line_dash="dot", line_color="gray",
                       annotation_text="Initial", annotation_position="top right")
    fig_hist.add_vline(x=var_95, line_dash="dash", line_color="#DC2626",
                       annotation_text="VaR 95%", annotation_position="top left")
    fig_hist.add_vline(x=cvar_95, line_dash="dash", line_color="#F97316",
                       annotation_text="CVaR 95%", annotation_position="top left")
    fig_hist.update_layout(
        title="Distribution of final portfolio value",
        xaxis_title="Portfolio value ($)",
        yaxis_title="Number of paths",
        margin=dict(l=0, r=0, t=50, b=0),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Individual asset stats ─────────────────────────────────────────────────
    with st.expander("Individual asset statistics (from historical data)"):
        asset_stats = pd.DataFrame({
            "Ticker":        valid_tickers,
            "Ann. Return (%)": (log_ret.mean() * 252 * 100).round(2).values,
            "Ann. Vol (%)":    (log_ret.std() * np.sqrt(252) * 100).round(2).values,
            "Sharpe (rf=0)":   ((log_ret.mean() * 252) / (log_ret.std() * np.sqrt(252))).round(3).values,
        })
        st.dataframe(asset_stats, use_container_width=True, hide_index=True)

        corr_df = log_ret.corr().round(3)
        st.markdown("**Return correlation matrix (1-year daily)**")
        st.dataframe(corr_df, use_container_width=True)

    with st.expander("Methodology"):
        st.markdown(
            """
            **Geometric Brownian Motion (GBM)**

            Each asset's daily log-return is modelled as:

            > ln(S_{t+1}/S_t) = (μ − ½σ²) + σ · Z_t

            where μ and σ are estimated from 1 year of historical daily log-returns,
            and Z_t is a standard normal random variable.

            **Correlated paths via Cholesky decomposition**

            Rather than simulating each asset independently, the historical correlation
            matrix is Cholesky-decomposed to produce a lower-triangular matrix L such
            that L·Lᵀ = Σ. Independent standard normal draws are then multiplied by Lᵀ
            to produce correlated shocks that respect the observed co-movement between
            assets. This means diversification (or lack of it) is captured correctly.

            **Portfolio construction**

            An equal-weighted portfolio is assumed. Each day's portfolio log-return is the
            weighted sum of individual asset log-returns. Final portfolio value is:

            > V_T = V_0 · exp(Σ r_t)

            **Risk metrics**

            - **VaR (95%)** — the loss threshold exceeded in only 5% of simulated paths
            - **CVaR (95%)** — the average loss across the worst 5% of paths (also called
              Expected Shortfall); a more conservative and coherent risk measure than VaR
            - **P(loss)** — fraction of paths ending below the initial investment

            **Limitations**

            GBM assumes constant μ and σ, normally distributed returns, and no
            jumps — all of which are violated by real equity returns. The model does
            not account for changing correlations in stress periods (correlations tend
            to spike during market crashes). Results should be treated as a stylised
            risk illustration, not a precise forecast.
            """
        )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    "Built by **Ethan Buckley** — "
    "[GitHub](https://github.com/ethanbuckley) · "
    "[LinkedIn](https://www.linkedin.com/in/ethan-buckley-b7ab6935b/)",
    unsafe_allow_html=False,
)
