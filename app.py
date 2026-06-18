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

conf_lo = float(np.floor(df["Confidence"].min() * 10) / 10)
conf_hi = float(np.ceil(df["Confidence"].max() * 10) / 10)
if conf_lo == conf_hi:  # single candidate / all-equal confidence -> avoid slider crash
    conf_hi = conf_lo + 0.1
conf_range = st.sidebar.slider(
    "Confidence (%)",
    min_value=conf_lo,
    max_value=conf_hi,
    value=(conf_lo, conf_hi),
    step=0.1,
)

sent_lo = float(np.floor(df["Sentiment_Score"].min() * 100) / 100)
sent_hi = float(np.ceil(df["Sentiment_Score"].max() * 100) / 100)
if sent_lo == sent_hi:
    sent_hi = sent_lo + 0.01
sent_range = st.sidebar.slider(
    "Sentiment Score",
    min_value=sent_lo,
    max_value=sent_hi,
    value=(sent_lo, sent_hi),
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
    col1.metric("Total candidates", len(df))
    col2.metric("Matching filters", len(filtered))
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
    """Download adjusted close prices for a tuple of tickers.

    Returns an empty DataFrame on any download failure (network error, rate
    limit, delisted/unknown ticker) so callers can degrade gracefully instead
    of crashing the dashboard.
    """
    try:
        raw = yf.download(
            list(tickers), period=period, progress=False, auto_adjust=True
        )
    except Exception as exc:  # noqa: BLE001 - surface any yfinance/network failure
        st.error(f"Price download failed (yfinance): {exc}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = list(tickers)
    return prices.dropna(how="all")


def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor of a correlation matrix.

    A tiny diagonal ridge absorbs floating-point noise (the fast path that
    handles almost all real return data). If the matrix is genuinely indefinite
    or rank-deficient — e.g. fewer observations than assets, or two perfectly
    co-moving tickers — fall back to a nearest-PSD projection via eigenvalue
    clipping (``np.linalg.eigh`` is reliable on symmetric matrices).
    """
    n = len(corr)
    try:
        return np.linalg.cholesky(corr + 1e-10 * np.eye(n))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(corr)
        vals = np.clip(vals, 1e-8, None)
        fixed = vecs @ np.diag(vals) @ vecs.T
        d = np.sqrt(np.diag(fixed))
        fixed = fixed / np.outer(d, d)            # renormalise to unit diagonal
        return np.linalg.cholesky(fixed)


@st.cache_data(ttl=3600, show_spinner=False)
def simulate_growth(
    log_ret: pd.DataFrame,
    horizon: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate correlated GBM growth factors for an equal-weighted portfolio.

    Returns *unit* growth factors (initial value = 1) of shape
    ``(n_paths, horizon + 1)``. The dollar initial value is applied by the
    caller, so changing it never re-runs the simulation. Cached on the return
    data, horizon and path count, so unrelated widget changes (sidebar filters,
    initial value) hit the cache instead of re-simulating.

    Co-movement is preserved with a Cholesky factor of the historical
    correlation matrix; scaling the correlated shocks by each asset's ``sigma``
    reproduces the full covariance.
    """
    rng = np.random.default_rng(seed)

    mu    = log_ret.mean().values          # daily mean log-return per asset
    sigma = log_ret.std().values           # daily vol per asset
    L     = _safe_cholesky(log_ret.corr().values)
    n_assets = len(mu)

    # iid standard normals, then correlate: (n_paths, horizon, n_assets)
    Z = rng.standard_normal((n_paths, horizon, n_assets))
    Z_corr = Z @ L.T

    # mu is the *mean of log-returns*, so log-returns are distributed N(mu, sigma^2)
    # and we simulate that directly. Subtracting an extra ½σ² here would apply the
    # Itô correction a second time and bias the drift downward (worse for volatile
    # names and long horizons).
    daily_log_ret = mu + sigma * Z_corr      # dt = 1 day

    # Equal-weighted (in log space) portfolio log-return each day
    w = np.ones(n_assets) / n_assets
    port_daily = daily_log_ret @ w                   # (n_paths, horizon)
    port_cum   = np.cumsum(port_daily, axis=1)       # (n_paths, horizon)

    growth = np.exp(
        np.concatenate([np.zeros((n_paths, 1)), port_cum], axis=1)
    )  # (n_paths, horizon + 1), initial value = 1

    return growth


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

    # Keep only tickers that returned data
    prices = prices[[t for t in chosen if t in prices.columns]].dropna()

    if prices.shape[1] == 0:
        st.error("Could not fetch price data for any selected ticker (yfinance).")
        st.stop()

    not_fetched = [t for t in chosen if t not in prices.columns]
    if not_fetched:
        st.warning(f"No price data for: {', '.join(not_fetched)}.")

    log_ret = np.log(prices / prices.shift(1)).dropna()

    # Drop assets with no price variation: a constant series produces NaN
    # correlations and would otherwise break the Cholesky factorisation.
    nonconstant = log_ret.std() > 0
    if (~nonconstant).any():
        flat = list(log_ret.columns[~nonconstant])
        st.warning(f"Ignoring assets with no price variation: {', '.join(flat)}.")
        log_ret = log_ret.loc[:, nonconstant]

    valid_tickers = list(log_ret.columns)

    if len(valid_tickers) == 0:
        st.error("Could not obtain usable price data for any selected ticker.")
        st.stop()

    if log_ret.shape[0] < 30:
        st.error(
            f"Only {log_ret.shape[0]} overlapping days of history — too few to "
            "estimate risk reliably. Try more established tickers or fewer names."
        )
        st.stop()

    with st.spinner(f"Running {n_paths:,} Monte Carlo paths …"):
        growth = simulate_growth(log_ret, horizon, n_paths)
        port_values = float(initial_value) * growth

    # ── Risk metrics ──────────────────────────────────────────────────────────
    final_values  = port_values[:, -1]
    final_returns = (final_values - initial_value) / initial_value

    var_95  = float(np.percentile(final_values, 5))
    cvar_95 = float(final_values[final_values <= var_95].mean())
    p_loss  = float((final_values < initial_value).mean())
    med_ret = float(np.median(final_returns) * 100)
    mean_ret = float(np.mean(final_returns) * 100)

    # Dollar losses vs the starting value, floored at 0: over long horizons a
    # positive drift can lift even the 5th-percentile outcome above the initial
    # value, which is a gain — not a "negative loss".
    var_loss  = max(initial_value - var_95, 0.0)
    cvar_loss = max(initial_value - cvar_95, 0.0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("VaR (95%)",  f"${var_loss:,.0f}",
              help="Loss not exceeded in 95% of scenarios — you lose less than "
                   "this (vs. the starting value) 95% of the time")
    m2.metric("CVaR (95%)", f"${cvar_loss:,.0f}",
              help="Average loss in the worst 5% of scenarios (Expected Shortfall)")
    m3.metric("P(loss)",    f"{p_loss*100:.1f}%",
              help="Fraction of simulated paths that end below the initial investment")
    m4.metric("Median return", f"{med_ret:+.1f}%")
    m5.metric("Mean return",   f"{mean_ret:+.1f}%")

    st.divider()

    st.info(
        f"Starting from \\${initial_value:,.0f}, in the worst 5% of simulated "
        f"{horizon}-day outcomes this portfolio ends near \\${var_95:,.0f} "
        f"(about a \\${var_loss:,.0f} loss). The median outcome is {med_ret:+.1f}%, "
        f"and {p_loss * 100:.0f}% of paths finish below the starting value."
    )

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

            > ln(S_{t+1}/S_t) = μ + σ · Z_t

            where μ and σ are the mean and standard deviation of 1 year of
            historical daily log-returns, and Z_t is a standard normal random
            variable. Because μ is estimated directly as the mean *log*-return,
            log-returns are simulated as N(μ, σ²); the Itô "−½σ²" term is not
            subtracted again (doing so would double-count it and bias the drift
            downward, increasingly so for volatile names and long horizons).

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
