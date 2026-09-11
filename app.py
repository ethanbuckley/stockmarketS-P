"""
app.py: S&P 500 AI Screener dashboard
Reads data/latest_signals.csv produced by generate_signals.py and the
validation artefacts written by evaluate.py.
Deploy to Streamlit Community Cloud; no heavy ML dependencies required.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# config.py is dependency-free, so importing it never pulls xgboost/torch
# into the deployed app.
from config import (
    BACKTEST_COST_BPS,
    BACKTEST_HOLD_DAYS,
    BOTTOM_N_CANDIDATES,
    CANDIDATE_PRICES_PATH,
    DATA_DIR,
    LONG_POOL,
    SHORT_POOL,
    SIGNALS_PATH,
    TOP_N_CANDIDATES,
    VALIDATION_CALIBRATION_PATH,
    VALIDATION_DAILY_PATH,
    VALIDATION_METRICS_PATH,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

LONG_LABEL = f"Long candidates (top-{TOP_N_CANDIDATES} pool, Sent > 0)"
SHORT_LABEL = f"Short candidates (bottom-{BOTTOM_N_CANDIDATES} pool, Sent < 0)"
POOL_LABEL = {
    LONG_POOL: f"Long pool (top {TOP_N_CANDIDATES})",
    SHORT_POOL: f"Short pool (bottom {BOTTOM_N_CANDIDATES})",
}

BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_daily.csv")

MC_HORIZONS = {"1 month (21 days)": 21, "3 months (63 days)": 63, "1 year (252 days)": 252}

ACCENT = "#6366F1"
MUTED = "#8B95A7"
_GRID = "rgba(255,255,255,0.06)"

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0B0E16; --card:#141A24; --card2:#10151E; --border:rgba(255,255,255,.07);
  --muted:#8B95A7; --text:#E6EAF2; --accent:#6366F1; --accent2:#818CF8;
  --pos:#34D399; --neg:#F87171;
}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,-apple-system,sans-serif;}
.stApp{background:radial-gradient(1100px 560px at 82% -8%,rgba(99,102,241,.10),transparent 60%),var(--bg);}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1320px;}

.hero-badge{display:inline-flex;align-items:center;gap:7px;font-size:.72rem;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;color:var(--accent2);background:rgba(99,102,241,.12);
  border:1px solid rgba(99,102,241,.25);padding:5px 12px;border-radius:999px;margin-bottom:14px;}
.hero-badge .dot{width:7px;height:7px;border-radius:50%;background:var(--pos);box-shadow:0 0 8px var(--pos);}
.hero h1{font-size:2.6rem;font-weight:800;letter-spacing:-.02em;margin:0 0 6px 0;line-height:1.1;
  background:linear-gradient(92deg,#fff 10%,#B9C0FF 60%,#818CF8 100%);-webkit-background-clip:text;
  background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{color:var(--muted);font-size:1.02rem;margin:0;max-width:760px;line-height:1.5;}

.disclaimer{margin:18px 0 6px 0;padding:11px 16px;font-size:.86rem;color:#E2D3AE;
  background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.22);
  border-left:3px solid #FBBF24;border-radius:10px;}

[data-testid="stMetric"]{background:linear-gradient(180deg,var(--card) 0%,var(--card2) 100%);
  border:1px solid var(--border);border-radius:16px;padding:18px 20px;
  box-shadow:0 1px 2px rgba(0,0,0,.35);transition:border-color .15s,transform .15s;}
[data-testid="stMetric"]:hover{border-color:rgba(99,102,241,.45);transform:translateY(-2px);}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:.72rem!important;font-weight:600!important;
  text-transform:uppercase;letter-spacing:.05em;}
[data-testid="stMetricValue"]{font-weight:700;font-size:1.7rem;letter-spacing:-.01em;}

[data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--border);}
button[data-baseweb="tab"]{font-weight:600;font-size:.95rem;padding:10px 4px;}
[data-baseweb="tab-highlight"]{background:var(--accent)!important;height:3px;border-radius:3px;}

[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:14px;overflow:hidden;}
[data-testid="stExpander"]{border:1px solid var(--border);border-radius:14px;background:rgba(255,255,255,.015);}

.footer{color:var(--muted);font-size:.85rem;text-align:center;padding-top:6px;}
.footer a{color:var(--accent2);text-decoration:none;}
.footer a:hover{text-decoration:underline;}
</style>
"""


def style_plotly(fig, height=None):
    """Shared Plotly styling so every chart matches the dark theme."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#C7D0DF", size=13),
        title_font=dict(size=15, color="#E6EAF2"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#141A24", font_size=12, font_family="Inter"),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID)
    return fig


def is_long(df: pd.DataFrame) -> pd.Series:
    return (df["Signal_Pool"] == LONG_POOL) & (df["Sentiment_Score"] > 0)


def is_short(df: pd.DataFrame) -> pd.Series:
    return (df["Signal_Pool"] == SHORT_POOL) & (df["Sentiment_Score"] < 0)


# =============================================================================
# DATA LOADING
# =============================================================================


@st.cache_data(ttl=3600)
def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Signal_Pool" not in df.columns:
        # Older signal files predate the column: the leaderboard is the top
        # TOP_N and bottom BOTTOM_N by confidence, so recover the pools by rank.
        ranked = df["Confidence"].rank(ascending=False, method="first")
        df["Signal_Pool"] = np.where(ranked <= TOP_N_CANDIDATES, LONG_POOL, SHORT_POOL)
    return df


@st.cache_data(ttl=3600)
def load_candidate_prices(path: str) -> pd.DataFrame:
    """Adjusted closes written by generate_signals.py; empty frame if absent."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])


@st.cache_data(ttl=3600)
def load_validation_artefacts():
    with open(VALIDATION_METRICS_PATH) as f:
        metrics = json.load(f)
    daily = pd.read_csv(VALIDATION_DAILY_PATH, parse_dates=["date"])
    calib = pd.read_csv(VALIDATION_CALIBRATION_PATH)
    backtest = pd.read_csv(BACKTEST_PATH, parse_dates=["date"]) if os.path.exists(BACKTEST_PATH) else pd.DataFrame()
    return metrics, daily, calib, backtest


# =============================================================================
# MONTE CARLO ENGINE
# =============================================================================


def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor of a correlation matrix.

    A tiny diagonal ridge absorbs floating-point noise (the fast path that
    handles almost all real return data). If the matrix is genuinely indefinite
    or rank-deficient (fewer observations than assets, or two perfectly
    co-moving tickers), fall back to a nearest-PSD projection via eigenvalue
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
        fixed = fixed / np.outer(d, d)  # renormalise to unit diagonal
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

    mu = log_ret.mean().values  # daily mean log-return per asset
    sigma = log_ret.std().values  # daily vol per asset
    L = _safe_cholesky(log_ret.corr().values)
    n_assets = len(mu)

    # iid standard normals, then correlate: (n_paths, horizon, n_assets)
    Z = rng.standard_normal((n_paths, horizon, n_assets))
    Z_corr = Z @ L.T

    # mu is the *mean of log-returns*, so log-returns are distributed N(mu, sigma^2)
    # and we simulate that directly. Subtracting an extra ½σ² here would apply the
    # Itô correction a second time and bias the drift downward (worse for volatile
    # names and long horizons).
    daily_log_ret = mu + sigma * Z_corr  # dt = 1 day

    # Equal-weighted (in log space) portfolio log-return each day
    w = np.ones(n_assets) / n_assets
    port_daily = daily_log_ret @ w  # (n_paths, horizon)
    port_cum = np.cumsum(port_daily, axis=1)  # (n_paths, horizon)

    growth = np.exp(
        np.concatenate([np.zeros((n_paths, 1)), port_cum], axis=1)
    )  # (n_paths, horizon + 1), initial value = 1

    return growth


# =============================================================================
# PAGE SECTIONS
# =============================================================================


def render_header() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div class="hero-badge"><span class="dot"></span> S&amp;P 500 · XGBoost + FinBERT · Updated weekly</div>
          <h1>S&amp;P 500 stock screener</h1>
          <p>XGBoost signals across the S&amp;P 500, combined with FinBERT news sentiment,
          with a Monte Carlo simulator for portfolio risk and walk-forward validation
          of the classifier.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">⚠️ <b>Educational project, not financial advice.</b> '
        "Past model performance does not guarantee future results; do not make "
        "investment decisions based on this tool.</div>",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.divider()
    st.markdown(
        '<div class="footer">Built by <b>Ethan Buckley</b> &nbsp;·&nbsp; '
        '<a href="https://github.com/ethanbuckley" target="_blank">GitHub</a> &nbsp;·&nbsp; '
        '<a href="https://www.linkedin.com/in/ethan-buckley/" target="_blank">LinkedIn</a></div>',
        unsafe_allow_html=True,
    )


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar controls; returns the filtered signals frame for the Screener tab."""
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

    # A blank Sentiment_Score means no news was available at generation time;
    # NaN rows are kept visible and must not break the slider bounds.
    sent_values = df["Sentiment_Score"].dropna()
    if sent_values.empty:
        sent_lo, sent_hi = -1.0, 1.0
    else:
        sent_lo = float(np.floor(sent_values.min() * 100) / 100)
        sent_hi = float(np.ceil(sent_values.max() * 100) / 100)
    if sent_lo == sent_hi:
        sent_hi = sent_lo + 0.01
    sent_range = st.sidebar.slider(
        "Sentiment Score",
        min_value=sent_lo,
        max_value=sent_hi,
        value=(sent_lo, sent_hi),
        step=0.01,
    )

    signal_filter = st.sidebar.radio("Signal type", options=["All", LONG_LABEL, SHORT_LABEL])

    # Rows with missing sentiment pass the sentiment filter so that
    # "no news available" does not silently hide a candidate.
    in_conf = df["Confidence"].between(conf_range[0], conf_range[1])
    in_sent = df["Sentiment_Score"].isna() | df["Sentiment_Score"].between(sent_range[0], sent_range[1])
    filtered = df[in_conf & in_sent].copy()

    if signal_filter == LONG_LABEL:
        filtered = filtered[is_long(filtered)]
    elif signal_filter == SHORT_LABEL:
        filtered = filtered[is_short(filtered)]
    return filtered


def render_screener(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total candidates", len(df))
    col2.metric("Matching filters", len(filtered))
    col3.metric("Long signals", int(is_long(df).sum()))
    col4.metric("Short signals", int(is_short(df).sum()))

    st.divider()
    st.subheader("Screener Leaderboard")

    display_cols = ["Ticker", "Close", "Confidence", "Sentiment_Score"]
    display_df = filtered[display_cols].sort_values("Confidence", ascending=False).reset_index(drop=True)
    display_df.index += 1

    st.dataframe(display_df, width="stretch", height=400)
    st.caption(
        "A blank sentiment score means no news was available for that ticker "
        "at generation time; it does not mean neutral sentiment."
    )

    st.divider()

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
        st.plotly_chart(style_plotly(fig_bar), width="stretch")

    with col_right:
        st.subheader("Confidence vs Sentiment")
        plot_df = df.assign(Pool=df["Signal_Pool"].map(POOL_LABEL))
        fig_scatter = px.scatter(
            plot_df,
            x="Sentiment_Score",
            y="Confidence",
            text="Ticker",
            color="Pool",
            color_discrete_map={POOL_LABEL[LONG_POOL]: "#34D399", POOL_LABEL[SHORT_POOL]: "#F87171"},
            labels={"Sentiment_Score": "FinBERT Sentiment Score", "Confidence": "XGBoost Confidence (%)"},
            title="Signal Map",
        )
        fig_scatter.update_traces(textposition="top center", marker_size=8)
        fig_scatter.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="neutral news",
            annotation_position="top",
        )
        fig_scatter.update_layout(legend=dict(orientation="h", y=1.12), margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(style_plotly(fig_scatter), width="stretch")
        st.caption(
            "Confidence is the model's probability that the +4% barrier is hit before "
            "the −4% barrier within 5 trading days. Compare it with the ~25–30% base rate, "
            "not with 50%: a well-calibrated 50% is roughly twice the market's hit rate."
        )

    st.divider()

    with st.expander("How this works"):
        st.markdown(
            f"""
            This screener combines two complementary signals to identify long and short
            candidates across the S&P 500.

            **Stage 1: XGBoost classifier (quantitative signal)**

            An XGBoost gradient-boosting model is trained on 10+ years of daily price data
            across all S&P 500 constituents, each from the date it joined the index
            (over a million labelled stock-days). The target label uses
            the *triple-barrier method*: for each trading day, the model asks whether the stock
            will hit a +4% take-profit *before* it hits a −4% stop-loss within the next 5 trading
            days. This is more realistic than simple N-day forward returns because it mirrors
            how a real trade with risk management plays out.

            Features include momentum indicators (RSI, MACD, lagged returns), volatility
            measures (Bollinger Band position, ATR ratio), volume signals (VWAP deviation,
            volume surge), macro context (SPY, QQQ, SMH, VIX, 10Y Treasury), and
            relative performance vs benchmarks. One model is trained across all tickers so
            it learns cross-sectional patterns rather than fitting to any single stock's history.

            **Stage 2: FinBERT sentiment analysis (qualitative signal)**

            The top 15 and bottom 5 candidates by XGBoost confidence are passed to
            [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on
            financial news, analyst reports, and earnings call transcripts. Live headlines
            are fetched for each candidate and scored. The final sentiment score is the
            mean signed score across up to 10 recent headlines.

            **Interpreting the output**

            | Signal | Condition |
            |--------|-----------|
            | Long candidate | In the top-{TOP_N_CANDIDATES} pool by confidence **and** Sentiment > 0 |
            | Short candidate | In the bottom-{BOTTOM_N_CANDIDATES} pool by confidence **and** Sentiment < 0 |

            The pools are rank-based, not threshold-based. The label asks whether +4% is
            hit before −4% within 5 days, which happens for only about a quarter to a
            third of stock-days, so a probability near 50% is already well above the
            base rate. Pool membership alone is not a buy signal: both the quantitative
            and qualitative signals should align. Mixed signals (long pool, negative
            sentiment) warrant caution.
            """
        )


def render_monte_carlo(df: pd.DataFrame) -> None:
    st.subheader("Monte Carlo Portfolio Risk Simulation")
    st.markdown(
        "Simulates thousands of possible future portfolio paths using geometric "
        "Brownian motion calibrated to the historical return distribution of the "
        "screener's top candidates. Correlated asset paths are generated via "
        "Cholesky decomposition of the empirical correlation matrix."
    )

    # Early exits use `return`, not st.stop(): stopping here would also blank
    # the Model Validation tab and the footer, which render after this tab.
    all_prices = load_candidate_prices(CANDIDATE_PRICES_PATH)
    if all_prices.empty:
        st.info(
            "No price history found at `data/candidate_prices.csv`. It is written "
            "alongside the signals by `python generate_signals.py`; commit both files."
        )
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    all_tickers = [t for t in df["Ticker"] if t in all_prices.columns]
    default_tickers = [t for t in df.nlargest(5, "Confidence")["Ticker"] if t in all_tickers]

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        chosen = st.multiselect(
            "Portfolio tickers (from screener candidates)",
            options=all_tickers,
            default=default_tickers,
            max_selections=10,
        )
    with c2:
        horizon = MC_HORIZONS[st.selectbox("Horizon", list(MC_HORIZONS))]
    with c3:
        n_paths = st.selectbox("Simulated paths", [5_000, 10_000, 50_000], index=1)
    with c4:
        initial_value = float(st.number_input("Initial portfolio ($)", value=10_000, step=1_000))

    if not chosen:
        st.info("Select at least one ticker above to run the simulation.")
        return

    # ── Simulate ──────────────────────────────────────────────────────────────
    # Aligned panel: a ticker with a shorter history shortens everyone's
    # window, so drop names with too few observations first.
    prices = all_prices[chosen].dropna(how="all")
    enough = prices.columns[prices.notna().sum() >= 60]
    too_short = [t for t in chosen if t not in enough]
    if too_short:
        st.warning(f"Too little price history for: {', '.join(too_short)}.")
    prices = prices[enough].dropna()

    if prices.shape[1] == 0:
        st.error("No selected ticker has enough price history to simulate.")
        return

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
        return

    if log_ret.shape[0] < 30:
        st.error(
            f"Only {log_ret.shape[0]} overlapping days of history, too few to "
            "estimate risk reliably. Try more established tickers or fewer names."
        )
        return

    with st.spinner(f"Running {n_paths:,} Monte Carlo paths …"):
        growth = simulate_growth(log_ret, horizon, n_paths)
        port_values = initial_value * growth

    # ── Risk metrics ──────────────────────────────────────────────────────────
    final_values = port_values[:, -1]
    final_returns = (final_values - initial_value) / initial_value

    var_95 = float(np.percentile(final_values, 5))
    cvar_95 = float(final_values[final_values <= var_95].mean())
    p_loss = float((final_values < initial_value).mean())
    med_ret = float(np.median(final_returns) * 100)
    mean_ret = float(np.mean(final_returns) * 100)

    # Dollar losses vs the starting value, floored at 0: over long horizons a
    # positive drift can lift even the 5th-percentile outcome above the initial
    # value, which is a gain, not a "negative loss".
    var_loss = max(initial_value - var_95, 0.0)
    cvar_loss = max(initial_value - cvar_95, 0.0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "VaR (95%)",
        f"${var_loss:,.0f}",
        help="Loss not exceeded in 95% of scenarios: you lose less than this (vs. the starting value) 95% of the time",
    )
    m2.metric("CVaR (95%)", f"${cvar_loss:,.0f}", help="Average loss in the worst 5% of scenarios (Expected Shortfall)")
    m3.metric(
        "P(loss)", f"{p_loss * 100:.1f}%", help="Fraction of simulated paths that end below the initial investment"
    )
    m4.metric("Median return", f"{med_ret:+.1f}%")
    m5.metric("Mean return", f"{mean_ret:+.1f}%")

    st.divider()

    st.info(
        f"Starting from \\${initial_value:,.0f}, in the worst 5% of simulated "
        f"{horizon}-day outcomes this portfolio ends near \\${var_95:,.0f} "
        f"(about a \\${var_loss:,.0f} loss). The median outcome is {med_ret:+.1f}%, "
        f"and {p_loss * 100:.0f}% of paths finish below the starting value."
    )

    # ── Fan chart ─────────────────────────────────────────────────────────────
    t_axis = list(range(horizon + 1))
    pcts = np.percentile(port_values, [5, 25, 50, 75, 95], axis=0)

    fig_fan = go.Figure()
    fig_fan.add_trace(
        go.Scatter(
            x=t_axis + t_axis[::-1],
            y=pcts[4].tolist() + pcts[0].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(99,102,241,0.16)",
            line=dict(width=0),
            name="5th–95th pct",
            showlegend=True,
        )
    )
    fig_fan.add_trace(
        go.Scatter(
            x=t_axis + t_axis[::-1],
            y=pcts[3].tolist() + pcts[1].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(99,102,241,0.28)",
            line=dict(width=0),
            name="25th–75th pct",
            showlegend=True,
        )
    )
    fig_fan.add_trace(
        go.Scatter(
            x=t_axis,
            y=pcts[2],
            name="Median",
            line=dict(color=ACCENT, width=2.5),
        )
    )
    fig_fan.add_hline(
        y=initial_value,
        line_dash="dot",
        line_color="gray",
        opacity=0.6,
        annotation_text="Initial value",
        annotation_position="right",
    )
    fig_fan.update_layout(
        title=f"Simulated portfolio paths, {horizon}-day horizon  "
        f"({n_paths:,} paths, equal-weighted: {', '.join(valid_tickers)})",
        xaxis_title="Trading days",
        yaxis_title="Portfolio value ($)",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=60, b=0),
        height=420,
    )
    st.plotly_chart(style_plotly(fig_fan), width="stretch")

    # ── Final value distribution ───────────────────────────────────────────────
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=80,
            marker_color=ACCENT,
            opacity=0.75,
            name="Final value",
        )
    )
    fig_hist.add_vline(
        x=initial_value, line_dash="dot", line_color="gray", annotation_text="Initial", annotation_position="top right"
    )
    fig_hist.add_vline(
        x=var_95, line_dash="dash", line_color="#DC2626", annotation_text="VaR 95%", annotation_position="top left"
    )
    fig_hist.add_vline(
        x=cvar_95, line_dash="dash", line_color="#F97316", annotation_text="CVaR 95%", annotation_position="top left"
    )
    fig_hist.update_layout(
        title="Distribution of final portfolio value",
        xaxis_title="Portfolio value ($)",
        yaxis_title="Number of paths",
        margin=dict(l=0, r=0, t=50, b=0),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(style_plotly(fig_hist), width="stretch")

    # ── Individual asset stats ─────────────────────────────────────────────────
    st.caption(
        f"Calibrated on {log_ret.shape[0]} trading days of adjusted closes ending "
        f"{prices.index.max():%Y-%m-%d}, written with the signals by generate_signals.py."
    )

    with st.expander("Individual asset statistics (from historical data)"):
        ann_ret = log_ret.mean() * 252
        ann_vol = log_ret.std() * np.sqrt(252)
        asset_stats = pd.DataFrame(
            {
                "Ticker": valid_tickers,
                "Ann. Return (%)": (ann_ret * 100).round(2).values,
                "Ann. Vol (%)": (ann_vol * 100).round(2).values,
                "Sharpe (rf=0)": (ann_ret / ann_vol).round(3).values,
            }
        )
        st.dataframe(asset_stats, width="stretch", hide_index=True)

        st.markdown("**Return correlation matrix (daily, committed history)**")
        st.dataframe(log_ret.corr().round(3), width="stretch")

    with st.expander("Methodology"):
        st.markdown(
            """
            **Geometric Brownian Motion (GBM)**

            Each asset's daily log-return is modelled as:

            > ln(S_{t+1}/S_t) = μ + σ · Z_t

            where μ and σ are the mean and standard deviation of the committed
            year of historical daily log-returns, and Z_t is a standard normal random
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

            - **VaR (95%)**: the loss threshold exceeded in only 5% of simulated paths
            - **CVaR (95%)**: the average loss across the worst 5% of paths (also called
              Expected Shortfall); a more conservative and coherent risk measure than VaR
            - **P(loss)**: fraction of paths ending below the initial investment

            **Limitations**

            GBM assumes constant μ and σ, normally distributed returns, and no
            jumps, all of which are violated by real equity returns. The model does
            not account for changing correlations in stress periods (correlations tend
            to spike during market crashes). Results should be treated as a stylised
            risk illustration, not a precise forecast.
            """
        )


def render_validation() -> None:
    st.subheader("Walk-Forward Validation")

    artefacts_present = all(
        os.path.exists(p) for p in (VALIDATION_METRICS_PATH, VALIDATION_DAILY_PATH, VALIDATION_CALIBRATION_PATH)
    )
    if not artefacts_present:
        st.info(
            "No validation artefacts found. Run `python evaluate.py` locally to "
            "generate `data/validation_metrics.json` and the daily/calibration "
            "CSVs, then commit them."
        )
        return

    metrics, val_daily, val_calib, val_backtest = load_validation_artefacts()
    pooled = metrics["results"]["pooled"]
    overall = metrics["results"]["overall_daily"]
    snapshot = metrics["data_snapshot"]
    scheme = metrics["validation_scheme"]

    st.markdown(
        "Out-of-sample performance of the XGBoost classifier, measured the "
        "way the screener is actually used: each test day, rank the whole "
        "S&P 500 cross-section and take the top 15. Expanding-window "
        "walk-forward folds with calendar-year test blocks; training rows "
        "whose 5-day label windows would overlap a test period are purged."
    )

    v1, v2, v3, v4, v5, v6 = st.columns(6)
    v1.metric(
        "ROC AUC (pooled)",
        f"{pooled['roc_auc']:.3f}",
        help="Across all out-of-sample test rows, each scored by its own fold's model. 0.5 is chance. "
        "Includes the between-day 'regime' effect; see the per-day AUC for pure ranking skill.",
    )
    v6.metric(
        "Per-day AUC (mean)",
        f"{overall.get('daily_auc_mean', float('nan')):.3f}",
        help="ROC AUC computed within each test day's cross-section, then averaged: the ranking skill a "
        "screener can actually use. A model that only knew which days were good would score 0.5 here.",
    )
    v2.metric(
        "Brier score",
        f"{pooled['brier']:.3f}",
        help="Mean squared error of the predicted probabilities; lower is better.",
    )
    v3.metric(
        "Precision@15 (daily mean)",
        f"{overall['precision_top15_mean'] * 100:.1f}%",
        delta=f"{overall['excess_precision_top15_mean'] * 100:+.1f} pts vs base rate",
        help="Fraction of each day's top-15 picks whose take-profit barrier was hit first, averaged over test days.",
    )
    v4.metric(
        "Daily base rate",
        f"{overall['base_rate_daily_mean'] * 100:.1f}%",
        help="Average fraction of all stocks that hit the take-profit barrier first on a given test day.",
    )
    v5.metric(
        "Top-decile lift",
        f"{overall['top_decile_lift_mean']:.2f}x",
        help="Hit rate of the top decile by predicted probability relative to the day's base rate.",
    )
    ci_lo, ci_hi = overall["excess_precision_top15_ci95"]
    universe = snapshot.get("universe", {})
    universe_note = (
        f"{universe['n_current']} current members + {universe['n_removed_with_price_data']} of "
        f"{universe['n_removed_since_start']} former members"
        if universe.get("type") == "point_in_time"
        else f"{snapshot['n_tickers']} tickers"
    )
    st.caption(
        f"Data snapshot: {snapshot['last_price_date']} "
        f"({universe_note}) · "
        f"generated {metrics['generated_at_utc']} · "
        f"top-15 beats the base rate on "
        f"{overall['frac_days_top15_beats_base'] * 100:.0f}% of "
        f"{overall['n_days']} test days · "
        f"95% CI on the mean excess: [{ci_lo * 100:+.1f}, {ci_hi * 100:+.1f}] pts"
    )

    atr = metrics["results"].get("baselines", {}).get("atr_rank")
    if atr:
        st.caption(
            f"Volatility-only baseline (rank each day by ATR ratio, no model): per-day AUC "
            f"{atr['daily_auc_mean']:.3f}, precision@15 {atr['precision_top15_mean'] * 100:.1f}% "
            f"({atr['excess_precision_top15_mean'] * 100:+.1f} pts vs base). The model's margin over this "
            f"is its value beyond 'buy the most volatile names'."
        )

    # The caveats ship inside the same artefact as the numbers, so they
    # are always rendered alongside them.
    st.warning("**Read before quoting these numbers**\n\n" + "\n".join(f"- {c}" for c in metrics["caveats"]))

    st.divider()

    fold_cols = ["fold_id", "test_start", "partial"] + (
        ["n_estimators"] if "n_estimators" in scheme["folds"][0] else []
    )
    folds_df = pd.DataFrame(scheme["folds"])[fold_cols]
    perf_df = pd.DataFrame(metrics["results"]["per_fold"])
    fold_table = folds_df.merge(perf_df, on="fold_id")
    fold_table["Test year"] = fold_table["test_start"].str[:4] + np.where(fold_table["partial"], " (partial)", "")
    year_label = dict(zip(fold_table["fold_id"], fold_table["Test year"], strict=True))

    st.subheader("Per-Fold Results")
    ci = fold_table["excess_precision_top15_ci95"].apply(lambda c: f"{c[0] * 100:+.1f} to {c[1] * 100:+.1f}")
    display = pd.DataFrame(
        {
            "Test year": fold_table["Test year"],
            "Trees": fold_table["n_estimators"] if "n_estimators" in fold_table else "",
            "Test rows": fold_table["n_test_rows"],
            "ROC AUC": fold_table["roc_auc"].round(3),
            "Per-day AUC": fold_table["daily_auc_mean"].round(3) if "daily_auc_mean" in fold_table else "",
            "Brier": fold_table["brier"].round(3),
            "Base rate (%)": (fold_table["base_rate"] * 100).round(1),
            "P@15 mean (%)": (fold_table["precision_top15_mean"] * 100).round(1),
            "Excess (pts)": (fold_table["excess_precision_top15_mean"] * 100).round(1),
            "Excess 95% CI (pts)": ci,
            "Days beating base (%)": (fold_table["frac_days_top15_beats_base"] * 100).round(0),
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)

    col_box, col_rel = st.columns(2)

    with col_box:
        box_df = val_daily.copy()
        box_df["Test year"] = box_df["fold_id"].map(year_label)
        box_df = box_df.melt(
            id_vars=["Test year"],
            value_vars=["precision_top15", "base_rate"],
            var_name="Metric",
            value_name="Value",
        )
        box_df["Metric"] = box_df["Metric"].map({"precision_top15": "Precision@15", "base_rate": "Base rate"})
        fig_box = px.box(
            box_df,
            x="Test year",
            y="Value",
            color="Metric",
            title="Daily precision@15 vs base rate, by test year",
            color_discrete_sequence=[ACCENT, MUTED],
        )
        fig_box.update_layout(
            yaxis_title="Daily hit rate",
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(style_plotly(fig_box, height=400), width="stretch")

    with col_rel:
        pooled_calib = val_calib[val_calib["fold_id"].astype(str) == "pooled"]
        lo = float(pooled_calib["mean_predicted"].min())
        hi = float(pooled_calib["mean_predicted"].max())
        fig_rel = go.Figure()
        fig_rel.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect calibration",
                line=dict(color="gray", dash="dash"),
            )
        )
        fig_rel.add_trace(
            go.Scatter(
                x=pooled_calib["mean_predicted"],
                y=pooled_calib["observed_rate"],
                mode="lines+markers",
                name="Model (pooled)",
                line=dict(color=ACCENT, width=2.5),
                marker_size=8,
                customdata=pooled_calib["count"],
                hovertemplate="Predicted %{x:.3f}<br>Observed %{y:.3f}<br>n=%{customdata}<extra></extra>",
            )
        )
        fig_rel.update_layout(
            title="Reliability curve (quantile bins, pooled test rows)",
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed positive rate",
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(style_plotly(fig_rel, height=400), width="stretch")

    render_backtest(metrics, val_backtest)

    with st.expander("How this validation works"):
        st.markdown(
            f"""
            **Protocol.** Expanding-window walk-forward validation with
            calendar-year test blocks, starting in
            {scheme["first_test_year"]}. For each fold, the model is
            retrained from scratch on all data up to the fold's training
            cutoff using the production training code and hyperparameters,
            then scores every day in the test year.

            **Leakage control.** The triple-barrier label for day *t* looks
            at the next {scheme["purge_trading_days"]} trading days, so the
            last {scheme["purge_trading_days"]} trading days before each
            test block are removed from training: their labels would peek
            into the test period. All features are strictly backward-looking
            (rolling windows, exponential averages, lags), which is enforced
            by an automated causality check that rebuilds features from
            truncated data and asserts they are unchanged.

            **Metrics.** Precision@15 mirrors deployment: rank the day's
            cross-section by predicted probability, take the top 15, and
            measure how many hit the take-profit barrier first. The base
            rate is the same quantity for the whole cross-section, so the
            excess is the value added by the ranking. Pooled ROC AUC and the
            Brier score are computed over all test rows; the per-day AUC is
            computed inside each day and averaged, which strips out the
            model's ability to tell good days from bad and leaves only
            ranking skill. The reliability curve shows whether predicted
            probabilities match observed frequencies. Confidence intervals
            use a moving-block bootstrap (block length
            {scheme["purge_trading_days"]}) because overlapping label
            windows make consecutive days dependent.

            The full implementation is in `evaluate.py`; every number on
            this page is read from `data/validation_metrics.json`, which
            records the data snapshot, fold boundaries, library versions
            and git commit that produced it.
            """
        )


def render_backtest(metrics: dict, bt: pd.DataFrame) -> None:
    block = metrics["results"].get("backtest")
    if not block or bt.empty:
        return

    st.divider()
    st.subheader("Portfolio Backtest (long-only, after costs)")
    a = block["assumptions"]
    st.markdown(
        f"Each test day's top {a['top_n']} picks form an equal-weighted tranche held "
        f"{a['hold_days']} trading days; {a['hold_days']} tranches overlap, so a fifth of the book rolls "
        f"daily. Costs: {a['cost_bps_per_side']:.0f} bps per side on every entry and exit. Compared with an "
        "equal-weighted, cost-free portfolio of the whole eligible cross-section (what the ranking chooses "
        "from) and with SPY. Same out-of-sample predictions as the metrics above."
    )

    series = block["series"]
    net, gross, ew, spy = series["strategy_net"], series["strategy_gross"], series["universe_ew"], series["spy"]
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric(
        "Ann. return (net)",
        f"{net['ann_return'] * 100:+.1f}%",
        delta=f"{(net['ann_return'] - ew['ann_return']) * 100:+.1f} pts vs universe EW",
    )
    b2.metric(
        "Ann. return (gross)",
        f"{gross['ann_return'] * 100:+.1f}%",
        help="Before costs. The gap to net is the price of rolling a fifth of the book every day.",
    )
    b3.metric("Sharpe (net, rf=0)", f"{net['sharpe']:.2f}", delta=f"{net['sharpe'] - ew['sharpe']:+.2f} vs universe EW")
    b4.metric("Max drawdown (net)", f"{net['max_drawdown'] * 100:.1f}%")
    b5.metric("SPY ann. return", f"{spy['ann_return'] * 100:+.1f}%", help="Same test window, buy and hold.")
    atr_bt = metrics["results"].get("baselines", {}).get("atr_rank", {}).get("backtest")
    if atr_bt:
        st.caption(
            f"Same book built from the volatility-only ranking: {atr_bt['strategy_net']['ann_return'] * 100:+.1f}%/yr "
            f"net (Sharpe {atr_bt['strategy_net']['sharpe']:.2f}, max drawdown "
            f"{atr_bt['strategy_net']['max_drawdown'] * 100:.1f}%)."
        )

    curve = bt.set_index("date")[["strategy_net", "strategy_gross", "universe_ew", "spy"]].fillna(0.0)
    equity = (1.0 + curve).cumprod()
    names = {
        "strategy_net": f"Top-{a['top_n']} book, net",
        "strategy_gross": f"Top-{a['top_n']} book, gross",
        "universe_ew": "Universe equal-weight",
        "spy": "SPY",
    }
    colors = {"strategy_net": ACCENT, "strategy_gross": "#A5B4FC", "universe_ew": MUTED, "spy": "#F59E0B"}
    fig_eq = go.Figure()
    for col in ["strategy_net", "strategy_gross", "universe_ew", "spy"]:
        fig_eq.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity[col],
                name=names[col],
                line=dict(
                    color=colors[col],
                    width=2.5 if col == "strategy_net" else 1.5,
                    dash="dot" if col == "strategy_gross" else None,
                ),
            )
        )
    fig_eq.update_layout(
        title="Growth of 1 (log scale), out-of-sample test period",
        yaxis_type="log",
        yaxis_title="Value",
        xaxis_title="",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    st.plotly_chart(style_plotly(fig_eq, height=420), width="stretch")

    by_year = pd.DataFrame(block["by_year"])
    table = pd.DataFrame(
        {
            "Year": by_year["year"].astype(str) + np.where(by_year["n_days"] < 200, " (partial)", ""),
            "Book net (%)": (by_year["strategy_net"] * 100).round(1),
            "Book gross (%)": (by_year["strategy_gross"] * 100).round(1),
            "Universe EW (%)": (by_year["universe_ew"] * 100).round(1),
            "SPY (%)": (by_year["spy"] * 100).round(1),
            "Net minus universe (pts)": ((by_year["strategy_net"] - by_year["universe_ew"]) * 100).round(1),
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        f"Close-to-close holding, no barrier exits, no slippage or capacity model; {BACKTEST_COST_BPS:.0f} bps per "
        f"side, {BACKTEST_HOLD_DAYS}-day hold. The universe benchmark carries the same survivorship bias as the book."
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="S&P 500 AI Screener", page_icon="📈", layout="wide")
    render_header()

    if not os.path.exists(SIGNALS_PATH):
        st.warning(
            "No signals file found at `data/latest_signals.csv`. "
            "Run `python generate_signals.py` locally to generate it, "
            "then commit the file to the repository."
        )
        st.stop()

    df = load_signals(SIGNALS_PATH)

    if "generated_at" in df.columns:
        st.caption(f"Last updated: **{df['generated_at'].iloc[0]} UTC**")
    else:
        st.caption("Last updated: timestamp not available")

    filtered = sidebar_filters(df)

    tab_screener, tab_mc, tab_validation = st.tabs(["Screener", "Monte Carlo Risk", "Model Validation"])
    with tab_screener:
        render_screener(df, filtered)
    with tab_mc:
        render_monte_carlo(df)
    with tab_validation:
        render_validation()

    render_footer()


main()
