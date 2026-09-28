"""
preregistered.py - executes alpha/PREREGISTRATION.md exactly.

Every element of the strategy is a module constant below. There is no CLI
option, environment variable or function argument that changes the model, the
features, the penalty, the smoothing, the weighting or the metric. The only
switch is which price source to read, which the pre-registration itself makes
conditional on entitlement.

    python3 alpha/preregistered.py baseline      # survivors-only arm (current data)
    python3 alpha/preregistered.py confirmatory  # survivorship-free arm (WRDS)

`baseline` is run BEFORE the WRDS data arrives and stored, so Test B is a
paired comparison on identical dates rather than two unanchored estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alpha.replicate_lps import monthly_components, nw_tstat, spread  # noqa: E402
from alpha.wide_panel import FEATS  # noqa: E402

RESULTS = REPO / "alpha" / "results"
BASELINE_JSON = RESULTS / "prereg_baseline.json"
CONFIRM_JSON = RESULTS / "prereg_confirmatory.json"

# ----------------------------------------------------------------- FROZEN
RIDGE_ALPHA = 100.0
TEST_YEARS = list(range(2020, 2027))
SMOOTH = 1
WEIGHT_QUANTILE = 10  # equal-weight top/bottom decile
GROSS_TRADED_PER_DAY = 2.0  # in at the close, out at the open
WINSOR = (0.005, 0.995)
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 1e6
DSR_PASS_N = 108  # every trial already spent
DSR_THRESHOLD = 0.95
TEST_A_T_THRESHOLD = -2.0
EULER = 0.5772156649015329
ANN = np.sqrt(252.0)
GATE_MIN_NAMES_PER_DAY = 300
GATE_MIN_COVERAGE = 0.90
GATE_MIN_NEW_TICKERS = 150
GATE_DECOMP_TOL = 1e-9
# -------------------------------------------------------------------------


def sd_sr_from_prior_search() -> float:
    """sd(SR) characterising the search ALREADY spent; frozen, not recomputed."""
    g = pd.read_csv(RESULTS / "deflated_grid.csv")
    return float(g["SR_daily"].std(ddof=1))


def oos_pred(X, y, years, dates):
    p = np.full(len(y), np.nan)
    for yr in TEST_YEARS:
        te, tr = years == yr, years < yr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        trd = np.unique(dates[tr])
        tr = tr & (dates < trd[-1])
        p[te] = Ridge(alpha=RIDGE_ALPHA).fit(X[tr], y[tr]).predict(X[te])
    return p


def decile_pnl(scores, actual, dates):
    s = pd.Series(scores, index=dates)
    r = s.groupby(level=0).rank(pct=True)
    w = pd.Series(0.0, index=s.index)
    w[r >= 1 - 1 / WEIGHT_QUANTILE] = 1.0
    w[r <= 1 / WEIGHT_QUANTILE] = -1.0
    g = w.abs().groupby(level=0).transform("sum").replace(0, np.nan)
    return pd.DataFrame({"d": dates, "x": (w / g).values * actual}).groupby("d")["x"].sum()


def deflated_sharpe(pnl: pd.Series, sd_sr: float, n_trials: int) -> float:
    x = pnl.dropna().values
    sr = x.mean() / x.std(ddof=1)
    skew = pd.Series(x).skew()
    kurt = pd.Series(x).kurtosis() + 3.0
    z1, z2 = norm.ppf(1 - 1 / n_trials), norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = sd_sr * ((1 - EULER) * z1 + EULER * z2)
    den = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / den))


def block_bootstrap_ci(x: np.ndarray, block: int = 20, reps: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = len(x)
    nb = max(T // block, 1)
    out = []
    for _ in range(reps):
        st = rng.integers(0, max(T - block, 1), nb)
        out.append(np.concatenate([x[s : s + block] for s in st]).mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# -------------------------------------------------------------------- gates
def run_gates(df: pd.DataFrame, membership: pd.DataFrame | None, known_tickers: set[str], arm: str) -> dict:
    g = {}
    chk = ((1 + df["r_overnight"]) * (1 + df["r_intraday"]) - 1 - df["r_cc"]).abs()
    g["decomposition_identity"] = {
        "value": float(chk.max()) if chk.notna().any() else 0.0,
        "pass": bool((chk.max() if chk.notna().any() else 0.0) < GATE_DECOMP_TOL),
    }

    npd = float(df.groupby("Date").size().median())
    g["names_per_day"] = {"value": npd, "pass": bool(npd >= GATE_MIN_NAMES_PER_DAY)}

    # Gates 2 and 4 check that the NEW data is genuinely survivorship-free, so
    # they apply to the confirmatory arm only (PREREGISTRATION.md Amendment 1).
    # The baseline arm IS the survivor-biased comparison point; requiring it to
    # have full point-in-time coverage is a category error.
    if arm == "confirmatory":
        if membership is not None and len(membership):
            have = set(df["Ticker"].unique())
            by = membership.assign(h=membership["Ticker"].isin(have)).groupby(membership["asof"].dt.year)["h"].mean()
            worst = float(by.min())
            g["pit_coverage_min_year"] = {"value": round(worst, 4), "pass": bool(worst >= GATE_MIN_COVERAGE)}
        else:
            g["pit_coverage_min_year"] = {"value": None, "pass": False}
        new = len(set(df["Ticker"].unique()) - known_tickers)
        g["delisted_tickers_present"] = {"value": new, "pass": bool(new >= GATE_MIN_NEW_TICKERS)}
    else:
        g["pit_coverage_min_year"] = {"value": "n/a (confirmatory only)", "pass": True}
        g["delisted_tickers_present"] = {"value": "n/a (confirmatory only)", "pass": True}
    return g


# -------------------------------------------------------------------- panel
def frozen_panel(price_files: list[Path], membership: pd.DataFrame | None) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in price_files], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"], keep="first").reset_index(drop=True)
    df["dv"] = df["Close"] * df["Volume"]
    df = df[(df["Close"] >= MIN_PRICE) & (df["Volume"] > 0)]
    med = df.groupby("Ticker", observed=True)["dv"].transform("median")
    df = df[med >= MIN_DOLLAR_VOLUME]

    g = df.groupby("Ticker", observed=True)
    for col, tag in (("r_overnight", "on"), ("r_intraday", "id")):
        df[f"{tag}_lag1"] = g[col].shift(1)
        df[f"{tag}_5"] = g[col].transform(lambda s: s.shift(1).rolling(5).sum())
        df[f"{tag}_21"] = g[col].transform(lambda s: s.shift(1).rolling(21).sum())
        df[f"{tag}_ewma"] = g[col].transform(lambda s: s.shift(1).ewm(halflife=60, min_periods=21).mean())
    df["tug"] = df["on_ewma"] - df["id_ewma"]
    df["tug_21"] = df["on_21"] - df["id_21"]
    df["ret_lag1"] = g["r_cc"].shift(1)
    df["ret_lag2"] = g["r_cc"].shift(2)
    df["ret_5"] = g["r_cc"].transform(lambda s: s.shift(1).rolling(5).sum())
    df["vol_21"] = g["r_cc"].transform(lambda s: s.shift(1).rolling(21).std())
    df["vol_surge"] = df["Volume"] / g["Volume"].transform(lambda s: s.shift(1).rolling(21).mean())
    df["dv_rank"] = df.groupby("Date", observed=True)["dv"].rank(pct=True)

    cal = pd.Index(sorted(df["Date"].unique()))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    df["dpos"] = df["Date"].map(pos)
    ok = (g["dpos"].shift(-1) - df["dpos"]) == 1
    df["y_on"] = g["r_overnight"].shift(-1).where(ok)
    df = df.dropna(subset=FEATS + ["y_on"])
    df["y_on"] = df.groupby("Date", observed=True)["y_on"].transform(
        lambda s: s.clip(s.quantile(WINSOR[0]), s.quantile(WINSOR[1]))
    )
    df["y_on"] = df["y_on"] - df.groupby("Date", observed=True)["y_on"].transform("mean")

    if membership is not None and len(membership):
        m = membership[["asof", "Ticker"]].copy()
        m["is_member"] = True
        df = pd.merge_asof(
            df.sort_values("Date"),
            m.sort_values("asof"),
            left_on="Date",
            right_on="asof",
            by="Ticker",
            direction="backward",
            tolerance=pd.Timedelta("200D"),
        )
        df = df[df["is_member"].fillna(False).astype(bool)]
    return df.reset_index(drop=True)


def run_arm(arm: str) -> dict:
    from alpha.source import membership_file, price_files

    t0 = time.time()
    memb = pd.read_parquet(membership_file()) if membership_file().exists() else None
    if memb is not None:
        memb["asof"] = pd.to_datetime(memb["asof"]).astype("datetime64[ns]")
        memb = memb[memb["asof"] >= "2015-01-01"]

    known = set()
    if arm == "confirmatory" and (REPO / "alpha" / "ohlc_open.parquet").exists():
        known = set(pd.read_parquet(REPO / "alpha" / "ohlc_open.parquet")["Ticker"].unique())

    files = price_files()
    print(f"arm={arm}  sources={[p.name for p in files]}", flush=True)
    df = frozen_panel(files, memb)
    gates = run_gates(df, memb, known, arm)
    print("\n=== data-quality gates (pre-specified) ===")
    for k, v in gates.items():
        print(f"  {k:<28} {str(v['value'])[:12]:>14}   {'PASS' if v['pass'] else 'FAIL'}")
    if not all(v["pass"] for v in gates.values()):
        print("\nGATE FAILURE -> per PREREGISTRATION.md no test is reported.")
        return {"arm": arm, "gates": gates, "tests_run": False}

    X = df.groupby("Date", observed=True)[FEATS].rank(pct=True).values - 0.5
    y = df["y_on"].values
    years = df["Date"].dt.year.values
    dates = df["Date"].values
    p = oos_pred(X, y, years, dates)
    m = np.isin(years, TEST_YEARS)
    pnl = decile_pnl(p[m], y[m], dates[m])
    ic = (
        pd.DataFrame({"p": p[m], "a": y[m], "d": dates[m]})
        .dropna()
        .groupby("d")
        .apply(lambda x: x["p"].corr(x["a"]), include_groups=False)
    )

    gross = float(pnl.mean() * 1e4)
    out = {
        "arm": arm,
        "gates": gates,
        "tests_run": True,
        "days": int(len(pnl)),
        "names_per_day": float(df.groupby("Date").size().median()),
        "tickers": int(df["Ticker"].nunique()),
        "mean_IC": float(ic.mean()),
        "IC_t": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))),
        "gross_bps_day": gross,
        "break_even_bps_side": gross / GROSS_TRADED_PER_DAY,
        "gross_SR_ann": float(pnl.mean() / pnl.std(ddof=1) * ANN),
        "pnl_dates": [str(pd.Timestamp(d).date()) for d in pnl.index],
        "pnl": [float(v) for v in pnl.values],
    }

    # Test A - confirmatory replication
    mc = monthly_components(df)
    res = spread(mc, "on")
    est = float(res["id"]["spread"].mean() * 100)
    t = float(nw_tstat(res["id"]["spread"].values))
    out["test_A"] = {"estimate_pct_per_month": est, "nw_t": t, "pass": bool(est < 0 and t < TEST_A_T_THRESHOLD)}

    # Test C - deflated Sharpe
    sd_sr = sd_sr_from_prior_search()
    out["test_C"] = {
        "DSR_N108": deflated_sharpe(pnl, sd_sr, DSR_PASS_N),
        "DSR_N2": deflated_sharpe(pnl, sd_sr, 2),
        "pass": bool(deflated_sharpe(pnl, sd_sr, DSR_PASS_N) >= DSR_THRESHOLD),
    }

    print(f"\n=== frozen configuration, arm = {arm} ===")
    print(f"  days {out['days']}  tickers {out['tickers']}  names/day {out['names_per_day']:.0f}")
    print(f"  IC {out['mean_IC']:.4f} (t={out['IC_t']:.2f})   gross {gross:.3f} bps/day   SR {out['gross_SR_ann']:.2f}")
    print(f"  BREAK-EVEN {out['break_even_bps_side']:.2f} bps/side")
    print(f"\n  Test A  LPS Panel A: {est:+.2f}%/mo  t={t:.2f}   {'PASS' if out['test_A']['pass'] else 'FAIL'}")
    print(
        f"  Test C  DSR@N=108 {out['test_C']['DSR_N108']:.3f} "
        f"(N=2: {out['test_C']['DSR_N2']:.3f})   "
        f"{'PASS' if out['test_C']['pass'] else 'FAIL'}"
    )
    print(f"\n  {time.time() - t0:.0f}s")
    return out


def test_b(baseline: dict, confirm: dict) -> dict:
    """Paired difference in break-even on identical dates."""
    b = pd.Series(baseline["pnl"], index=pd.to_datetime(baseline["pnl_dates"]))
    c = pd.Series(confirm["pnl"], index=pd.to_datetime(confirm["pnl_dates"]))
    common = b.index.intersection(c.index)
    d = (c[common] - b[common]).values * 1e4 / GROSS_TRADED_PER_DAY
    lo, hi = block_bootstrap_ci(d)
    return {
        "delta_break_even_bps_side": float(d.mean()),
        "ci95": [lo, hi],
        "paired_days": int(len(common)),
        "baseline_break_even": baseline["break_even_bps_side"],
        "confirmatory_break_even": confirm["break_even_bps_side"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Runs PREREGISTRATION.md. No strategy options.")
    ap.add_argument("arm", choices=["baseline", "confirmatory"])
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = run_arm(a.arm)
    dest = BASELINE_JSON if a.arm == "baseline" else CONFIRM_JSON
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")

    if a.arm == "confirmatory" and BASELINE_JSON.exists() and out.get("tests_run"):
        b = json.loads(BASELINE_JSON.read_text())
        if b.get("tests_run"):
            tb = test_b(b, out)
            print("\n=== Test B - what the DELISTED half of survivorship bias is worth ===")
            print(f"  survivors-only break-even   {tb['baseline_break_even']:.2f} bps/side")
            print(f"  survivorship-free break-even {tb['confirmatory_break_even']:.2f} bps/side")
            print(
                f"  delta {tb['delta_break_even_bps_side']:+.3f} "
                f"95% CI [{tb['ci95'][0]:+.3f}, {tb['ci95'][1]:+.3f}]  "
                f"over {tb['paired_days']} paired days"
            )
            out["test_B"] = tb
            dest.write_text(json.dumps(out, indent=2))
