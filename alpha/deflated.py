"""
deflated.py - does the overnight result survive the search that found it?

Every t-statistic reported in this module treats its configuration as if it
were the only one tried. It was not. We searched over targets, models,
regularisation, smoothing windows, weighting schemes and universes, then
reported the best. Under that procedure the maximum Sharpe across trials is
biased upward even when no strategy has any edge at all: with N independent
trials on pure noise, the expected best Sharpe grows roughly as
sqrt(2 log N) x sd(SR).

The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014, "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality")
corrects for exactly this, and additionally for the skew and fat tails of the
return series, which inflate a naive Sharpe t-test.

    SR0  = sd(SR_n) * [ (1-g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) ]      g = Euler-Mascheroni
    DSR  = Z[ (SR - SR0) sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) ]

sd(SR_n) is taken from the ACTUAL grid searched here rather than assumed, and N
is the size of that grid. DSR is the probability the true Sharpe exceeds zero
after deflation; below 0.95 the result does not survive its own search.

All Sharpes here are in per-period (daily) units, as the formula requires.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from alpha.decomposed import build_panel, rank_x  # noqa: E402
from alpha.pit_analysis import attach_membership  # noqa: E402
from alpha.source import membership_file  # noqa: E402

TEST = list(range(2020, 2027))
ANN = np.sqrt(252.0)
EULER = 0.5772156649015329

TARGETS = ["y_on", "y_cc"]
ALPHAS = [10.0, 100.0, 1000.0]
SMOOTHS = [1, 5, 20]
WEIGHTS = ["proportional", "decile", "quintile"]
UNIVERSES = ["all", "pit"]
EFF_N = [2, 5, 10, 20, 50, 108]


def oos_pred(X, y, years, dates, alpha):
    p = np.full(len(y), np.nan)
    for yr in TEST:
        te, tr = years == yr, years < yr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        trd = np.unique(dates[tr])
        tr = tr & (dates < trd[-1])
        p[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
    return p


def book(scores, actual, dates, tickers, smooth, scheme):
    d = pd.DataFrame({"p": scores, "a": actual, "d": dates, "t": tickers}).dropna()
    if smooth > 1:
        d = d.sort_values(["t", "d"])
        d["p"] = d.groupby("t", observed=True)["p"].transform(
            lambda x, k=smooth: x.rolling(k, min_periods=1).mean())
    s = pd.Series(d["p"].values, index=d["d"].values)
    if scheme == "proportional":
        w = s - s.groupby(level=0).transform("mean")
    else:
        q = 10 if scheme == "decile" else 5
        r = s.groupby(level=0).rank(pct=True)
        w = pd.Series(0.0, index=s.index)
        w[r >= 1 - 1 / q] = 1.0
        w[r <= 1 / q] = -1.0
    g = w.abs().groupby(level=0).transform("sum").replace(0, np.nan)
    w = w / g
    return pd.DataFrame({"d": d["d"].values,
                         "x": w.values * d["a"].values}).groupby("d")["x"].sum()


def deflated_sharpe(pnl: pd.Series, sr_trials: np.ndarray, n_trials: int) -> dict:
    """DSR for one strategy given the spread of Sharpes across the grid."""
    x = pnl.dropna().values
    T = len(x)
    sr = x.mean() / x.std(ddof=1)                       # daily units
    skew = pd.Series(x).skew()
    kurt = pd.Series(x).kurtosis() + 3.0                # scipy/pandas give excess
    sd_sr = np.std(sr_trials, ddof=1)

    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr0 = sd_sr * ((1 - EULER) * z1 + EULER * z2)

    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    return {"SR_daily": sr, "SR_ann": sr * ANN, "SR0_daily": sr0,
            "SR0_ann": sr0 * ANN, "sd_SR_trials": sd_sr, "N": n_trials,
            "T": T, "skew": skew, "kurtosis": kurt, "DSR": dsr}


def main():
    t0 = time.time()
    df, feats = build_panel()
    if membership_file().exists():
        df = attach_membership(df, pd.read_parquet(membership_file()))
    else:
        df["is_member"] = True
    X = rank_x(df, feats).values
    years = df["Date"].dt.year.values
    dates, tick = df["Date"].values, df["Ticker"].values
    mask_test = np.isin(years, TEST)

    rows, pnls = [], {}
    for uni in UNIVERSES:
        keep = np.ones(len(df), bool) if uni == "all" else df["is_member"].values
        for target in TARGETS:
            y = df[target].values
            for a in ALPHAS:
                p = np.full(len(df), np.nan)
                p[keep] = oos_pred(X[keep], y[keep], years[keep], dates[keep], a)
                sel = keep & mask_test
                for sm in SMOOTHS:
                    for wt in WEIGHTS:
                        pnl = book(p[sel], y[sel], dates[sel], tick[sel], sm, wt)
                        key = f"{uni}|{target}|a{a:g}|s{sm}|{wt}"
                        pnls[key] = pnl
                        rows.append({"config": key, "universe": uni, "target": target,
                                     "alpha": a, "smooth": sm, "weight": wt,
                                     "SR_daily": pnl.mean() / pnl.std(ddof=1),
                                     "SR_ann": pnl.mean() / pnl.std(ddof=1) * ANN,
                                     "bps_day": pnl.mean() * 1e4})
                print(f"  {uni:<4} {target} alpha={a:<6g} done "
                      f"({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows).sort_values("SR_ann", ascending=False)
    res.to_csv(REPO / "alpha" / "results" / "deflated_grid.csv", index=False)
    N = len(res)
    sr_trials = res["SR_daily"].values
    sd_sr_daily = float(np.std(sr_trials, ddof=1))

    print(f"\n=== grid actually searched: N = {N} configurations ===")
    print(f"annualised Sharpe across the grid: "
          f"min {res.SR_ann.min():.2f}  median {res.SR_ann.median():.2f}  "
          f"max {res.SR_ann.max():.2f}  sd {res.SR_ann.std(ddof=1):.2f}")
    print("\ntop 5 by Sharpe:")
    print(res.head(5)[["config", "SR_ann", "bps_day"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n=== deflated Sharpe ===")
    print(f"{'strategy':<34}{'SR ann':>9}{'SR0 ann':>10}{'skew':>8}{'kurt':>8}"
          f"{'DSR':>9}{'verdict':>12}")
    headline = [res.iloc[0]["config"],
                "pit|y_on|a100|s1|decile", "all|y_on|a100|s1|decile",
                "pit|y_cc|a100|s20|proportional"]
    for key in dict.fromkeys(headline):
        if key not in pnls:
            continue
        d = deflated_sharpe(pnls[key], sr_trials, N)
        verdict = "survives" if d["DSR"] >= 0.95 else "FAILS"
        print(f"{key:<34}{d['SR_ann']:>9.2f}{d['SR0_ann']:>10.2f}{d['skew']:>8.2f}"
              f"{d['kurtosis']:>8.1f}{d['DSR']:>9.3f}{verdict:>12}")
    # The 108 grid points are NOT independent: nested models on one dataset.
    # But the real search was wider than the grid (XGBoost, the reversal
    # baseline, the flip, ownership and liquidity terciles, the wide universe).
    # The effective trial count is unknown and the verdict depends on it, so
    # report the whole curve rather than one number.
    print("\n=== sensitivity to the effective number of independent trials ===")
    print(f"{'strategy':<34}{'SR ann':>8}" + "".join(f"{'N=' + str(n):>8}" for n in EFF_N))
    for key in dict.fromkeys(headline):
        if key not in pnls:
            continue
        x = pnls[key].dropna().values
        sr = x.mean() / x.std(ddof=1)
        skew = pd.Series(x).skew()
        kurt = pd.Series(x).kurtosis() + 3.0
        den = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
        line = f"{key:<34}{sr * ANN:>8.2f}"
        for n in EFF_N:
            z1, z2 = norm.ppf(1 - 1 / n), norm.ppf(1 - 1 / (n * np.e))
            sr0 = sd_sr_daily * ((1 - EULER) * z1 + EULER * z2)
            line += f"{norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / den):>8.3f}"
        print(line)

    print("\nSR0 is the Sharpe expected from the BEST of N trials under the null.")
    print("DSR >= 0.95 means the result survives its own search.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
