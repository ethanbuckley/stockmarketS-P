"""Does smoothing the signal cut turnover fast enough to beat the IC it costs?

A daily cross-sectional signal rebalances ~130% of gross per day. Averaging the
score over k days lowers turnover roughly as 1/k while IC decays only as fast as
the signal's own autocorrelation. Where those two curves cross is the answer to
'is this tradeable'.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpha.lab import evaluate

P = Path(__file__).parent
pred = pd.read_pickle(P / "oos_predictions.pkl")  # cols: Date,Ticker,model,pred,actual

rows = []
for m in pred["model"].unique():
    d = pred[pred.model == m].sort_values(["Ticker", "Date"]).copy()
    for k in (1, 2, 3, 5, 10, 20):
        if k == 1:
            s = d["pred"].values
        else:
            s = (
                d.groupby("Ticker", observed=True)["pred"]
                .transform(lambda x, k=k: x.rolling(k, min_periods=1).mean())
                .values
            )
        r, pnl, turn, ic = evaluate(s, d["actual"].values, d["Date"].values, d["Ticker"].values, f"{m}_k{k}")
        r["smooth_k"] = k
        r["base"] = m
        rows.append(r)

res = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:,.4f}")
cols = ["base", "smooth_k", "mean_IC", "rankIC", "gross_bps", "gross_SR", "turnover", "SR@1bp", "SR@2bp", "SR@5bp"]
print(res[cols].to_string(index=False))
res.to_csv(P / "smooth_results.csv", index=False)
