# `alpha/` — cross-sectional next-day return prediction

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

The screener in the repo root measures **classifier quality** (AUC, precision@15). Its README is explicit that this is not a portfolio backtest:

> No transaction costs or portfolio backtest. The validation measures per-prediction classifier quality, not tradeable returns.

This module fills that gap. It predicts next-day returns, converts the predictions into a dollar-neutral book, and charges it transaction costs.

## The scoring rule

Summing `predicted × actual` across stocks is the P&L of a book whose position sizes are the predictions. Taken raw it cannot rank models, because it scales with the size of the predictions: in the run below the reversal baseline scores 30.98 and ridge scores 0.06, and ridge is the better model. Two corrections make it interpretable.

1. **Demean predictions each day**, so the book is dollar-neutral and measures stock-picking rather than market timing.
2. **Divide by gross exposure**, so daily P&L is a return on $1 and Sharpe means something.

Note that daily cross-sectional IC is *identical* whether or not the target is demeaned, because demeaning subtracts a per-day constant and correlation is shift-invariant. The raw sum is the one quantity that changes, which is the tell that it measures market exposure rather than skill.

## Validation

Expanding-window walk-forward, calendar-year test blocks, 2020–2026, 1,634 out-of-sample days. One day is purged before each test block because the label looks one day ahead. Every run carries a **shuffled-label null**: it returns IC ≈ 0.000, so there is no leakage.

## Close-to-close results

| model | IC | IC t | gross SR | turnover/day | SR @2bp | SR @5bp |
| --------------------- | ------- | ----- | -------- | ------------ | ------- | ------- |
| reversal baseline | 0.0111 | 2.28 | 0.49 | 1.32 | −0.36 | −1.63 |
| **ridge** | 0.0123 | 2.36 | 0.83 | 0.71 | 0.45 | −0.12 |
| XGBoost | 0.0072 | 1.58 | 0.44 | 1.19 | −0.20 | −1.16 |
| shuffled labels (null) | −0.0001 | −0.10 | 0.10 | 1.33 | −4.41 | −11.17 |

Ridge beats XGBoost on accuracy *and* turnover. At this signal-to-noise ratio the boosted trees fit noise.

Every model is profitable gross and dead after costs. Smoothing the score over 20 days cuts turnover 8× while IC only halves, which is what rescues it:

| ridge | turnover | IC | SR @5bp |
| ---------- | -------- | ------ | ------- |
| unsmoothed | 0.713 | 0.0123 | −0.12 |
| 20-day | 0.088 | 0.0063 | **+0.49** |

A block bootstrap puts that Sharpe at **0.49, 95% CI [−0.13, 1.13]**. Six and a half years is not enough to establish it. Per-year ICs run from −0.002 (2022) to +0.024 (2023). In 2020 the *shuffled null* posted a gross Sharpe of 1.52, which is the clearest available demonstration that one year of backtest Sharpe means nothing.

## Overnight / intraday decomposition

Following Lou, Polk & Skouras (2019), *A tug of war: Overnight versus intraday expected returns*, JFE 134:192–213. Close-to-close return is the compound of two components whose predictability has opposite sign, so a model trained on the sum fits the residual of two partly-cancelling signals.

`overnight.py` downloads `Open` prices and decomposes exactly (max error 4.4e-16):

```
r_intraday,t  = Close_t / Open_t      - 1
r_overnight,t = Open_t  / Close_{t-1} - 1
```

**Replication of their Table 1 on 2015–2026** (their sample ends 2013; ours is equal-weighted, S&P 500 only, survivorship-biased):

| sort → measured leg | ours | theirs (1993–2013) |
| --------------------------- | ------------------- | ------------------ |
| overnight → next overnight | +0.68%/mo (t=3.00) | +3.47% (t=16.57) |
| overnight → next intraday | −1.02%/mo (t=−4.57) | −3.24% (t=−9.34) |
| intraday → next intraday | +0.48% (t=1.53) | +2.19% (t=6.72) |
| intraday → next overnight | −0.21% (t=−0.66) | −1.81% (t=−8.44) |

Panel A survives with correct signs at roughly a quarter of the magnitude. Panel B does not survive.

**Predicting the legs separately** is the single biggest improvement in this module:

| target | IC | IC t | gross SR |
| -------------- | ---------- | -------- | -------- |
| overnight | **0.0293** | **5.32** | **1.30** |
| intraday | 0.0034 | 0.81 | 0.56 |
| close-to-close | 0.0102 | 2.03 | 0.58 |

`diagnose_overnight.py` rules out bid-ask bounce, the obvious artefact when using raw open prices: the signal survives lagging every feature an extra day (IC 0.0274), *improves* when close-return features are dropped (0.0298), and is **strongest in the highest** dollar-volume tertile (0.0360), the opposite of the bounce pattern.

## What decides tradeability

The two book types need different cost accounting, and getting it wrong flatters the strategy badly.

- **Close-to-close** holds through the day and trades only the change in weights: cost = `c × turnover`, turnover 0.09–0.76/day.
- **Overnight-only** must be flat during the day to avoid the intraday leg, so it buys 1.0 gross at the close and sells 1.0 gross at the open **every day regardless of how slowly the signal moves**: 2.0 gross traded/day. Smoothing cannot rescue it, because smoothing cuts rebalancing turnover, not the daily in-and-out.

Break-even one-way cost for the overnight book, by weighting scheme:

| weighting | gross bps/day | break-even bps/side |
| ------------------------- | ------------- | ------------------- |
| score-proportional | 2.442 | 1.22 |
| **equal-weight top/bottom decile** | **3.627** | **1.81** |
| top/bottom quintile | 2.389 | 1.19 |
| top/bottom tercile | 1.968 | 0.98 |

The "flip" (hold overnight, reverse intraday) earns +55% more gross but trades +73% more, so its break-even is *worse* at 1.09.

Adding the overnight/intraday features to the **close-to-close** book makes it worse (SR@2bp 0.43 → 0.17). They predict the components, not the sum.

## Institutional ownership (SEC Form 13F)

LPS argue the tug of war is a **clientele** effect: institutions trade at and near the close, individuals near the open. Their Tables 6–8 show institutional ownership conditions the effect (momentum overnight spread 1.15%/mo, t=5.39, between high and low institutional active weight). If that holds on our data, the overnight signal should concentrate in institution-dominated names, and we could trade only those, raising alpha per unit of gross traded.

`thirteenf.py` ingests SEC's Form 13F structured data sets (50 quarterly archives, 2014–2026), `cusip_map.py` maps our tickers to CUSIPs, and `shares_outstanding.py` supplies the denominator.

Three things had to be got right, each of which silently corrupts the panel otherwise:

- **Filing windows, not report periods.** SEC keys archives by *filing* date, so one report quarter's filers are split across several files: 40% of `(period, CUSIP)` pairs appear in more than one. Deduping keeps one file's filers and drops the rest; naively summing double-counts amendments. Fixed by carrying the filer's CIK and keeping each manager's **latest** filing per `(period, issuer)`, which handles both at once.
- **`VALUE` units changed** from thousands of dollars to whole dollars for filings from Jan 2023. Detected per file from the implied price rather than hardcoding a date.
- **Nested archive members.** Some ZIPs store `INFOTABLE.tsv` under a directory prefix rather than at the root. An exact-name lookup silently skipped an entire quarter; resolution is now by basename.

The mapping is validated against six publicly documented CUSIP6 prefixes (6/6 correct) and covers 447/503 tickers. Sanity check: AAPL reads ~5,600–6,000 filers holding 9.4bn of ~14.8bn shares, i.e. ~63% institutional, matching the commonly cited figure.

**No lookahead.** A 13F for quarter-end Q is not public for up to 45 days. We use Q + 60 days as the availability date and merge as-of on that, not on the report date. Median staleness in the attached panel is 46 days.

### Result: the mechanism does not carry over

| conditioning | low | mid | high |
| ------------------------- | ------ | ------ | ------ |
| overnight IC by ownership | 0.0308 | 0.0274 | 0.0245 |
| overnight IC by breadth | 0.0265 | 0.0221 | 0.0330 |

Ownership runs the **wrong way** (IC falls as institutional ownership rises); breadth is U-shaped. The two measures disagree, so neither shows a real conditioning effect. Adding ownership features to the model makes it slightly worse (IC 0.0273 → 0.0266), and restricting the book to high-ownership names does not raise break-even cost:

| universe | names/day | gross bps/day | break-even bps/side |
| ------------------ | --------- | ------------- | ------------------- |
| all names | 418 | 3.242 | 1.62 |
| io high tercile | 139 | 3.218 | 1.61 |
| io low tercile | 140 | 3.554 | 1.78 |

**The most likely reason is that the conditioning variable has almost no range here.** LPS work across the whole CRSP cross-section, where institutional ownership runs from near zero to near one. Our universe is the S&P 500: the *lowest* ownership tercile still averages 65% institutional. There is little variation left to sort on, so the test is weak rather than the theory wrong. Testing it properly needs a universe with genuine small-cap variation, which is also where survivorship bias would bite hardest.

**Known measurement limitation.** Summed institutional shares exceed shares outstanding for 7.5% of ticker-quarters, systematically per name (ON Semiconductor reads 1.2–1.35 in every quarter). This is 13F co-manager double counting: when managers share investment discretion over a block, each may report it. `INFOTABLE` carries `INVESTMENTDISCRETION` and `OTHERMANAGER` to resolve it; we do not retain those fields, so we cap at 1.0 and rely on rank-based conditioning, with denominator-free `breadth` as the check. Retaining those two fields on a re-ingest is the clean fix.


## Execution timing

The application the paper itself recommends, and the only free one. The book has to rebalance; each trade can execute in the closing auction of day *t* or the opening auction of *t+1*, and the price difference between them *is* the overnight return.

| execution rule | bps/day | bps/yr |
| --------------------------- | ------- | ------- |
| always at the close | 0.000 | 0.0 |
| always at the open | −0.076 | −19.2 |
| **model-timed** | **+0.048** | **+12.0** |
| perfect foresight (ceiling) | 5.275 | 1329.3 |

+12 bps/year, t = 1.92. Small because the smoothed book only turns over 9%/day; the gain scales with turnover.

## Point-in-time constituents

The root README lists survivorship bias as uncorrected. `constituents.py` corrects half of it. Wikipedia's page no longer carries a parseable change log, but its **revision history** does the job better: fetch the article as it stood on a past date and read the constituent table exactly as it was then. The 2016 snapshot contains ATVI, the 2020 snapshot ABMD; both were later acquired and are absent from today's list. 47 quarterly snapshots, 2015–2026.

**784 tickers were S&P 500 members at some point in that window. We have prices for 499 of them (64%).** Coverage of true members degrades sharply going back, because Yahoo serves no history at all for delisted tickers — 11 of 11 removed names I tested (ATVI, TWTR, XLNX, CERN, SIVB, FRC, …) return nothing:

| year | price coverage of true members |
| ---- | ------------------------------ |
| 2015 | 61.5% |
| 2018 | 71.6% |
| 2021 | 81.0% |
| 2024 | 91.6% |
| 2026 | 98.5% |

Restricting the panel to point-in-time members (81.1% of rows survive) costs about a quarter of everything:

| target | universe | IC | IC t | gross SR | break-even bps/side |
| -------------- | ---------------- | ------ | -------- | -------- | ------------------- |
| close-to-close | all names | 0.0102 | 2.03 | 0.58 | 1.48 |
| close-to-close | **PIT members** | 0.0083 | **1.73** | 0.44 | 0.87 |
| overnight | all names | 0.0293 | 5.32 | 1.30 | 1.81 |
| overnight | **PIT members** | 0.0220 | 4.08 | 0.99 | **1.35** |

The close-to-close model's t-statistic falls below 2 and it stops being significant. The overnight model survives but its break-even drops from 1.81 to 1.35 bps/side. **This is still an upper bound**: it removes pre-inclusion history (companies contributing the growth that got them into the index) but cannot resurrect the 276 members that were delisted, and those are disproportionately losers.

## Small-cap extension

The S&P 500 could not test the clientele mechanism: institutional breadth spans only 5.9x from its 5th to 95th percentile there, against 83x across all 13F issuers, with 88% of the tradeable 13F universe sitting below the S&P 500's 5th percentile. `download_wide.py` and `wide_panel.py` extend to 4,901 tickers (6.8M rows, ~2,270 names/day, 28x breadth range).

Features are rebuilt from OHLC alone, since `master_cache.pkl` covers only the 503. That is not a compromise: `diagnose_overnight.py` found the overnight model *improves* when close-return features are dropped.

**The mechanism does not hold, and runs the other way.** Overnight IC by institutional breadth:

| breadth bucket | mean filers | IC | IC t | gross SR |
| -------------- | ----------- | ------ | ----- | -------- |
| low | 111 | 0.0626 | 21.96 | 6.82 |
| mid | 256 | 0.0309 | 8.25 | 2.29 |
| high | 969 | 0.0260 | 5.52 | 1.49 |

Monotonically decreasing: the effect is 2.4x stronger in the *least* institutionally-held names, the opposite of LPS. It survives a double sort on dollar volume, so it is not purely size.

### But it is not tradeable, and that is the real finding

A gross Sharpe of 6.8 should not be believed without testing it. Two checks:

**Not one-day bounce.** Pushing every feature back an extra day retains 89% / 94% / 97% of the IC across the three buckets. The statistical signal is real.

**But it cannot pay the spread it must cross.** The Corwin–Schultz (2012) high–low estimator, which needs only daily highs and lows, against the break-even each bucket can afford:

| breadth bucket | gross bps/day | break-even bps/side | est. spread/side | verdict |
| -------------- | ------------- | ------------------- | ---------------- | ---------- |
| low | 14.911 | 7.46 | 17.0 | not viable |
| mid | 5.440 | 2.72 | 11.9 | not viable |
| high | 4.015 | 2.01 | 6.7 | not viable |

The alpha rises into small caps and the cost of reaching it rises faster. This also explains the inverted mechanism result: low breadth tracks low liquidity, and illiquidity inflates *measured* overnight predictability without making it reachable. The clientele story is not needed to explain the pattern.

Caveat on the estimator: Corwin–Schultz measures continuous-session effective spread and is known to overstate it for very liquid names. Closing and opening auctions often execute at or inside the spread, so the large-cap numbers should not be read as a death sentence for the S&P 500 book. For the small-cap buckets the gap (17.0 against 7.46) is wide enough to survive that caveat.


## Compustat via WRDS (staged, awaiting account approval)

`wrds_source.py` replaces the yfinance/Wikipedia data layer with Compustat, which retains delisted companies and carries an **open** price — the two things that block a proper fix. `comp.idxcst_his` supplies index membership with real from/thru dates, replacing the Wikipedia revision scrape.

Nothing here has touched live data yet: the account is pending approval. What *is* verified:

- **Connection parameters.** With a username set and no `~/.pgpass`, the connection resolves `wrds-pgdata.wharton.upenn.edu`, reaches port 9737, and fails at the auth handshake. Host, port, database and TLS are therefore correct; only credentials are missing.
- **The transformation.** Eight tests in `tests/test_wrds_source.py` exercise the Compustat→panel logic against synthetic `comp.secd`-shaped rows: the decomposition identity is exact, a 2-for-1 split gives a *zero* overnight return rather than −50%, a pure dividend lands entirely in the overnight leg (LPS p.196 n.9), the intraday leg is adjustment-invariant, calendar gaps produce no overnight return, and the output schema is a drop-in for `ohlc_open.parquet`.

When approval lands:

```bash
export WRDS_USERNAME=yourusername
printf 'wrds-pgdata.wharton.upenn.edu:9737:wrds:YOURUSER:YOURPASS\n' >> ~/.pgpass
chmod 600 ~/.pgpass          # postgres silently ignores the file otherwise
python3 alpha/wrds_source.py probe     # entitlement check
python3 alpha/wrds_source.py all       # pull prices + constituents
```

`probe` query-tests each table rather than reading the menu. WRDS lists every product it *sells*, subscribed or not, and entitlement only surfaces when you actually query — so it also tests `crsp.dsf` and `crsp.dsp500list` to settle whether UCL has CRSP, which its library pages do not mention.

### Switching sources

`source.py` routes every downstream module, so this is a flag rather than an edit in five places:

| `ALPHA_SOURCE` | behaviour |
| -------------- | --------------------------------------------------- |
| `auto` (default) | Compustat if its files exist, else yfinance |
| `compustat` | require Compustat; fail loudly if absent |
| `yfinance` | force the original layer, for A/B comparison |

The A/B is the point: re-running `pit_analysis.py` under both sources measures what the delisted half of survivorship bias was actually worth, which is the one number this project has never been able to see.

### A dependency note

We deliberately do **not** use the official `wrds` PyPI package. It pins `pandas<2.3`, which downgrades this repo to pandas 2.2.3 and makes `master_cache.pkl` unreadable (`StringDtype.__init__() takes from 1 to 2 positional arguments`). The package is a thin wrapper over psycopg2 plus `read_sql`, so `wrds_source.py` calls those directly and the repo stays on pandas 3.


## Deflated Sharpe: does any of this survive the search that found it?

Every t-statistic above treats its configuration as if it were the only one tried. It was not. `deflated.py` searches a 108-point grid (2 targets x 3 ridge penalties x 3 smoothing windows x 3 weighting schemes x 2 universes) and applies the Deflated Sharpe Ratio of Bailey & López de Prado (2014), which corrects a Sharpe for selection across N trials **and** for the skew and fat tails that inflate a naive Sharpe t-test. Our daily P&L has kurtosis of 11–15, so the second correction is not cosmetic.

Grid Sharpes run 0.21 to 1.52, sd 0.44. With N = 108 the Sharpe expected from the **best of pure noise** is **1.12 annualised**.

| strategy | SR ann | SR0 ann | skew | kurt | DSR | verdict |
| ------------------------------ | ------ | ------- | ----- | ---- | ----- | ------- |
| best of grid (all, overnight, 5d) | 1.52 | 1.12 | −0.64 | 11.5 | 0.836 | fails |
| **overnight, PIT, decile** | **1.00** | 1.12 | −0.46 | 13.9 | **0.387** | **fails** |
| overnight, all names, decile | 1.35 | 1.12 | −0.87 | 12.5 | 0.716 | fails |
| close-to-close, PIT, 20d smoothed | 0.24 | 1.12 | 0.50 | 15.1 | 0.013 | fails |

**The survivorship-corrected overnight strategy scores 1.00, below the 1.12 you would expect from the best of 108 noise trials.** Nothing here clears DSR ≥ 0.95.

The grid points are not independent — nested models on one dataset — so the effective trial count is lower than 108. But the real search was *wider* than the grid: XGBoost, the reversal baseline, the flip, ownership and liquidity terciles, the small-cap universe. The verdict depends on a number nobody can observe, so the whole curve is reported:

| strategy | SR ann | N=2 | N=5 | N=10 | N=20 | N=50 | N=108 |
| --------------------------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- |
| best of grid | 1.52 | 0.999 | 0.992 | 0.978 | 0.953 | 0.899 | 0.836 |
| overnight, all names, decile | 1.35 | 0.997 | 0.978 | 0.947 | 0.897 | 0.807 | 0.716 |
| **overnight, PIT, decile** | 1.00 | 0.973 | 0.884 | 0.783 | 0.666 | 0.508 | 0.387 |
| close-to-close, PIT, smoothed | 0.24 | 0.515 | 0.238 | 0.128 | 0.067 | 0.028 | 0.013 |

The best uncorrected configuration survives only if you believe fewer than about 20 independent things were tried. The survivorship-corrected one needs fewer than 5, which is not credible.

### What still stands

Selection bias makes results look **better**, so findings that are negative or pre-specified are not undermined by it:

- **The LPS replication.** Panel A is a hypothesis specified by a published paper, not a maximised quantity: sorting on past overnight returns forecasts next month's intraday return at −1.02%/mo, t = −4.57. One pre-specified test.
- **The survivorship measurement.** 64% coverage of true membership, ~25% inflation of measured performance. A measurement, not a selected result.
- **Every negative tradeability result.** Small-cap overnight alpha sitting inside the Corwin–Schultz spread (17.0 against 7.46 bps/side), the overnight book's 2.0 gross/day floor, the flip being worse than the leg alone. Selection bias would have hidden these, not created them.

What does **not** stand is the claim that any configuration here is a tradeable edge.


## The pre-registered result (2026-09-11)

Run on **CRSP**, which UCL turned out to license despite it appearing nowhere on the library's pages — the protocol picked the source by query-test rather than by documentation, which is the only reason we found it.

All four gates passed, point-in-time coverage exact at 1.000 (736 price series against 736 permnos ever in the index), decomposition exact to 3.3e-16, delisting returns applied to 127 observations.

| test | result | predicted in advance | |
| ---- | ------------------------------------- | -------------------- | ---- |
| **A** LPS Panel A replication | −0.90%/mo, t = −4.04 | pass, below −1.02% | **PASS** |
| **B** value of the delisted half | Δ −0.104 bps/side, CI [−0.615, +0.290] | −0.10 to −0.45 | — |
| **C** deflated Sharpe | 0.736 at N=108 | fail | **FAIL** |

**Test B is the finding: the delisted half of survivorship bias is worth approximately nothing here.** Break-even is 1.57 bps/side on survivor-only data and 1.57 on survivorship-free data, and the paired difference cannot be distinguished from zero.

That inverts the usual warning, for a structural reason. Survivorship bias is severe for long-only strategies, where the missing names are failures you would have held to the end. This book is dollar-neutral and decile-weighted, so 159 recovered names entering a 510-name daily cross-section dilute rather than dominate. The half that mattered was **pre-inclusion** history, which the baseline already corrected.

Test C failing is the durable conclusion of the whole module: with clean data, real delisting returns and genuine index membership, the strategy still does not survive correction for the search that found it.

**Limitation:** UCL's CRSP licence ends 2024-12-31, so the confirmatory arm covers 2020–2024 against the baseline's 2020–2026. Test B is paired on the 1,257 common days; nothing here speaks to 2025 or 2026.


## Limitations

These inherit the root README's limitations and add two.

- **Survivorship bias — now measured, not just corrected.** The pre-registered CRSP test puts the delisted half at −0.104 bps/side with a CI spanning zero. What follows describes the pre-CRSP state.
- **Survivorship bias — partly corrected (pre-CRSP).** `constituents.py` gives genuine point-in-time membership, which removes pre-inclusion history and costs ~25% of measured performance. The other half is not fixable with free data: Yahoo serves no history for delisted tickers, so 276 of 784 true members (36%) are simply absent, and they are disproportionately losers. Fully fixing this needs CRSP (via WRDS, which UCL may provide) or a paid survivorship-bias-free feed such as Sharadar SEP.
- **The small-cap extension is more survivorship-biased, not less.** It is drawn from currently-listed SEC tickers, and small-cap delisting rates are several times large-cap ones.
- **Raw open prices.** LPS use a first-half-hour VWAP specifically because the raw open can be set by very small orders. We use the raw open and test the resulting artefact directly rather than avoiding it.
- **Cost model is a constant per unit traded.** No market impact, no spread that widens with size, no borrow cost on the short leg.
- **Break-even, not profit.** 1.81 bps/side is the level at which the overnight book stops making money. Whether your execution beats it is an empirical question about your broker, not about this code.

## Files

| file | purpose |
| ----------------------- | -------------------------------------------------------- |
| `lab.py` | data loading, book construction, evaluation metrics |
| `walkforward.py` | expanding-window walk-forward over close-to-close targets |
| `smoothing.py` | turnover-vs-IC curve |
| `overnight.py` | download `Open`, decompose into the two legs |
| `replicate_lps.py` | replication of the paper's Table 1 |
| `decomposed.py` | predict each leg separately |
| `diagnose_overnight.py` | bid-ask-bounce artefact tests |
| `tradeable.py` | correct cost accounting for both book types |
| `concentration.py` | weighting schemes and the flip |
| `execution_timing.py` | open-vs-close order routing |
| `thirteenf.py` | SEC Form 13F ingest (two passes) |
| `cusip_map.py` | ticker to CUSIP6 by issuer name |
| `shares_outstanding.py` | historical share counts (ownership denominator) |
| `ownership.py` | ownership panel, attached without lookahead |
| `mechanism.py` | does ownership condition the overnight effect? |
| `constituents.py` | point-in-time S&P 500 membership from Wikipedia revisions |
| `pit_analysis.py` | survivorship-bias correction and its cost |
| `download_wide.py` | OHLC for the extended universe |
| `wide_panel.py` | lean OHLC-only feature panel |
| `mechanism_wide.py` | mechanism test with real ownership range |
| `diagnose_wide.py` | spread vs break-even on the small-cap result |
| `wrds_source.py` | Compustat prices and index membership via WRDS |
| `source.py` | which price/membership files the analysis reads |
| `deflated.py` | deflated Sharpe across the grid actually searched |
| `PREREGISTRATION.md` | the frozen WRDS test, committed before the data |
| `preregistered.py` | executes it; no strategy options |

Reproduce with:

```bash
python3 alpha/overnight.py          # ~100s, downloads 503 tickers
python3 alpha/replicate_lps.py
python3 alpha/tradeable.py

export SEC_USER_AGENT="Your Name you@domain.com"   # SEC 403s without this
python3 alpha/thirteenf.py ingest      # pass 1: name map inputs
python3 alpha/cusip_map.py
python3 alpha/thirteenf.py holdings    # pass 2: CIK-level, our universe
python3 alpha/shares_outstanding.py
python3 alpha/mechanism.py
```

**This is research tooling, not investment advice.**
