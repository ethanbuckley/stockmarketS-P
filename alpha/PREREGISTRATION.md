# Pre-registration: the WRDS survivorship-free test

**Committed before the WRDS account was approved and before any Compustat or CRSP data was retrieved.** The git timestamp on this file is the evidence. If it is edited after data arrives, the test is void.

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

## Why this document exists

`deflated.py` established that nothing in this module survives correction for the search that produced it. The best grid configuration scored a Sharpe of 1.52 against 1.12 expected from the best of 108 noise trials (DSR 0.836); the survivorship-corrected overnight book scored 1.00, *below* the noise expectation (DSR 0.387).

At an IC of ~0.01 the search is the dominant source of apparent performance. A 109th trial on new data is worth nothing. One test, fixed in advance, is worth something. This freezes that test.

## The frozen configuration

No element below may be changed after data arrives. All are constants in `preregistered.py`; there are no tunable parameters and no command-line options that alter the strategy.

| element | frozen value | why fixed now |
| --------------- | ------------------------------------------------------------ | ---------------------------------- |
| price source | CRSP `crsp.dsf` if entitled, else Compustat `comp.secd` | CRSP has true delisting returns |
| universe | point-in-time S&P 500 members only | the bias we are correcting |
| features | the 16 OHLC-only features in `wide_panel.FEATS` | delisted names have no `master_cache.pkl` features |
| target | `y_on`, next-day overnight, cross-sectionally demeaned | the leg the whole module is about |
| winsorisation | daily 0.5 / 99.5 percentile on the target | as used throughout |
| model | `Ridge(alpha=100.0)` | the mid value of the searched grid |
| validation | expanding walk-forward, calendar-year blocks 2020–2026, 1-day purge | as used throughout |
| smoothing | none (k=1) | smoothing cannot cut an overnight book's 2.0 gross/day floor |
| weighting | equal-weight top/bottom decile | highest break-even in the search |
| cost model | 2.0 gross traded per day (in at close, out at open) | forced by the strategy, not chosen |
| headline metric | break-even one-way cost = gross bps/day ÷ 2.0 | the number that decides usability |

**The feature set is the one deviation forced on us**, because `master_cache.pkl` covers only the 503 surviving names. The baseline below is therefore re-run on the *current* data with the same 16 OHLC-only features, so the comparison is paired and exact.

## The three tests

### Test A — confirmatory replication (pass/fail)

LPS Table 1 Panel A on a survivorship-free universe: sort into deciles on lagged one-month overnight return, measure the 10−1 spread's **next-month intraday** return, Newey–West t-statistic with 12 lags.

- **Passes if** the point estimate is negative **and** t < −2.0.
- This is a hypothesis specified by a published paper, not a quantity we maximised. It is the one finding that survived deflation (−1.02%/mo, t = −4.57 on survivor-biased data).
- **Stated prediction:** passes, with |estimate| smaller than the survivor-biased −1.02%/mo.

### Test B — the estimand (no pass/fail)

How much is the *delisted* half of survivorship bias worth?

Δ = break-even(survivorship-free) − break-even(survivors-only), same period, same frozen configuration, paired on identical dates.

- Reported as a point estimate with a moving-block bootstrap 95% CI (block length 20).
- This is the number this project has never been able to see. It is an estimate, not a hypothesis test; there is no way to "fail" it.
- **Stated prediction:** Δ is negative and between −0.10 and −0.45 bps/side, i.e. break-even falls from the 1.35 baseline to somewhere in 0.90–1.25.

### Test C — the economic claim (pass/fail)

Deflated Sharpe of the frozen configuration on the survivorship-free universe.

- **Passes if** DSR ≥ 0.95 at effective **N = 108**, carrying forward every trial already spent.
- N = 2 is reported alongside as the optimistic bound, but the pass/fail is on N = 108. The Compustat sample overlaps the yfinance sample heavily — only 276 delisted names and the pre-inclusion exclusions are new — so this is **not** an independent sample and does not earn an N = 1 test.
- **Stated prediction:** fails.

## Data-quality gates

These run **before** any test. They are pre-specified, so fixing a gate failure and re-running is not p-hacking; changing a gate after seeing a test result is.

Gates 1 and 3 apply to both arms. Gates 2 and 4 are checks that the *new* data is genuinely survivorship-free, so they apply to the **confirmatory arm only** — see Amendment 1.

1. *(both arms)* Decomposition identity `(1+overnight)(1+intraday) − 1 = close-to-close` holds to < 1e-9.
2. *(confirmatory only)* Point-in-time membership coverage ≥ 90% of index members in every year 2015–2026.
3. *(both arms)* Median ≥ 300 names per day in the test window.
4. *(confirmatory only)* The universe contains at least 150 tickers absent from the current yfinance panel, i.e. the delisted names are actually present. **If this fails the whole exercise is pointless** and no test is reported.

## Amendment log

Amendments are recorded here rather than applied silently. An amendment made after a test statistic has been observed voids the test.

### Amendment 1 — 2026-09-06, before any test statistic was computed

**Change.** Gates 2 and 4 apply to the confirmatory arm only, not to both arms.

**Reason.** A specification error, found on the first run of the baseline arm. The baseline arm *is* the survivorship-biased data — that is its entire purpose as the paired comparison point — so a gate requiring ≥ 90% coverage of point-in-time membership can never pass on it. Its observed coverage is 0.711 in its worst year, which is not a defect to be fixed but the very quantity the confirmatory arm exists to correct. Gate 4 has the same structure and was already written as confirmatory-only in the code.

**Why this is not result-driven.** The gate fired *before* any test ran, so no value of Test A, B or C had been observed when the error was found. Nothing about the strategy, the metric or the thresholds changed. The frozen configuration table is untouched.

## What we will not do

- Not change any frozen value, including "just to check".
- Not add, remove or transform features.
- Not re-tune the ridge penalty, smoothing window or weighting scheme.
- Not swap the metric if break-even is unflattering.
- Not restrict the date range, universe or sector after seeing results.
- Not report a failed test as "directionally encouraging".

If the frozen configuration fails, **that is the result**. Any further analysis is labelled exploratory, reported separately, and never described as confirmatory.

## Void conditions

The test does not run, and no result is reported, if: neither CRSP nor Compustat is entitled; any data-quality gate fails and cannot be fixed without altering a frozen value; or this file is modified after data retrieval.

## Baseline arm (survivors-only), recorded before the WRDS data exists

Run with `ALPHA_SOURCE=yfinance python3 alpha/preregistered.py baseline`. Stored in `alpha/results/prereg_baseline.json`, including the daily P&L series, so Test B is a paired comparison on identical dates.

| quantity | value |
| --------------------- | ---------------------- |
| days / tickers | 1,677 / 577 |
| names per day | 441 |
| IC | 0.0214 (t = 4.01) |
| gross | 3.141 bps/day |
| gross Sharpe | 1.13 |
| **break-even** | **1.57 bps/side** |
| Test A | −0.79%/mo, t = −3.41, **PASS** |
| Test C | DSR@N=108 = 0.511, **FAIL** |

Both predictions recorded above held: Test A passes with a magnitude smaller than the survivor-biased −1.02%/mo, and Test C fails.

Note the ticker count is 577, not the 499 quoted elsewhere in the README. The small-cap download recovered former index members that were *removed from the index but are still listed*; those are not delisted and Yahoo still serves them. The genuinely missing population is narrower than 276, and Test B will measure what that narrower population is worth.

---

# Results (2026-09-11)

Source: **CRSP** (`crsp.dsf`), as the pre-registration mandates where entitled. UCL turned out to license CRSP despite it appearing nowhere on the library's pages, which is why the protocol picked the source by query-test rather than by documentation.

Using CRSP also let the decomposition follow LPS's own definition (p.196) instead of reconstructing it from adjusted prices:

```
r_intraday  = Close / Open - 1
r_overnight = (1 + r_close_to_close) / (1 + r_intraday) - 1
```

`crsp.dsf.ret` is a total return including dividends, and merged with `crsp.dsedelist` it includes the delisting return, so the identity holds by construction (max error 3.3e-16) and corporate actions land in the overnight leg exactly as the paper assumes. Delisting returns were applied to **127** observations. Identity is `permno`, not ticker, because tickers are reused after a delisting.

### Gates

| gate | value | |
| ------------------------ | ------ | ---- |
| decomposition identity | 3.3e-16 | PASS |
| names per day | 510 | PASS |
| point-in-time coverage | **1.000** | PASS |
| delisted tickers present | 736 | PASS |

Coverage is exact: 736 price series against 736 permnos ever in the index.

### The three tests

| | result | predicted | |
| ------ | ------------------------------------ | --------- | ---- |
| **A** | −0.90%/mo, t = **−4.04** | pass, magnitude below −1.02% | **PASS** |
| **B** | Δ = **−0.104** bps/side, 95% CI **[−0.615, +0.290]** | −0.10 to −0.45 | — |
| **C** | DSR@N=108 = **0.736** (N=2: 0.996) | fail | **FAIL** |

Frozen configuration on survivorship-free data: 1,257 days, 736 tickers, 510 names/day, IC 0.0204 (t = 3.49), gross 3.138 bps/day, Sharpe 1.40, **break-even 1.57 bps/side**.

### What Test B says

**The delisted half of survivorship bias is worth approximately nothing.** Break-even is 1.57 bps/side in both arms. The paired difference is −0.104 with a confidence interval spanning zero, so it is not distinguishable from no effect at all. My predicted range was −0.10 to −0.45; the point estimate sits on its boundary, but the interval is what matters and it does not exclude zero.

That is the opposite of the received wisdom, and the reason is structural. Survivorship bias is punishing for long-only strategies, where the missing names are the failures you would have held. This book is dollar-neutral and decile-weighted: the 159 extra names enter a 510-name daily cross-section, mostly on one side, and dilute rather than dominate. The correction that actually mattered was the **pre-inclusion** half — not using a company's history from before it joined the index — and the baseline already carried it.

### Limitation

UCL's CRSP licence ends **2024-12-31**, so the confirmatory arm covers 2020-01-02 to 2024-12-30 while the yfinance baseline runs to 2026-09-03. Test B is paired on the 1,257 common days, so the comparison is valid, but nothing here speaks to 2025 or 2026.

### Amendment log, continued

### Amendment 2 — 2026-09-11, before any test statistic was computed

**Change.** The CRSP price query applies no share-code or exchange filter.

**Reason.** The first confirmatory run failed the point-in-time coverage gate at 0.886. The cause was a `shrcd in (10,11)` screen in the CRSP query, which is the conventional US-common-stock filter and is wrong here: it deletes 51 foreign-incorporated members (Linde, Medtronic, Aon) and 38 REITs (Simon Property, Prologis, American Tower), all of which are genuine S&P 500 constituents. The frozen universe is "point-in-time S&P 500 members", and membership *is* the definition, so any further screen contradicts it.

**Why this is not result-driven.** The gate fired before any test ran, which is what gates are for. The filter was never part of the frozen configuration: the CRSP path did not exist when the protocol was frozen, because entitlement was unknown. Removing it took coverage from 0.886 to 1.000. No threshold was changed.
