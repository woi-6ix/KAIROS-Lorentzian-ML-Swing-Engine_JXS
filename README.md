# *KAIROS Lorentzian Swing Engine V5*

![Pine Script](https://img.shields.io/badge/Pine%20Script-v6-blue)
![Python](https://img.shields.io/badge/Python-QuantConnect-3776AB)
![Platform](https://img.shields.io/badge/Platform-TradingView-black)
![License](https://img.shields.io/badge/License-MPL--2.0-purple)

**KAIROS V5 is a TradingView Strategy** combining Nadaraya–Watson kernel swing triggers with optional Lorentzian Classification (LC), ML vote confirmation, EMA/SMA entry filters, LC environment filters, and adaptive Hidden Markov Model (HMM) regimes. It includes slope/Volatility Push (VP) markers, fixed ATR exits, editable shares, cash loss control, session settings, filled-entry alerts, LC bar colours/prediction values, a trade-entry checklist, and a navy/gold theme.

**Inspiration:** KAIROS was inspired by **jdehorty's open-source Lorentzian Classification**. KAIROS adapts this foundation into the strategy's entry confirmations, position protection, session controls, and presentation. V4.1 adds a separate Gaussian HMM for directional regime confirmation. Original work and supporting sources are linked under References.

**Why KAIROS?** Kairos personifies the opportune moment in Greek mythology. The name reflects waiting until the kernel's direction, enabled confirmations, and trading hours align.

## Updates in LC V5

**V5 adds an entry checklist to the statistics table.** It shows which existing conditions pass for a long or short entry, with a separate view of the checks recorded for the last filled trade. The checklist reads the strategy's entry conditions; it does not add filters or change the inherited trading rules.

| Addition | Function |
| --- | --- |
| 21 checklist rows | Direction, session, kernel, slope, VP, LC/ML, averages, regimes, HMM and position/quantity checks |
| Current bar view | Compare LONG (`L`) and SHORT (`S`) on the latest confirmed decision bar |
| Last filled trade view | Review the entry-submission checks preserved for an actual filled trade |
| Status and detail | ✅ / ❌, OFF or WARMUP, with vote counts, slope and HMM values where applicable |

Use the [V5 source](KAIROS_LC_Swing_Engine_V5_JXS_918.pine) for the current version. Checklist settings and the meaning of each status are explained under [V5 entry checklist](#v5-entry-checklist).

### Retained from V4.1

| Addition | Function |
| --- | --- |
| LC bar colours / prediction values | Optional candle tint and raw ML vote labels |
| LC regime, volatility and ADX filters | Independent checks for trend/range conditions, ATR expansion and trend strength |
| Adaptive three-state Gaussian HMM | Learns return distributions and state transitions, then filters entries by the current regime |
| Regime diagnostics | Eight additional statistics rows, optional chart background and Data Window probabilities |

The [V4.1 source](KAIROS_LC_Swing_Engine_V4_1_JXS_918.pine) remains available as the pre-checklist version. V5 retains its LC environment filters and Gaussian HMM. The [regime guide](KAIROS_V4_1_REGIMES.md) covers the inherited model, settings and verification in more detail. Regime changes filter new entries; they do not independently trigger orders, alter position size, or add exit rules.

## Overview: How the Machine Learning is Applied

**Machine learning (ML)** in KAIROS has two separate parts: a **Lorentzian-distance approximate neighbor classifier**, derived from k-nearest neighbors (kNN), and an **unsupervised three-state Gaussian HMM** for regimes. The classifier combines labels from stored historical examples; the HMM learns return-distribution states without directional training labels. Here, **ANN means Approximate Nearest Neighbors**, not an artificial neural network. [1, 10] The neighbor classifier describes the current market state using stored historical examples. Each bar becomes a **feature vector**: a list of normalized indicator readings describing momentum, price deviation, and trend strength. A **label** is the directional value attached to a historical example.

### Euclidean Distance versus Lorentzian Distance

**Euclidean distance** measures straight-line separation between feature vectors. It squares each feature difference, adds those squares, and takes the square root. Large differences can dominate the resulting distance. [2]

KAIROS's **Lorentzian Classification (LC)** uses a logarithmic distance instead. For each enabled normalized feature, let `difference = current value − historical value`:

```text
Euclidean: sqrt(sum(difference²))
LC:        sum(log(1 + abs(difference)))
```

<img width="3200" height="2000" alt="image" src="https://github.com/user-attachments/assets/2149d41f-04dc-4f86-b380-21b599ddbcd2" />

The logarithm compresses large differences, changing which historical states look similar. The motivation is to limit the influence of unusual readings during noisy markets or major events such as FOMC Days. KAIROS uses indicator values rather than news or event schedules. The logarithmic metric follows the original LC implementation. [3] Research on Lorentzian classifiers provides broader classification background, but does not establish that this metric improves KAIROS trading performance. [5, 6]

### Approximate neighbors in KAIROS

KAIROS uses **approximate nearest neighbors (ANN)**: a chronological scan with a moving acceptance threshold and a persistent vote queue. Neighbor Count determines how many votes are retained. Unlike conventional **k‑nearest neighbors (kNN)**, it does not sort history to find the exact `k` smallest distances. Normalizing features helps prevent an indicator’s numerical scale from dominating comparisons. Historical feature comparisons then produce the ML score that confirms a kernel trigger. [1, 3]

<p align="center">
  <img src="https://github.com/user-attachments/assets/40ad1a41-a82b-4ba8-badf-5b5a6430f844"
       style="height:500px; width:auto;" />
</p>

The image shows how KAIROS compares the current candle with historical patterns, collects votes, and uses them to confirm a trade signal.
1. **Start with the idea of k-nearest neighbors (kNN)**: Each candle becomes a set of indicator values, such as:
- `RSI` (**Relative Strength Index**: *evaluates buying/selling pressure within fixed boundaries*)
- `CCI` (**Commodity Channel Index**: *detects deviations from statistical moving averages to capture cyclical shifts*)
- `ADX` (**Average Directional Index**: *measures overall trend strength regardless of price direction*). 
Think of these values as its coordinates in the first diagram.
2. **KAIROS uses an approximate search instead**: KAIROS doesn’t run exact kNN first and then convert the result into ANN. ANN replaces the exact neighbor-selection step.
The gold star represents the current candle. The other dots represent historical patterns. KAIROS compares them using Lorentzian distance, which compresses large differences between indicator values.
The circled dots illustrate accepted candidates; they aren’t necessarily the closest dots to the star.
3. **Scan historical candidates in time order**: The candle strip shows the chronological scan: older → newer.
KAIROS checks eligible historical candidates against a moving distance threshold. In the current implementation, a candidate can be accepted when its distance is at least the current threshold. The threshold changes as votes enter the queue.
That makes this a chronological acceptance process rather than a sorted list of the smallest distances.
4. **Keep a bounded queue of votes**
Accepted candidates contribute their stored training labels:
- +1: long-direction vote
- −1: short-direction vote
- 0: neutral vote. Neighbor Count controls how many votes the queue retains. When it exceeds that count, the oldest retained vote is removed. The queue persists between candles, with expired historical votes also removed.
The image’s example is:
+1 − 1 + 1 = +1 overall vote
These are illustrative votes. KAIROS defaults to 8 retained votes.
5. Use the score to confirm the kernel trigger
A positive score supports a long direction; a negative score supports a short direction. Its magnitude measures the vote balance; not a probability of success.
KAIROS then checks whether that direction agrees with the kernel flip, the required vote threshold, the trading session, and any other enabled filters.
When those conditions align, the trade can qualify. The ML vote alone doesn’t trigger an entry.

### Nadaraya–Watson kernel regression

The Nadaraya–Watson (NW) estimator smooths price using distance-dependent weights. Gaussian and Rational Quadratic kernels give different weights to historical observations, allowing the estimate to respond differently from a simple moving average. The resulting line still has smoothing lag; its response depends on the selected kernel settings. [4]

In KAIROS, the kernel estimates price through a weighted average: `sum(weight × price) / sum(weight)`. Rational Quadratic weights smooth price across past observations; a rising estimate is bullish and a falling estimate bearish. Optional Enhanced Smoothing compares Gaussian and Rational Quadratic estimates to determine direction. **The kernel flip triggers a trade; LC confirms it.** This price estimator and the feature-based classifier perform separate jobs. [4]

## ML features and prediction

Features are normalized through the imported `MLExtensions/2` library. Each slot has an independent Enable checkbox, feature selector and editable parameters.

| Slot | Default feature | A / B | Information represented |
| --- | --- | --- | --- |
| 1 | Relative Strength Index (RSI) | 14 / 1 | Momentum |
| 2 | WaveTrend (WT) | 10 / 11 | Smoothed momentum |
| 3 | Commodity Channel Index (CCI) | 20 / 1 | Price deviation / momentum |
| 4 | Average Directional Index (ADX) | 20 / 2 | Trend strength; B ignored |
| 5 | RSI | 9 / 1 | Faster momentum |

**Feature Slots to Consider** uses the first N slots and their Enable switches. Different RSI settings create distinct features. LC requires at least one active feature.

V5 searches prior samples within a bounded rolling history. Its chronological ANN loop accepts distances against a moving threshold, skips historical bar indices divisible by four and retains a bounded vote queue. This is an approximate persistent search, rather than a sorted exact k-nearest-neighbor search; samples can be revisited.

The original LC label convention is retained: `source[4] < source` gives **−1**, `source[4] > source` gives **+1**, equality gives **0**. These labels use a known historical four-bar comparison, not a newly constructed future-return target. The current sample is added **after** prediction, so it cannot vote on itself.

Votes sum to the ML score: positive classifies long, negative short, zero retains direction. For example, six +1 votes and two −1 votes produce a score of +4. ML Prediction confirmation blocks zero/insufficient votes. LC requires completed warmup and a full vote queue. **Vote strength** is `abs(net votes) / votes used × 100`, not a probability of profit.

## Switches and ML defaults

**EMA** is an exponential moving average, weighting recent prices more heavily; **SMA** is a simple moving average of the selected number of closes. Both filters compare price with an average; they are separate from the ML vote threshold.

| Section / setting | Default | Effect |
| --- | --- | --- |
| LC Direction Filter | ON | Master switch; require a ready matching classification |
| Neighbor Count | 8; range 1–100 | Maximum retained votes |
| Maximum Training History | 2,000 bars; range 100–5,000 | Rolling candidate/history window |
| Feature Slots / Training Label Source | 5 / close | Considered features and four-bar label input |
| ML Prediction Filter / Minimum Net Votes | ON / 1 | Long score ≥ threshold; short score ≤ negative threshold |
| EMA Entry Filter | OFF / 200 | Long close above EMA; short below |
| SMA Entry Filter | OFF / 200 | Long close above SMA; short below |

LC master OFF bypasses the neighbor classifier and its ML vote check. EMA/SMA, LC environment filters and HMM remain independently switchable. To restore V3-style entry checks, also disable EMA, SMA, LC Regime, LC Volatility, LC ADX and HMM, and match the remaining V3 settings. A vote threshold cannot exceed neighbor count when its check is active. Equal close/average passes neither direction; missing data blocks only enabled filters.

## Regime filters: identifying the market environment

**A regime describes the conditions in which a signal occurs.** V4.1 adds two separate checks: the original LC environment filters and an adaptive Gaussian HMM. Both must pass when enabled. They are independent of the **LC Direction Filter**; turning off the neighbor classifier does not turn these filters off.

### Original LC environment filters

These checks use the imported `MLExtensions/2` functions. The regime filter evaluates the normalized slope of an adaptive curve to screen trending/ranging conditions; it does not assign long/short direction. The optional volatility filter compares ATR(1) with ATR(10), while the ADX filter tests trend strength. The LC volatility check is separate from the existing Volatility Push module. [8, 9]

| Setting | V5 default | Effect |
| --- | --- | --- |
| Enable LC Regime Filter / threshold | ON / −0.1 | Require the original LC trend/range condition on `ohlc4` |
| Enable LC Volatility Filter | OFF | Require ATR(1) > ATR(10) |
| Enable LC ADX Filter / threshold | OFF / 20 | Require LC's ADX value to exceed the threshold; length 14 |

### Adaptive Hidden Markov Model (HMM)

**The HMM is an unsupervised statistical sequence model.** It estimates three hidden states, each with a Gaussian observation distribution, and a 3×3 matrix describing transitions between states. **Baum–Welch**, an Expectation-Maximization (EM) method, fits the parameters from observations. KAIROS implements this in Pine; `hmmlearn` is a background reference, not a runtime dependency. [10]

At each eligible confirmed close, the observation is `(close − close[1]) / ATR[1]`, using ATR(14) by default. Values are clipped to [−6, +6] for numerical stability. The model first fits on 300 **prior** valid observations, then refits every 100 new observations using the latest 300-sample window. The current observation is scored before it is added to training history.

States are ranked by their fitted mean. Their current probabilities come from **forward filtering**, using information available up to the current bar. Historical fitting uses forward/backward calculations within the prior window; later refits do not rewrite previously emitted regime values or trades.

| Inferred state | Entry condition at defaults |
| --- | --- |
| Highest-mean state | Longs only when its mean exceeds +0.05 ATR and probability/persistence checks pass |
| Middle-mean state | Neutral/range proxy; blocks both directions |
| Lowest-mean state | Shorts only when its mean is below −0.05 ATR and probability/persistence checks pass |

The signed-mean checks prevent treating an all-negative fit as bullish or an all-positive fit as bearish. These states are statistical return clusters, not verified economic regimes or a separate crisis detector.

| HMM setting | V5 default | Effect |
| --- | --- | --- |
| Enable HMM Regime Filter | ON | Require a qualified directional state |
| Rolling Training Observations | 300; range 100–750 | Prior observations used per fit |
| Refit Every N Observations | 100; range 25–2,000 | Re-estimate parameters periodically |
| Baum-Welch Iterations | 5; range 1–10 | Fixed fitting budget; convergence is not guaranteed |
| Observation ATR Length | 14 | Scale the close-to-close change by previous ATR |
| Minimum State Probability | 0.55 | Minimum probability of the current winning state |
| Consecutive Regime Bars | 2 | Require the same winning state for two scored observations |
| Minimum Directional Mean | 0.05 ATR | Minimum signed drift required for the directional state |
| Emission Variance Floor | 0.01 | Prevent degenerate Gaussian variances |

Warmup, missing observations, a neutral state, weak directional means, insufficient persistence or low probability block entries while HMM is enabled. Persistence restarts after each refit; the probability threshold applies to the current observation. **State probability is not trade win probability.**

The rolling model adapts as loaded bars are processed, including bars outside the entry session, with no daily reset. It does not retain training across chart reloads or permanently store years of data. A 300-bar window depends on chart timeframe; it is not a multi-year training sample. HMM fitting does not optimize trade profit or use future trade outcomes as labels.

To restore the pre-regime V4.1 entry rules, switch **all four new filters OFF**: LC Regime, LC Volatility, LC ADX and HMM. Match the other settings to the earlier run. Hiding regime rows, background or Data Window values changes display only.

## How KAIROS takes trades

All enabled checks must agree on the **kernel-flip bar**:

| Check | Long | Short |
| --- | --- | --- |
| Direction enabled | Enable Longs | Enable Shorts |
| Kernel trigger | Newly bullish state | Newly bearish state |
| Session | Bar close inside entry window | Same |
| Slope | Normalized kernel slope ≥ minimum | Slope ≤ negative minimum |
| VP | Qualifying bullish push | Qualifying bearish push |
| LC / ML | Long classification and sufficient positive votes | Short classification and sufficient negative votes |
| EMA / SMA | Close above each enabled average | Close below each enabled average |
| LC environment | Each enabled regime, volatility and ADX check passes | Same |
| HMM | Qualified highest-mean state | Qualified lowest-mean state |
| Position / size | Flat or short; feasible quantity | Flat or long; feasible quantity |

A rejected flip does not become a pending entry. ML, average or regime changes alone do not trigger trades or exits. No same-direction pyramiding is allowed; a qualified opposite entry closes the old direction and opens the requested new quantity.

### VP: Volatility Push

**VP means Volatility Push**: the strategy's confirmation of a strong directional candle during expanding market activity. **ATR (Average True Range)** measures recent price movement and provides the scale for its range/body tests.

VP awards one point each for fast/slow ATR expansion, candle range relative to slow ATR, and volume relative to its moving average. A bullish push also requires an up candle, sufficient body size and a close near its high; a bearish push requires the opposite. The matching push must fall within the backward-looking confirmation window and be newer than the opposing push. At the default score of 2, ATR and range alone can qualify: **volume expansion is not independently mandatory**.

V5 uses **chart-timeframe VP and fixed targets**. [V2](KAIROS_V2_Usage.md) separately contains adaptive partial profits and one-minute continuation checks.

## Inherited strategy parameters

| Setting | Pine V5 default |
| --- | --- |
| Trading session / timezone | 09:30–before 15:00 / America/New_York; intraday charts |
| Longs / shorts / opposite-flip exit / session close | All ON |
| Kernel source / lookback / relative weighting | Close / 8 / 8.0 |
| Regression Level | 25; passed to KernelFunctions' `_startAtBar` |
| Enhanced Smoothing / lag | ON / 6; Gaussian bandwidth `max(1, lookback − lag)` |
| Slope filter / lookback / ATR / minimum | OFF / 2 bars / 14 / 0.05 ATR |
| Require VP / confirmation window | ON / 10 bars; matching push must be newer than the opposing push |
| VP fast / slow ATR | 5 / 20 |
| Minimum ATR ratio / range-to-ATR | 1.10 / 1.25 |
| Volume SMA / minimum volume ratio | 20 / 1.50 |
| Minimum VP score | 2 of 3 |
| Minimum body-to-ATR / body-to-range | 0.80 / 0.65 |
| Bull / bear close location | ≥0.70 / ≤0.30 |
| Profit / Stop / risk ATR | ON / 14 |
| Stop ATR multiple / target | 1.5 / 0.4R |
| Entry quantity | 2,500 underlying shares/contracts; editable quantity, not a $2,500 cash budget |
| Maximum Cash Loss / amount / box | OFF / $500 USD / hidden, bottom left |

These are the current V5 source defaults, inherited from V4.1. The separate V4 file and QuantConnect V3 port use different presets.

## Position protection and exits

After a fill, `1R = risk ATR × stop multiple`. The target is entry price ± `1R × target R`; the ATR stop is one R away. Cash protection converts the configured account-currency loss into a price distance using **actual filled quantity**, symbol point value and tick size. The tighter ATR/cash stop and target share one exit bracket. Tightening the cash stop leaves the ATR-based target unchanged.

Quantity overrides Properties sizing and rounds down to the symbol minimum. An infeasible one-tick cash stop blocks entry. Cash protection works with Profit / Stop disabled; hiding the loss box does not disable it. Gaps, slippage and fees can exceed the configured cap.

An opposite kernel flip reverses only when the opposite entry qualifies; otherwise the optional opposite-flip exit closes the trade. Optional session close liquidates at the first bar closing at/after cutoff. LC/ML, EMA/SMA and regime filters do not add exit rules. An open position is not closed merely because HMM changes state or an LC environment check fails.

## Metrics, appearance and alerts

The table uses executed TradingView `strategy.*` results:

| Metric | Meaning |
| --- | --- |
| Closed trades; wins/losses/breakevens | Executed outcomes |
| Win rate / profit factor | Wins ÷ closed trades; gross profit ÷ absolute gross loss |
| Net profit / net return | Net P/L; net P/L ÷ initial capital |
| Average trade / maximum drawdown | Net P/L ÷ closed trades; maximum equity drawdown |
| Position / target | Current direction and configured R |
| LC / votes / neighbors / averages | Current classification, raw vote score, model usage and filter states |
| LC Regime / LC Volatility & ADX | Pass/block status of enabled environment checks |
| HMM Regime / State Probability / Entry Gate | Current state, winning-state probability and reason entries pass or are blocked |
| HMM History / Fits / Means / Standard Deviations | Rolling observations, fit count and learned distributions in bearish/neutral/bullish mean order |
| Entry checklist | Per-condition status for the latest confirmed bar or last filled trade |

Kernel/VP markers, stop/target lines and the loss box are configurable. Table fonts and bottom watermark text, size, font family and color are editable.

| Display setting | V5 default | Function |
| --- | --- | --- |
| Show Bar Colours | OFF | Tint confirmed candles by the existing LC prediction |
| Show Bar Prediction Value | OFF | Show the raw LC vote above positive bars and below negative/neutral bars; up to the 500-label limit |
| Show Regime Rows in Statistics | ON | Add eight rows for LC environment checks and HMM diagnostics |
| Show HMM Background | OFF | Tint the chart by the inferred HMM regime |
| Show HMM Data Window Values | ON | Expose ranked state (−1/0/+1), winning-state probability and all three filtered state probabilities |

LC colours and labels require the LC classifier to be enabled and ready. They do not affect entries. HMM display controls likewise do not disable the HMM entry filter.

### V5 entry checklist

The **ENTRY CHECKLIST** section appears below the existing statistics and regime rows, using the same navy/gold theme and selected table font family. Keep **Show Statistics** enabled to see it.

| Setting under V5 Entry Checklist | Default | Function |
| --- | --- | --- |
| Show Entry Checklist in Statistics | ON | Show or hide the checklist section |
| Checklist View | Current bar | Choose Current bar or Last filled trade |
| Checklist Font Size | Tiny | Choose Tiny, Small or Normal independently of the main statistics font size |

**Current bar** displays `L` and `S` results side by side. It captures the latest eligible confirmed decision bar before orders, and retains that snapshot through fill recalculations. It does not update on every tick of an open candle. The timestamp is the decision bar's closing time in the selected trading timezone. The final row shows **✅ READY** or **❌ BLOCKED** for each direction; READY describes the entry conditions, not an order-fill confirmation.

**Last filled trade** displays one direction. Checks are captured when an entry is submitted and promoted to this view only after the strategy observes a matching actual fill. This preserves the conditions before position size and fill-callback flags change. The snapshot remains after exit until another entry fills. Hover over a checklist value to see the submission time and actual fill time/price. Before the first recorded fill, the view shows **NO FILLED TRADE**.

| Status | Meaning |
| --- | --- |
| ✅ | The enabled check passed for the displayed direction and snapshot |
| ❌ | The check failed or a required readiness condition was unavailable |
| OFF | That optional filter was bypassed; it is not a passed check |
| WARMUP | A dependent check lacks ready model or indicator data |
| — | No snapshot is available yet |

The 21 rows cover the following:

| Checklist rows | What is checked |
| --- | --- |
| Direction enabled; Trading session; Kernel flip | Allowed direction, bar close inside the entry window and a new directional kernel state |
| Kernel slope | Existing ATR-normalized kernel slope over **Slope Lookback Bars**, compared with the directional threshold |
| Volatility Push | A qualifying matching push within the confirmation window and newer than the opposing push; displays push age/window |
| LC ready / neighbors; LC direction; LC vote threshold | Model readiness and retained-vote count, matching classification and the enabled minimum signed net vote |
| EMA entry filter; SMA entry filter | Close above each enabled average for longs, below for shorts |
| LC regime; LC volatility; LC ADX | Each independently enabled environment check |
| HMM ready; HMM direction | Current usable HMM score and the directional latent state |
| HMM probability; HMM persistence; HMM signed mean | State probability, consecutive-state count and fitted directional mean against their settings |
| Position eligible; Quantity / cash stop | Position permits an entry/reversal and the requested quantity fits the enabled cash-stop constraint |
| All entry conditions | Combined entry eligibility in Current bar view; **✅ FILLED** for the saved filled-trade snapshot |

The slope row reports the existing kernel-slope filter; it does not introduce a requirement for every preceding candle to point in the same direction. Vote, slope, VP and HMM details belong to the displayed snapshot, so a last-entry value may differ from the current chart. A gold value indicates mixed long/short pass/fail results; passed values are green, failed values red, and inactive/warmup values gray.

The expanded table can be shortened by hiding the checklist, existing ML rows or regime rows in settings. Changing checklist visibility, view or font size affects display only.

Use **Order fills** alerts for V5 order messages. Including **alert() function calls** adds optional filled-entry notifications and can duplicate entry messages. Long/short notification switches are separate. Recreate alerts after code or settings changes.

## Note on "repainting"

V5 updates the LC classifier and HMM on eligible confirmed bars, excludes the current sample from its own training/prediction step, and guards against duplicate model updates during fill callbacks. HMM scores use causal forward probabilities; later fits do not retrospectively relabel earlier outputs. The kernel uses current/past price inputs.

These are source-level design choices, not a guarantee of identical historical and live behavior. KAIROS uses `process_orders_on_close=true` and `calc_on_order_fills=true`; order-fill recalculation, historical intrabar assumptions and `varip` rollback behavior need separate validation. Changing the loaded history, chart or settings rebuilds the rolling models and can change results. [7, 11]

## Use and validation

Copy [V5 Pine](KAIROS_LC_Swing_Engine_V5_JXS_918.pine) into TradingView's Pine Editor and add it to a standard intraday chart. Check Strategy Tester with realistic quantity, fees and slippage. Compare the pre-regime baseline, LC regime only, HMM only, and both regime filters using the same chart and costs. Freeze selected settings before evaluating later unseen periods.

In V5, open **V5 Entry Checklist** in Inputs to select the view and text size. Entry/exit alert messages and the default bottom watermark now identify V5. Recreate existing TradingView alerts when switching scripts.

The [V4.1 regime guide](KAIROS_V4_1_REGIMES.md) documents the inherited model and numeric/source checks. This command targets the V4.1 regime implementation, not the V5 checklist:

```bash
python -m unittest discover -s tests -p 'test_kairos_v41_regimes.py' -v
```

The 13 focused checks cover HMM calculations, probability normalization, bounded history, chronological scoring, filter bypasses and preservation of existing defaults/exits. They do not compile or execute Pine, verify live behavior or establish profitability. The guide records two pre-existing failures in the broader V4 test suite; the focused regime checks are separate.

Pine has per-loop and total execution limits. If HMM processing times out, reduce the training window/iterations, increase the refit interval or disable HMM. [12] Defaults are starting values, not optimized trading parameters. [V4 source](KAIROS_LC_Swing_Engine_V4_JXS_918.pine) and [V4 usage](KAIROS_V4_Usage.md) remain available for the earlier version.

## Separate QuantConnect V3 options port

The [Python options port](quantconnect/kairos_v3_spy_3min_backtest.py) remains **V3**, without the V4/V4.1 neighbor classifier, regime additions or V5 checklist. It buys closest-ATM same-day SPY calls/puts with a $2,500 premium budget, using 3-minute signals and one-minute SPY exit checks.

Its preset differs from Pine: smoothing ON/lag 6, slope OFF, VP window 10, opposite-flip exit OFF, stop 1.5× ATR, target 0.45R, cash cap OFF. Corrected options-mode performance remains unverified. See [backtest settings and results](quantconnect/KAIROS_V3_BACKTEST.md).

## Author and license

**Author:** JXS_918 · [@woi-6ix](https://github.com/woi-6ix). Adapted code is covered by [MPL 2.0](LICENSE); see [third-party notices](THIRD_PARTY_NOTICES.md). [V3 source](KAIROS_LC_Swing_Engine_V3_JXS_918.pine) and earlier KAIROS versions remain available.

## References

1. **scikit-learn:** [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html) — feature-space similarity, kNN and classification by neighbor votes.
2. **SciPy:** [Euclidean distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html) — mathematical definition of the comparison metric.
3. **jdehorty:** [Machine Learning: Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/) — original inspiration, logarithmic feature distance, feature engineering and ANN logic.
4. **jdehorty:** [Nadaraya–Watson Rational Quadratic Kernel](https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/) and [KernelFunctions](https://www.tradingview.com/script/e0Ek9x99-KernelFunctions/) — kernel estimator and imported functions.
5. **Kerimbekov, Bilge & Uğurlu (2016):** [The use of Lorentzian distance metric in classification problems](https://doi.org/10.1016/j.patrec.2016.09.006).
6. **Kerimbekov & Bilge (2017):** [Lorentzian Distance Classifier for Multiple Features](https://www.scitepress.org/Papers/2017/61970/) — DOI: 10.5220/0006197004930501.
7. **TradingView:** [Pine strategies and order-fill recalculation](https://www.tradingview.com/pine-script-docs/concepts/strategies/#calc_on_order_fills) — execution and backtest behavior.
8. **jdehorty:** [MLExtensions](https://www.tradingview.com/script/ia5ozyMF-MLExtensions/) — imported feature normalization, adaptive regime, volatility and ADX filter functions.
9. **Lorentzian Classification source:** [Pine implementation](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/ports/pinescript/lorentzian-classification-v2.pine) — original filter calls and LC display logic.
10. **hmmlearn:** [Hidden Markov Model tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html) — Gaussian emissions, transition probabilities, forward/backward inference and Baum–Welch parameter estimation. Background reference; KAIROS implements its HMM directly in Pine.
11. **TradingView:** [Pine execution model](https://www.tradingview.com/pine-script-docs/language/execution-model/) — confirmed bars, rollback, persistent variables and chart reloads.
12. **TradingView:** [Pine limitations](https://www.tradingview.com/pine-script-docs/writing/limitations/) — runtime and loop budgets relevant to rolling HMM fits.

The research papers and library documentation provide mathematical background. They do not validate KAIROS's trading performance. Current implementation details and defaults are documented in the [V5 source](KAIROS_LC_Swing_Engine_V5_JXS_918.pine); the [regime guide](KAIROS_V4_1_REGIMES.md) covers the inherited V4.1 model.
