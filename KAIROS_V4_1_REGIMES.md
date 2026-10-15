# KAIROS V4.1 — LC filters and adaptive HMM regimes

The existing `KAIROS_LC_Swing_Engine_V4_1_JXS_918.pine` now includes independent LC environment filters and a three-state Gaussian hidden Markov model (HMM). The version remains **V4.1**. These are entry filters, not an automatic strategy optimizer or a guarantee of better returns.

## What changed

- **Original LC regime filter:** calls `ml.regime_filter(ohlc4, threshold, enabled)` from the existing `jdehorty/MLExtensions/2` import. It compares the normalized slope of its adaptive curve with the threshold to screen trending/ranging conditions. It does not supply long/short direction.
- **Optional original LC volatility and ADX filters:** `ml.filter_volatility(1, 10, enabled)` and `ml.filter_adx(close, 14, threshold, enabled)`. The volatility filter is separate from the existing Volatility Push module.
- **Adaptive HMM:** estimates three Gaussian emission distributions and a learned 3×3 transition matrix using bounded rolling Baum-Welch fits. Current regime probabilities come from causal forward filtering, not a future-smoothed/Viterbi classification of the entire chart.
- **Diagnostics:** eight additional statistics rows, optional regime background, and Data Window state/probability series. Hiding diagnostics does not turn off protection.

The old settings, LC neighbor classifier, kernel trigger, order size, 09:30–15:00 New York session, 1.5×ATR stop, 0.4R target, cash cap, kernel-flip exits, and alert structure remain unchanged. A regime change does **not** independently open or close a position. A newly blocked kernel flip is not saved for a delayed entry.

## Toggles and defaults

All new entry filters are independent of **Enable LC Direction Filter**. Turning the LC classifier off does not silently turn the HMM or LC environment filters off.

| Setting | Default | Meaning |
|---|---:|---|
| Enable LC Regime Filter | On | Require the original LC trend/range condition |
| LC Regime Threshold | −0.1 | Original LC threshold convention |
| Enable LC Volatility Filter | Off | Require ATR(1) > ATR(10) |
| Enable LC ADX Filter | Off | Require LC's ADX calculation to exceed its threshold |
| LC ADX Threshold | 20 | ADX length remains 14 |
| Enable HMM Regime Filter | On | Require a qualified directional HMM state |
| Rolling Training Observations | 300 | Prior normalized-return observations used per fit; range 100–750 |
| Refit Every N Observations | 100 | Retrain periodically, not on every candle; range 25–2,000 |
| Baum-Welch Iterations | 5 | Fixed iteration budget; range 1–10, no convergence guarantee |
| Observation ATR Length | 14 | Scale current close-to-close move by the **previous** ATR |
| Minimum State Probability | 0.55 | Winning latent-state probability, **not** trade win probability |
| Consecutive Regime Bars | 2 | Same most-probable state for this many scored observations |
| Minimum Directional Mean | 0.05 ATR | Highest-mean state must exceed +0.05 for longs; lowest must be below −0.05 for shorts |
| Emission Variance Floor | 0.01 | Prevent degenerate Gaussian variances |
| Show Regime Rows in Statistics | On | Show filter status, regime, confidence, gate, fit count, means/stds |
| Show HMM Background | Off | Tint chart by current inferred regime |
| Show HMM Data Window Values | On | Export ranked state and all three filtered probabilities |

To restore the pre-regime **entry rules**, switch **all four new filter toggles off**: LC Regime, LC Volatility, LC ADX, and HMM. No code deletion is needed. Other existing settings must also match the old run. Added diagnostic visuals can be hidden separately.

## How the HMM adapts

1. At each eligible confirmed close, form `x = (close - close[1]) / ATR[1]`. Clip `x` to [−6, +6] for numerical robustness. This is not a trading-loss limit.
2. Once 300 prior valid observations exist, initialize and fit the three-state Gaussian HMM on **only those prior observations**. Subsequent refits repeat every 100 new observations on the most recent 300 samples. Initialization is deterministic; means, variances, start probabilities, and transitions are estimated, not fixed indicator thresholds pretending to be an HMM.
3. Sort states by fitted emission mean and permute both rows and columns of the transition matrix. Lowest mean is the bearish candidate, middle mean is the neutral/range proxy, and highest mean is the bullish candidate. Sign/minimum-drift checks prevent calling an all-positive fit bearish or an all-negative fit bullish.
4. Refilter the historical window under its final parameters to obtain the prior state distribution. Score the **current** observation with a single forward update, then append it to the bounded training history. Historical regime outputs and trades are not rewritten when later fits occur.
5. Admit longs only in the qualified highest-mean state; admit shorts only in the qualified lowest-mean state. The middle state, weak signed means, low probability, insufficient persistence, and HMM warmup block both directions. Persistence restarts after every refit. Its probability threshold applies to the current scored observation; it is not required on every preceding confirmation bar.

Learning and scoring run once per confirmed bar, with fill-callback/duplicate-bar guards. Model state survives same-bar rollback via primitive `varip` arrays. All valid loaded chart bars contribute, including bars outside the entry session. There is no daily reset; chart/session selection changes the sample sequence. Missing observations block current scoring and reset the persistence count.

### What “adapts over years” does and does not mean

As a multi-year chart is processed chronologically, rolling refits can change the fitted distributions and transition probabilities with the market. The HMM does **not** permanently learn across reloads, retain every past year in its 300-bar window, optimize net profit, or download history not available to the chart. Every reload reconstructs the model from loaded data. Bar count is not calendar duration: 300 one-minute observations are a short intraday window, not a multi-year training sample.

These latent states are statistical return-distribution clusters, not verified economic regimes. The middle-mean state is only a neutral/range proxy; this is not a separate high-volatility/crisis detector. Gaussian assumptions and rolling windows can misclassify markets, and confidence can be high when the model is wrong. Filtering can remove profitable trades as well as losing ones.

## Verification and limits

Run the numeric/source checks from the repository root:

```bash
python -m unittest discover -s tests -p 'test_kairos_v41_regimes.py' -v
```

The Python mirror checks scaled forward/backward expectations against brute-force latent-state paths, probability normalization, finite behavior on flat/extreme data, synthetic regime separation, chronological sample exclusion, bounded history, repeated fills, toggle bypasses, and preservation of original input defaults and exit code. These checks **do not compile or execute Pine**, simulate actual market orders, or establish profitability.

At implementation time all 13 new regime checks passed. Running the broader existing suite produced 24 passes and two pre-existing failures in `test_kairos_v4_logic.py`: its V3/V4 default-equality checks disagree with the already-existing font-size and target-input minimum differences. Those tests read only the unchanged V3/V4 files, not the updated V4.1 file. They were left untouched; the broader suite is not fully green.

TradingView compilation, realtime/historical equivalence, runtime on your actual chart, and multi-year out-of-sample performance remain to be verified. Pine has per-loop and total execution limits. If a chart times out, reduce the training window/iteration budget, increase the refit interval, or disable HMM. Large existing LC history settings also consume runtime. The supplied defaults are unoptimized starting values, not recommended or validated trading parameters.

For evaluation, compare unchanged V4.1, LC-only, HMM-only, and both filters with the same chart, costs, session, and position sizing. Choose settings using a training period, then freeze them for a later holdout period; repeat walk-forward across multiple years. Compare net expectancy, drawdown, trade count, and costs—not just win rate. HMM fitting is unsupervised and does not use future trade outcomes as labels.

## References

- [jdehorty MLExtensions](https://www.tradingview.com/script/ia5ozyMF-MLExtensions/)
- [LC source and filter calls](https://github.com/artificial-intelligence-edge/lorentzian-classification/blob/main/ports/pinescript/lorentzian-classification-v2.pine)
- [HMM definitions, forward/backward inference, and Baum-Welch](https://hmmlearn.readthedocs.io/en/stable/tutorial.html)
- [Pine execution limits](https://www.tradingview.com/pine-script-docs/writing/limitations/)
