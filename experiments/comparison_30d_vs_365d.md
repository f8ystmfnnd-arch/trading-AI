# 30d vs 365d Experiment Comparison

## A. Data Scale

| Metric | 30d Baseline | 365d Run |
|---|---|---|
| raw 1m rows | ~43,200 (30d expected) | 525,075 |
| 15m feature rows | 1,931 | 34,087 |
| risk target rows | 1,927 | 34,083 |
| test rows | 290 | 5,114 directional / 5,113 volatility |

## B. Directional Model

| Metric | 30d Baseline | 365d Run |
|---|---|---|
| accuracy | 0.517241 | 0.506844 |
| precision | 0.509728 | 0.493344 |
| recall | 0.903448 | 0.749595 |
| f1 | 0.651741 | 0.595055 |
| roc_auc | 0.606849 | 0.514879 |
| log_loss | 0.771222 | 0.718183 |
| pred_label=1 ratio | 88.62% | 73.45% |
| pred_proba_up mean | 0.684000 | 0.561657 |

## C. Directional Backtest

| Metric | 30d Baseline | 365d Run |
|---|---|---|
| best threshold | 0.65 | 0.75 |
| total return | -3.97% | -11.09% |
| buy and hold return | -2.30% | -27.11% |
| max drawdown | -5.57% | -13.74% |
| number of trades | 113 | 167 |
| win rate | 54.39% | 48.81% |
| profit factor | 1.44 | 0.894401 |

## D. Risk Target Ratios

| Target | 30d Baseline | 365d Run |
|---|---|---|
| target_big_move_next_4 | 3.74% | 4.48% |
| target_drop_next_4 | 1.66% | 2.39% |
| target_pump_next_4 | 2.08% | 2.09% |
| target_volatility_high_next_4 | 29.99% | 30.00% |

## E. High-Volatility Model

| Metric | 30d Baseline | 365d Run |
|---|---|---|
| accuracy | 0.406897 | 0.598279 |
| precision | 0.364706 | 0.401684 |
| recall | 0.902913 | 0.856945 |
| f1 | 0.519553 | 0.546978 |
| roc_auc | 0.537719 | 0.778481 |
| average_precision | 0.461486 | 0.631077 |
| log_loss | 1.034779 | 0.666922 |
| pred_label=1 ratio | 87.93% | 60.38% |
| pred_proba_high_vol mean | - | 0.554660 |

## F. Interpretation

1. **Upside prediction crowding eased in the 365d run.** The directional pred_label=1 ratio fell from about 88.62% to 73.45%, and recall fell from 0.903448 to 0.749595. The model is still tilted long, but the extreme 30d long bias is reduced.
2. **High-volatility probability crowding eased materially.** The 365d pred_proba_high_vol mean is 0.554660 and the distribution is much wider than the 30d behavior, where probabilities were crowded high.
3. **Directional model quality did not improve.** Directional roc_auc dropped from 0.606849 to 0.514879, which is close to random. More data reduced the obvious probability bias, but did not create a strong short-horizon directional edge.
4. **The directional short-term strategy still has fee and trade-count pressure.** In the 365d test window, the best threshold strategy lost less than buy-and-hold during a weak market window, but total return was still negative and the strategy remains fragile.
5. **The high-volatility classifier is the more promising risk-management model.** roc_auc improved from 0.537719 to 0.778481, average_precision improved from 0.461486 to 0.631077, and threshold tuning now gives a usable precision/recall tradeoff.
6. **Recommended next work:** fix the raw 1m collection gap pattern, add walk-forward validation, calibrate probabilities, backtest the volatility model as an entry block / position-size filter, and shift more modeling effort toward regime/risk targets rather than direct direction prediction.
