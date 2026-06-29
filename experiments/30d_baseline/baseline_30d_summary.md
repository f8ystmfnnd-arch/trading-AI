# 30d Baseline Summary

This baseline preserves the existing 30-day experiment results before rerunning the pipeline with 365 days of 1-minute data.

## Directional XGBoost Model

| Metric | Value |
|---|---:|
| accuracy | 0.517241 |
| precision | 0.509728 |
| recall | 0.903448 |
| f1 | 0.651741 |
| roc_auc | 0.606849 |
| log_loss | 0.771222 |
| pred_proba_up mean | ~0.684 |
| pred_label=1 ratio | ~88.62% |
| actual target_up_next ratio | 0/1 each 50% |

## XGBoost Directional Backtest

| Metric | Value |
|---|---:|
| best threshold | 0.65 |
| total_return | -3.97% |
| buy_and_hold_return | -2.30% |
| max_drawdown | -5.57% |
| number_of_trades | 113 |
| win_rate | 54.39% |
| profit_factor | 1.44 |

## Risk Target Creation

| Metric | Value |
|---|---:|
| final rows | 1,927 |
| target_big_move_next_4 positive ratio | 3.74% |
| target_drop_next_4 positive ratio | 1.66% |
| target_pump_next_4 positive ratio | 2.08% |
| target_volatility_high_next_4 positive ratio | 29.99% |
| volatility threshold q=0.70 | 0.00213342 |

## High-Volatility Classifier

| Metric | Value |
|---|---:|
| total rows | 1,927 |
| feature count | 26 |
| train high vol ratio | 28.93% |
| validation high vol ratio | 29.41% |
| test high vol ratio | 35.52% |
| scale_pos_weight | 2.456410 |
| accuracy | 0.406897 |
| precision | 0.364706 |
| recall | 0.902913 |
| f1 | 0.519553 |
| roc_auc | 0.537719 |
| average_precision | 0.461486 |
| log_loss | 1.034779 |
| confusion matrix | [[25, 162], [10, 93]] |

## Baseline Question

The 365-day rerun will check whether short-window bias decreases, especially directional upside prediction crowding and high-volatility probability crowding.
