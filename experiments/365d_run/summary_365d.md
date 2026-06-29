# 365d Experiment Summary

## Data

- Raw 1m rows: 525,075
- Raw 1m period: 2025-06-29 13:56:00+00:00 ~ 2026-06-29 13:54:00+00:00
- 15m resampled rows: 35,041 (2025-06-29 13:45:00+00:00 ~ 2026-06-29 13:45:00+00:00)
- 15m feature rows: 34,087
- Risk target rows: 34,083
- Data quality note: raw 1m had 524 two-minute interval gaps; resampled 5m+ intervals had 0 interval issues.

## Risk Target Ratios

| Target | Positive Ratio |
|---|---|
| target_big_move_next_4 | 4.48% |
| target_drop_next_4 | 2.39% |
| target_pump_next_4 | 2.09% |
| target_volatility_high_next_4 | 30.00% |
| target_volatility_next_4 q=0.70 threshold | 0.00206928 |

## Directional Model Test Metrics

| Metric | Value |
|---|---|
| accuracy | 0.506844 |
| precision | 0.493344 |
| recall | 0.749595 |
| f1 | 0.595055 |
| roc_auc | 0.514879 |
| log_loss | 0.718183 |
| pred_label_1_ratio | 73.45% |
| pred_proba_up_mean | 0.561657 |

## Directional Backtest

| Metric | Value |
|---|---|
| best_threshold | 0.750000 |
| total_return | -11.09% |
| buy_and_hold_return | -27.11% |
| max_drawdown | -13.74% |
| number_of_trades | 167 |
| win_rate | 48.81% |
| profit_factor | 0.894401 |

## High-Volatility Classifier Test Metrics

| Metric | Value |
|---|---|
| accuracy | 0.598279 |
| precision | 0.401684 |
| recall | 0.856945 |
| f1 | 0.546978 |
| roc_auc | 0.778481 |
| average_precision | 0.631077 |
| log_loss | 0.666922 |
| pred_label_1_ratio | 60.38% |
| pred_proba_high_vol_mean | 0.554660 |
| confusion_matrix | [[1819, 1847], [207, 1240]] |

## Volatility Threshold Analysis

| threshold | predicted_high_count | precision | recall | f1 |
|---|---|---|---|---|
| 0.30 | 4,453 | 0.315967 | 0.972357 | 0.476949 |
| 0.40 | 3,877 | 0.347949 | 0.932274 | 0.506762 |
| 0.50 | 3,087 | 0.401684 | 0.856945 | 0.546978 |
| 0.60 | 2,219 | 0.471384 | 0.722875 | 0.570649 |
| 0.70 | 1,317 | 0.580866 | 0.528680 | 0.553546 |
| 0.80 | 631 | 0.730586 | 0.318590 | 0.443696 |

## Volatility Feature Importance Top 10

| feature | importance |
|---|---|
| high_low_range | 0.269269 |
| volatility_20 | 0.140877 |
| volume_ma_20 | 0.051436 |
| h1_ma_60 | 0.032649 |
| h1_ma_20 | 0.030778 |
| ma_60 | 0.030257 |
| ma_20 | 0.029690 |
| volatility_60 | 0.029271 |
| h1_trend_direction | 0.029200 |
| h4_ma_20 | 0.026362 |
