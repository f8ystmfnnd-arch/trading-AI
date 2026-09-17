# 데이터와 연구 파이프라인

명령은 저장소 루트에서 실행한다. 현재 진입점은 루트 `*.py`이며 `scripts/` 경로는 사용하지 않는다.

## 운용 경로

`refresh_live_data.py --watch`는 spot 1분봉 이후 데이터를 증분 수집하고,
완결 봉 검증 → 리샘플링 → 미래 target 없는 최신 피처 → 기존 모델 추론 순서로 실행한다.
학습이나 전체 백테스트를 실행하지 않는다.

```powershell
python refresh_live_data.py --seed-raw "D:/artifacts/BTCUSDT_1m.csv" --model-dir "D:/artifacts/model" --watch
```

출력은 기본 `data/operational/spot/`이다.

| 하위 경로 | 내용 |
| --- | --- |
| `raw/BTCUSDT_1m.csv` | 보존된 분봉과 증분 수집 결과 |
| `resampled/BTCUSDT_*.csv` | 모든 구성 분봉이 존재하는 완결 봉 |
| `processed/BTCUSDT_15m_features_with_indicators.csv` | 최신 최대 1,000개 피처 |
| `processed/BTCUSDT_15m_volatility_predictions_with_indicators.csv` | 고변동 확률과 최신성 metadata |
| `processed/BTCUSDT_15m_drop_risk_predictions.csv` | 급락 확률과 최신성 metadata |
| `operational_status.json` | 갱신 성공·실패, 시각, instrument |

파일은 임시 파일에서 atomic 교체한다. 기존 seed와 모델은 읽기만 한다.
최신성 및 실패 처리는 [로컬 운용 문서](live-operation.md)를 따른다.
similarity 결과는 이 경로에서 새로 생성하지 않는다.

## 연구 경로

별도로 실행하는 기존 연구 스크립트 목록이다. 전체 명령을 무조건 연속 실행하지 않는다.
입력·출력과 기존 파일 보존 여부를 확인하고 필요한 단계만 실행한다.

| 단계 | 루트 스크립트 | 주요 입력·출력 |
| --- | --- | --- |
| 수집 | `collect_bybit_1m.py --update` | `data/raw/BTCUSDT_1m.csv` 이후 분봉 |
| 초기 수집 | `collect_bybit_1m.py --days 30` | 원본이 없는 환경의 제한된 기간 |
| 리샘플·품질 | `resample_ohlcv.py`, `check_data_quality.py` | `data/raw/` → `data/resampled/` |
| 피처 | `create_features_multi_timeframe.py` | `data/processed/BTCUSDT_15m_features.csv` |
| 리스크 타깃 | `create_risk_targets.py`, `create_swing_targets.py` | training 구간 임계값으로 생성 |
| 기술지표 | `create_features_with_indicators.py` | 지표를 추가한 피처·타깃 |
| 학습 | `train_volatility_classifier_with_indicators.py`, `train_drop_risk_classifier.py`, `train_xgboost_multi_timeframe.py` | `model/` 및 prediction 산출물 |
| 유사도 | `create_similarity_dataset.py`, `analyze_similar_patterns.py` | `data/processed/similarity/` |
| 백테스트 | `backtest_similarity_risk.py`, `backtest_volatility_risk_filter.py`, `backtest_xgboost_signal.py` | 연구용 거래·자산·요약 |

similarity 생성은 새 run이며 기존 CSV에 이어붙이지 않는다.
대용량 생성과 `--force-refresh`는 일상 운용 명령이 아니다.
학습 결과를 운용에 반영하기 전 시간 순서 검증과 모델 provenance 확인이 필요하다.

## 남은 검증

- 기존 모델을 누수 방지 구조로 재검증하고 확률 calibration을 평가한다.
- F04 similarity 백테스트의 연속 시간축 accounting을 수정한다.
- F05 futures TP/SL PnL 중복 계산을 수정한다.
- fee·MDD·Sharpe 공통 지표를 정리한다.

문서 정리는 연구 결함 수정이나 모델 성능 검증 완료를 의미하지 않는다.
