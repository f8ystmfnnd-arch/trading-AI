# Pipeline

## 원칙

모든 명령은 프로젝트 루트에서 실행한다.

```powershell
cd C:\Users\MinJae\Documents\코덱스
```

데이터, 모델, 대용량 결과 파일은 GitHub에 올리지 않는 방향으로 관리한다. 중요한 해석과 결론은 `docs/` 또는 작은 Markdown 요약으로 남긴다.

## 1. 1m 데이터 수집

주요 스크립트:

- `scripts/collect/collect_bybit_1m.py`

목표 입력:

- Bybit `BTCUSDT` 1m OHLCV

목표 출력:

- `data/raw/BTCUSDT_1m.csv`
- `data/raw/backups/`

우선 구현 방향:

```powershell
python scripts/collect/collect_bybit_1m.py --days 365
python scripts/collect/collect_bybit_1m.py --update
```

`--days`는 지정한 기간만큼 과거 데이터를 수집한다. `--update`는 기존 `data/raw/BTCUSDT_1m.csv`의 마지막 `timestamp` 이후 데이터만 받아 append하고, 중복 제거와 정렬을 수행한다.

`--watch`는 나중에 실시간 Risk Dashboard 단계에서 다룬다.

## 2. 리샘플

주요 스크립트:

- `scripts/collect/resample_ohlcv.py`

입력 파일:

- `data/raw/BTCUSDT_1m.csv`

출력 파일:

- `data/resampled/BTCUSDT_5m.csv`
- `data/resampled/BTCUSDT_15m.csv`
- `data/resampled/BTCUSDT_1h.csv`
- `data/resampled/BTCUSDT_4h.csv`
- `data/resampled/BTCUSDT_1d.csv`

실행:

```powershell
python scripts/collect/resample_ohlcv.py
python scripts/collect/check_data_quality.py
```

## 3. 멀티타임프레임 피처 생성

주요 스크립트:

- `scripts/features/create_features_multi_timeframe.py`

입력 파일:

- `data/resampled/BTCUSDT_5m.csv`
- `data/resampled/BTCUSDT_15m.csv`
- `data/resampled/BTCUSDT_1h.csv`
- `data/resampled/BTCUSDT_4h.csv`
- `data/resampled/BTCUSDT_1d.csv`

출력 파일:

- `data/processed/BTCUSDT_15m_features.csv`

실행:

```powershell
python scripts/features/create_features_multi_timeframe.py
```

## 4. 리스크 타깃 생성

주요 스크립트:

- `scripts/features/create_risk_targets.py`
- `scripts/features/create_swing_targets.py`

입력 파일:

- `data/processed/BTCUSDT_15m_features.csv`
- `data/resampled/BTCUSDT_15m.csv`

출력 파일:

- `data/processed/BTCUSDT_15m_risk_targets.csv`
- 스윙 매매용 타깃 파일

핵심 타깃:

- 15m 단기 방향성
- 1h 고변동 가능성
- 1h 급락 위험
- 1d 시장 국면과 리스크 상태

## 5. 유사 패턴 데이터셋

주요 스크립트:

- `scripts/features/create_similarity_dataset.py`
- `scripts/analysis/analyze_similar_patterns.py`

입력 파일:

- `data/processed/BTCUSDT_15m_features.csv`

출력 파일:

- `data/processed/similarity/`

목적:

최근 시장 패턴과 과거 유사 구간을 비교해 이후 고변동, 급락, 큰 움직임이 얼마나 자주 발생했는지 참고한다.

## 6. 모델 학습

주요 스크립트:

- `scripts/train/train_xgboost_multi_timeframe.py`
- `scripts/train/train_volatility_classifier.py`
- `scripts/train/train_drop_risk_classifier.py`

입력 파일:

- `data/processed/BTCUSDT_15m_features.csv`
- `data/processed/BTCUSDT_15m_risk_targets.csv`

출력 파일:

- `model/xgb_multi_timeframe_classifier.json`
- `model/xgb_volatility_classifier.json`
- `model/xgb_drop_risk_classifier.json`
- `data/processed/*_predictions.csv`
- feature importance plot

## 7. 백테스트

주요 스크립트:

- `scripts/backtest/backtest_xgboost_signal.py`
- `scripts/backtest/backtest_volatility_risk_filter.py`
- `scripts/backtest/backtest_similarity_risk.py`

입력 파일:

- `data/processed/*_predictions.csv`
- `data/processed/BTCUSDT_15m_risk_targets.csv`
- `data/resampled/BTCUSDT_15m.csv`

출력 파일:

- `data/processed/*_backtest.csv`
- `data/processed/*_summary.csv`
- `model/*_equity_curve.png`

평가 관점:

- 단순 수익률보다 MDD 감소
- 고변동 구간 회피
- 급락 구간 회피
- exposure 조절
- fee impact 관리
