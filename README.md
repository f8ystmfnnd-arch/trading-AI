# BTC Market Regime & Risk Guard AI

BTC Market Regime & Risk Guard AI는 BTCUSDT 시장의 가격 방향을 맞히는 자동매매 봇이 아니라, 시장 국면과 위험 상태를 읽기 위한 리스크 관리 보조 도구입니다.

이 프로젝트의 핵심은 다음 질문에 답하는 것입니다.

- 지금 시장이 정상적인 구간인가, 고변동 구간인가?
- 다음 1시간 동안 변동성이 커질 가능성이 높은가?
- 급락 위험이 커지고 있는가?
- 현재 차트와 비슷했던 과거 구간에서는 이후에 어떤 일이 자주 일어났는가?
- 신규 진입, 포지션 축소, 관망 같은 판단을 할 때 어떤 위험 정보를 참고해야 하는가?

방향성 예측 모델은 보조 신호입니다. 핵심은 고변동 위험, 급락 위험, 유사패턴 위험, 시장 판세를 함께 보여주는 `Risk Guard`입니다.

## What This Project Is / Is Not

### This Project Is

- BTCUSDT 시장 국면과 리스크를 분석하는 도구
- 고변동 가능성 판단 보조 도구
- 급락 위험 판단 보조 도구
- 과거 유사 패턴 기반 리스크 참고 도구
- Streamlit 기반 대시보드
- 데이트레이딩/스윙 판단 전에 시장 분위기를 확인하는 보조 시스템

### This Project Is Not

- 수익을 보장하는 시스템
- 자동 매수/매도 봇
- 단독으로 진입과 청산을 결정하는 신호 시스템
- 금융 조언 또는 투자 자문

## Current Features

- Bybit `BTCUSDT` 1분봉 데이터 수집과 업데이트
- 1분봉 원본 데이터 기반 `5m`, `15m`, `1h`, `4h`, `1d` 리샘플링
- 15분봉 중심 멀티타임프레임 feature 생성
- 기술지표 feature 생성
  - Bollinger Band
  - ATR
  - MA slope
  - RSI slope
- 다음 1시간 고변동 예측 classifier
- 다음 1시간 급락 위험 classifier
- Historical Similarity Pattern Analysis
- Streamlit dashboard
- TradingView-style candlestick chart
- Bybit WebSocket live price display
- WebSocket 실패 시 REST fallback 가격 표시

## Quick Start

새 노트북이나 새 컴퓨터에서는 먼저 repo를 clone합니다.

```powershell
git clone https://github.com/f8ystmfnnd-arch/trading-AI.git
cd trading-AI
```

Python 가상환경을 만들고 패키지를 설치합니다.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

대시보드를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

## Data Update / Rebuild Pipeline

`data/`, `model/`, `experiments/`는 GitHub에 올리지 않는 로컬 산출물입니다. 새 컴퓨터에서는 기존 산출물을 복사하거나 아래 파이프라인으로 재생성해야 합니다.

모든 명령은 프로젝트 루트에서 실행합니다.

```powershell
cd C:\Users\skana\dev\trading-AI
```

기존 `data/raw/BTCUSDT_1m.csv`가 있다면 증분 업데이트를 사용합니다.

```powershell
.\.venv\Scripts\python.exe collect_bybit_1m.py --update
```

처음부터 테스트 데이터를 만들 때는 기간을 지정할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe collect_bybit_1m.py --days 365 --force-refresh
```

주의: `--force-refresh`는 기존 raw CSV를 백업한 뒤 새로 수집하는 모드입니다. 기존 데이터를 덮어쓰기 전에 백업 경로를 확인하세요.

전체 재생성 순서:

```powershell
.\.venv\Scripts\python.exe collect_bybit_1m.py --update
.\.venv\Scripts\python.exe resample_ohlcv.py
.\.venv\Scripts\python.exe check_data_quality.py
.\.venv\Scripts\python.exe create_features_multi_timeframe.py
.\.venv\Scripts\python.exe create_risk_targets.py
.\.venv\Scripts\python.exe create_swing_targets.py
.\.venv\Scripts\python.exe create_features_with_indicators.py
.\.venv\Scripts\python.exe train_volatility_classifier_with_indicators.py
.\.venv\Scripts\python.exe train_drop_risk_classifier.py
.\.venv\Scripts\python.exe create_similarity_dataset.py
.\.venv\Scripts\python.exe analyze_similar_patterns.py --pattern-length 48 --top-k 100
```

`analyze_similar_patterns.py`는 아래와 같은 raw similarity dataset을 먼저 필요로 합니다.

```text
data/processed/similarity/BTCUSDT_15m_similarity_raw_L48.csv
```

따라서 새 컴퓨터처럼 `data/processed/similarity/`가 비어 있는 환경에서는 `create_similarity_dataset.py`를 먼저 실행해야 합니다. 이미 similarity dataset이 있다면 이 단계는 생략하고 `analyze_similar_patterns.py`만 다시 실행해도 됩니다.

5년치 기준 `create_similarity_dataset.py`는 오래 걸릴 수 있고 raw vector CSV 파일도 매우 커질 수 있습니다. 처음 실행할 때는 디스크 용량과 실행 시간을 먼저 확인하세요.

## Dashboard

실행 명령:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

접속 주소:

```text
http://localhost:8501
```

대시보드에서 확인할 수 있는 정보:

- BTCUSDT candlestick chart
- timeframe selector: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- high-vol probability
- drop risk probability
- ATR ratio
- Bollinger Band width
- moving averages
- action hint
- latest similarity pattern summary

대시보드가 주로 참조하는 파일:

```text
data/raw/BTCUSDT_1m.csv
data/resampled/BTCUSDT_5m.csv
data/resampled/BTCUSDT_15m.csv
data/resampled/BTCUSDT_1h.csv
data/resampled/BTCUSDT_4h.csv
data/resampled/BTCUSDT_1d.csv
data/processed/BTCUSDT_15m_features_with_indicators.csv
data/processed/BTCUSDT_15m_volatility_predictions_with_indicators.csv
data/processed/BTCUSDT_15m_drop_risk_predictions.csv
data/processed/similarity/
```

Bybit WebSocket live price는 화면 표시용입니다. 현재 구조는 live price를 CSV에 자동 저장하는 collector가 아닙니다. 모델 판단은 사전에 생성된 15분봉 feature와 prediction CSV를 기준으로 합니다.

## Local Artifacts and Git Policy

아래 폴더는 로컬 산출물입니다.

```text
data/
model/
models/
experiments/
```

이 폴더들은 보통 GitHub에 올리지 않습니다. GitHub에는 코드와 문서 중심으로 관리합니다.

새 컴퓨터에서 실행하려면 선택지는 두 가지입니다.

1. 기존 컴퓨터의 `data/`와 `model/` 산출물을 복사한다.
2. 이 README의 pipeline 순서대로 데이터를 다시 수집하고 feature/model/prediction 파일을 재생성한다.

대용량 데이터와 모델 파일을 정리할 때는 삭제보다 백업을 우선합니다.

## Documentation Map

- `docs/project-overview.md`: 프로젝트 방향과 문제의식
- `docs/architecture.md`: 전체 구조와 의사결정 계층
- `docs/pipeline.md`: 데이터 수집, feature, model, backtest 흐름
- `docs/evaluation.md`: 평가 철학과 리스크 중심 지표
- `docs/roadmap.md`: 개발 단계와 다음 기능
- `docs/notes/`: 날짜별 개발 메모
- `AGENTS.md`: Codex 작업 규칙

## Roadmap

- real-time WebSocket collector
- real-time 1m/15m candle builder
- dashboard indicator on/off controls
- high-vol probability sub-chart
- similarity risk cards
- drop-risk and high-vol combined risk score
- Day Trading Mode / Swing Trading Mode 화면 분리
- mental/behavior guard features

## Important Notes

- 이 프로젝트는 자동매매 수익 보장 시스템이 아닙니다.
- 대시보드의 `action_hint`는 매수/매도 신호가 아니라 리스크 참고 상태입니다.
- time-series 데이터는 시간 순서를 유지해야 합니다.
- feature 생성, target 생성, backtest에서는 lookahead bias가 생기지 않도록 주의해야 합니다.
- 방향성 모델은 보조 신호이며, 핵심 판단은 고변동 위험, 급락 위험, 유사패턴 위험, 시장 국면을 종합해서 봅니다.
