# BTC Market Regime & Risk Guard AI

Bybit **spot BTCUSDT**의 시장 위험을 확인하는 Streamlit 대시보드입니다.
고변동·급락 가능성과 과거 유사 구간을 참고해 사람이 위험을 판단하도록 돕습니다.

## 현재 상태

| 영역 | 구현 및 제한 |
| --- | --- |
| 데이터 | 1분봉 증분 수집, 완결 봉만 5m·15m·1h·4h·1d 리샘플링 |
| 실시간 추론 | 기존 모델로 최신 완결 15분봉의 고변동·급락 확률 계산 |
| 위험 정책 | `NORMAL`, `CAUTION`, `NO_TRADE`, `UNKNOWN`, `STALE` 공통 정책 |
| 대시보드 | 일반 차트, 15m Risk Overlay, 1m Live Micro View |
| 유사도 | 기존 분석·백테스트 제공. 최신 운용 갱신에 자동 연결되지 않음 |
| 모델 검증 | 기존 모델은 `legacy_pre_fix`. 누수 수정 후 성능 재검증 필요 |

2026-09-16 로컬 운용 연결 시 약 273만 개의 1분봉을 확인했습니다.
CSV와 모델이 모두 GitHub에 포함되는 것은 아니므로 clone만으로 예측이 생성되지는 않습니다.

## 시작하기

프로젝트 루트에서 실행합니다.

```powershell
git clone https://github.com/f8ystmfnnd-arch/trading-AI.git
cd trading-AI
py -3 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 현재 PC의 로컬 운용

```powershell
./start_dashboard.ps1
```

접속: [http://localhost:8504](http://localhost:8504)

실행 스크립트의 기본 Python·원본·모델 경로는 현재 PC의 기존 환경입니다.
다른 PC에서는 아래처럼 경로를 명시하고 두 프로세스를 각각 실행하세요.

### 다른 PC에서 실행

기존 spot 1분봉 원본과 아래 두 모델을 별도로 준비합니다.

- `xgb_volatility_classifier_with_indicators.json`
- `xgb_drop_risk_classifier.json`

첫 번째 터미널:

```powershell
./.venv/Scripts/python.exe refresh_live_data.py --seed-raw "D:/artifacts/BTCUSDT_1m.csv" --model-dir "D:/artifacts/model" --watch
```

두 번째 터미널:

```powershell
$env:TRADING_AI_DATA_DIR = Join-Path (Get-Location) 'data/operational/spot'
./.venv/Scripts/python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8504
```

첫 실행이 성공하면 `data/operational/spot/operational_status.json`이 생성됩니다.
최근 12일의 연속 분봉과 모델의 피처 스키마가 필요합니다.
기존 산출물 없이 시작하려면 [연구 파이프라인](docs/pipeline.md)을 참고하세요.
학습과 전체 재생성은 실시간 실행의 필수 명령으로 묶지 않았습니다.

## 데이터와 판단의 최신성

- 수집은 갱신 완료 후 60초 간격, 리스크 카드는 30초 간격으로 확인합니다.
- `feature_asof`는 완결된 15분봉 종료 시각입니다.
- 예측은 다음 15분봉 종료까지 유효하며 재실행으로 유효 시간을 연장하지 않습니다.
- 예측 부재·불일치·검증 실패는 `UNKNOWN`, 만료 또는 수집 상태 지연은 `STALE`입니다.
- 최신 추론 실행은 모델 성능 재검증 완료를 의미하지 않습니다.

## 구조와 문서

| 경로 | 역할 |
| --- | --- |
| 루트 `*.py` | 수집·피처·학습·분석·백테스트 진입점 |
| `dashboard/` | 화면과 차트 |
| `market/` | 거래소 category·symbol 공통 상수 |
| `risk/` | 위험 임계값과 공통 정책 |
| `evaluation/` | 시간 분할·scaler·target·threshold 검증 |
| `tests/` | 작은 synthetic fixture 기반 회귀 테스트 |
| `docs/` | 운용 안내·설계·계획·개발일지 |

- [로컬 운용](docs/live-operation.md)
- [데이터·연구 파이프라인](docs/pipeline.md)
- [현재 구조와 계획 구분](docs/architecture.md)
- [저장소 관리 및 파일 안내](docs/repository-guide.md)
- [평가 기준](docs/evaluation.md) · [로드맵](docs/roadmap.md)
- [개발일지](docs/notes/)

## 테스트

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests
```

2026-09-17 기준 17개 테스트 통과. 대형 데이터 생성·모델 학습 없이 실행합니다.

## 산출물 관리

새 데이터·모델·실험·로그·패키지는 Git 추적 대상에서 제외합니다.
과거부터 추적 중인 `data/btc_15m*.csv`, `model/` 및 실험 요약은 보존합니다.
이 legacy 파일들은 현재 운용 입력 전체를 대신하지 않습니다.

이 도구는 자동 주문을 실행하지 않습니다. 위험 상태는 매수·매도 지시나 수익 보장이 아닙니다.
