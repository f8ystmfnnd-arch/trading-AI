# 저장소 관리

## 기준 저장소

실제 코드와 개발 기록은 `f8ystmfnnd-arch/trading-AI`에서 관리한다.
`trading_AI`는 별도의 빈 저장소이며 이 프로젝트의 clone·push 대상이 아니다.

## 코드 위치

진입점은 기존 import와 명령 호환성을 위해 루트에 유지한다.
정리 목적으로 파일을 임의 이동하거나 기존 기록을 삭제하지 않는다.

| 구분 | 파일 |
| --- | --- |
| 현재 운용 | `refresh_live_data.py`, `start_dashboard.ps1`, `dashboard/app.py` |
| 공통 모듈 | `market/`, `risk/`, `evaluation/` |
| 수집 | `collect_bybit_1m.py`, `resample_ohlcv.py`, `check_data_quality.py` |
| 피처·타깃 | `create_features_*.py`, `create_risk_targets.py`, `create_swing_targets.py` |
| 연구 | `train_*.py`, `create_similarity_dataset.py`, `analyze_similar_patterns.py`, `backtest_*.py` |
| 초기 15분봉 실험 | `collect_bybit_btc_15m.py`, `calculate_features_5y.py`, `train_xgboost.py`, `run_backtest.py` |

초기 실험의 `data/btc_15m*.csv`와 `model/xgb_model.json`은 최신 대시보드 입력과 다르다.
현재 운용은 별도 `data/operational/spot/`을 사용한다.

## 문서 위치

- README: 프로젝트 상태·실행 안내·문서 링크
- `docs/live-operation.md`: 최신성·실패 처리·운용 로그
- `docs/pipeline.md`: 실제 진입점과 연구 단계
- `docs/architecture.md`: 구현된 구조와 향후 설계 구분
- `docs/notes/`: 날짜별 개발 기록과 과거 계획 보존
- `docs/memos/`: 기존 메모 원문 보존. 유사 제목이 있어도 삭제하지 않음
- `docs/roadmap.md`: 다음 우선순위와 장기 계획

## Git 관리

새 `data/`, `model/`, `models/`, `experiments/`, `work/`, 로컬 패키지는 제외한다.
이미 추적 중인 과거 CSV·모델·실험 요약은 ignore 설정으로 사라지지 않으며 보존한다.
자격증명과 `.env`는 올리지 않는다.
Git history 재작성과 모델 재학습은 이번 정리에 포함하지 않는다.
패키지 버전 고정은 새 환경 검증 후 별도 수행한다. `requirements.txt`는 직접 의존성 목록이다.
