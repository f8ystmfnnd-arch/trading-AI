# Roadmap

## 2026-09-17 현재 우선순위

spot 증분 수집·최신 추론·UNKNOWN/STALE 정책·로컬 대시보드 연결은 구현됐다.
아래 Phase 목록은 장기 방향으로 보존하며 모든 항목을 미구현으로 해석하지 않는다.

1. 누수 수정 이후 기존 모델을 시간 순서로 재검증하고 calibration 및 provenance를 확인한다.
2. F04 similarity accounting, F05 futures TP/SL PnL 중복 계산을 수정한다.
3. fee·MDD·Sharpe 공통 지표와 최신 similarity query 경로를 정리한다.

뉴스·새 모델·투자선 기능보다 현재 결과를 검증하는 작업을 우선한다.

## 개발 방향

이 프로젝트는 가격 예측 AI가 아니라 `BTC Market Regime & Risk Guard AI`이다. 개발 우선순위는 수익률 극대화보다 리스크 판단 정확도와 백테스트 검증에 둔다.

## Phase 1. 데이터 수집 안정화

우선 `collect_bybit_1m.py`를 안정적인 데이터 수집기로 만든다.

- `--days` 옵션으로 지정 기간만큼 과거 데이터를 수집한다.
- `--update` 옵션으로 기존 CSV 이후 데이터만 증분 수집한다.
- 중복 `timestamp` 제거, 시간순 정렬, 저장 안정성을 보장한다.
- `data/raw/BTCUSDT_1m.csv`를 기준 원천 데이터로 관리한다.

`--watch`는 이 단계의 핵심이 아니다. 실시간 Risk Dashboard를 만들 때 다룬다.

## Phase 2. 리샘플과 데이터 품질 점검

- 1m 데이터를 5m / 15m / 1h / 4h / 1d로 리샘플한다.
- 결측 구간, 중복 timestamp, 비정상 OHLCV를 점검한다.
- `check_data_quality.py`를 파이프라인 검증 도구로 사용한다.

## Phase 3. 멀티타임프레임 피처와 리스크 타깃

- 15m / 1h / 1d 멀티호라이즌 리스크 예측을 고려한다.
- 단기 방향성은 보조 신호로 둔다.
- 고변동 가능성, 급락 위험, 시장 국면 타깃을 핵심으로 만든다.
- `Day Trading Mode`와 `Swing Trading Mode`에 필요한 타깃을 분리한다.

## Phase 4. 리스크 모델 학습

- 방향성 모델은 보조 신호로 유지한다.
- 고변동 예측 모델을 리스크 필터로 평가한다.
- 급락 위험 예측 모델을 신규 진입 금지와 포지션 축소 판단에 연결한다.
- 시장 국면 분류 모델을 스윙 매매 판단에 연결한다.

## Phase 5. 백테스트와 평가 체계

- 단순 수익률이 아니라 MDD 감소를 핵심 평가 지표로 둔다.
- 고변동 구간 회피, 급락 구간 회피, exposure, fee impact를 함께 본다.
- buy and hold 대비 성과뿐 아니라 위험 감소 효과를 검증한다.
- walk-forward 검증과 확률 calibration을 추가한다.

## Phase 6. 뉴스 기반 이벤트 리스크

뉴스는 매수/매도 신호가 아니다. 변동성, 불확실성, 이벤트 위험도 판단에 사용한다.

- BTC 관련성 점수
- 시장 전체 관련성 점수
- 심각도 점수
- sentiment
- market impact
- volatility risk
- risk action

차트 리스크와 뉴스 리스크를 결합해 최종 risk score를 계산한다.

## Phase 7. Trading Modes 분리

### Day Trading Mode

- 1m / 5m / 15m / 1h 중심
- 신규 진입 위험, 단기 고변동, 급락 위험 판단
- 포지션 크기 조절과 단기 손절/익절 보조

### Swing Trading Mode

- 15m / 1h / 4h / 1d 중심
- 며칠 단위의 시장 국면과 포지션 유지 위험 판단
- 기존 포지션 유지, 축소, 관망 판단

## Phase 8. 실시간 Risk Dashboard

마지막 단계에서 실시간 차트와 뉴스 업데이트를 통합한다.

- `--watch` 기반 실시간 수집
- 실시간 리스크 점수 업데이트
- `normal`, `caution`, `reduce_position`, `no_trade` 상태 표시
- Day Trading / Swing Trading 모드별 화면 분리
