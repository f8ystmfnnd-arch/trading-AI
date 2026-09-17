# 구조

## 현재 운용 흐름

```text
Bybit spot BTCUSDT + 보존된 1분봉
  -> refresh_live_data.py: 증분 수집·완결 봉 검증
  -> 기존 feature 함수: 멀티타임프레임·기술지표
  -> 기존 XGBoost 모델: 고변동·급락 확률
  -> data/operational/spot: CSV·최신성 metadata·상태
  -> dashboard/app.py + risk/policy.py: 위험 상태·차트
```

- `market/instrument.py`: 과거 수집과 현재 표시의 spot category·symbol을 통일한다.
- `risk/policy.py`: 고변동·급락 threshold와 monotonic 보호 정책을 관리한다.
- `evaluation/`: training-only scaler·target threshold, 시간 순서 split·purge, validation 기반 threshold 선택을 제공한다.
- `tests/`: CSV 재실행, 정책, 최신성, 누수 방지, 운용 분봉 검증을 작은 fixture로 확인한다.

대시보드는 가격 표시와 예측 시각을 구분한다. 현재가는 예측의 유효 시간을 연장하지 않는다.
갱신 오류나 timestamp 불일치는 정상 상태를 만들지 않는다.
기존 모델의 `legacy_pre_fix` 경고는 모델 재검증 전까지 유지한다.

## 별도 연구 경로

루트 수집·학습·유사도·백테스트 스크립트는 별도로 실행한다.
운용 프로세스는 학습, 전체 similarity 생성, 자동 주문을 실행하지 않는다.
상세 파일 안내는 [repository-guide.md](repository-guide.md)를 참고한다.

## 향후 설계

시장 국면 분류, 뉴스 심각도·거시 이벤트 리스크, funding·open interest,
Day Trading / Swing Trading 모드, 확률 calibration은 추가 검증 및 구현 계획이다.
현재 운용에 모두 구현된 기능으로 설명하지 않는다.
뉴스는 변동성·불확실성을 평가하는 보조 정보로 사용한다.

계획은 [roadmap.md](roadmap.md), 평가 기준은 [evaluation.md](evaluation.md)에 둔다.
