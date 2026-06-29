# 개발 진행 방향성 메모

## 프로젝트 이름 후보

```text
BTC Market Regime & Risk Guard AI
```

한국어 설명:

```text
비트코인 시장 국면 판단 및 리스크 관리 AI
```

## 프로젝트 방향 재정의

초기에는 BTC의 단기 방향성 또는 단기 가격 움직임을 예측하는 모델을 중심으로 실험했다.

하지만 30일 데이터와 365일 데이터 실험을 비교한 결과, 단기 방향성 예측은 일반화가 약하고 특정 기간 장세에 쉽게 영향을 받는 것으로 보였다.

반면 고변동 예측 모델은 365일 데이터에서 성능이 크게 개선되었다.

따라서 프로젝트 방향을 아래처럼 재정의한다.

```text
단기 가격 자체를 맞히는 AI가 아니라,
시장 국면, 변동성 확대 가능성, 급락 위험, 뉴스 기반 위험도를 분석하여
포지션 축소, 신규 진입 금지, 전략 선택, Tilt Guard에 활용하는 리스크 관리 시스템을 만든다.
```

## 핵심 철학

```text
가격 예측보다 위험 제어가 우선이다.
```

단기 가격/방향 예측은 외부 요인, 뉴스, 거시경제 이벤트, 수급, 청산, 거래소 이슈 등에 크게 흔들린다.

따라서 모델의 목표를 다음처럼 설정한다.

```text
나쁜 구간을 피하는 것
쓸데없는 진입을 줄이는 것
고변동/급락 위험을 미리 감지하는 것
장세에 따라 전략을 켜고 끄는 것
감정적 매매와 틸트 상황을 줄이는 것
```

## 최종 시스템 구상

```text
[Data Engine]
- Bybit BTCUSDT OHLCV 수집
- 1m raw data 저장
- 5m / 15m / 1h / 4h / 1d 리샘플링
- 마지막 수집 timestamp 이후부터 이어받는 증분 수집
- 필요 시 watch 모드로 실시간 수집

[Feature Engine]
- 멀티타임프레임 차트 피처
- 방향성 피처
- 변동성 피처
- 거래량 피처
- 추세/레짐 피처
- 뉴스 기반 위험도 피처

[Model Layer]
- 단기 방향성 모델
- 고변동 예측 모델
- 급락 위험 모델
- 스윙용 리스크 모델
- 데이트레이딩용 리스크 모델
- 시장 국면 분류 모델

[Risk Engine]
- chart_risk_score
- volatility_risk_score
- drop_risk_score
- news_risk_score
- regime_score

[Decision Layer]
- 정상 거래 가능
- 주의
- 포지션 축소
- 신규 진입 금지
- 전략 비활성화
- 강제 휴식 / Tilt Guard
```

## 매매 스타일 구분

장기투자는 제외하고, 두 가지 매매 스타일을 중심으로 설계한다.

### 1. Day Trading Mode

목표:

```text
하루 안에서의 단기 진입 위험과 고변동 위험 판단
```

사용 타임프레임:

```text
1m
5m
15m
1h 보조
```

주요 예측 대상:

```text
다음 15분 방향성
다음 1시간 고변동 가능성
다음 1시간 급락 위험
당일 변동성 확대 가능성
```

활용 방식:

```text
진입 허용 / 진입 주의 / 신규 진입 금지
짧은 손절/익절 환경 판단
고변동 구간에서 포지션 축소
수수료와 거래 빈도를 고려한 필터링
```

### 2. Swing Trading Mode

목표:

```text
며칠 단위 포지션 유지/축소/진입 여부 판단
```

사용 타임프레임:

```text
15m
1h
4h
1d
```

주요 예측 대상:

```text
다음 4시간 변동성
다음 1일 변동성
다음 1일 급락 위험
시장 국면: 상승 / 하락 / 횡보 / 고변동
```

활용 방식:

```text
스윙 신규 진입 가능 여부
보유 포지션 유지 여부
포지션 축소 여부
강한 위험 구간에서 관망
```

## 실시간 뉴스 데이터 연동 계획

차트 데이터만으로는 외부 이벤트를 빠르게 반영하기 어렵다.

따라서 뉴스 데이터를 수집하고 AI가 심각성/시장 관련성/위험도를 분석하는 기능을 장기 목표로 둔다.

뉴스 분석 항목:

```text
timestamp
source
title
summary
url
btc_relevance_score
severity_score
sentiment
market_impact
volatility_risk_score
impact_horizon
risk_action
```

뉴스 AI의 역할:

```text
긍정 뉴스라서 매수, 부정 뉴스라서 매도처럼 단순 판단하지 않는다.
뉴스가 시장 변동성과 리스크를 높이는지 분석한다.
```

예시 판단:

```text
CPI 예상치 상회
→ 심각도 높음
→ BTC 관련성 높음
→ 단기 변동성 위험 증가
→ 신규 진입 금지 또는 포지션 축소 검토
```

## 멀티호라이즌 예측 계획

단순히 가격을 맞히는 것이 아니라, 시간 구간별 리스크를 나눠서 판단한다.

```text
15분 예측:
- 초단기 변동성
- 진입 주의 여부
- 단기 방향성 보조 신호

1시간 예측:
- 고변동 가능성
- 급락 위험
- 데이트레이딩 전략 ON/OFF

1일 예측:
- 시장 국면
- 스윙 진입 가능 여부
- 리스크 온/오프 판단
```

## 데이터 수집 자동화 계획

현재는 데이터를 매번 새로 수집하는 과정이 번거롭다.

따라서 `collect_bybit_1m.py`를 개선해 다음 기능을 추가할 계획이다.

```text
--days 365
--update
--force-refresh
--watch
```

### --days

지정한 일수만큼 과거 데이터를 수집한다.

예시:

```powershell
python collect_bybit_1m.py --days 365
```

### --update

기존 CSV의 마지막 timestamp를 읽고, 그 다음 시점부터 현재까지 이어서 수집한다.

예시:

```powershell
python collect_bybit_1m.py --update
```

동작 방식:

```text
1. 기존 data/raw/BTCUSDT_1m.csv 확인
2. 마지막 timestamp 읽기
3. 마지막 timestamp 이후부터 현재까지 새 데이터 수집
4. 기존 CSV에 append
5. timestamp 중복 제거
6. timestamp 기준 정렬
7. 저장
```

### --force-refresh

기존 파일을 무시하고 지정한 기간을 새로 수집한다.
단, 기존 파일 삭제는 조심해야 하므로 백업 또는 명시적 확인이 필요하다.

### --watch

프로그램을 계속 실행해두고 1분마다 새 캔들을 수집한다.
실시간 대시보드와 연결할 때 사용한다.

## 개발 우선순위

```text
1. collect_bybit_1m.py에 --days / --update 기능 추가
2. 고변동 모델을 리스크 필터로 붙인 백테스트
3. Day Trading / Swing Trading 구조 분리
4. 스윙용 risk target 생성
5. 실시간 차트 기반 risk dashboard
6. 뉴스 수집기 생성
7. 뉴스 AI 심각도 분석
8. 뉴스 점수와 차트 feature 결합
9. 15m / 1h / 1d 멀티호라이즌 리스크 예측
10. 웹 대시보드 또는 앱 형태로 확장
```

## 다음 작업 후보

가장 먼저 할 작업:

```text
collect_bybit_1m.py 개선
```

목표:

```text
--days 지원
--update 지원
기존 CSV 마지막 timestamp 이후만 수집
중복 제거
정렬
gap 체크
```

그 다음 작업:

```text
backtest_volatility_risk_filter.py
```

목표:

```text
고변동 확률이 높은 구간에서 cash 또는 포지션 축소를 적용했을 때
MDD가 줄어드는지 확인한다.
```

## 최종 한 줄 정의

```text
BTC Market Regime & Risk Guard AI는
실시간 차트 데이터와 뉴스 데이터를 기반으로
BTC 시장의 위험 국면, 고변동 가능성, 급락 위험을 판단하고,
데이트레이딩과 스윙 매매 각각에 맞는 리스크 관리 결정을 지원하는 시스템이다.
```
