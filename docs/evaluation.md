# Evaluation

## 평가 철학

이 프로젝트는 단순히 높은 수익률을 내는 모델을 찾는 것이 아니다. 핵심은 위험을 줄이고, 거래하기 나쁜 구간을 피하며, 포지션 노출을 더 합리적으로 조절하는 것이다.

따라서 평가의 중심은 다음에 둔다.

- MDD 감소
- 고변동 구간 회피
- 급락 구간 회피
- exposure 조절
- fee impact 관리
- buy and hold 대비 위험 조정 성과

## 기본 지표

백테스트 결과는 최소한 다음 지표를 포함해야 한다.

- `total return`
- buy and hold 대비 성과
- `maximum drawdown`
- volatility
- Sharpe-like ratio
- market exposure
- trade count
- fee impact

다만 `total return`만으로 모델을 평가하지 않는다. 리스크 관리 시스템에서는 수익률이 조금 낮아도 MDD가 크게 줄고 급락 구간을 잘 피하면 의미 있는 개선으로 볼 수 있다.

## MDD 감소

MDD 감소는 이 프로젝트의 핵심 평가 기준이다.

확인할 질문:

- 리스크 필터를 적용했을 때 `maximum drawdown`이 줄었는가?
- MDD 감소가 단순히 exposure를 과도하게 줄인 결과는 아닌가?
- MDD 감소 대비 수익률 손실이 합리적인가?
- 특정 기간에만 우연히 좋아진 결과는 아닌가?

## 고변동 구간 회피

고변동 모델은 직접 수익률을 올리는 모델이 아니라 위험 구간을 줄이는 모델로 평가한다.

확인할 질문:

- 고변동 예측 확률이 높은 구간에서 실제 변동성이 높았는가?
- 해당 구간에서 포지션을 줄이면 손실 변동성이 줄어드는가?
- 고변동 구간 회피가 지나치게 많은 기회를 제거하지는 않는가?
- threshold별 trade-off가 명확한가?

## 급락 구간 회피

급락 위험 모델은 신규 진입 금지, 포지션 축소, 관망 판단에 사용한다.

확인할 질문:

- 급락 발생 전 위험 점수가 높아지는가?
- 급락 구간에서 exposure가 줄어드는가?
- false positive가 너무 많아 정상 구간까지 과도하게 막지 않는가?
- 급락 회피가 MDD 감소에 실제로 기여하는가?

## Exposure

`market exposure`는 리스크 필터의 핵심 결과다.

확인할 질문:

- exposure가 줄어든 만큼 MDD도 함께 줄었는가?
- exposure가 너무 낮아져 전략이 사실상 거래하지 않는 상태가 되지는 않았는가?
- `Day Trading Mode`와 `Swing Trading Mode`에서 적절한 exposure 기준이 다른가?

## Fee Impact

데이터레이딩 성격이 강할수록 fee impact는 매우 중요하다.

확인할 질문:

- trade count 증가가 수익률을 훼손하지 않는가?
- threshold 변경이 거래 횟수와 수수료에 어떤 영향을 주는가?
- 수수료를 반영해도 리스크 감소 효과가 유지되는가?

## 방향성 예측 평가

방향성 예측은 보조 신호로만 평가한다.

- 단독 매수/매도 신호로 사용하지 않는다.
- 고변동, 급락, 시장 국면 판단과 결합했을 때 의미가 있는지 본다.
- 정확도보다 리스크 엔진에 추가했을 때의 MDD, exposure, fee impact 변화를 본다.

## 뉴스 리스크 평가

뉴스는 긍정이면 매수, 부정이면 매도하는 방식으로 평가하지 않는다.

뉴스 리스크는 다음 관점으로 평가한다.

- 변동성 확대 전조를 잘 포착하는가?
- 불확실성이 큰 구간에서 `caution` 또는 `no_trade` 판단에 도움이 되는가?
- 차트 기반 리스크 점수와 결합했을 때 과도한 진입을 줄이는가?

## 검증 방식

- time-series split을 사용한다.
- `shuffle`을 사용하지 않는다.
- train 기간보다 미래인 validation/test 기간으로 평가한다.
- walk-forward 검증을 도입한다.
- threshold와 probability calibration을 별도로 검증한다.
