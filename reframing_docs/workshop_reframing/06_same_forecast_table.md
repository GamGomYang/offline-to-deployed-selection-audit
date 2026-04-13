# Same-Forecast / Different-Decision-Quality Table Specification

## Metadata

- Project: ICML Workshop Reframing - Direct Evidence for Implementation, Not Prediction
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 워크숍 논문에서 가장 중요한 보강 분석인, 같은 forecasting information 또는 거의 같은 forecast quality 아래에서도 decision interface가 realized decision quality를 바꾼다는 점을 직접 표로 제시하기 위한 구현 명세다.

이 분석은 새로운 main result를 만드는 것이 아니다. 역할은 현재 이미 가능한 해석인 implementation / translation / accounting effect를 더 직접적이고 비방어적인 증거로 바꾸는 것이다.

## Main Question

> 같은 선형 forecast map을 유지할 때, forecast quality는 거의 같지만 realized decision quality는 달라지는가?

이 질문은 워크숍 framing에 매우 중요하다. 핵심 메시지를 아래처럼 더 직접적으로 만들어주기 때문이다.

- 기존 메시지: implementation, not prediction
- 강화된 메시지: forecast quality is nearly unchanged, but decision quality changes materially

## Why This Analysis Matters

현재도 원고만으로 아래 해석은 가능하다.

- [ ] RL selected-point result는 positive-cost에서 좋아지고 `κ=0`에서는 거의 효과가 없음
- [ ] gain은 lower realized turnover와 lower realized cost에서 옴
- [ ] target-vs-executed gap이 존재함
- [ ] therefore alpha improvement보다 translation / accounting improvement에 가깝다

하지만 reviewer 관점에서 이 해석은 여전히 간접 해석일 수 있다. 특히 RL만 보면 policy regularization effect가 작동한 것 아니냐는 질문이 남을 수 있다. 그래서 이 표의 역할은 예측 자체보다 예측을 의사결정으로 번역하는 계층이 달라졌기 때문에 결과가 달라졌다는 점을 직관적으로 보여주는 것이다.

## Fixed Analysis Target

### Comparator

- `CC-TA-LBIP` only

### Why CC-TA-LBIP

- [ ] same 918-dimensional state
- [ ] fixed ridge forecast map
- [ ] `c`만 validation에서 선택
- [ ] `κ=0` collapse 구조 보유
- [ ] RL보다 forecast-side interpretation이 더 직접적임

## Fixed Comparison Conditions

### Rows

- `c = 0`
- `c = 3000`

### Cost Regimes

- `κ = 0`
- `κ = 5e-4`
- `κ = 1e-3`

### Forecast Map Condition

- [ ] ridge forecast map은 refit하지 않는다.
- [ ] 같은 fitted forecast outputs 또는 같은 fitted forecast model을 사용한다.
- [ ] 바뀌는 것은 decision-layer cost anchor strength뿐이다.

## Table Specification

### Table Name

- `table_same_forecast_diff_decision.csv`
- `table_same_forecast_diff_decision.tex`

### Default Title

> Similar forecasting information, different realized decision quality.

이 제목을 기본으로 둔다. 실제 숫자를 뽑아보면 forecast metric이 완전히 identical일 수도 있지만 약간 다를 수도 있기 때문이다.

### Stronger Title Candidate

forecast metric 차이가 거의 0에 수렴할 때만 아래 제목으로 올린다.

> Same forecast, different realized decision quality.

### Required Columns

- forecast MSE
- rank IC 또는 sign accuracy
- average executed turnover
- realized cost
- net Sharpe

### Optional Columns

- gross Sharpe
- utility improvement
- turnover reduction %
- cost reduction %

### Recommended Order

1. forecast MSE
2. rank IC 또는 sign accuracy
3. `TOexec`
4. realized cost
5. net Sharpe

## Forecast Metric Rules

forecast metric은 decision layer 이전의 forecast outputs 기준으로 계산한다.

- per-asset predicted signal
- predicted expected return vector
- optimizer input forecast vector

### Allowed Forecast Metrics

- Option A: forecast MSE against realized next-period returns
- Option B: cross-sectional rank IC
- Option C: sign accuracy

### Recommended Priority

1. forecast MSE
2. rank IC
3. sign accuracy

## Analysis Procedure

### Step 1

- [ ] 동일한 fitted ridge forecast model에서 날짜별 forecast outputs를 추출한다.

### Step 2

- [ ] 각 날짜의 realized next-period returns와 매칭한다.

### Step 3

- [ ] forecast metric을 계산한다.
- [ ] 전체 held-out period 기준으로 계산한다.
- [ ] 필요하면 `κ`와 무관한 forecast metric으로 한 번만 계산한다.

### Step 4

같은 forecast outputs를 사용한 상태에서 아래 두 조건의 decision metrics를 계산한다.

- `c = 0`
- `c = 3000`

### Step 5

- [ ] 표를 만든다.

## Evaluation Tiers

### Tier A - Best Outcome

- [ ] forecast metric 차이는 매우 작다.
- [ ] decision metrics 차이는 뚜렷하다.

이 경우 본문에서는 아래 문장을 사용할 수 있다.

> Holding the forecast map fixed, cost-aware decision translation materially changes realized performance without materially changing forecast quality.

### Tier B - Good Outcome

- [ ] forecast metric 차이는 작지만 완전히 무시할 정도는 아니다.
- [ ] decision metric 차이는 훨씬 크다.

이 경우 아래 제목과 문장을 사용한다.

> Similar forecasting information, different realized decision quality.

> Differences in realized decision quality are much larger than differences in forecast quality, supporting an implementation-side interpretation.

### Tier C - Weak Outcome

- [ ] forecast metric 차이도 꽤 존재한다.
- [ ] decision metric 차이와 함께 움직인다.

이 경우 이 표는 본문에서 빼고 appendix로 내린다.

## Success Criteria

### Full Pass

- [ ] forecast metric 차이 << decision metric 차이
- [ ] positive-cost에서 `TOexec`, realized cost, net Sharpe 차이가 선명
- [ ] `κ=0`에서는 collapse 또는 near-collapse
- [ ] 본문에서 implementation, not prediction을 직접 증거로 쓸 수 있음

### Soft Pass

- [ ] forecast metric 차이는 다소 있지만 작음
- [ ] decision metric 차이는 여전히 훨씬 큼
- [ ] 본문에서 similar forecasting information 표현으로 안전하게 사용 가능

### Fail

- [ ] forecast metric 차이가 decision metric 차이만큼 큼
- [ ] forecast map fixed라는 구조가 구현상 불명확
- [ ] `κ=0` collapse 구조가 깨짐

## Result Paragraph Templates

### Template A - Strongest

> With the linear forecast map held fixed, forecast quality remains nearly unchanged across `c=0` and `c=3000`, while executed turnover, realized cost, and net Sharpe change materially in positive-cost regimes. This turns the implementation, not prediction interpretation into direct evidence rather than a purely indirect reading.

### Template B - Safer

> Using the same fitted linear forecast map, differences in realized decision quality across `c=0` and `c=3000` are much larger than differences in forecast quality, supporting an implementation-side interpretation of the positive-cost gains.

### Template C - Weakest Acceptable

> Even when the forecasting information is broadly similar, cost-aware decision translation can materially alter realized portfolio outcomes under frictions.

## Forbidden Wording

- [ ] forecast is exactly identical라는 확인 없는 단정
- [ ] this proves prediction never matters
- [ ] there is zero forecast difference라는 과한 문장
- [ ] therefore all gains are only execution 같은 과도한 단정

## Deliverables

- `forecast_outputs_eval.csv`
- `table_same_forecast_diff_decision.csv`
- `table_same_forecast_diff_decision.tex`
- `forecast_metric_analysis.md`
- `same_forecast_table_paragraph.md`

## Caption Drafts

### Conservative Caption

> Table X. Similar forecasting information, different realized decision quality. Using the same fitted linear forecast map, the cost-aware decision layer changes executed turnover, realized cost, and net Sharpe much more than it changes forecast-quality metrics. This supports an implementation-side interpretation of the positive-cost gains.

### Stronger Caption

> Table X. Same forecast, different realized decision quality. Holding the fitted linear forecast map fixed, the cost-aware decision layer leaves forecast quality essentially unchanged while materially altering realized portfolio outcomes under transaction costs.

## Final Review Checklist

- [ ] forecast map is truly fixed
- [ ] forecast metrics were computed before the decision layer
- [ ] decision metrics were computed after the decision layer
- [ ] forecast-side differences are small relative to decision-side differences
- [ ] the table strengthens implementation, not prediction without overclaiming
