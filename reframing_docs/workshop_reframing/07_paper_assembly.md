# Paper Assembly Specification

## Metadata

- Project: ICML Workshop Reframing - Workshop Paper Assembly and Writing Order
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 앞에서 만든 모든 패키지를 실제 ICML 워크숍용 짧은 논문으로 조립하기 위한 편집 및 서술 명세다.

목적은 많은 결과를 다 넣는 것이 아니다. 현재 가장 강한 클레임에 가장 직접적으로 기여하는 결과만 남기고, 나머지는 appendix 또는 supplementary로 내리는 것이다.

## Final Framing Sentence

논문 조립의 모든 결정은 아래 문장을 기준으로 한다.

> In cost-sensitive portfolio decision systems, holding the predictive signal fixed, the forecast-to-execution interface materially changes realized decision quality.

이 문장에서 벗어나는 내용은 본문에 올리지 않는다.

## Title Options

아래 후보 중 하나만 사용한다.

1. From Forecasts to Portfolios Under Frictions: Execution-Aware Evaluation Improves Realized Decision Quality
2. Forecast-to-Execution Interfaces Matter in Cost-Sensitive Portfolio Decisions
3. Execution-Aware Evaluation for Cost-Sensitive Portfolio Decision Systems
4. When Forecasts Meet Frictions: Why Execution-Aware Portfolio Evaluation Matters
5. Translation, Not Just Prediction: Execution-Aware Portfolio Decisions Under Costs

### Recommended Priority

- Option 1
- Option 2
- Option 5

### Forbidden Title Directions

- [ ] Reinforcement Learning만 전면에 둔 제목
- [ ] SOTA 느낌 제목
- [ ] general forecasting systems처럼 범위를 과하게 넓힌 제목

## Paper Structure

워크숍용 본문은 아래 5개 섹션으로 고정한다.

1. Introduction
2. Forecast-to-Execution Interface
3. Experimental Setup
4. Results
5. Conclusion

부록과 보조자료는 별도로 둔다.

## Section Roles

### 1. Introduction

반드시 아래 네 가지를 한다.

- [ ] forecasting-driven decisions often stop at proposed targets
- [ ] under costs, realized positions differ from targets
- [ ] therefore primary evaluation should attach to realized decisions
- [ ] we study this in cost-sensitive portfolio decisions using fixed predictive signals

반드시 넣을 문장:

- target vs executed distinction
- realized-path evaluation necessity
- implementation-side question
- not a stronger target generator

금지:

- [ ] PRL 내부 상세 과다
- [ ] 관련연구 과다
- [ ] broad forecasting theory claim
- [ ] new RL method claim

### 2. Forecast-to-Execution Interface

짧고 날카롭게 쓴다.

넣을 것:

- target portfolio `w_t^tgt`
- executed portfolio `w_t^exec`
- partial execution update
- executed-path primary evaluation
- target-based quantities are diagnostics only

금지:

- [ ] 긴 theoretical section
- [ ] proposition proof full expansion
- [ ] adaptive `η` discussion

### 3. Experimental Setup

복잡성을 최대한 줄인다.

넣을 것:

- U27 canonical split
- frozen-policy RL protocol
- validation-selected `η=0.5`
- `κ` grid
- auxiliary CC-TA-LBIP comparator
- executed-path metrics

금지:

- [ ] heuristic baseline 동물원
- [ ] retraining variants 상세
- [ ] rolling windows 상세
- [ ] U36 상세
- [ ] implementation appendix를 본문에 끌어오기

### 4. Results

여기가 논문 중심이다. 순서는 고정한다.

1. RL frozen-policy main table
2. accounting gap figure or table
3. dense friction curve
4. CC-TA-LBIP auxiliary table
5. same-forecast table

same-forecast table은 quality가 좋을 때만 본문에 넣고, 아니면 appendix로 내린다.

### 5. Conclusion

한 가지 메시지만 반복한다.

- evaluation must follow realized decisions
- interface matters under costs
- gains are implementation-side, not necessarily alpha-side
- evidence is from a portfolio decision case study

금지:

- [ ] broad universality
- [ ] production-readiness claim
- [ ] benchmark dominance claim

## Final Figure and Table Package

### Main Items

- `Main Figure 1`: Forecast-to-execution pipeline schematic
- `Main Table 1`: RL frozen-policy selected-point result
- `Main Figure 2`: Dense friction sensitivity curve
- `Main Figure 3`: Target-vs-executed accounting gap figure
- `Main Table 2`: CC-TA-LBIP auxiliary comparator

### Optional Main Item

- `Optional Main Table 3`: same-forecast / different-decision-quality table

## What Must Leave the Main Text

아래는 본문에서 제거하거나 appendix로 보낸다.

- [ ] heuristic baselines detailed table
- [ ] buy-and-hold comparison 전면화
- [ ] rolling-window full details
- [ ] U36 replication full details
- [ ] `η`-aligned retraining full details
- [ ] long proof section
- [ ] detailed environment implementation
- [ ] multiple secondary diagnostic figures

## Results Paragraph Order

Results 안 문단 배치는 아래 순서를 따른다.

1. selected-point RL result
2. implementation-side explanation via turnover and cost
3. accounting-gap diagnostic
4. friction-sensitive curve
5. auxiliary non-RL comparator
6. same-forecast direct evidence, conditional

이 순서를 유지하면 독자는 자연스럽게 좋아졌다, 왜 좋아졌는지 보인다, 왜 target-path로 보면 안 되는지 이해한다, 비용이 커질수록 더 중요하다는 걸 본다, RL 밖에서도 비슷하다는 걸 본다, prediction보다 translation이 핵심이었다는 걸 확인한다는 흐름으로 읽게 된다.

## Abstract Rules

초록은 5문장 구조로 고정한다.

1. Forecasting-driven decision systems are often evaluated as if proposed targets were immediately realized.
2. Under transaction costs, this conflates target decisions with realized positions and can misstate realized decision quality.
3. We study a simple forecast-to-execution interface that separates target and executed portfolios and evaluates performance on the realized path.
4. Under a frozen learned policy, the validation-selected interior operating point improves net Sharpe in positive-cost regimes while roughly halving executed turnover, with negligible zero-cost effect.
5. The gain is implementation-side rather than purely predictive, and a linear forecast-plus-convex-optimization comparator provides same-direction supporting evidence.

## Introduction Opening Template

> Many forecasting-driven decision systems stop at proposed targets. In cost-sensitive portfolio decisions, however, realized positions need not equal proposed targets, because trading frictions intervene between what is suggested and what is actually carried into the next return interval. This distinction changes what turnover is realized, what costs are paid, and which path should be evaluated. We therefore study a forecast-to-execution interface in which target and executed portfolios are separated and primary evaluation is attached to realized executed paths rather than counterfactual target paths.

## Contribution Rules

contribution은 3개만 쓴다.

1. A forecast-to-execution interface for cost-sensitive portfolio decisions that separates target and executed portfolios and attaches primary evaluation to realized paths.
2. Frozen-policy evidence that the validation-selected interior point improves positive-cost held-out decision quality while sharply reducing executed turnover.
3. Supporting evidence from accounting diagnostics, friction sensitivity, and a linear forecast-plus-convex-optimization comparator.

## Discussion and Limitation Rules

한계는 숨기지 말고 짧게 명시한다.

반드시 적을 한계:

- [ ] portfolio-domain case study
- [ ] fixed 27-name snapshot
- [ ] frozen-policy identification, not end-to-end retraining study
- [ ] auxiliary comparator is not the core identification result
- [ ] passive benchmark dominance is not claimed

## Appendix Layout

- Appendix A: additional diagnostics
- Appendix B: external baselines / context only
- Appendix C: rolling-window robustness
- Appendix D: U36 and retraining checks
- Appendix E: implementation details

## Assembly Success Criteria

- [ ] title and abstract do not overclaim
- [ ] RL main result is clearly the center
- [ ] accounting gap and friction sensitivity support the same story
- [ ] CC-TA-LBIP remains auxiliary
- [ ] the paper reads as a focused workshop paper, not a shrunken full paper

## Assembly Failure Signals

아래 느낌이 들면 다시 줄여야 한다.

- [ ] 이 논문이 무엇을 주장하는지 한 문장으로 말하기 어렵다.
- [ ] RL, forecasting, comparator 이야기가 뒤섞여 중심이 흐린다.
- [ ] 메인 결과가 무엇인지 약하다.
- [ ] 본론보다 부록감 결과가 더 많아 보인다.

이 경우 해야 할 일은 결과 추가가 아니라 삭제다.

## Deliverables

- `paper_workshop_outline.md`
- `abstract_workshop_v1.md`
- `intro_workshop_v1.md`
- `results_workshop_v1.md`
- `discussion_limitations_v1.md`
- `figure_table_order.md`

## Editing Principle

> 모든 편집 결정은 "이 문장이 최종 framing sentence에 직접 기여하는가?"라는 질문으로 판단한다. 기여하지 않으면 본문에서 제거한다.
