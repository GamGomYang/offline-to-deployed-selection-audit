# Claim Freeze Document

## Metadata

- Project: ICML Workshop Reframing - Forecast-to-Execution Interface in Cost-Sensitive Portfolio Decision Systems
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 워크숍용 재구성 논문의 최종 주장 범위, 허용되는 해석 범위, 금지되는 post-hoc 변경, 그리고 실험 성공 판정 규칙을 고정하기 위한 문서다.

이 프로젝트의 목적은 원하는 결론이 나오도록 실험을 비트는 것이 아니다.
목적은 이미 확보된 strongest evidence에 가장 정확하게 부합하는 질문을 고정하고, 그 질문에 대한 증거만 순차적으로 패키징하는 것이다.

## Final Claim

### Primary Claim

> In cost-sensitive portfolio decision systems, holding the predictive signal fixed, the forecast-to-execution interface materially changes realized decision quality.

### Korean Working Version

거래비용이 있는 포트폴리오 의사결정 시스템에서는, 예측 신호가 같더라도 forecast-to-execution interface에 따라 최종 realized decision quality가 실질적으로 달라질 수 있다.

이 클레임은 현재 원고와 정합적이다. 원고는 target과 executed portfolio를 분리하고, primary evaluation을 executed turnover와 realized executed path에 부착하며, frozen-policy / validation-first protocol 하에서 `η=0.5`가 positive-cost에서 turnover를 줄이고 net Sharpe를 개선한다고 설명한다. 또한 이득은 gross return 향상이 아니라 lower realized turnover와 lower realized cost에서 온다고 적고 있다.

## Core Research Question

본 프로젝트에서 답하려는 질문은 오직 아래 하나다.

> 같은 predictive signal 또는 같은 learned target path를 유지할 때, execution interface의 변화가 positive-cost regime에서 realized performance를 바꾸는가?

이 질문은 현재 원고의 frozen-policy identification question과 정확히 맞는다. 원고는 learned policy를 고정한 채 execution mapping만 바꾸어 realized turnover, realized cost, realized net performance가 어떻게 달라지는지를 본다고 명시한다.

## Allowed Claim Scope

아래 범위까지만 허용한다.

- [ ] execution-aware interface는 positive-cost regime에서 realized net performance를 개선할 수 있다.
- [ ] 이 효과는 better alpha generation이라기보다 implementation / translation / accounting effect로 해석된다.
- [ ] 비용이 클수록 execution frontier의 중요성이 커질 수 있다.
- [ ] 이 현상은 RL 본체만의 특수 현상이 아니라, linear forecast + convex decision interface에서도 같은 방향의 보조 증거가 존재한다.

이 네 가지는 모두 현재 원고가 직접적으로 지지한다. 원고는 positive-cost에서 `η=0.5`의 net Sharpe 개선, `κ=0`에서의 negligible effect, dense friction grid에서 `κ`가 커질수록 gain이 커지는 패턴, 그리고 CC-TA-LBIP comparator의 same-direction evidence를 보고한다.

## Forbidden Claim Scope

아래 주장은 금지한다.

- [ ] forecasting systems in general에 대한 일반 이론을 제시한다.
- [ ] 새로운 RL method를 제안한다.
- [ ] better alpha generation 또는 superior forecasting model을 제시한다.
- [ ] 실전 배포 가능한 trading system 우위를 입증했다고 쓴다.
- [ ] passive benchmark 또는 strong classical baseline을 전반적으로 이겼다고 쓴다.

이 금지 규칙은 현재 원고의 범위와 한계에 기반한다. 원고는 fixed 27-name large-cap snapshot이라는 controlled benchmark를 사용하며, point-in-time constituent reconstruction이 아니라고 명시한다. 또한 external heuristics는 contextual only이고 core identification strategy가 아니라고 적고 있으며, buy-and-hold equal weight가 selected RL arm보다 강한 구간이 있음을 스스로 인정한다.

## Evidence Hierarchy

아래 순서를 고정한다.

### Tier 1 - Core Evidence

1. RL frozen-policy selected-point result
2. target-vs-executed accounting gap
3. dense friction sensitivity

### Tier 2 - Supporting Evidence

1. CC-TA-LBIP auxiliary comparator

### Tier 3 - Optional Supporting Analysis

1. same forecast / different decision quality table
2. rolling-window robustness
3. `η`-aligned retraining
4. U36 replication

이 순서를 고정하는 이유는 간단하다. 논문 본체는 이미 frozen-policy RL selected-point evidence와 executed-path accounting argument만으로도 설 수 있다. auxiliary evidence는 본체를 돕는 역할만 해야 한다.

## Interpretation Frame

모든 결과 해석은 아래 프레임 안에서만 수행한다.

- [ ] primary evaluation object는 executed turnover와 executed-path return이다.
- [ ] target-path quantity는 diagnostic only다.
- [ ] improvement는 prediction improvement가 아니라 translation / accounting improvement로 해석한다.
- [ ] positive-cost에서 gain이 크고 `κ=0`에서는 효과가 작다면, 이는 claim을 지지하는 패턴이다.
- [ ] gross-path improvement가 없더라도 논문 실패로 판단하지 않는다.

이 해석 프레임은 현재 원고의 구조와 완전히 일치한다. 원고는 executed path를 primary evaluation object로 명시하고, target-based return과 `TOtgt`는 diagnostics only라고 적는다. 또한 target-based cost charging은 `η<1`일 때 friction scale을 과대평가한다고 설명한다.

## Fixed Experiment Rules

### Frozen RL Main Experiment

- [ ] predictive model 또는 learned policy는 고정한다.
- [ ] main selected operating point는 validation-only rule로 선택한다.
- [ ] test 결과를 보고 `η`를 다시 고르지 않는다.
- [ ] main 비교는 `η=1.0` vs selected `η=0.5`로 고정한다.
- [ ] `κ` grid는 `{0, 5e-4, 1e-3}`로 고정한다.

### Accounting Rules

- [ ] primary metric은 executed-path net Sharpe다.
- [ ] `TOexec`와 realized cost는 main supporting metric이다.
- [ ] `TOtgt`, target-path return, tracking discrepancy는 diagnostic only다.
- [ ] target-path metric을 main result로 승격하지 않는다.

### Comparator Rules

- [ ] CC-TA-LBIP는 auxiliary comparator다.
- [ ] comparator 결과는 main result와 동급으로 다루지 않는다.
- [ ] deterministic single-run character를 숨기지 않는다.
- [ ] `κ=0`에서 `c=0`과 collapse되는 구조를 명시한다.

## Success Tiers

### Success Tier 1 - Full Success

- [ ] `κ=5e-4, 1e-3`에서 `ΔSharpe > 0`
- [ ] executed turnover reduction이 동반된다.
- [ ] `κ=0`에서 효과가 negligible하다.
- [ ] accounting gap diagnostics가 `TOtgt/TOexec ≈ 2`, small-but-nonzero tracking, cost-sensitive final equity gap을 재확인한다.

### Success Tier 2 - Moderate Success

positive-cost 두 구간 중 한 구간에서만 강한 gain이 보이면, claim strength를 아래처럼 낮춘다.

> Execution-aware interfaces can improve realized decision quality in positive-cost regimes.

### Success Tier 3 - Accounting-Centered Success

Sharpe gain이 약하지만 accounting gap이 분명하면, 논문 중심을 아래처럼 전환한다.

> Cost-aligned evaluation is necessary because target-based accounting can materially distort realized decision assessment under partial execution.

## Failure Triggers

아래 경우는 즉시 추가 튜닝이 아니라 원인 분석으로 이동한다.

- [ ] validation-selected `η`가 재현되지 않는다.
- [ ] test selected-point result가 기존 수치와 유의미하게 다르다.
- [ ] `κ=0` row가 갑자기 크게 좋아지거나 나빠져 해석 구조가 무너진다.
- [ ] `TOtgt / TOexec` 구조가 재생성되지 않는다.
- [ ] effective evaluation dates가 맞지 않는다.

이 경우 다음 단계로 넘어가지 않는다. 먼저 데이터 버전, seed list, artifact directory, trace source, cost column, selection logic을 재검증한다.

## Prohibited Post-hoc Changes

아래 변경은 금지한다.

- [ ] test를 보고 `η` grid 수정
- [ ] `κ` grid 수정
- [ ] metric 우선순위 변경
- [ ] selected point 대신 raw best tiny-`η` point를 main으로 승격
- [ ] 좋게 나온 split만 본문 채택
- [ ] heuristic baseline 비교를 본문 중심으로 승격
- [ ] comparator를 main identification experiment처럼 포장

## Preferred Wording

### Prefer

- realized decision quality
- executed-path evaluation
- cost-sensitive implementation
- forecast-to-execution interface
- translation / accounting effect
- frozen-policy identification
- auxiliary comparator
- supporting evidence

### Avoid

- better predictor
- superior alpha
- SOTA
- generally applicable forecasting theorem
- broadly dominant trading system

## Operational Principle

> This project does not aim to manufacture a desired conclusion. It freezes the narrowest claim already supported by the strongest existing evidence, and only allows experiments or analyses that directly test, reproduce, or interpret that claim. Post-hoc selection changes, test-driven retuning, and claim expansion are prohibited.
