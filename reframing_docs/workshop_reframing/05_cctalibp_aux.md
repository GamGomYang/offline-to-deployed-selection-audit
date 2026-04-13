# CC-TA-LBIP Auxiliary Package Specification

## Metadata

- Project: ICML Workshop Reframing - Linear Forecast + Cost-Aware Decision Comparator Package
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 워크숍 논문에서 RL 외부에서도 같은 방향의 implementation effect가 보인다는 보조 근거를 만들기 위한 CC-TA-LBIP auxiliary package의 구현 명세다.

이 패키지는 절대 main identification package가 아니다. 역할은 forecast-to-decision interface effect가 RL 내부 현상만은 아니라는 supporting evidence를 제공하는 것이다.

## Main Question

> 같은 state information과 같은 accounting 아래에서도 cost-aware decision interface가 positive-cost에서 realized decision quality를 바꾸는가?

이 질문은 RL 메인 질문보다 약하다. 여기서는 causal-looking main evidence를 만들지 않고, same-direction supporting evidence만 얻는 것이 목표다.

## Comparator Definition

### Comparator Name

- `CC-TA-LBIP`
- Cost-Calibrated Turnover-Aware Linear Information-Parity Baseline

### Structure Summary

- same 918-dimensional state as RL
- linear ridge forecast map
- long-only convex optimizer with explicit turnover anchor
- same proportional cost grid
- same executed-path accounting
- validation-only tuning of cost-scaling constant `c`

## Fixed Setup

### State and Information

- [ ] RL과 같은 918-dimensional state
- [ ] same return window
- [ ] same volatility vector
- [ ] same previous executed weights
- [ ] same frozen signal channels

### Forecast Map

- [ ] ridge alpha fixed at `30`
- [ ] linear forecast map is fixed
- [ ] `c` tuning은 forecast map refit을 의미하지 않음

### Decision Layer

- [ ] convex optimizer with turnover anchor
- [ ] `c grid = {0, 2000, 3000, 4000, 6000, 8000, 12000}`
- [ ] selected `c = 3000`

### Accounting

- [ ] same `κ` grid
- [ ] same executed-path net-return definition
- [ ] same Sharpe annualization
- [ ] `κ=0`에서는 selected-`c` arm이 `c=0` baseline으로 collapse

## Package Role

이 패키지는 아래 역할만 맡는다.

- [ ] RL main result의 범용성 암시 보조
- [ ] interface / implementation matters라는 메시지의 RL 외부 보조 근거
- [ ] forecasting workshop 맥락에서 forecast map + decision layer 구조를 더 직접적으로 보여주는 장치

이 패키지는 아래 역할을 맡지 않는다.

- [ ] main claim replacement
- [ ] benchmark dominance claim
- [ ] deterministic optimizer superiority claim
- [ ] RL frozen-policy result보다 더 강한 증거

## Expected Structure

숫자 크기 자체보다 아래 구조가 핵심이다.

- [ ] positive-cost에서만 improvement
- [ ] `κ=0`에서는 no free smoothing
- [ ] selected `c = 3000`
- [ ] same-direction evidence
- [ ] auxiliary only
- [ ] forecast map은 고정
- [ ] cost-aware turnover control이 decision layer에서 작동
- [ ] improvement는 cost-sensitive decision-interface effect로 읽음
- [ ] deterministic single-run character는 숨기지 않음

## Success Criteria

### Full Pass

- [ ] selected `c = 3000` 재현
- [ ] `κ=0` row에서 collapse condition 유지
- [ ] positive-cost rows에서 same-direction improvement
- [ ] executed-path accounting matched
- [ ] auxiliary positioning을 유지한 채 결과 서술 가능

### Soft Pass

- [ ] selected `c`가 같음
- [ ] `κ=0` collapse 유지
- [ ] positive-cost improvement가 존재하지만 수치가 약간 흔들림

### Fail

- [ ] selected `c`가 달라짐
- [ ] `κ=0` collapse 구조가 사라짐
- [ ] positive-cost에서도 improvement sign이 불안정
- [ ] forecast map refit 여부가 불명확
- [ ] deterministic single-run 한계를 감추지 않으면 안 되는 상황

## Table Specification

### Table Name

- `table_cctalibp_aux.csv`
- `table_cctalibp_aux.tex`

### Title Draft

> Auxiliary evidence from a linear forecast-plus-convex-optimization comparator. CC-TA-LBIP uses the same state information and the same executed-path accounting as the RL experiments, while tuning only the cost-scaling constant on validation.

### Columns

- `κ`
- selected `c`
- `Net Sharpe (c=0)`
- `Net Sharpe (c=3000)`
- `ΔSharpe`
- optional `TOexec`
- optional realized cost

### Rows

- `κ=0`
- `κ=5e-4`
- `κ=1e-3`

## Interpretation Rules

1. As a supporting check outside RL, we evaluate a linear forecast-plus-convex-optimization comparator under the same accounting rules.
2. The comparator keeps the forecast map fixed and tunes only the cost-scaling constant `c` on validation, selecting `c=3000`.
3. Because the turnover anchor is proportional to `κ`, the selected arm collapses to the unregularized `c=0` baseline at `κ=0`, avoiding any free zero-cost smoothing.
4. Positive-cost improvements in this comparator therefore support the same directional message: cost-aware decision interfaces matter under frictions.

## Result Paragraph Templates

### Template A

> As auxiliary evidence beyond RL, CC-TA-LBIP uses the same 918-dimensional state and the same executed-path accounting, while fixing the ridge forecast map and tuning only the cost-scaling constant `c` on validation. The selected comparator `c=3000` collapses to the `c=0` baseline at `κ=0`, so any positive-cost gain supports the same implementation-side interpretation without introducing free zero-cost smoothing.

### Template B

> This comparator does not replace the frozen-policy RL result, but it shows that friction-calibrated turnover control can also matter for a stronger classical optimizer under information parity.

## Forbidden Wording

- [ ] "CC-TA-LBIP proves the same result more strongly than RL"
- [ ] "this is our second main identification result"
- [ ] "the comparator dominates all baselines"
- [ ] "the gain is purely forecast-driven"
- [ ] "deterministic single-run is not a limitation"

## Connected Analysis

이 패키지는 아래 문서와 직접 연결된다.

- `workshop_reframing/06_same_forecast_table.md`

이유는 간단하다. CC-TA-LBIP는 forecast map 고정, `c`만 변경 구조를 가장 직관적으로 보여줄 수 있어서 implementation, not prediction 메시지를 직접 표로 만들기 가장 좋은 기반이다.

## Deliverables

- `table_cctalibp_aux.csv`
- `table_cctalibp_aux.tex`
- `cctalibp_aux_caption.md`
- `cctalibp_aux_paragraph.md`

## Caption Draft

> Table X. Auxiliary evidence from a linear forecast-plus-convex-optimization comparator. CC-TA-LBIP uses the same 918-dimensional state and the same executed-path accounting as the RL experiments, while fixing the ridge forecast map and tuning only the cost-scaling constant `c` on validation. The comparator remains auxiliary and is not part of the paper's core frozen-policy identification strategy.

## Final Review Checklist

- [ ] same state information is preserved
- [ ] ridge forecast map is fixed
- [ ] selected `c` is still `3000`
- [ ] `κ=0` collapse condition is preserved
- [ ] result can be written as supporting evidence, not main evidence
