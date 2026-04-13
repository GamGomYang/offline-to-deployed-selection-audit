# Friction Curve Package Specification

## Metadata

- Project: ICML Workshop Reframing - Dense Friction Sensitivity Package
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 워크숍 논문에서 execution interface effect가 비용 수준에 민감하게 커지는 구조적 현상인지를 보여주기 위한 dense friction sensitivity package의 구현 명세다.

이 패키지는 새로운 선택 규칙을 만들기 위한 것이 아니다. 이미 selected-point main result가 보여준 positive-cost gain이 우연한 한 점짜리 결과가 아니라 `κ`와 함께 커지는 friction-sensitive pattern임을 보여주는 데 목적이 있다.

## Main Question

> 거래비용이 커질수록 execution interface의 realized-performance 효과도 커지는가?

## Package Role

- [ ] selected-point main result의 구조적 해석 강화
- [ ] "그냥 smoothing regularization win 아니냐"라는 질문에 대한 보조 반박
- [ ] `κ=0`에서는 미약하고 positive-cost에서 커지는 regime-sensitive evidence 제공
- [ ] cost-sensitive decision systems라는 framing 정당화

## Fixed Setup

### Policy and Selection

- [ ] learned policy는 고정
- [ ] validation-first protocol 유지
- [ ] selected operating point는 global selector 기준 `η=0.5` 유지
- [ ] dense friction expansion은 diagnostic only
- [ ] test 결과를 보고 `η`를 다시 고르지 않음

### Friction Grid

- `κ ∈ {2e-4, 5e-4, 1e-3, 2e-3}`

### Main Curves

- curve 1: selected `η=0.5` vs `η=1.0`의 paired-median `ΔSharpe`
- curve 2: best interior diagnostic gain vs `η=1.0`
- supporting quantity: executed turnover reduction

## Reference Values

| Item | Reference |
| --- | --- |
| global selected `η` | `0.5` |
| selected-point `ΔSharpe` at `κ=2e-4` | `+0.0041` |
| selected-point `ΔSharpe` at `κ=5e-4` | `+0.0105` |
| selected-point `ΔSharpe` at `κ=1e-3` | `+0.0213` |
| selected-point `ΔSharpe` at `κ=2e-3` | `+0.0409` |
| selected-point median `TOexec` reduction | `≈ 0.01105` |
| best interior gain at `κ=2e-4` | `+0.0113` |
| best interior gain at `κ=5e-4` | `+0.0230` |
| best interior gain at `κ=1e-3` | `+0.0424` |
| best interior gain at `κ=2e-3` | `+0.0761` |

## Success Criteria

### Full Pass

- [ ] expanded positive-cost set에서도 selected `η = 0.5` 유지
- [ ] selected-point `ΔSharpe`가 `κ`와 함께 증가
- [ ] best-interior diagnostic gain도 `κ`와 함께 증가
- [ ] turnover reduction sign이 유지

### Soft Pass

- [ ] low-cost에서 작고 high-cost에서 큰 gain
- [ ] selected-point gain curve가 비감소 또는 대체로 증가
- [ ] best-interior diagnostic도 같은 방향

Soft pass일 때는 아래 문장을 쓴다.

> The execution frontier becomes more consequential as proportional trading frictions rise.

### Fail

- [ ] selected `η`가 dense grid에서 흔들림
- [ ] selected-point gain이 `κ`와 무관하거나 반대로 움직임
- [ ] best interior gain도 monotonic pattern이 전혀 없음

## Figure Specification

### Figure Name

- `fig_kappa_curve.pdf`

### Title Draft

> The execution frontier steepens as friction grows.

### Layout

- x-axis: transaction cost `κ`
- y-axis: paired-median `ΔSharpe` vs `η=1.0`
- line 1: validation-selected `η=0.5`
- line 2: best interior diagnostic
- optional annotation: selected `η` remains `0.5` across expanded positive-cost set

### Visualization Rules

- [ ] x축은 실제 `κ` 값 그대로 사용
- [ ] y축은 `ΔSharpe`
- [ ] main line은 selected-point
- [ ] best interior line은 범례에 `diagnostic only`라고 명시

## Interpretation Rules

1. The dense friction expansion is diagnostic only and does not alter the main selected-point protocol.
2. Even on the expanded positive-cost grid, the global validation selector remains at `η=0.5`.
3. The selected-point gain increases from `+0.0041` to `+0.0409` as `κ` rises, while the best-interior diagnostic gain steepens even more strongly.
4. This supports a friction-sensitive interpretation: the execution interface matters more as proportional trading frictions rise.

## Result Paragraph Templates

### Template A

> On the dense canonical friction grid, the validation-selected operating point remains `η=0.5`, and its held-out paired-median `ΔSharpe` rises from `+0.0041` at `κ=2×10^-4` to `+0.0409` at `κ=2×10^-3`. The best-interior diagnostic frontier steepens in the same direction, indicating that the execution effect becomes more consequential as trading frictions increase.

### Template B

> The dense friction diagnostic shows that the result is not a one-off selected-point accident: interior execution becomes progressively more attractive as proportional trading costs grow.

## Forbidden Wording

- [ ] "best `η` tuning paper"
- [ ] "adaptive `η` is implied by this figure"
- [ ] "this proves universal monotonic improvement"
- [ ] "the selected point should always shrink with `κ`"
- [ ] "the diagnostic best-interior line is the new main result"

## Connected Packages

- `workshop_reframing/02_rl_main_package.md`
- `workshop_reframing/03_accounting_gap.md`
- `workshop_reframing/04_friction_curve.md`

이 figure는 main result보다 먼저 오면 안 된다. 역할은 main result의 구조적 해석 강화이지, 논문 중심의 교체가 아니다.

## Deliverables

- `dense_friction_repro.csv`
- `fig_kappa_curve.pdf`
- `friction_curve_caption.md`
- `friction_curve_paragraph.md`

## Caption Draft

> Figure X. The execution frontier steepens as transaction costs rise. On the expanded canonical friction grid, the validation-selected operating point remains `η=0.5`. Its held-out paired-median net-Sharpe gain increases with `κ`, and the best-interior diagnostic gain steepens even more strongly, indicating that the execution effect becomes more consequential in higher-friction regimes.

## Final Review Checklist

- [ ] dense grid에서도 selected `η` is still `0.5`
- [ ] selected-point gain grows with `κ`
- [ ] best-interior diagnostic gain grows with `κ`
- [ ] figure strengthens a cost-sensitive interpretation
- [ ] figure remains diagnostic, not a replacement of the main result
