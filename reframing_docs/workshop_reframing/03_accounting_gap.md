# Accounting Gap Package Specification

## Metadata

- Project: ICML Workshop Reframing - Target-vs-Executed Accounting Gap Package
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 워크숍 논문에서 왜 target-based evaluation이 아니라 executed-path evaluation이 primary여야 하는지를 보여주는 diagnostic package의 구현 명세다.

이 패키지는 메인 RL result를 대체하지 않는다. 역할은 메인 결과를 해석 가능하게 만들고, "implementation, not prediction" 메시지를 직관적으로 뒷받침하는 것이다.

## Main Question

> 같은 learned policy라도 target-path와 executed-path를 혼동하면 무엇이 왜곡되는가?

## Package Role

이 패키지는 논문에서 아래 역할을 맡는다.

- [ ] executed-path primary evaluation의 정당화
- [ ] translation / accounting interpretation의 직접 증거
- [ ] "그냥 smoothing 아니냐"라는 질문에 대한 반박 보조
- [ ] positive-cost gain이 왜 gross-path gain이 아닌지 설명하는 장치

## Fixed Comparison

- [ ] selected operating point: `η = 0.5`
- [ ] reference baseline: target-path hypothetical immediate execution
- [ ] `κ ∈ {0, 5e-4, 1e-3}`

원고의 trace-based diagnostics 구조와 맞추기 위해, target-path return과 realized executed-path return을 분리하고 selected `η`에 대한 trace diagnostics를 사용한다.

## Core Diagnostics

### Required Metrics

- `TOexec`
- `TOtgt`
- `TOtgt / TOexec`
- tracking discrepancy
- mean absolute return gap
- max daily absolute gap
- final equity gap

### Optional Diagnostics

- executed-path equity trace
- target-path equity trace

## Reference Values

| Item | Reference |
| --- | --- |
| `TOexec` | `0.01095` |
| `TOtgt` | `0.02190` |
| `TOtgt / TOexec` | `2.00` |
| tracking discrepancy | `0.00259` |
| final equity gap at `κ=0` | `0.00067` |
| final equity gap at `κ=5e-4` | `0.00369` |
| final equity gap at `κ=1e-3` | `0.00739` |

## Figure Specification

### Figure Name

- `fig_accounting_gap.pdf`

### Title Draft

> Target-path and executed-path diverge under partial execution. Diagnostic traces for the validation-selected operating point `η=0.5` across cost regimes.

### Preferred Layout

#### Option A - Equity Trace Figure

- x-axis: time
- y-axis: cumulative equity
- line 1: executed-path equity
- line 2: target-path hypothetical equity
- panels: `κ=0`, `5e-4`, `1e-3`

#### Option B - Compact Diagnostic Summary

- `TOtgt / TOexec`
- final equity gap
- tracking discrepancy

워크숍 본문에서는 A가 더 직관적이고, appendix에는 B를 보조로 넣는 것이 좋다.

## Table Specification

### Table Name

- `diagnostic_gap_table.csv`
- `diagnostic_gap_table.tex`

### Title Draft

> Executed-path diagnostics for the validation-selected operating point. Target-based quantities are reported for diagnosis only and are not used as primary evaluation objects.

### Columns

- `κ`
- `TOexec`
- `TOtgt`
- `TOtgt / TOexec`
- tracking discrepancy
- final equity gap
- mean abs return gap

### Rows

- `κ = 0`
- `κ = 5e-4`
- `κ = 1e-3`

## Interpretation Rules

아래 문장 흐름을 따른다.

1. Target-path quantities are counterfactual diagnostics rather than primary evaluation objects.
2. At the selected operating point, target turnover is about twice executed turnover, consistent with the partial execution rule.
3. Tracking discrepancy remains small but nonzero, indicating that execution is neither identical to the target nor arbitrarily detached from it.
4. The final path gap increases with cost, showing that target-based accounting increasingly misstates realized decision outcomes in positive-cost regimes.

## Result Paragraph Templates

### Template A

> At the validation-selected operating point `η=0.5`, target turnover is about twice realized executed turnover (`0.02190` vs `0.01095`; ratio `≈ 2.00`), while tracking remains small but nonzero (`0.00259`). The resulting final equity gap grows with cost, from `0.00067` at `κ=0` to `0.00739` at `κ=10^-3`, supporting an accounting / translation interpretation rather than a pure alpha-improvement story.

### Template B

> The target path is a useful diagnostic but not the correct primary evaluation object: under partial execution, target turnover overstates realized trading, and the divergence between target and executed paths becomes more consequential as costs rise.

## Success Criteria

### Full Pass

- [ ] `TOtgt / TOexec ≈ 2`
- [ ] tracking discrepancy is small but nonzero
- [ ] final equity gap increases with `κ`
- [ ] executed-path and target-path trace can be stably reconstructed

### Soft Pass

- [ ] target turnover > executed turnover
- [ ] tracking is small but nonzero
- [ ] cost increases path gap

### Fail

- [ ] target and executed quantities are numerically almost identical
- [ ] `TOtgt / TOexec` 구조가 사라짐
- [ ] cost-sensitive final equity gap 구조가 없음
- [ ] trace reconstruction 실패

## Forbidden Uses

- [ ] target-path Sharpe를 main result처럼 사용
- [ ] diagnostic quantity를 primary metric처럼 승격
- [ ] "difference exists, therefore always better" 식으로 해석
- [ ] tracking discrepancy가 작다는 이유만으로 effect를 trivial하다고 서술

## Deliverables

- `diagnostic_gap_table.csv`
- `diagnostic_gap_table.tex`
- `fig_accounting_gap.pdf`
- `accounting_gap_caption.md`
- `accounting_gap_paragraph.md`

## Caption Drafts

### Figure Caption Draft

> Figure X. Target-path and executed-path diverge under partial execution. For the validation-selected operating point `η=0.5`, executed and target traces remain close but not identical. The gap becomes more consequential as transaction costs increase, supporting executed-path evaluation as the primary accounting object.

### Table Caption Draft

> Table X. Executed-path diagnostics for the validation-selected operating point. Target-based quantities are included for diagnosis only. Under partial execution, target turnover is about twice executed turnover, tracking remains small but nonzero, and final path divergence grows with cost.

## Final Review Checklist

- [ ] target-vs-executed difference is visible
- [ ] target turnover meaningfully exceeds executed turnover
- [ ] tracking is small but nonzero
- [ ] cost increases the path gap
- [ ] the package supports an accounting / translation interpretation
