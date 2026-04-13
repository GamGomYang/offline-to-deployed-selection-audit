# RL Main Package Specification

## Metadata

- Project: ICML Workshop Reframing - Frozen-Policy Main Result Package
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 워크숍 논문의 가장 강한 중심 증거인 RL frozen-policy selected-point result를 재생성하고, 본문용 표와 문장으로 고정하기 위한 구현 명세다.

본 패키지의 목적은 가장 잘 나온 `η`를 찾는 것이 아니다. validation-only로 이미 선택된 `η=0.5` operating point를 기준으로, 같은 learned target path에서 execution interface만 바뀌었을 때 realized decision quality가 어떻게 달라지는지를 가장 간단하고 설득력 있게 보여주는 것이 목적이다.

## Main Question

> 고정된 learned policy에서 validation-selected execution point가 positive-cost held-out test에서 realized decision quality를 개선하는가?

## Package Role

이 패키지는 논문 전체에서 아래 역할을 맡는다.

- [ ] 핵심 causal-looking evidence
- [ ] 가장 먼저 제시되는 결과
- [ ] 논문 제목, 초록, 결론을 지탱하는 기준 증거
- [ ] 다른 모든 보조 결과의 기준점

## Fixed Experimental Setup

### Policy Condition

- [ ] learned policy는 고정한다.
- [ ] retraining 금지.
- [ ] test 결과를 보고 policy를 다시 고르지 않는다.

### Execution Comparison

- [ ] baseline: `η = 1.0`
- [ ] selected operating point: `η = 0.5`
- [ ] main body에서는 이 둘만 직접 비교한다.

### Cost Grid and Metrics

- [ ] `κ ∈ {0, 5e-4, 1e-3}`
- [ ] primary metric: executed-path net Sharpe
- [ ] supporting metrics: `TOexec`, realized cost, paired-median `ΔSharpe`, optional win-rate
- [ ] `κ=0` negligible effect는 오히려 좋은 패턴으로 해석한다.
- [ ] gross-path improvement 부재는 실패가 아니다.

## Reference Values

| Item | Reference |
| --- | --- |
| selected `η` | `0.5` |
| `ΔSharpe` at `κ=5e-4` | `+0.0105` |
| `ΔSharpe` at `κ=1e-3` | `+0.0213` |
| `ΔSharpe` at `κ=0` | `-0.00025` or near zero |
| baseline `TOexec` | `0.02200` |
| selected `TOexec` | `0.01095` |

## Success Criteria

### Full Pass

- [ ] selected `η = 0.5` 재현
- [ ] `κ=5e-4`에서 `ΔSharpe > 0`
- [ ] `κ=1e-3`에서 `ΔSharpe > 0`
- [ ] `κ=0`에서 effect가 near zero
- [ ] `TOexec`가 baseline 대비 크게 감소

### Soft Pass

- [ ] positive-cost 두 구간 중 하나만 확실한 개선
- [ ] 다른 한 구간은 near-zero 또는 작은 positive
- [ ] `TOexec` reduction 유지
- [ ] `κ=0` effect remains small

Soft pass일 때는 아래 문장을 쓴다.

> The validation-selected interior point can improve realized decision quality in positive-cost regimes while substantially reducing executed turnover.

### Fail

- [ ] selected `η`가 달라짐
- [ ] positive-cost 두 구간 모두 non-positive
- [ ] `κ=0`에서만 좋아지고 cost regime에서 약함
- [ ] `TOexec` reduction이 거의 없음

## Main Table Specification

### Table Name

- `table_rl_main.csv`
- `table_rl_main.tex`

### Title Draft

> Held-out selected-point comparison under fixed learned policies. Validation-selected `η=0.5` is compared against the immediate-execution baseline `η=1.0` on the `2024-02-14` to `2025-12-31` held-out window.

### Columns

- `κ`
- `Net Sharpe (η=1.0)`
- `Net Sharpe (η=0.5)`
- `Paired-median ΔSharpe`
- `TOexec (η=1.0)`
- `TOexec (η=0.5)`
- optional realized cost ratio or cost reduction

### Rows

- `κ = 0`
- `κ = 5e-4`
- `κ = 1e-3`

### Emphasis

- [ ] positive-cost rows 강조
- [ ] `η=0.5` positive-cost improvement 강조
- [ ] `TOexec` reduction 강조

## Interpretation Rules

아래 순서로 해석한다.

1. Validation-selected `η=0.5` preserves the positive-cost direction on held-out test data.
2. The improvement is concentrated in positive-cost regimes and is negligible at `κ=0`.
3. Executed turnover is nearly halved, consistent with an implementation-driven rather than alpha-driven gain.

## Result Paragraph Templates

### Template A - Strongest

> Under a frozen learned policy, the validation-selected interior point `η=0.5` improves held-out net Sharpe in both positive-cost regimes (`+0.0105` at `κ=5×10^-4` and `+0.0213` at `κ=10^-3`) while reducing average executed turnover from `0.02200` to `0.01095`; the `κ=0` effect remains negligible.

### Template B - Moderate

> Under a frozen learned policy, the validation-selected interior point preserves the positive-cost direction on held-out data while substantially reducing executed turnover, with little to no gain at `κ=0`.

## Forbidden Wording

- [ ] "best `η` is small, therefore smaller is always better"
- [ ] "this proves a new RL algorithm"
- [ ] "the policy became better"
- [ ] "the model learns superior alpha"
- [ ] "the selected point dominates all baselines"

원고의 selected-point logic은 raw best tiny-`η`가 아니라 validation-selected operating point이며, core claim도 stronger target generator가 아니라 execution-aware accounting / implementation effect다.

## Connected Packages

이 패키지는 아래 문서와 연결된다.

- `reframing_docs/workshop_reframing/03_accounting_gap.md`
- `reframing_docs/workshop_reframing/04_friction_curve.md`

연결 논리:

- Table 1이 selected-point improvement를 보여준다.
- accounting gap figure가 왜 executed-path evaluation이 필요한지를 보여준다.
- friction curve가 왜 이것이 cost-sensitive structure인지를 보여준다.

## Deliverables

- `table_rl_main.csv`
- `table_rl_main.tex`
- `rl_main_numbers_check.md`
- `rl_main_result_paragraph.md`
- `rl_main_caption.md`

## Final Review Checklist

- [ ] selected point is still `η=0.5`
- [ ] positive-cost gains are still positive
- [ ] `κ=0` remains negligible
- [ ] `TOexec` reduction remains large
- [ ] interpretation stays implementation / translation rather than alpha
