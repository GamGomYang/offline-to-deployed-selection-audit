# Reproduction Checklist

## Metadata

- Project: ICML Workshop Reframing - Reproduction Gate Before Any New Analysis
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Purpose

이 문서는 새로운 실험이나 추가 분석을 시작하기 전에, 기존 원고의 strongest evidence가 현재 환경에서 재생성되는지 검증하기 위한 체크리스트다.

새 분석은 이 문서를 통과한 뒤에만 시작한다.
이 문서를 통과하지 못하면 어떤 추가 실험도 본문용 evidence로 승격하지 않는다.

## Facts That Must Reproduce

### A. Protocol Facts

- [ ] frozen-policy design
- [ ] validation-only `η` selection
- [ ] fixed `η` grid
- [ ] primary evaluation on executed path
- [ ] effective held-out dates: `2024-02-14` to `2025-12-31`

### B. Main RL Facts

- [ ] validation-selected operating point = `η = 0.5`
- [ ] positive-cost held-out gain at `κ = 5e-4, 1e-3`
- [ ] negligible `κ = 0` effect
- [ ] executed turnover roughly halves at the selected point

### C. Accounting Diagnostics

- [ ] `TOtgt / TOexec ≈ 2.00`
- [ ] tracking discrepancy `≈ 0.00259`
- [ ] final equity gap increases with cost

### D. Friction Sensitivity

- [ ] dense friction grid에서도 selected `η = 0.5` 유지
- [ ] `κ`가 커질수록 selected-point `ΔSharpe`가 커짐

### E. CC-TA-LBIP Facts

- [ ] same 918-dimensional state
- [ ] fixed ridge forecast map
- [ ] validation-selected `c = 3000`
- [ ] `κ=0`에서 `c=0` baseline으로 collapse
- [ ] auxiliary only

## Hard Gate Values

| Item                           | Target      |
| ------------------------------ | ----------- |
| selected `η`                   | `0.5`       |
| held-out `ΔSharpe` at `κ=5e-4` | `> 0`       |
| held-out `ΔSharpe` at `κ=1e-3` | `> 0`       |
| `κ=0` effect                   | near zero   |
| selected `TOexec`              | `≈ 0.01095` |
| `TOtgt / TOexec`               | `≈ 2.00`    |
| CC-TA-LBIP selected `c`        | `3000`      |

재생성은 완전히 동일한 소수점이 아니라 해석상 같은 패턴이면 통과다. 다만 위 값들은 원고 기준과 매우 가깝게 맞아야 한다.

## Reproduction Order

### Step 1 - Artifact and Environment Lock

확인 항목:

- [ ] 코드 브랜치
- [ ] 데이터 snapshot 경로
- [ ] outputs root
- [ ] config 버전
- [ ] seed list
- [ ] date split
- [ ] trace 저장 형식

산출물:

- `env_lock.txt`
- `artifact_root.txt`

### Step 2 - Selected-Point Validation Logic Check

확인 항목:

- [ ] `η` grid가 `{1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02}`인지
- [ ] positive-cost validation set가 `{5e-4, 1e-3}`인지
- [ ] `95% largest-qualifying` selection logic이 그대로인지
- [ ] test 결과를 selection에 사용하지 않았는지

산출물:

- `validation_selection_check.md`

### Step 3 - RL Main Result Reproduction

확인 항목:

- [ ] selected `η = 0.5`
- [ ] held-out positive-cost rows의 `ΔSharpe`
- [ ] `κ=0` row effect
- [ ] selected row `TOexec`

예상 패턴:

- `κ=5e-4`: positive
- `κ=1e-3`: stronger positive
- `κ=0`: near-flat
- `TOexec`: roughly half of `η=1.0` baseline

산출물:

- `rl_selected_vs_eta1_repro.csv`
- `rl_selected_vs_eta1_repro.md`

### Step 4 - Accounting Diagnostic Reproduction

확인 항목:

- [ ] `TOexec`
- [ ] `TOtgt`
- [ ] `TOtgt / TOexec`
- [ ] tracking discrepancy
- [ ] mean absolute return gap
- [ ] final equity gap
- [ ] max daily gap

예상 값:

- `TOtgt / TOexec ≈ 2.00`
- tracking `≈ 0.00259`
- final equity gap은 `κ` 증가에 따라 증가

산출물:

- `diagnostic_selected_eta_repro.csv`
- `fig_selected_trace_repro.png`

### Step 5 - Dense Friction Curve Reproduction

확인 항목:

- [ ] `κ` grid = `{2e-4, 5e-4, 1e-3, 2e-3}`
- [ ] selected `η` remains `0.5`
- [ ] selected-point `ΔSharpe` increases with `κ`

예상 값:

- `+0.0041`
- `+0.0105`
- `+0.0213`
- `+0.0409`

산출물:

- `dense_friction_repro.csv`
- `fig_kappa_curve_repro.pdf`

### Step 6 - CC-TA-LBIP Reproduction

확인 항목:

- [ ] same state features
- [ ] ridge alpha fixed
- [ ] `c` grid
- [ ] selected `c = 3000`
- [ ] `κ=0` row collapse condition
- [ ] auxiliary positioning

예상 구조:

- positive-cost only improvement
- zero-cost no free smoothing
- main RL result의 보조 증거로 사용 가능

산출물:

- `cctalibp_repro.csv`
- `cctalibp_repro.md`

## File-by-File Questions

### RL Selected Result File

- [ ] 이 파일이 selected `η` vs `η=1.0` 비교인지
- [ ] metric이 executed-path net Sharpe인지
- [ ] paired-seed median 기준인지
- [ ] `κ=0 / 5e-4 / 1e-3` 세 줄이 모두 존재하는지

### Validation Selection File

- [ ] validation-only selection인지
- [ ] test metric이 selection에 들어가 있지 않은지
- [ ] largest qualifying `η` rule이 맞는지

### Trace Diagnostic File

- [ ] `metrics.csv`가 아니라 `trace.parquet` 기준 재구성이 가능한지
- [ ] target-path와 executed-path가 모두 복구되는지
- [ ] effective held-out date alignment가 맞는지

### CC-TA-LBIP File

- [ ] forecast map이 `c`에 따라 refit되지 않았는지
- [ ] `c`만 바뀌는 구조인지
- [ ] `κ=0`일 때 `c=0` row와 selected-`c` row가 collapse하는지

## Reference Table

| Item                           | Reference  |
| ------------------------------ | ---------- |
| selected `η`                   | `0.5`      |
| held-out `ΔSharpe` at `κ=5e-4` | `+0.0105`  |
| held-out `ΔSharpe` at `κ=1e-3` | `+0.0213`  |
| `κ=0` effect                   | negligible |
| selected `TOexec`              | `0.01095`  |
| selected `TOtgt`               | `0.02190`  |
| `TOtgt / TOexec`               | `2.00`     |
| tracking discrepancy           | `0.00259`  |
| final equity gap at `κ=0`      | `0.00067`  |
| final equity gap at `κ=5e-4`   | `0.00369`  |
| final equity gap at `κ=1e-3`   | `0.00739`  |
| dense-grid `ΔSharpe` at `2e-4` | `+0.0041`  |
| dense-grid `ΔSharpe` at `5e-4` | `+0.0105`  |
| dense-grid `ΔSharpe` at `1e-3` | `+0.0213`  |
| dense-grid `ΔSharpe` at `2e-3` | `+0.0409`  |
| CC-TA-LBIP selected `c`        | `3000`     |

## Mismatch Types

### Type A - Configuration Mismatch

- [ ] 잘못된 config 파일
- [ ] 잘못된 `κ` grid
- [ ] `η` grid 누락
- [ ] selected rule 오구현

### Type B - Data or Split Mismatch

- [ ] wrong effective start date
- [ ] wrong validation/test split
- [ ] rolling warmup misalignment
- [ ] universe mismatch

### Type C - Metric or Accounting Mismatch

- [ ] target cost를 main cost로 사용
- [ ] target-path return을 main Sharpe에 사용
- [ ] net/gross confusion
- [ ] turnover column mismatch

### Type D - Artifact Mismatch

- [ ] stale metrics file
- [ ] stale trace path
- [ ] wrong output directory
- [ ] old branch artifacts mixed in

## Pass or Stop Rules

### Pass to Next Stage

아래가 모두 충족되면 다음 단계로 넘어간다.

- [ ] selected `η` reproduced
- [ ] selected-point positive-cost gain sign preserved
- [ ] accounting gap pattern preserved
- [ ] dense friction curve pattern preserved

### Stop Condition

아래 중 하나라도 발생하면 즉시 중단하고 원인 분석 문서를 쓴다.

- [ ] selected `η`가 `0.5`가 아님
- [ ] `κ=0` row가 크게 흔들림
- [ ] `TOtgt / TOexec` 구조가 사라짐
- [ ] CC-TA-LBIP selected `c`가 달라짐
- [ ] effective date range가 어긋남

산출물:

- `repro_failure_report.md`

## Summary Template

> The canonical frozen-policy result reproduces the workshop-relevant pattern: the validation-selected interior point remains `η=0.5`, positive-cost held-out gains are preserved, the `κ=0` effect remains negligible, and executed-path diagnostics continue to support a translation/accounting interpretation rather than a pure alpha-improvement interpretation.

## Next Documents

이 문서를 통과한 뒤에만 다음 문서를 연다.

- `workshop_reframing/02_rl_main_package.md`
- `workshop_reframing/03_accounting_gap.md`
- `workshop_reframing/04_friction_curve.md`
