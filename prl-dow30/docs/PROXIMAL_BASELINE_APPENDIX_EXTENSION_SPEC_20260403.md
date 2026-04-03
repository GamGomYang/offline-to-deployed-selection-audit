# Proximal Simplicity / Information-Parity Baseline / Adaptive-eta Appendix Spec (2026-04-03)

## 목적

이 문서는 다음 3가지를 한 번에 정리한다.

1. 논문 본문에 넣을 수 있는 `paste-ready` 수정 문안
2. repo 기준의 `Linear Baseline for Information Parity (LBIP)` 구현/실험 설계
3. 기존 코드가 이미 지원하는 `rule_vol` 인프라를 활용한 appendix용 adaptive-eta 진단 계획

핵심 방향은 다음과 같다.

- Proximal Execution Control의 단순함을 `임의의 휴리스틱`이 아니라 `minimal regularized execution rule`로 재위치시킨다.
- 선형 convex baseline은 `같은 정보, 다른 매핑` 원칙으로 설계한다.
- adaptive eta는 메인 claim을 흔들지 않도록 `appendix exploratory diagnostic`으로만 둔다.

---

## 0. 이미 반영한 즉시 보강

다음 두 문장은 이미 논문 본문에 반영했다.

- fixed `eta`는 simplification이 아니라 `one-parameter execution frontier`를 보존하는 identification choice라는 점
- proximal 해의 convex-combination이 simplex 안에서 자동으로 feasible하다는 점

현재 반영 위치:
- `/workspace/execution-aware-portfolio-rl/paper.tex:204`
- `/workspace/execution-aware-portfolio-rl/paper.tex:224`
- `/workspace/execution-aware-portfolio-rl/paper.tex:226`

현재 들어간 핵심 문장:
- `Fixed eta removes schedule-design confounds, preserves a one-parameter execution frontier...`
- `Since both wtgt_t and wexec_{t-1} lie in the simplex, their convex combination is naturally feasible without further projection.`
- `That same convex-combination form also makes the rule computationally light and feasibility-preserving under the simplex constraint.`

이건 중요하다. 단순함을 약점이 아니라
- 계산 효율성
- feasibility preservation
- reproducibility
- one-parameter frontier interpretability
로 뒤집는 첫 번째 방어선이기 때문이다.

---

## 1. 논문 섹션별 실제 수정안

이 섹션은 논문에 추가할 문장을 섹션별로 바로 붙여넣을 수 있도록 적어둔다.

### 1-1. Introduction / Contribution framing

추가 목표:
- novelty를 `controller sophistication`에서 `execution/accounting interface`로 옮긴다.
- linear rule의 단순함을 `minimal principled choice`로 정의한다.

권장 삽입 문안:

```tex
The proposed execution rule is intentionally minimal. We do not present fixed-$\eta$ partial adjustment as a globally optimal market-microstructure controller. We present it as the exact solution to a one-step regularized execution problem that preserves feasibility on the simplex, yields a transparent one-parameter frontier, and cleanly isolates execution-layer accounting effects.
```

```tex
The empirical contribution is therefore not a more elaborate execution policy. It is the identification of how realized performance changes when the target generator is held fixed and the execution/accounting interface is made explicit.
```

### 1-2. Related Work: turnover-aware convex optimization 축 추가

현재 Related Work에는 execution / aim portfolio / portfolio RL 축이 있으므로, 여기에 `transaction-cost-aware convex optimization` 축을 더 또렷하게 넣는다.

권장 소절 제목:
- `Transaction-cost-aware convex portfolio optimization`

권장 문안:

```tex
A separate line of work studies transaction-cost-aware portfolio updates through convex optimization, typically by combining a return forecast, a risk model, and an explicit turnover penalty or no-trade regularization. That literature is directly relevant here because it already distinguishes desired allocation changes from costly realized trading. Our paper does not claim to replace those methods. Instead, it asks a narrower question that becomes especially important in portfolio RL settings: when the policy output is treated as a target rather than as an already-executed holding, how do execution-aware accounting and partial adjustment change realized net performance?
```

그리고 아래 문장으로 RL과 연결:

```tex
To make that comparison fair, the natural external benchmark is not a signal-free heuristic alone, but a linear cost-aware optimizer built from the same information set. We use that logic to motivate the Linear Baseline for Information Parity introduced in the experimental design.
```

### 1-3. Method / Theory: proximal rule의 정당화 문안

권장 문안:

```tex
The fixed-$\eta$ update is chosen because it is the exact solution to a minimal proximal execution problem, not because adaptive execution is impossible. Its value for the present paper is that it preserves a one-parameter execution frontier while remaining feasible by construction on the long-only simplex.
```

그리고 Proposition 1 다음 remark 보강 문안:

```tex
This feasibility-preserving property is practically important. Since both the previous executed portfolio and the current target lie in the simplex, the proximal update stays in the simplex without any additional projection, constrained optimization substep, or post-hoc repair.
```

### 1-4. Experimental Setup: Information-parity baseline 정의

이 부분은 실제 baseline 실험을 돌리기 전까지는 `planned comparator`로만 쓰고, 결과가 나오면 `implemented comparator`로 바꾼다.

권장 이름:
- `Linear Baseline for Information Parity (LBIP)`

권장 이유:
- `Linearized Oracle`는 내부 별칭으로는 괜찮지만, main text에서는 과장처럼 들릴 수 있다.
- reviewer-facing label은 `LBIP`가 더 신뢰감 있고 방어 가능하다.

권장 문안:

```tex
To benchmark the value of a nonlinear learned target generator against a stronger classical alternative, we define a Linear Baseline for Information Parity (LBIP). LBIP uses the same state input $s_t$ as the RL controller, but maps that state to per-asset return forecasts through a linear ridge model rather than through a nonlinear policy network. The resulting forecast vector $\hat\mu_t$ is then passed to a long-only convex portfolio optimizer under the same transaction-cost accounting convention.
```

그리고 fairness를 더 강하게:

```tex
The point of LBIP is not to give the convex baseline extra information. It is to enforce information parity: same state, different mapping. This lets the comparison focus on whether nonlinear target generation adds value beyond a linear state-to-alpha map once both methods are evaluated under the same executed-path accounting.
```

### 1-5. Results: 결과가 잘 나왔을 때의 문안 템플릿

#### 경우 A: RL+PEC > LBIP+PEC

```tex
Under information parity, the nonlinear RL target generator remains stronger than the linear convex baseline in positive-cost net performance, while both methods exhibit the same qualitative benefit from execution-aware accounting. This pattern suggests that the accounting contribution is not unique to RL, but that RL retains an additional advantage in how it maps the shared state input to target allocations.
```

#### 경우 B: RL+PEC ~= LBIP+PEC

```tex
Under information parity, the linear convex baseline is competitive with the RL controller, but the key execution-accounting pattern is shared: in both cases, separating target and executed holdings improves positive-cost realized performance relative to immediate execution. This strengthens the paper's core accounting claim even though it softens any stronger claim of nonlinear target-generation superiority.
```

#### 경우 C: LBIP+PEC > RL+PEC

```tex
The information-parity convex baseline is stronger on this benchmark, but the main execution-accounting result is unchanged: for both target generators, realized performance depends materially on whether target and executed holdings are separated. We therefore interpret the linear benchmark primarily as an external scale calibration and as evidence that the accounting contribution is not tied to one model family.
```

### 1-6. Discussion: 단순성 비판 대응 문안

```tex
The execution rule is intentionally simple in form, but that simplicity is part of its value here. Because the update is the exact solution to a minimal proximal problem, it preserves feasibility on the simplex, introduces only one execution-timescale parameter, and yields a frontier that can be interpreted without mixing accounting effects with controller-design complexity.
```

### 1-7. Limitations: 단순성 관련 정직한 제한 문안

```tex
We do not claim that fixed linear partial adjustment is globally optimal under realistic microstructure. More expressive adaptive controllers may outperform it. The present paper uses the minimal fixed-$\eta$ rule because it preserves a clean one-parameter frontier and keeps the empirical claim focused on execution-aware accounting rather than on adaptive schedule design.
```

---

## 2. Convex Baseline 설계: `Linear Baseline for Information Parity (LBIP)`

### 2-1. paper-facing naming

권장 메인 이름:
- `Linear Baseline for Information Parity (LBIP)`

내부 별칭으로만 허용:
- `linearized_oracle`

권장 이유:
- `oracle`는 reviewer에 따라 불필요하게 공격 포인트가 될 수 있다.
- main paper에서는 `LBIP`가 더 차분하고 공정하게 들린다.

### 2-2. 설계 원리

LBIP의 핵심 원리는 다음 한 줄이다.

- `same signals, same state, different mapping`

즉 RL이 쓰는 입력이 `s_t`라면, convex baseline도 같은 `s_t`를 사용한다. 다만 RL은 nonlinear policy로 target weights를 만들고, LBIP는 linear ridge regression으로 `\hat\mu_t`를 만든 뒤 convex optimizer에 넣는다.

### 2-3. strict information parity 정의

`strict parity` variant의 입력은 RL observation과 동일하다.

상태 벡터 구성:
- flattened 30-day log-return window
- current 30-day rolling volatility vector
- previous executed weights
- frozen signal channels (`reversal_5d`, `short_term_reversal`)

즉 차원은 canonical U27에서 RL과 같은 `918`차원이다.

중요한 원칙:
- alpha model 입력에 `s_t` 전체를 사용한다.
- turnover/cost는 RL과 동일하게 executed path에 붙인다.
- optimizer의 제약집합도 long-only simplex로 고정한다.

### 2-4. alpha model 정의

권장 기본형:
- asset별 one-step-ahead linear ridge regression
- target: next-day arithmetic asset return `r_{t+1,i}`
- feature: same state vector `s_t`

학습 방식:
- train split에서만 fit
- validation split에서 ridge penalty 선택
- held-out test는 pure evaluation only

권장 표기:

```tex
\hat\mu_{t,i} = \beta_i^\top s_t,
```

with ridge shrinkage fit on the training split.

권장 하이퍼파라미터 grid:
- ridge alpha in `{1e-4, 1e-3, 1e-2, 1e-1, 1, 10}`

선택 기준:
- validation positive-cost mean score over `kappa in {5e-4, 1e-3}`
- tie-breaker는 더 작은 ridge가 아니라 더 큰 retained-score / 더 낮은 turnover sensitivity가 아니라 `pre-registered smallest model complexity change`로 잡지 말고, 그냥 best validation score로 단순화

### 2-5. convex optimizer 정의

권장 target-construction objective:

```tex
\w_t^{\mathrm{LBIP}} = \arg\max_{w \in \Delta}
\hat\mu_t^\top w
- \lambda_r w^\top \hat\Sigma_t w
- \lambda_c \|w - \wexec_{t-1}\|_1.
```

여기서
- `\Delta`는 long-only fully invested simplex
- `\hat\mu_t`는 same-state linear ridge alpha
- `\hat\Sigma_t`는 rolling covariance
- `\lambda_c`는 turnover penalty

권장 risk/covariance 설정:
- covariance lookback grid `{60, 126, 252}`
- history minimum `60`
- risk aversion grid `{5, 10, 20}`
- turnover penalty multiplier grid `{0.5, 1.0, 2.0, 4.0}` applied to `kappa`

가장 안전한 validation selection rule:
- 전체 grid를 validation에서만 선택
- best positive-cost validation score variant 1개만 held-out report
- paper에는 selected LBIP variant만 main comparator로 보고
- 나머지는 appendix table에 `selection grid summary`로 보관

### 2-6. 두 가지 baseline arm을 모두 두는 것이 좋다

권장 arm matrix:

1. `LBIP-immediate`
- LBIP target
- immediate execution (`eta=1.0`)

2. `LBIP+PEC`
- LBIP target
- same fixed execution layer, ideally validation-selected `eta=0.5` or a validation-selected interior eta inside LBIP family

이 구조의 장점:
- RL vs convex target-generator 차이
- immediate vs PEC execution-layer 차이
를 분리해서 볼 수 있다.

### 2-7. 논문에 유리한 비교 매트릭스

권장 표 구조:

- RL target + immediate execution
- RL target + selected PEC
- LBIP target + immediate execution
- LBIP target + selected PEC

이 비교가 좋은 이유:
- PEC가 RL만의 trick이 아니라는 점도 볼 수 있음
- 동시에 RL nonlinear mapping의 추가 가치도 측정 가능
- reviewer가 `RL만 자기들끼리 비교했다`고 말하기 어려워짐

### 2-8. repo 기준 구현 경로

권장 새 파일:
- `/workspace/execution-aware-portfolio-rl/prl-dow30/prl/linear_information_parity.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/run_information_parity_baselines.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/configs/exp/paper_u27_lbip_validation.yaml`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/configs/exp/paper_u27_lbip_final.yaml`

권장 수정 파일:
- `/workspace/execution-aware-portfolio-rl/prl-dow30/prl/baselines.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/prl/eval.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/build_control_eta_validation_first_tables.py`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/run_external_heuristic_baselines.py` (혹은 새 script로 분리)

### 2-9. 함수 단위 구현 명세

#### A. state builder

새 함수 권장:
- `build_state_matrix_like_env(...)`

입력:
- returns frame
- volatility frame
- signal feature frame
- window size
- initial previous-weight convention (equal-weight)

출력:
- index-aligned `X_t` matrix
- next-day return target matrix `Y_{t+1}`

원칙:
- env observation order와 동일해야 함
- RL과 strict information parity를 맞추려면 state serialization이 동일해야 함

#### B. linear alpha fit

새 함수 권장:
- `fit_linear_state_alpha_models(X_train, Y_train, ridge_alpha)`

형태:
- per-asset ridge regression
- standardized features
- intercept allowed

출력:
- coefficients
- scaler stats
- per-asset fit diagnostics

#### C. daily prediction

새 함수 권장:
- `predict_linear_state_alpha(models, X_slice)`

출력:
- `mu_hat_t` vector

#### D. convex solve

새 함수 권장:
- `solve_turnover_penalized_mean_variance(mu_hat_t, Sigma_hat_t, w_prev, risk_aversion, turnover_penalty)`

solver 권장:
- 1순위: `cvxpy` + OSQP/SCS
- fallback: projected proximal gradient in numpy

논문/리뷰 대응 관점에서는 1순위가 더 좋다. 문제식이 명시적인 convex program이기 때문이다.

### 2-10. output artifact 명세

baseline run root 예시:
- `outputs/paper_u27_lbip_validation`
- `outputs/paper_u27_lbip_final`

필수 산출물:
- `aggregate.csv`
- `metrics.csv`
- `protocol.json`
- `selection.json`
- per-arm `trace.parquet`
- optional `alpha_fit_summary.csv`

논문 표 연결용 필드:
- `sharpe_net_lin`
- `cagr`
- `avg_turnover_exec`
- `tracking_error_l2_mean` (LBIP+PEC variant일 경우)
- `selected_variant`
- `ridge_alpha`
- `cov_lookback`
- `risk_aversion`
- `turnover_penalty_mult`

### 2-11. favorable but defendable selection rule

가장 좋은 운영법:
- validation에서 LBIP grid를 먼저 고정 선택
- held-out에는 selected LBIP variant 1개만 올림
- main text에 모든 convex variant를 나열하지 않음

이게 좋은 이유:
- 숫자 좋은 variant를 test에서 사후 선택했다는 비판을 피함
- 그래도 convex family 안에서 가장 강한 comparator를 공정하게 고를 수 있음

---

## 3. Adaptive-eta appendix diagnostic 계획

### 3-1. 원칙

adaptive eta는 메인 텍스트가 아니라 appendix diagnostic으로만 둔다.

이유:
- adaptive schedule을 메인에 올리면 논문의 object가 execution accounting에서 controller design으로 이동함
- reviewer에게는 `we considered richer rules` 신호만 주면 충분함

### 3-2. repo에서 이미 되는 것

현재 코드에는 이미 `rule_vol` mode가 있다.

관련 파일:
- `/workspace/execution-aware-portfolio-rl/prl-dow30/prl/envs.py`
- `EnvConfig.eta_mode = rule_vol`
- `rule_vol_window`, `rule_vol_a`, `eta_clip_min`, `eta_clip_max`

즉 appendix diagnostic 1차 버전은 새 algorithm 구현 없이 바로 가능하다.

### 3-3. 권장 appendix diagnostic

#### Diagnostic A: volatility-triggered eta

정의:
- `eta_t = clip(1 / (1 + 2 a sigma_t), eta_min, eta_max)`

권장 이유:
- 이미 코드가 존재함
- fixed eta보다 조금 더 표현력이 있지만, 여전히 해석 가능
- high-vol에서는 trading speed가 낮아지는 직관이 자연스러움

### 3-4. appendix용 validation grid

권장 grid:
- `rule_vol_a in {12, 24, 48}`
- `eta_clip in {[0.35, 0.65], [0.30, 0.70]}`
- `rule_vol_window = 20`

권장 이유:
- selected fixed point `eta=0.5` 주변에서 평균 eta가 움직이도록 유도
- 너무 공격적인 small-eta regime을 피해서 appendix 결과가 망가질 확률을 줄임
- fixed eta=0.5를 부정하는 diagnostic이 아니라, 그 근처의 state-dependent modulation만 점검하는 구조가 됨

### 3-5. appendix 성공 기준

adaptive rule은 `must beat fixed eta=0.5`가 아니다.

appendix diagnostic에서 보고 싶은 건 다음 중 하나면 충분하다.

1. fixed `eta=0.5`와 비슷한 positive-cost Sharpe를 유지하면서 turnover를 더 줄임
2. fixed `eta=0.5`보다 조금 낫지만 해석은 유지됨
3. fixed `eta=0.5`보다 크게 낫지 않음 -> 오히려 메인 fixed rule의 minimality가 더 설득력 있음

즉 appendix에서는 어떤 결과가 나와도 메인 스토리를 흔들지 않도록 설계해야 한다.

### 3-6. appendix 문안 템플릿

```tex
As an exploratory appendix check, we also tested a simple volatility-triggered execution schedule using the already-implemented rule-vol mode. The purpose of this check is not to replace the fixed-$\eta$ frontier, but to verify that the main accounting message is not an artifact of forbidding any state dependence in execution speed.
```

결과가 비슷할 때:

```tex
The adaptive rule does not overturn the fixed-$\eta$ reading: it produces comparable positive-cost performance, but the simpler fixed frontier remains easier to interpret and report.
```

결과가 조금 좋을 때:

```tex
The adaptive appendix rule modestly improves upon the fixed operating point, but the gain is incremental rather than structural. The fixed-$\eta$ frontier therefore remains the main object because it isolates the accounting effect with less controller-design confounding.
```

---

## 4. 실제 구현 순서

가장 효율적인 순서는 아래다.

### Phase 1. 문장 선반영

1. Introduction / Related Work / Method / Theory에 위 문안 반영
2. 논문 내 baseline framing을 `planned LBIP comparator` 수준으로 미리 정리

### Phase 2. LBIP 구현

1. `prl/linear_information_parity.py` 추가
2. state builder 구현
3. ridge alpha fit/predict 구현
4. convex optimizer solve 구현
5. `run_information_parity_baselines.py` 스크립트 추가
6. validation selection output 저장

### Phase 3. LBIP 결과 연결

1. validation-selected LBIP variant locked
2. held-out final run
3. paper table 추가
4. Results / Discussion / Limitations 반영

### Phase 4. adaptive eta appendix diagnostic

1. existing `rule_vol` mode의 narrow grid validation
2. selected appendix rule 1개만 held-out run
3. appendix figure/table로만 추가

---

## 5. 가장 추천하는 최종 experimental matrix

### Main text

- RL immediate (`eta=1.0`)
- RL selected PEC (`eta=0.5`)
- LBIP immediate
- LBIP + selected PEC

### Appendix only

- rule-vol adaptive eta diagnostic
- optional threshold/no-trade rule은 다음 라운드로 미룸

---

## 6. reviewer 대응 포인트 요약

이 설계를 쓰면 두 비판에 각각 이렇게 답할 수 있다.

### 비판 1: PEC가 너무 단순한 휴리스틱 아닌가?

답변 구조:
- exact proximal solution
- simplex feasibility preservation
- no extra projection
- one-parameter identifiable frontier
- adaptive control은 appendix exploratory로 분리

### 비판 2: RL끼리만 비교한 것 아닌가?

답변 구조:
- `Linear Baseline for Information Parity` 도입
- same state, different mapping
- same long-only simplex
- same executed-path accounting
- same validation/test lock

---

## 7. 바로 실행할 때의 실무 메모

가장 먼저 구현할 것은 adaptive eta가 아니라 LBIP다.

이유:
- reviewer가 더 직접적으로 요구하는 것은 stronger classical comparator임
- adaptive eta는 appendix diagnostic이라 없어도 submission 가능
- LBIP는 결과가 좋으면 논문 신뢰도를 크게 올리고, 결과가 중립이어도 discussion 재료가 된다

권장 paper-facing 표현 요약:
- `minimal proximal execution rule`
- `feasibility-preserving on the simplex`
- `Linear Baseline for Information Parity (LBIP)`
- `same state, different mapping`

