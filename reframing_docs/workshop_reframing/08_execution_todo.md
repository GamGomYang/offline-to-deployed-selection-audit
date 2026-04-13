# Execution TODO Checklist

## Metadata

- Project: ICML Workshop Reframing - Forecast-to-Execution Interface in Cost-Sensitive Portfolio Decision Systems
- Version: `v1.0`
- Owner: `[이름]`
- Last Updated: `[날짜]`

## Path Convention

이 문서의 경로 표기는 저장소 루트 기준 상대경로로 통일한다.
프로젝트 문서와 산출물은 모두 `reframing_docs/workshop_reframing/...` 형식을 사용한다.

## Scope Lock

이 워크숍 리프레이밍 작업의 기본 축은 current incumbent 설정이 아니라 canonical paper rebuild다.
즉, 메인 클레임과 패키징 기준은 validation-selected `η = 0.5` canonical result를 중심으로 고정한다.

현재 기준 작업 구조는 아래와 같다.

```text
reframing_docs/workshop_reframing/
reframing_docs/workshop_reframing/prompts/
reframing_docs/workshop_reframing/outputs/
reframing_docs/workshop_reframing/outputs/tables/
reframing_docs/workshop_reframing/outputs/figures/
reframing_docs/workshop_reframing/outputs/checks/
reframing_docs/workshop_reframing/outputs/logs/
reframing_docs/workshop_reframing/outputs/appendix/
```

## 0. Operating Principles

### 0.1 Top-Level Principles

- [ ] 이 프로젝트의 목적은 원하는 결론을 만들기 위한 실험이 아니라, 이미 가장 강하게 서 있는 질문에 대한 증거를 순차적으로 잠그는 것이다.
- [ ] test 결과를 보고 `η` selection rule을 바꾸지 않는다.
- [ ] test 결과를 보고 `κ` grid를 바꾸지 않는다.
- [ ] test 결과를 보고 main metric을 바꾸지 않는다.
- [ ] test 결과를 보고 claim 범위를 넓히지 않는다.
- [ ] 좋게 나온 일부 split이나 일부 표만 본문에 cherry-pick 하지 않는다.
- [ ] 추가 분석은 오직 메인 메시지의 직접 검증 또는 해석 보강을 위해서만 수행한다.

### 0.2 Final Framing Re-Check

> In cost-sensitive portfolio decision systems, holding the predictive signal fixed, the forecast-to-execution interface materially changes realized decision quality.

- [ ] 위 문장에서 벗어나는 실험은 본문용으로 진행하지 않는다.
- [ ] `better predictor`, `new RL method`, `general forecasting systems`, `benchmark dominance` 같은 과대 framing을 금지한다.

### 0.3 What Success Means

- [ ] positive-cost에서 selected-point gain이 유지되면 성공이다.
- [ ] `κ=0`에서 effect가 거의 없으면 오히려 좋은 패턴이다.
- [ ] gross-path improvement가 약해도 실패가 아니다.
- [ ] improvement가 turnover / realized cost 감소와 연결되면 메인 해석은 유지된다.

## 1. Workspace and Document Structure

### 1.1 Required Documents

- [ ] `reframing_docs/workshop_reframing/00_claim_freeze.md`
- [ ] `reframing_docs/workshop_reframing/01_repro_checklist.md`
- [ ] `reframing_docs/workshop_reframing/02_rl_main_package.md`
- [ ] `reframing_docs/workshop_reframing/03_accounting_gap.md`
- [ ] `reframing_docs/workshop_reframing/04_friction_curve.md`
- [ ] `reframing_docs/workshop_reframing/05_cctalibp_aux.md`
- [ ] `reframing_docs/workshop_reframing/06_same_forecast_table.md`
- [ ] `reframing_docs/workshop_reframing/07_paper_assembly.md`
- [ ] `reframing_docs/workshop_reframing/08_execution_todo.md`

### 1.2 Working Folders

- [ ] `reframing_docs/workshop_reframing/`
- [ ] `reframing_docs/workshop_reframing/prompts/`
- [ ] `reframing_docs/workshop_reframing/outputs/`
- [ ] `reframing_docs/workshop_reframing/outputs/tables/`
- [ ] `reframing_docs/workshop_reframing/outputs/figures/`
- [ ] `reframing_docs/workshop_reframing/outputs/checks/`
- [ ] `reframing_docs/workshop_reframing/outputs/logs/`
- [ ] `reframing_docs/workshop_reframing/outputs/appendix/`

### 1.3 Naming Rules

- [ ] 표 파일은 `table_*.csv`, `table_*.tex`
- [ ] 그림 파일은 `fig_*.pdf`, `fig_*.png`
- [ ] 설명 문서는 `*_notes.md`, `*_paragraph.md`, `*_caption.md`
- [ ] 재현 로그는 `*_repro.md`, `*_check.md`

## 2. Environment Lock

### 2.1 Code Environment Lock

- [ ] 현재 git branch 기록
- [ ] commit hash 기록
- [ ] remote status 기록
- [ ] dirty working tree 여부 기록
- [ ] 사용 python 버전 기록
- [ ] 패키지 버전 freeze 기록
- [ ] random seed policy 기록

### 2.2 Data Environment Lock

- [ ] data snapshot 경로 기록
- [ ] universe 정의 파일 기록
- [ ] split 정의 파일 기록
- [ ] validation window 기록
- [ ] held-out test effective date 기록
- [ ] rolling warmup 처리 방식 기록

현재 원고는 effective validation/test trace dates, fixed 27-name large-cap snapshot, rolling-feature warmup, close-to-close convention을 분명히 적고 있다.

### 2.3 Artifact Path Lock

- [ ] canonical run root 기록
- [ ] selected-point stats file 위치 기록
- [ ] validation selection file 위치 기록
- [ ] trace parquet 위치 기록
- [ ] dense friction outputs 위치 기록
- [ ] CC-TA-LBIP outputs 위치 기록
- [ ] stale artifact 혼입 여부 점검

### 2.4 Environment Lock Deliverables

- [ ] `env_lock.txt`
- [ ] `artifact_root.txt`
- [ ] `run_manifest.md`

## 3. Phase 1 - Reproduction Gate

### 3.1 Validation-Only Selection Reproduction

- [ ] `η` grid가 정확히 `{1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02}`인지 확인
- [ ] positive-cost validation set가 `{5e-4, 1e-3}`인지 확인
- [ ] selection rule이 `largest η within locked near-best threshold`인지 확인
- [ ] test 결과가 selection에 들어가지 않았는지 확인
- [ ] selected `η = 0.5` 재현 확인

### 3.2 RL Main Result Reproduction

- [ ] `η=1.0` vs `η=0.5` 비교 파일 로드
- [ ] `κ=0` row 존재 확인
- [ ] `κ=5e-4` row 존재 확인
- [ ] `κ=1e-3` row 존재 확인
- [ ] paired-median `ΔSharpe` 계산 방식 확인
- [ ] executed-path net Sharpe 기준인지 확인
- [ ] selected `η=0.5`의 positive-cost gain 확인
- [ ] `κ=0` effect near zero 확인
- [ ] `TOexec 0.02200 -> 0.01095` 근처 구조 확인

### 3.3 Accounting Diagnostics Reproduction

- [ ] trace source가 살아 있는지 확인
- [ ] executed-path vs target-path 모두 재구성 가능한지 확인
- [ ] `TOexec` 추출 가능
- [ ] `TOtgt` 추출 가능
- [ ] `TOtgt / TOexec` 계산 가능
- [ ] tracking discrepancy 계산 가능
- [ ] final equity gap 계산 가능
- [ ] `κ` 증가에 따른 gap 증가 구조 확인
- [ ] `TOtgt / TOexec ≈ 2.00` 구조 확인
- [ ] tracking `≈ 0.00259` 근처 구조 확인

### 3.4 Dense Friction Reproduction

- [ ] `κ` grid가 `{2e-4, 5e-4, 1e-3, 2e-3}`인지 확인
- [ ] selected `η`가 dense grid에서도 `0.5`인지 확인
- [ ] selected-point `ΔSharpe`가 `κ`와 함께 커지는지 확인
- [ ] best interior diagnostic gain도 같은 방향인지 확인
- [ ] figure source file 존재 확인

### 3.5 CC-TA-LBIP Reproduction

- [ ] same 918-dimensional state 사용 여부 확인
- [ ] ridge `alpha=30` 고정 여부 확인
- [ ] `c` grid 확인
- [ ] selected `c = 3000` 재현 확인
- [ ] `κ=0` collapse 구조 확인
- [ ] executed-path accounting 일치 확인
- [ ] auxiliary positioning 재확인

### 3.6 Reproduction Deliverables

- [ ] `validation_selection_check.md`
- [ ] `rl_selected_vs_eta1_repro.csv`
- [ ] `diagnostic_selected_eta_repro.csv`
- [ ] `dense_friction_repro.csv`
- [ ] `cctalibp_repro.csv`
- [ ] `repro_summary.md`

### 3.7 Stop Conditions

- [ ] selected `η != 0.5`이면 중단
- [ ] positive-cost gain sign이 깨지면 중단
- [ ] `κ=0` row 해석이 무너지면 중단
- [ ] `TOtgt / TOexec` 구조가 사라지면 중단
- [ ] selected `c != 3000`이면 중단
- [ ] 중단 시 `repro_failure_report.md` 작성

## 4. Phase 2 - RL Main Package

### 4.1 Data Freeze

- [ ] selected-point comparison 파일 최종 버전 확정
- [ ] 사용 metrics가 executed-path 기준인지 최종 확인
- [ ] paired-median 사용 여부 확인
- [ ] `κ` 세 줄만 main table에 사용할지 확정

### 4.2 Main Table

- [ ] `table_rl_main.csv` 생성
- [ ] `table_rl_main.tex` 생성
- [ ] `κ`, `Net Sharpe(η=1.0)`, `Net Sharpe(η=0.5)`, `ΔSharpe`, `TOexec(η=1.0)`, `TOexec(η=0.5)` 포함
- [ ] positive-cost rows 강조
- [ ] `κ=0` row는 negligible effect 설명용으로 유지

### 4.3 Result Paragraphs

- [ ] strongest template 작성
- [ ] moderate template 작성
- [ ] safe fallback template 작성
- [ ] implementation-side gain 문장 포함
- [ ] not necessarily alpha improvement 문장 포함

### 4.4 Review

- [ ] positive-cost gain이 분명한가
- [ ] `κ=0` effect가 약한가
- [ ] `TOexec` reduction이 충분히 큰가
- [ ] 본문 첫 결과로 올릴 수 있는가

### 4.5 Deliverables

- [ ] `table_rl_main.csv`
- [ ] `table_rl_main.tex`
- [ ] `rl_main_result_paragraph.md`
- [ ] `rl_main_caption.md`

## 5. Phase 3 - Accounting Gap Package

### 5.1 Diagnostic Source Prep

- [ ] selected `η` trace 파일 경로 확정
- [ ] target-path diagnostics 계산 스크립트 경로 확정
- [ ] executed equity / target equity 재구성 가능 여부 점검
- [ ] metrics vs trace source 충돌 없는지 확인

### 5.2 Table

- [ ] `diagnostic_gap_table.csv` 생성
- [ ] `diagnostic_gap_table.tex` 생성
- [ ] `κ=0`, `5e-4`, `1e-3` 세 줄 구성
- [ ] `TOexec`, `TOtgt`, `TOtgt/TOexec`, tracking discrepancy, final equity gap 포함
- [ ] optional로 mean abs return gap 포함

### 5.3 Figure

- [ ] `fig_accounting_gap.pdf` 생성
- [ ] executed-path equity vs target-path equity trace 포함
- [ ] `κ`별 패널 구성
- [ ] 시각적으로 gap이 cost와 함께 커지는지 확인

### 5.4 Interpretation

- [ ] target-path는 diagnostic only라고 명시
- [ ] executed-path가 primary evaluation object라고 명시
- [ ] `TOtgt > TOexec` 구조를 직관적으로 설명
- [ ] tracking small but nonzero 문장 포함
- [ ] cost 상승에 따라 final path gap 증가 문장 포함

### 5.5 Review

- [ ] `TOtgt / TOexec ≈ 2` 근처인가
- [ ] tracking이 작지만 0은 아닌가
- [ ] final equity gap이 `κ`와 함께 커지는가
- [ ] figure가 본문에서 직관적으로 먹히는가

### 5.6 Deliverables

- [ ] `diagnostic_gap_table.csv`
- [ ] `diagnostic_gap_table.tex`
- [ ] `fig_accounting_gap.pdf`
- [ ] `accounting_gap_paragraph.md`
- [ ] `accounting_gap_caption.md`

## 6. Phase 4 - Dense Friction Curve Package

### 6.1 Data Freeze

- [ ] dense friction results source 확정
- [ ] `κ` grid 재확인
- [ ] selected `η=0.5` 유지 재확인
- [ ] best-interior diagnostic curve source 확인

### 6.2 Figure

- [ ] `fig_kappa_curve.pdf` 생성
- [ ] x축 `κ`, y축 `ΔSharpe`
- [ ] line 1: selected `η=0.5`
- [ ] line 2: best interior diagnostic
- [ ] 범례에 diagnostic curve임을 명시

### 6.3 Interpretation

- [ ] diagnostic only라는 표현 포함
- [ ] selected `η` remains `0.5` 문장 포함
- [ ] gain increases with friction 문장 포함
- [ ] cost-sensitive structure 문장 포함
- [ ] adaptive `η` or tuning paper처럼 보이지 않게 점검

### 6.4 Review

- [ ] selected-point curve가 `κ`와 함께 증가하는가
- [ ] best interior diagnostic도 같은 방향인가
- [ ] main result 대체가 아니라 해석 강화로 보이는가

### 6.5 Deliverables

- [ ] `fig_kappa_curve.pdf`
- [ ] `friction_curve_paragraph.md`
- [ ] `friction_curve_caption.md`

## 7. Phase 5 - CC-TA-LBIP Auxiliary Package

### 7.1 Comparator Validation

- [ ] same 918-dimensional state 확인
- [ ] ridge `alpha=30` 고정 확인
- [ ] `c` grid 확인
- [ ] selected `c=3000` 재확인
- [ ] `κ=0` collapse 조건 재확인
- [ ] `c` tuning이 forecast refit을 의미하지 않음을 확인

### 7.2 Table

- [ ] `table_cctalibp_aux.csv` 생성
- [ ] `table_cctalibp_aux.tex` 생성
- [ ] `κ=0`, `5e-4`, `1e-3` 포함
- [ ] `c=0` vs `c=3000` 중심 비교 구성
- [ ] optional로 `TOexec`, realized cost 포함

### 7.3 Interpretation

- [ ] auxiliary comparator 표현 포함
- [ ] same-direction supporting evidence 문장 포함
- [ ] deterministic single-run 성격 숨기지 않기
- [ ] not part of the core frozen-policy identification strategy 문장 포함

### 7.4 Review

- [ ] 본문 보조 문단으로 충분한가
- [ ] RL 메인 결과보다 앞에 나오지 않게 배치했는가
- [ ] benchmark dominance처럼 안 보이는가

### 7.5 Deliverables

- [ ] `table_cctalibp_aux.csv`
- [ ] `table_cctalibp_aux.tex`
- [ ] `cctalibp_aux_paragraph.md`
- [ ] `cctalibp_aux_caption.md`

## 8. Phase 6 - Same Forecast / Different Decision Quality Analysis

### 8.1 Pre-Check

- [ ] 이 분석은 추가 분석 1회라는 점 확인
- [ ] forecast metric이 archive에 바로 없다는 점 확인
- [ ] 본문에 올릴지 appendix로 내릴지 결과 보고 판정할 계획 세움
- [ ] 제목은 일단 conservative version으로 시작

기본 제목:

> Similar forecasting information, different realized decision quality

### 8.2 Forecast Output Extraction

- [ ] same fitted ridge forecast map outputs 추출
- [ ] 날짜별 predicted signal 저장
- [ ] realized next-period returns와 정렬
- [ ] forecast metric 계산용 intermediate file 저장

### 8.3 Forecast Metrics

- [ ] forecast MSE 계산
- [ ] rank IC 계산 가능 여부 확인
- [ ] sign accuracy 계산 가능 여부 확인
- [ ] 최소한 forecast MSE + 하나의 직관적 metric 확보

### 8.4 Decision Metrics

- [ ] `c=0` 조건 결과 정리
- [ ] `c=3000` 조건 결과 정리
- [ ] `TOexec` 계산
- [ ] realized cost 계산
- [ ] net Sharpe 계산
- [ ] `κ=0` collapse 확인

### 8.5 Table

- [ ] `table_same_forecast_diff_decision.csv` 생성
- [ ] `table_same_forecast_diff_decision.tex` 생성
- [ ] rows: `c=0`, `c=3000`
- [ ] columns: forecast MSE, rank IC or sign accuracy, `TOexec`, realized cost, net Sharpe

### 8.6 Decision Rule

- [ ] forecast-side 차이가 decision-side 차이보다 훨씬 작은가
- [ ] 그렇다면 본문 채택
- [ ] 아니면 appendix로 이동
- [ ] stronger title을 쓸지 conservative title을 유지할지 결정

### 8.7 Title Rule

- [ ] forecast metric 차이가 매우 작으면 `Same forecast, different realized decision quality`
- [ ] forecast metric 차이가 작지만 완전히 동일하진 않으면 `Similar forecasting information, different realized decision quality`
- [ ] forecast metric 차이도 꽤 크면 appendix-only

### 8.8 Deliverables

- [ ] `forecast_outputs_eval.csv`
- [ ] `table_same_forecast_diff_decision.csv`
- [ ] `table_same_forecast_diff_decision.tex`
- [ ] `forecast_metric_analysis.md`
- [ ] `same_forecast_table_paragraph.md`

## 9. Phase 7 - Paper Assembly Review

### 9.1 Main-Text Items

- [ ] RL frozen-policy main table
- [ ] accounting gap figure or table
- [ ] dense friction curve
- [ ] CC-TA-LBIP auxiliary table
- [ ] same-forecast table는 quality 좋을 때만 본문

### 9.2 Appendix Items

- [ ] external heuristics detailed table
- [ ] buy-and-hold dominant rows
- [ ] rolling-window full details
- [ ] U36 replication detailed table
- [ ] retraining details
- [ ] implementation appendix

### 9.3 Result Flow

- [ ] 결과 1: selected-point improvement
- [ ] 결과 2: why executed-path matters
- [ ] 결과 3: why it is friction-sensitive
- [ ] 결과 4: why this is not RL-only
- [ ] 결과 5: why this is implementation, not merely prediction

## 10. Abstract Assembly TODO

### 10.1 Sentence Structure

- [ ] sentence 1: proposed targets vs realized positions
- [ ] sentence 2: cost makes conflation problematic
- [ ] sentence 3: forecast-to-execution interface
- [ ] sentence 4: selected `η=0.5` + positive-cost gain + turnover halving
- [ ] sentence 5: implementation-side interpretation + auxiliary support

### 10.2 Number Selection

- [ ] `+0.0105`, `+0.0213`를 넣을지 결정
- [ ] turnover `0.02200 -> 0.01095`를 넣을지 결정
- [ ] 숫자가 너무 많으면 하나만 남길지 결정

### 10.3 Forbidden Claims

- [ ] broader-than-portfolio claim 금지
- [ ] new RL method claim 금지
- [ ] benchmark dominance claim 금지

## 11. Introduction Assembly TODO

### 11.1 First Paragraph

- [ ] forecasting-driven decision systems framing
- [ ] target vs executed distinction
- [ ] realized-path evaluation necessity
- [ ] cost-sensitive portfolio decision case study

### 11.2 Contribution Paragraph

- [ ] interface contribution
- [ ] frozen-policy main evidence
- [ ] accounting, friction, comparator supporting evidence

### 11.3 Forbidden Style

- [ ] related work 장황하게 쓰지 않기
- [ ] PRL 내부 구조 과하게 쓰지 않기
- [ ] full-paper처럼 넓게 벌리지 않기

## 12. Results Section Assembly TODO

### 12.1 Paragraph Order

- [ ] paragraph 1: RL main result
- [ ] paragraph 2: turnover / cost interpretation
- [ ] paragraph 3: accounting gap
- [ ] paragraph 4: friction sensitivity
- [ ] paragraph 5: CC-TA-LBIP auxiliary
- [ ] paragraph 6: same-forecast direct evidence, 조건부

### 12.2 Results Tone

- [ ] improves positive-cost held-out decision quality
- [ ] negligible zero-cost effect
- [ ] implementation-side gain
- [ ] diagnostic only
- [ ] auxiliary comparator

### 12.3 Forbidden Claims

- [ ] SOTA
- [ ] learns better alpha
- [ ] general theorem
- [ ] always better
- [ ] dominates baselines

## 13. Conclusion and Limitations TODO

### 13.1 Must Include in Conclusion

- [ ] realized decisions must be evaluated on realized paths
- [ ] interface matters under frictions
- [ ] gains are implementation-side
- [ ] portfolio-domain case study

### 13.2 Must Include in Limitations

- [ ] fixed 27-name snapshot
- [ ] portfolio-domain case study
- [ ] frozen-policy identification, not full retraining study
- [ ] auxiliary comparator is not core evidence
- [ ] passive benchmark dominance not claimed

## 14. Final Self-Review Checklist

### 14.1 Paper-Level Checks

- [ ] 이 논문이 무엇을 주장하는지 한 문장으로 말할 수 있는가
- [ ] RL main result가 분명히 중심인가
- [ ] accounting gap이 해석을 도와주는가
- [ ] friction figure가 structure를 보여주는가
- [ ] CC-TA-LBIP가 auxiliary로만 보이는가
- [ ] same-forecast table이 도움될 때만 본문에 있는가

### 14.2 Overclaim Checks

- [ ] 제목이 과장되지 않았는가
- [ ] 초록이 범위를 넘지 않았는가
- [ ] 결론이 benchmark superiority로 읽히지 않는가
- [ ] general forecasting systems claim이 숨어들지 않았는가

### 14.3 Reading Flow Checks

- [ ] 먼저 좋아졌음을 보여주는가
- [ ] 그다음 왜 좋아졌는지 설명하는가
- [ ] target-based accounting이 왜 문제인지 보여주는가
- [ ] cost가 커질수록 더 중요하다는 걸 보여주는가
- [ ] RL 밖에서도 같은 방향이라는 걸 짧게 확인하는가

## 15. Priority Order

### 15.1 Must Be First

- [ ] 재현 게이트 통과
- [ ] RL 메인 표 생성
- [ ] accounting gap figure 생성
- [ ] dense friction figure 생성

### 15.2 Next

- [ ] CC-TA-LBIP auxiliary 표 생성

### 15.3 Last

- [ ] same-forecast table 추가 분석
- [ ] 본문 채택 여부 결정

이 순서를 지키는 이유는 간단하다. 새 추가 분석이 흔들려도 메인 논문은 이미 설 수 있어야 한다.

## 16. Stop or Hold Rules

### 16.1 Stop Immediately

- [ ] reproduction mismatch 발생
- [ ] selected `η` mismatch
- [ ] primary metric mismatch
- [ ] trace reconstruction failure
- [ ] CC-TA-LBIP forecast refit ambiguity

### 16.2 Hold for Appendix

- [ ] same-forecast table가 애매할 때
- [ ] rolling-window figure가 흐릴 때
- [ ] external baselines가 본문 흐름을 해칠 때
- [ ] retraining checks가 focus를 흐릴 때

## 17. Final Deliverable Checklist

### 17.1 Tables and Figures

- [ ] `table_rl_main.tex`
- [ ] `diagnostic_gap_table.tex`
- [ ] `fig_accounting_gap.pdf`
- [ ] `fig_kappa_curve.pdf`
- [ ] `table_cctalibp_aux.tex`
- [ ] `table_same_forecast_diff_decision.tex` 또는 appendix 이동

### 17.2 Writing Files

- [ ] `abstract_workshop_v1.md`
- [ ] `intro_workshop_v1.md`
- [ ] `results_workshop_v1.md`
- [ ] `discussion_limitations_v1.md`
- [ ] `figure_table_order.md`

### 17.3 Status and Review Files

- [ ] `repro_summary.md`
- [ ] `reframing_docs/workshop_reframing/00_claim_freeze.md`
- [ ] `run_manifest.md`

## 18. Final Confirmation

아래 문장을 읽고 맞으면 체크한다.

- [ ] 나는 지금 결과를 좋게 만들기보다, 이미 강하게 서 있는 질문에 가장 직접적인 증거만 남기고 있다.
- [ ] 나는 새 실험으로 논문을 넓히지 않고, 현재 strongest evidence를 더 명확히 보여주고 있다.
- [ ] 내가 만드는 워크숍 논문은 full paper 축소판이 아니라, 메시지가 선명한 case-study workshop paper다.
