# Must Fix Before Submit v1

This note separates true pre-submission fixes from acceptable limitations and appendix-only issues.

## Must Fix Before Submission

### 1. Tighten forecasting-paper identity in the main narrative

- Why: this is the highest-risk reviewer attack.
- Fix standard: the paper must read first as a forecasting-to-decision evaluation paper and only second as a portfolio case study.
- Practical action:
  - keep the title, abstract, and first results paragraph forecasting-first
  - keep `executed-path evaluation is primary` as the central evaluation message
  - ensure dense-friction and comparator references do not pull the narrative back into finance-execution framing

### 2. Make the executed-vs-target disagreement support easier to see

- Why: otherwise the `executed-path is primary` claim may feel definitional.
- Fix standard: the reader should quickly see that positive-cost interpretation differs across target-based and executed-based evaluation.
- Practical action:
  - keep the accounting diagnostic prominent
  - consider using the target-vs-executed support table in discussion, appendix, or rebuttal-ready material
  - keep the wording narrow and diagnostic

### 3. Keep the same-forecast package explicitly conservative

- Why: overclaim here would trigger an easy reviewer attack.
- Fix standard: never imply exact forecast identity.
- Practical action:
  - use only `similar forecasting information`
  - keep the metric-level limitation explicit
  - do not let this package appear stronger than supporting evidence

### 4. Remove internal build/provenance jargon from submission-facing markdown

- Why: this is a credibility and presentation issue that reviewers notice quickly.
- Files currently affected:
  - `paper/forecasting_workshop/paper_forecasting_workshop_v1.md`
  - `paper/forecasting_workshop/appendix/appendix_forecasting_workshop_v1.md`
  - `paper/forecasting_workshop/appendix/captions_appendix_v1.md`
- Practical action:
  - replace `evaluation audit`, `forecast-similarity audit`, `regenerated canonical`, `locked protocol`, `workshop bundle`, `final audit`, and `guardrails` with plain paper-facing wording

### 5. Do not ship LaTeX build artifacts

- Why: `.fls` contains an absolute local path.
- Practical action:
  - exclude `.fls`, `.log`, `.aux`, and `.out` from any submission archive

## Acceptable Limitations

These are real weaknesses, but they do not need to be "fixed" before submission as long as the framing stays narrow.

- fixed 27-name snapshot and single-domain case study
- frozen-policy rather than end-to-end retraining evidence
- domain-specific realized metric in a finance case study
- no universality claim across forecasting domains

## Appendix-Only Issues

These should stay subordinate and must not be promoted to solve the paper's main persuasion problem.

- dense-friction sensitivity details
- dense-friction provenance details
- CC-TA-LBIP auxiliary evidence
- local `c`-ablation robustness
- extended forecast-similarity audit details

## Decision Rule

If the paper can satisfy the five must-fix items above, the remaining weaknesses are best handled as honest limitations rather than as fatal flaws. If it cannot, the most likely reviewer outcome is not a technical contradiction but a framing rejection: "interesting case study, but not yet convincingly a forecasting workshop paper."
