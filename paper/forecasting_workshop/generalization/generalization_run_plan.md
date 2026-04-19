# Generalization Master Run Plan

This document fixes the scope, wording, success criteria, and output layout for the forecasting workshop paper's next generalization support package.

## Background Summary

The compact temporal multi-split pilot was mixed. Accordingly, this package does **not** claim broad temporal robustness from that pilot, and it does not restore temporal robustness as the main generalization axis.

Instead, the next support package targets recurrence of the same narrow implementation-side reading across:

- fixed universes
- execution-layer and decision-architecture variants
- repeated target-vs-executed disagreement settings
- a minimal toy setting that travels beyond the portfolio-domain details

The generalization question is therefore not "does the full paper broadly generalize across time?" The narrower question is "does the same executed-path interpretation recur across fixed universes, decision architectures, and evaluation-object checks?"

## Package Scope

This package is support-only unless explicitly promoted later.

- executed-path quantities remain primary
- target-level quantities remain diagnostic
- no support experiment in this package becomes a new main empirical result by default
- no output from this package should be used to widen the title, abstract, or headline claim without a separate explicit promotion decision

## Workstreams

### 1. Multi-Universe Robustness

Purpose:
check whether the same narrow executed-path reading recurs across additional fixed universes without changing the paper's core identity.

Default reading:

- keep the domain close to the current case study
- look for recurrence across fixed universe definitions, not broad market-wide coverage
- treat the result as support for recurrence, not for universality

### 2. Decision-Architecture Robustness

Purpose:
check whether the implementation-side reading survives across architecture choices rather than only inside one trained RL pipeline.

Default scope:

- RL main pipeline variants where relevant
- linear comparator families where they sharpen the same reading
- execution-layer and decision-translation variants remain in scope

### 3. Target-vs-Executed Disagreement Audit

Purpose:
check whether target-level and executed-path evaluation disagreement repeats across settings and continues to justify executed-path primacy.

Default reading:

- executed-path evaluation remains the primary object
- target-based evaluation remains diagnostic only
- repeated disagreement is evidence for the evaluation-object argument, not evidence for a stronger predictor

### 4. Minimal Domain-General Toy Example

Purpose:
provide a small non-portfolio illustration showing how forecast-side targets and realized executed actions can diverge once an implementation layer intervenes.

Default reading:

- the toy example must stay minimal
- it exists to clarify the paper's general evaluation logic
- it should not become a second theory paper or a broad empirical package

## Main Success Pattern

The main success pattern is intentionally narrow.

- `kappa = 0` stays near-flat, or at least clearly weaker than the positive-cost rows
- positive-cost rows show an executed-path advantage, or at minimum a more informative executed-path interpretation than the target-based one
- target-vs-executed disagreement repeats across settings rather than appearing only once
- turnover and cost diagnostics remain directionally compatible with the implementation-side reading

This package does **not** require identical Sharpe magnitudes across settings. It only requires recurrence of the same qualitative interpretation under conservative reading rules.

## Conservative Failure Handling

Failure handling must remain conservative.

- if results are mixed, keep them appendix-facing or support-only
- if results contradict the narrow success pattern, lower the claim rather than forcing a robustness statement
- do not revive the temporal compact pilot as a fallback robustness headline
- do not translate noisy or contradictory support into a stronger abstract claim

Working rules:

- mixed support means no promotion
- contradictory support means claim narrowing
- absence of recurrence is informative and must be written honestly

## Paper-Facing Wording Constraints

Use conservative paper-facing wording throughout this package.

Forbidden wording:

- `broad robustness`
- `universal`
- `universality`
- `stronger predictor`
- `predictive dominance`
- `benchmark dominance`
- `general theorem`

Allowed wording:

- `narrow recurrence`
- `support-only`
- `implementation-side compatibility`
- `executed-path interpretation`
- `evaluation-object disagreement`
- `recurs across tested fixed settings`

Writing rule:

When in doubt, prefer "directionally compatible support" over any wording that sounds like a broad robustness claim.

## Stable Output Locations

Future outputs should be organized under the following reserved axis-first paths:

- `paper/forecasting_workshop/generalization/outputs/multi_universe/`
- `paper/forecasting_workshop/generalization/outputs/architecture/`
- `paper/forecasting_workshop/generalization/outputs/target_vs_executed/`
- `paper/forecasting_workshop/generalization/outputs/toy_example/`

These locations are the stable paper-facing output roots for the new support package. For the current skeleton task, only the top-level `outputs/` directory is created.

## Run Ordering

Run the package in the following order:

1. multi-universe robustness
2. decision-architecture robustness
3. target-vs-executed disagreement audit
4. minimal domain-general toy example

Why this order:

- first establish recurrence across fixed universes
- then check whether the reading survives architecture changes
- then audit the evaluation-object disagreement directly across settings
- finally add the minimal toy example as a compact explanatory support layer

## Promotion Rule

Support-only is the default.

- no result from this package enters the main paper claim without an explicit promotion decision
- appendix use is the default home for mixed or merely compatible support
- main-text promotion is allowed only if a later explicit decision says so

Until such a promotion decision exists, this package should be read as support structure rather than as claim expansion.

## Legacy Handling

The following legacy material remains in place and is not modified by this plan:

- `multi_split_*`
- `cost_sweep_*`
- `generalization_decision_rules_v1.md`
- associated historical results, logs, and notes already present under `paper/forecasting_workshop/generalization/`

How to treat that material:

- it is historical or legacy temporal-support material
- it is not the current default generalization axis
- it should not be reused as the basis for a broad temporal-robustness claim

## Final Constraint

Do not modify the main paper yet. This run plan defines the support package only; it does not authorize changes to `paper_forecasting_workshop_v1.tex`, `paper_forecasting_workshop_v1.md`, or the current caption files.
