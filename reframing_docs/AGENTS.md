# AGENTS.md

## Project Identity

This repository contains a workshop-reframing project built from an existing portfolio-RL paper.
The goal is not to invent a new RL method.
The goal is to package the strongest existing evidence into a focused workshop paper about forecast-to-execution interfaces under transaction costs.

## Path Convention

All project-document paths below are relative to the repository root.
Use the `workshop_reframing/...` form for project documents and outputs.

## Primary Framing

Use this sentence as the highest-level framing for all work:

**In cost-sensitive portfolio decision systems, holding the predictive signal fixed, the forecast-to-execution interface materially changes realized decision quality.**

Do not widen the scope beyond this framing unless explicitly instructed by the human.

## Hard Constraints

1. Do not manufacture a desired conclusion.
2. Do not change claim scope based on favorable or unfavorable test outcomes.
3. Do not change the eta selection rule after looking at held-out results.
4. Do not change the main metric after looking at results.
5. Do not promote diagnostic quantities into primary metrics.
6. Do not present auxiliary comparator evidence as the main identification result.
7. Do not rewrite the project into a "new RL algorithm" paper.
8. Do not make benchmark-dominance claims unless explicitly supported and requested.

## Allowed Claim Scope

Allowed:

- execution-aware interface
- realized-path evaluation
- cost-sensitive decision quality
- implementation / translation / accounting interpretation
- supporting evidence from a linear forecast-plus-optimization comparator

Not allowed:

- general forecasting-systems theorem
- stronger predictor claim
- better alpha generation claim
- SOTA claim
- production-ready trading system claim
- passive-benchmark dominance claim

## Evidence Hierarchy

Always preserve this evidence order:

### Tier 1: core evidence

1. RL frozen-policy selected-point result
2. target-vs-executed accounting gap
3. dense friction sensitivity

### Tier 2: supporting evidence

4. CC-TA-LBIP auxiliary comparator

### Tier 3: optional supporting analysis

5. same-forecast / different-decision-quality table
6. rolling-window robustness
7. retraining checks
8. second-universe checks

Do not invert this hierarchy.

## Source-of-Truth Documents

Read and follow these files when relevant:

- `workshop_reframing/00_claim_freeze.md`
- `workshop_reframing/01_repro_checklist.md`
- `workshop_reframing/02_rl_main_package.md`
- `workshop_reframing/03_accounting_gap.md`
- `workshop_reframing/04_friction_curve.md`
- `workshop_reframing/05_cctalibp_aux.md`
- `workshop_reframing/06_same_forecast_table.md`
- `workshop_reframing/07_paper_assembly.md`
- `workshop_reframing/08_execution_todo.md`

If a task conflicts with one of these documents, stop and surface the conflict instead of improvising.

## Reproduction-First Rule

Before any new analysis, pass the reproduction gate in:

- `workshop_reframing/01_repro_checklist.md`

Do not start new experiments until the following are reproduced or explicitly waived by the human:

- selected eta = 0.5
- positive-cost held-out gains remain positive
- kappa=0 effect remains negligible
- target-vs-executed diagnostic pattern remains intact
- selected c = 3000 for the auxiliary comparator

If reproduction fails, stop and write a failure report.
Do not continue to later milestones.

## Main Metric Rules

Primary metric:

- executed-path net Sharpe

Main supporting metrics:

- executed turnover
- realized cost
- paired-median delta Sharpe

Diagnostic-only quantities:

- target-path return
- target turnover
- tracking discrepancy
- target-vs-executed gaps

Never promote diagnostic-only quantities to the headline result.

## Package-by-Package Execution Order

Work in this order unless the human explicitly changes it:

1. reproduction gate
2. RL main package
3. accounting-gap package
4. friction-curve package
5. auxiliary CC-TA-LBIP package
6. same-forecast table
7. paper assembly

Do not skip forward when an earlier package is unresolved.

## Writing Rules

When editing prose:

- keep the workshop paper narrow
- prefer precise, non-hyped language
- describe gains as implementation-side unless evidence clearly supports stronger language
- keep CC-TA-LBIP auxiliary
- keep target-path metrics diagnostic-only
- mention limitations honestly and briefly

Preferred phrases:

- realized decision quality
- executed-path evaluation
- forecast-to-execution interface
- implementation-side gain
- cost-sensitive decision system
- auxiliary comparator
- supporting evidence

Avoid:

- better predictor
- superior alpha
- universally applicable
- SOTA
- dominates baselines
- generally proves

## Output Discipline

For each task:

1. Read the relevant spec files.
2. State the exact files you will produce.
3. Make only the requested changes.
4. Run the minimum validation needed.
5. Save outputs in the expected folder.
6. Write a short status note describing:
   - what was produced
   - whether the package passed
   - any deviations or ambiguities

Do not silently change configs, naming schemes, metrics, or file locations.

## File Output Conventions

Use:

- `workshop_reframing/outputs/tables/`
- `workshop_reframing/outputs/figures/`
- `workshop_reframing/outputs/checks/`
- `workshop_reframing/outputs/logs/`

Tables:

- `table_*.csv`
- `table_*.tex`

Figures:

- `fig_*.pdf`
- `fig_*.png`

Notes:

- `*_paragraph.md`
- `*_caption.md`
- `*_notes.md`
- `*_repro.md`

## Failure Behavior

If any of the following occurs, stop and report instead of patching around it:

- selected eta is not reproduced
- positive-cost sign pattern breaks
- kappa=0 row changes interpretation
- target-vs-executed diagnostic structure disappears
- auxiliary comparator no longer preserves its intended setup
- source-of-truth documents conflict with repository reality

When stopping, write:

- the failing condition
- likely causes
- files inspected
- what must be clarified before continuing

## Human Interaction Rule

If the human asks for one milestone, do only that milestone.
Do not opportunistically expand scope.
Do not "improve" the framing without permission.
Do not add new experiments unless the current spec explicitly calls for them.

## Default Task Style

For complex work:

- plan briefly
- execute narrowly
- validate
- update status/log files
- stop at the requested milestone

## End Condition

A task is complete only when:

- the requested artifacts exist,
- the relevant validation has been run,
- the short status note has been written,
- and the outputs match the active spec documents.
