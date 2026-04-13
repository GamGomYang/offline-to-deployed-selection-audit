Read reframing_docs/AGENTS.md and inspect the current repository structure.

Observed project structure should be treated as the source of truth:

- reframing_docs/AGENTS.md
- reframing_docs/workshop_reframing/00_claim_freeze.md
- reframing_docs/workshop_reframing/01_repro_checklist.md
- reframing_docs/workshop_reframing/02_rl_main_package.md
- reframing_docs/workshop_reframing/03_accounting_gap.md
- reframing_docs/workshop_reframing/04_friction_curve.md
- reframing_docs/workshop_reframing/05_cctalibp_aux.md
- reframing_docs/workshop_reframing/06_same_forecast_table.md
- reframing_docs/workshop_reframing/07_paper_assembly.md
- reframing_docs/workshop_reframing/08_execution_todo.md
- reframing_docs/workshop_reframing/outputs/...
- reframing_docs/workshop_reframing/prompts/...

Task:
Perform only a preflight repair so the repository instructions match the actual structure.

Requirements:

- Do not change the project claim, metric hierarchy, selection logic, or evidence hierarchy.
- Do not run experiments.
- Do not create paper outputs.
- Only fix instruction-path consistency and prompt-file readiness.

Do all of the following:

1. Update AGENTS.md so all source-of-truth paths point to `reframing_docs/workshop_reframing/...`.
2. Check whether reframing_docs/workshop_reframing/prompts/\*.md are empty.
3. Create or populate missing prompt files with placeholder headings only if needed.
4. Write a short note describing exactly what was repaired.

Deliverables:

- updated reframing_docs/AGENTS.md
- reframing_docs/workshop_reframing/outputs/logs/preflight_fix_report.md

Validation:

- reframing_docs/AGENTS.md contains only reframing_docs/workshop_reframing/... paths for project docs
- no old `docs/`-prefixed project-document paths remain
- prompt directory paths are consistent with the actual repository structure

Stop after this milestone.
