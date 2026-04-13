Read AGENTS.md and inspect the current repository structure.

Observed project structure should be treated as the source of truth:

- AGENTS.md
- workshop_reframing/00_claim_freeze.md
- workshop_reframing/01_repro_checklist.md
- workshop_reframing/02_rl_main_package.md
- workshop_reframing/03_accounting_gap.md
- workshop_reframing/04_friction_curve.md
- workshop_reframing/05_cctalibp_aux.md
- workshop_reframing/06_same_forecast_table.md
- workshop_reframing/07_paper_assembly.md
- workshop_reframing/08_execution_todo.md
- workshop_reframing/outputs/...
- workshop_reframing/prompts/...

Task:
Perform only a preflight repair so the repository instructions match the actual structure.

Requirements:

- Do not change the project claim, metric hierarchy, selection logic, or evidence hierarchy.
- Do not run experiments.
- Do not create paper outputs.
- Only fix instruction-path consistency and prompt-file readiness.

Do all of the following:

1. Update AGENTS.md so all source-of-truth paths point to `workshop_reframing/...`.
2. Check whether workshop_reframing/prompts/\*.md are empty.
3. Create or populate missing prompt files with placeholder headings only if needed.
4. Write a short note describing exactly what was repaired.

Deliverables:

- updated AGENTS.md
- workshop_reframing/outputs/logs/preflight_fix_report.md

Validation:

- AGENTS.md contains only workshop_reframing/... paths for project docs
- no old `docs/`-prefixed project-document paths remain
- prompt directory paths are consistent with the actual repository structure

Stop after this milestone.
