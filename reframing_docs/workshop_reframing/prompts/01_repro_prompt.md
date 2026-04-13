Read reframing_docs/AGENTS.md and the following source-of-truth documents:

- reframing_docs/workshop_reframing/00_claim_freeze.md
- reframing_docs/workshop_reframing/01_repro_checklist.md
- reframing_docs/workshop_reframing/08_execution_todo.md

Task:
Perform only the reproduction gate for the workshop-reframing project.

Requirements:

- Do not change configs, metrics, eta selection logic, kappa grid, or claim wording.
- Do not introduce new experiments.
- Do not edit the package specifications.
- Only inspect code, configs, scripts, and existing artifacts needed for reproduction.

Required checks:

1. validation-only eta selection
2. selected eta = 0.5
3. RL selected-point held-out pattern
4. accounting diagnostic pattern
5. dense friction sensitivity pattern
6. CC-TA-LBIP selected c = 3000 and kappa=0 collapse logic

Deliverables:

- reframing_docs/workshop_reframing/outputs/checks/validation_selection_check.md
- reframing_docs/workshop_reframing/outputs/checks/rl_selected_vs_eta1_repro.csv
- reframing_docs/workshop_reframing/outputs/checks/diagnostic_selected_eta_repro.csv
- reframing_docs/workshop_reframing/outputs/checks/dense_friction_repro.csv
- reframing_docs/workshop_reframing/outputs/checks/cctalibp_repro.csv
- reframing_docs/workshop_reframing/outputs/logs/repro_summary.md

If reproduction fails:

- write reframing_docs/workshop_reframing/outputs/logs/repro_failure_report.md
- stop immediately

Stop after this milestone.
