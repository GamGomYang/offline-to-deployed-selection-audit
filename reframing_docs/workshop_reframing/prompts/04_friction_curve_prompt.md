Read AGENTS.md and the following source-of-truth documents:

- workshop_reframing/00_claim_freeze.md
- workshop_reframing/04_friction_curve.md
- workshop_reframing/08_execution_todo.md
- workshop_reframing/outputs/logs/repro_summary.md

Task:
Build only the dense friction sensitivity package.

Requirements:

- Treat this package as diagnostic only.
- Do not change the selected-point protocol.
- Do not introduce adaptive eta logic.
- Keep the selected-point line centered on eta=0.5.
- Keep the best-interior line clearly marked as diagnostic.

Deliverables:

- workshop_reframing/outputs/figures/fig_kappa_curve.pdf
- workshop_reframing/outputs/logs/friction_curve_paragraph.md
- workshop_reframing/outputs/logs/friction_curve_caption.md

Validation:

- selected eta remains 0.5 on the dense friction diagnostic
- selected-point gain increases with kappa or preserves the expected friction-sensitive direction
- best-interior gain is clearly labeled diagnostic-only

Stop after this milestone.
