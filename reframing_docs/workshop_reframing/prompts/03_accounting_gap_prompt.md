Read AGENTS.md and the following source-of-truth documents:

- workshop_reframing/00_claim_freeze.md
- workshop_reframing/03_accounting_gap.md
- workshop_reframing/08_execution_todo.md
- workshop_reframing/outputs/logs/repro_summary.md

Task:
Build only the target-vs-executed accounting-gap package.

Requirements:

- Use the validation-selected operating point eta=0.5.
- Treat target-path quantities as diagnostic only.
- Do not replace the RL main result.
- Do not modify accounting definitions.
- Reconstruct traces only from valid source artifacts.

Deliverables:

- workshop_reframing/outputs/tables/diagnostic_gap_table.csv
- workshop_reframing/outputs/tables/diagnostic_gap_table.tex
- workshop_reframing/outputs/figures/fig_accounting_gap.pdf
- workshop_reframing/outputs/logs/accounting_gap_paragraph.md
- workshop_reframing/outputs/logs/accounting_gap_caption.md

Validation:

- TOtgt/TOexec remains meaningfully above 1 and close to the expected pattern
- tracking remains small but nonzero
- final path gap increases with kappa
- target-path stays diagnostic-only in all wording

Stop after this milestone.
