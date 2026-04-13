Read AGENTS.md and the following source-of-truth documents:

- workshop_reframing/00_claim_freeze.md
- workshop_reframing/02_rl_main_package.md
- workshop_reframing/08_execution_todo.md
- workshop_reframing/outputs/logs/repro_summary.md

Task:
Build only the RL frozen-policy main package using reproduced artifacts.

Requirements:

- Use only reproduced artifacts from the reproduction gate.
- Keep eta=1.0 as the baseline.
- Keep eta=0.5 as the selected operating point.
- Keep kappa rows at {0, 5e-4, 1e-3}.
- Use executed-path net Sharpe as the primary metric.
- Do not promote diagnostic quantities into main headline metrics.
- Do not introduce new experiments or additional tuning.

Deliverables:

- workshop_reframing/outputs/tables/table_rl_main.csv
- workshop_reframing/outputs/tables/table_rl_main.tex
- workshop_reframing/outputs/logs/rl_main_numbers_check.md
- workshop_reframing/outputs/logs/rl_main_result_paragraph.md
- workshop_reframing/outputs/logs/rl_main_caption.md

Validation:

- positive-cost gains remain positive
- kappa=0 remains negligible or near-flat
- turnover reduction remains large
- wording stays implementation-side, not alpha-side

Stop after this milestone.
