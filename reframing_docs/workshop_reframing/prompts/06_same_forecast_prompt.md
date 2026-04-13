Read reframing_docs/AGENTS.md and the following source-of-truth documents:

- reframing_docs/workshop_reframing/00_claim_freeze.md
- reframing_docs/workshop_reframing/06_same_forecast_table.md
- reframing_docs/workshop_reframing/08_execution_todo.md
- reframing_docs/workshop_reframing/outputs/logs/repro_summary.md

Task:
Run only the same-forecast / different-decision-quality analysis for the auxiliary comparator.

Requirements:

- Use CC-TA-LBIP only.
- Keep the fitted ridge forecast map fixed.
- Compute forecast metrics before the decision layer.
- Compute decision metrics after the decision layer.
- Compare only c=0 vs c=3000.
- Do not force the result into the main paper if the evidence is weak.

Deliverables:

- reframing_docs/workshop_reframing/outputs/checks/forecast_outputs_eval.csv
- reframing_docs/workshop_reframing/outputs/tables/table_same_forecast_diff_decision.csv
- reframing_docs/workshop_reframing/outputs/tables/table_same_forecast_diff_decision.tex
- reframing_docs/workshop_reframing/outputs/logs/forecast_metric_analysis.md
- reframing_docs/workshop_reframing/outputs/logs/same_forecast_table_paragraph.md

Validation:

- explicitly state whether the package is strong enough for main text or should move to appendix
- forecast-side differences must be interpreted conservatively
- do not overclaim exact forecast identity unless directly verified

Stop after this milestone.
