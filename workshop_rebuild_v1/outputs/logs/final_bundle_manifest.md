# Final Bundle Manifest

## Bundle Identity

- bundle root: `workshop_rebuild_v1`
- bundle role: `single final submission bundle candidate for the workshop build`
- claim track: `canonical paper rebuild centered on validation-selected eta=0.5`
- bundle classification: `final canonical workshop bundle candidate`

## Canonical Writing Files

The following files are the canonical writing set for final submission use:

- `workshop_rebuild_v1/outputs/logs/abstract_workshop_v2.md`
- `workshop_rebuild_v1/outputs/logs/intro_workshop_v2.md`
- `workshop_rebuild_v1/outputs/logs/results_workshop_v2.md`
- `workshop_rebuild_v1/outputs/logs/discussion_limitations_v2.md`
- `workshop_rebuild_v1/outputs/logs/figure_table_order_v2.md`
- `workshop_rebuild_v1/outputs/logs/paper_workshop_outline_v2.md`

## Canonical Results Package

The following files constitute the canonical results package inside `workshop_rebuild_v1`:

### RL main package

- `workshop_rebuild_v1/outputs/tables/table_rl_main.csv`
- `workshop_rebuild_v1/outputs/tables/table_rl_main.tex`
- `workshop_rebuild_v1/outputs/logs/rl_main_result_paragraph.md`
- `workshop_rebuild_v1/outputs/logs/rl_main_caption.md`
- `workshop_rebuild_v1/outputs/logs/rl_main_numbers_check.md`

### Accounting diagnostic package

- `workshop_rebuild_v1/outputs/tables/diagnostic_gap_table.csv`
- `workshop_rebuild_v1/outputs/tables/diagnostic_gap_table.tex`
- `workshop_rebuild_v1/outputs/figures/fig_accounting_gap.pdf`
- `workshop_rebuild_v1/outputs/logs/accounting_gap_paragraph.md`
- `workshop_rebuild_v1/outputs/logs/accounting_gap_caption.md`

### Dense-friction canonical diagnostic bundle

- `workshop_rebuild_v1/outputs/checks/dense_friction_regen.csv`
- `workshop_rebuild_v1/outputs/figures/fig_kappa_curve.pdf`
- `workshop_rebuild_v1/outputs/logs/dense_friction_manifest.md`
- `workshop_rebuild_v1/outputs/logs/dense_friction_hashes.md`
- `workshop_rebuild_v1/outputs/logs/dense_friction_bundle_status.md`
- `workshop_rebuild_v1/outputs/logs/dense_friction_regen_report.md`
- `workshop_rebuild_v1/outputs/logs/friction_curve_paragraph.md`
- `workshop_rebuild_v1/outputs/logs/friction_curve_caption.md`

### CC-TA-LBIP auxiliary package

- `workshop_rebuild_v1/outputs/tables/table_cctalibp_aux.csv`
- `workshop_rebuild_v1/outputs/tables/table_cctalibp_aux.tex`
- `workshop_rebuild_v1/outputs/logs/cctalibp_aux_paragraph.md`
- `workshop_rebuild_v1/outputs/logs/cctalibp_aux_caption.md`
- `workshop_rebuild_v1/outputs/logs/cctalibp_aux_robustness_note.md`
- `workshop_rebuild_v1/outputs/tables/table_cctalibp_c_ablation.csv`
- `workshop_rebuild_v1/outputs/tables/table_cctalibp_c_ablation.tex`

### Same-forecast supporting package

- `workshop_rebuild_v1/outputs/checks/forecast_outputs_eval.csv`
- `workshop_rebuild_v1/outputs/tables/table_same_forecast_diff_decision_v2.csv`
- `workshop_rebuild_v1/outputs/tables/table_same_forecast_diff_decision_v2.tex`
- `workshop_rebuild_v1/outputs/logs/forecast_metric_analysis_v2.md`
- `workshop_rebuild_v1/outputs/logs/same_forecast_table_paragraph_v2.md`

## Superseded But Retained Files

The following are retained for traceability inside the same lane but are not the canonical submission-facing versions:

- all `*_v1.md` writing files
- `table_same_forecast_diff_decision.csv`
- `table_same_forecast_diff_decision.tex`
- `forecast_metric_analysis.md`
- `same_forecast_table_paragraph.md`
- `workshop_rebuild_v1/outputs/logs/same_forecast_run.log`
- `workshop_rebuild_v1/outputs/logs/same_forecast_run.done`

These retained files are non-blocking historical artifacts. They do not prevent `workshop_rebuild_v1` from serving as the final submission bundle.

## Old-Lane Dependence Check

Essential submission outputs now exist inside `workshop_rebuild_v1`.

- essential outputs missing from `workshop_rebuild_v1`: `No`
- reliance on old lane for essential submission artifacts: `No`
- reliance on old lane for background provenance context only: `Yes`

The only remaining old-lane references are historical provenance notes for dense-friction artifact recovery. Those are background context rather than essential submission outputs.
