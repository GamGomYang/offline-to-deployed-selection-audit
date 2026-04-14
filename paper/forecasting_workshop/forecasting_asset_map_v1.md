# Forecasting Asset Map v1

This inventory maps the current workshop-build artifacts into `Main text`, `Appendix`, or `Internal only` for the forecasting workshop submission. The paper remains a forecasting-to-decision evaluation paper; portfolio decisions are only the concrete case study.

## Recommended Main-Text Asset Budget

Final approved main-text asset list for a short forecasting workshop paper:

1. conceptual forecast-to-decision figure
2. `workshop_rebuild_v1/outputs/tables/table_rl_main.tex`
3. `workshop_rebuild_v1/outputs/figures/fig_accounting_gap.pdf`

Reserved but not yet present in the inspected outputs:

- conceptual forecast-to-decision figure, if created later

Main-text guidance:

- Keep the RL selected-point table as the only main empirical result.
- Keep the accounting figure as the mechanism explanation for why executed-path evaluation is primary.
- Keep the same-forecast package in the appendix as conservative forecasting relevance support.
- Move dense-friction, CC-TA-LBIP, `c`-ablation, provenance, and audit materials out of the main paper.

## Inventory

| File path | Role | Classification | One-line reason |
| --- | --- | --- | --- |
| `workshop_rebuild_v1/outputs/logs/paper_workshop_submission_v1.md` | Assembled manuscript draft | Main text | This is the current single-file workshop draft and the best reference for the main paper narrative. |
| `workshop_rebuild_v1/outputs/tables/table_rl_main.tex` | RL selected-point main table | Main text | This is the only main empirical result and must stay central in the workshop paper. |
| `workshop_rebuild_v1/outputs/figures/fig_accounting_gap.pdf` | Accounting diagnostic figure | Main text | This is the clearest mechanism figure for why executed-path evaluation is primary. |
| `workshop_rebuild_v1/outputs/tables/table_same_forecast_diff_decision_v2.tex` | Same-forecast support table | Appendix | This is useful forecasting support, but it is safer in the appendix because only metric-level similarity is established. |
| `workshop_rebuild_v1/outputs/appendix/appendix_outline_v1.md` | Appendix structure note | Appendix | This organizes supporting materials without competing with the main claim. |
| `workshop_rebuild_v1/outputs/appendix/provenance_note_v1.md` | Dense-friction provenance note | Appendix | Provenance is necessary to keep honest language but should not appear in the main paper. |
| `workshop_rebuild_v1/outputs/appendix/supporting_tables_and_notes_v1.md` | Appendix support note | Appendix | This is the correct home for supporting evidence that should not dominate the workshop framing. |
| `workshop_rebuild_v1/outputs/figures/fig_kappa_curve.pdf` | Dense-friction figure | Appendix | Dense-friction is useful support but too provenance-heavy and secondary for the 4-page main text. |
| `workshop_rebuild_v1/outputs/logs/cctalibp_aux_robustness_note.md` | Comparator robustness note | Appendix | This is a reviewer-facing robustness note for an auxiliary comparator, not a main-paper result. |
| `workshop_rebuild_v1/outputs/tables/diagnostic_gap_table.tex` | Accounting companion table | Appendix | The accounting figure is enough for the main paper, so the companion table belongs in the appendix. |
| `workshop_rebuild_v1/outputs/tables/table_cctalibp_aux.tex` | CC-TA-LBIP auxiliary table | Appendix | The comparator must remain auxiliary and should not look like a second main result. |
| `workshop_rebuild_v1/outputs/tables/table_cctalibp_c_ablation.tex` | `c`-ablation table | Appendix | The narrow `c`-ablation is robustness detail that should stay out of the main paper. |
| `workshop_rebuild_v1/outputs/logs/abstract_workshop_v1.md` | Superseded abstract fragment | Internal only | Historical writing fragment retained only for traceability. |
| `workshop_rebuild_v1/outputs/logs/abstract_workshop_v2.md` | Current abstract source fragment | Internal only | Source fragment is useful for editing but should not be cited as a standalone asset once the draft is assembled. |
| `workshop_rebuild_v1/outputs/logs/accounting_gap_caption.md` | Accounting caption source | Internal only | Caption working file is part of assembly, not a paper-facing artifact. |
| `workshop_rebuild_v1/outputs/logs/accounting_gap_paragraph.md` | Accounting paragraph source | Internal only | Source paragraph is already integrated into the draft and should not appear separately. |
| `workshop_rebuild_v1/outputs/logs/cctalibp_aux_caption.md` | Comparator caption source | Internal only | Caption working file is assembly support only. |
| `workshop_rebuild_v1/outputs/logs/cctalibp_aux_paragraph.md` | Comparator paragraph source | Internal only | Paragraph source is for assembly and should not be promoted as a separate asset. |
| `workshop_rebuild_v1/outputs/logs/dense_friction_bundle_status.md` | Dense-friction archive status | Internal only | Archival status is provenance bookkeeping rather than paper content. |
| `workshop_rebuild_v1/outputs/logs/dense_friction_hashes.md` | Dense-friction hashes | Internal only | Hashes are audit metadata only. |
| `workshop_rebuild_v1/outputs/logs/dense_friction_manifest.md` | Dense-friction manifest | Internal only | Full regeneration manifest is too detailed for the paper and belongs to internal provenance records. |
| `workshop_rebuild_v1/outputs/logs/dense_friction_regen_report.md` | Dense-friction regeneration report | Internal only | This is rebuild process documentation rather than submission content. |
| `workshop_rebuild_v1/outputs/logs/discussion_limitations_v1.md` | Superseded discussion fragment | Internal only | Historical draft fragment retained only for traceability. |
| `workshop_rebuild_v1/outputs/logs/discussion_limitations_v2.md` | Current discussion source fragment | Internal only | Source fragment is already represented in the assembled draft. |
| `workshop_rebuild_v1/outputs/logs/figure_table_callouts_v1.md` | Callout assembly note | Internal only | This is manuscript-assembly support only. |
| `workshop_rebuild_v1/outputs/logs/figure_table_integration_notes.md` | Figure/table integration note | Internal only | Internal integration memo, not paper content. |
| `workshop_rebuild_v1/outputs/logs/figure_table_order.md` | Superseded order note | Internal only | Historical assembly memo retained only for traceability. |
| `workshop_rebuild_v1/outputs/logs/figure_table_order_v2.md` | Current order note | Internal only | Internal ordering memo for assembly, not a submission asset. |
| `workshop_rebuild_v1/outputs/logs/final_audit_report.md` | Final audit report | Internal only | Audit record must stay out of the paper. |
| `workshop_rebuild_v1/outputs/logs/final_bundle_manifest.md` | Final bundle manifest | Internal only | Bundle bookkeeping belongs to internal submission control. |
| `workshop_rebuild_v1/outputs/logs/final_captions_v1.md` | Consolidated captions | Internal only | Caption assembly file is not a standalone paper artifact. |
| `workshop_rebuild_v1/outputs/logs/final_readiness_report.md` | Final readiness report | Internal only | Readiness status is internal submission metadata. |
| `workshop_rebuild_v1/outputs/logs/final_submission_guardrails.md` | Guardrail memo | Internal only | Editorial guardrails should inform drafting but not appear in the paper. |
| `workshop_rebuild_v1/outputs/logs/forecast_metric_analysis.md` | Superseded forecast-analysis note | Internal only | Historical analysis note retained only for traceability. |
| `workshop_rebuild_v1/outputs/logs/forecast_metric_analysis_v2.md` | Current forecast-analysis note | Internal only | The note supports the same-forecast table but should not be cited as a standalone main-paper asset. |
| `workshop_rebuild_v1/outputs/logs/forecasting_asset_map_v1.md` | Prior asset map | Internal only | Previous routing memo is superseded by this standard-path version. |
| `workshop_rebuild_v1/outputs/logs/forecasting_claim_freeze_v1.md` | Prior claim-freeze memo | Internal only | Historical claim-freeze note retained for traceability after moving to the standard paper path. |
| `workshop_rebuild_v1/outputs/logs/forecasting_reframing_memo_v1.md` | Reframing memo | Internal only | Useful for internal positioning but not a paper or appendix artifact. |
| `workshop_rebuild_v1/outputs/logs/friction_curve_caption.md` | Dense-friction caption source | Internal only | Caption source is assembly support for appendix-only material. |
| `workshop_rebuild_v1/outputs/logs/friction_curve_paragraph.md` | Dense-friction paragraph source | Internal only | Paragraph source is assembly support for appendix-only material. |
| `workshop_rebuild_v1/outputs/logs/intro_workshop_v1.md` | Superseded intro fragment | Internal only | Historical draft fragment retained only for traceability. |
| `workshop_rebuild_v1/outputs/logs/intro_workshop_v2.md` | Current intro source fragment | Internal only | Source fragment is already represented in the assembled draft. |
| `workshop_rebuild_v1/outputs/logs/paper_workshop_outline.md` | Superseded outline | Internal only | Historical planning memo only. |
| `workshop_rebuild_v1/outputs/logs/paper_workshop_outline_v2.md` | Current outline | Internal only | Internal assembly note rather than submission content. |
| `workshop_rebuild_v1/outputs/logs/pre_submission_checklist_v1.md` | Pre-submission checklist | Internal only | Internal audit checklist belongs outside the paper. |
| `workshop_rebuild_v1/outputs/logs/pre_submission_ready_or_not_v1.md` | Pre-submission readiness note | Internal only | Internal release-control memo only. |
| `workshop_rebuild_v1/outputs/logs/pre_submission_risks_v1.md` | Pre-submission risks note | Internal only | Internal risk memo should not appear in the paper. |
| `workshop_rebuild_v1/outputs/logs/preflight_setup_report.md` | Preflight setup report | Internal only | Build-lane setup record only. |
| `workshop_rebuild_v1/outputs/logs/professor_review_checklist_v1.md` | Supervisor review checklist | Internal only | Review aid is outside the submission package. |
| `workshop_rebuild_v1/outputs/logs/professor_review_cover_note_v1.md` | Supervisor cover note | Internal only | Cover note is a review aid, not paper content. |
| `workshop_rebuild_v1/outputs/logs/professor_review_summary_v1.md` | Supervisor summary | Internal only | Review summary is not part of the submission package. |
| `workshop_rebuild_v1/outputs/logs/results_workshop_v1.md` | Superseded results fragment | Internal only | Historical results draft retained only for traceability. |
| `workshop_rebuild_v1/outputs/logs/results_workshop_v2.md` | Current results source fragment | Internal only | Source fragment is already integrated into the current draft. |
| `workshop_rebuild_v1/outputs/logs/rl_main_caption.md` | RL caption source | Internal only | Caption source is assembly support only. |
| `workshop_rebuild_v1/outputs/logs/rl_main_numbers_check.md` | RL number check | Internal only | Numeric verification note belongs to internal audit records. |
| `workshop_rebuild_v1/outputs/logs/rl_main_result_paragraph.md` | RL paragraph source | Internal only | Paragraph source is already represented in the assembled draft. |
| `workshop_rebuild_v1/outputs/logs/run_manifest.md` | Lane run manifest | Internal only | Build-lane bookkeeping only. |
| `workshop_rebuild_v1/outputs/logs/same_forecast_run.done` | Run completion marker | Internal only | Execution marker has no paper role. |
| `workshop_rebuild_v1/outputs/logs/same_forecast_run.log` | Same-forecast run log | Internal only | Run log is process trace only. |
| `workshop_rebuild_v1/outputs/logs/same_forecast_table_paragraph.md` | Superseded same-forecast paragraph | Internal only | Historical text fragment retained only for traceability. |
| `workshop_rebuild_v1/outputs/logs/same_forecast_table_paragraph_v2.md` | Current same-forecast paragraph source | Internal only | Paragraph source is already represented in the table and manuscript. |
| `workshop_rebuild_v1/outputs/logs/status.md` | Lane status note | Internal only | Internal process tracking only. |
| `workshop_rebuild_v1/outputs/tables/diagnostic_gap_table.csv` | Accounting backing data | Internal only | Data backing file for the appendix accounting table. |
| `workshop_rebuild_v1/outputs/tables/table_cctalibp_aux.csv` | Comparator backing data | Internal only | Data backing file for appendix comparator material. |
| `workshop_rebuild_v1/outputs/tables/table_cctalibp_c_ablation.csv` | `c`-ablation backing data | Internal only | Data backing file for appendix robustness material. |
| `workshop_rebuild_v1/outputs/tables/table_rl_main.csv` | RL backing data | Internal only | Data backing file for the approved main-text RL table. |
| `workshop_rebuild_v1/outputs/tables/table_same_forecast_diff_decision.csv` | Superseded same-forecast table | Internal only | Historical typesetting/data pair retained only for traceability. |
| `workshop_rebuild_v1/outputs/tables/table_same_forecast_diff_decision.tex` | Superseded same-forecast table | Internal only | Historical typesetting/data pair retained only for traceability. |
| `workshop_rebuild_v1/outputs/tables/table_same_forecast_diff_decision_v2.csv` | Same-forecast backing data | Internal only | Data backing file for the approved main-text support table. |

## Provenance and Audit Routing

The following materials are not approved for the main paper:

- provenance note: appendix-only
- dense-friction manifest, hashes, bundle status, regeneration report: internal only
- final audit, final readiness, pre-submission checklist, risks, ready-or-not: internal only
- guardrail and claim-freeze memos: internal only

## Final Approved Main-Text Asset List

Approved main-text figures/tables for the forecasting workshop paper:

1. `workshop_rebuild_v1/outputs/tables/table_rl_main.tex`
2. `workshop_rebuild_v1/outputs/figures/fig_accounting_gap.pdf`

Everything else should remain in the appendix or stay internal only.

The conceptual forecast-to-decision figure should occupy the remaining main-text slot before any same-forecast, dense-friction, or comparator asset is promoted into the core paper.
