# Baseline Protocol

This repository's frozen paper baseline is the validation-first frozen-policy execution study anchored to [`paper.tex`](/workspace/execution-aware-portfolio-rl/paper.tex) and the canonical run root [`paper_rebuild_20260324T065755Z`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z). The authoritative machine-readable source of truth is [`manifests/baselines/paper_v3_frozen.json`](/workspace/execution-aware-portfolio-rl/manifests/baselines/paper_v3_frozen.json). If any ad hoc note disagrees with that manifest, the manifest wins.

## Frozen Scope

- Baseline ID: `paper_v3_frozen_control_eta_20260324`
- Manuscript scope: frozen-policy execution study only
- Canonical paper file: [`paper.tex`](/workspace/execution-aware-portfolio-rl/paper.tex)
- Canonical figures used by the paper:
  - [`fig_frontier.png`](/workspace/execution-aware-portfolio-rl/fig_frontier.png)
  - [`fig_misalignment.png`](/workspace/execution-aware-portfolio-rl/fig_misalignment.png)
  - [`fig_seed_scatter.png`](/workspace/execution-aware-portfolio-rl/fig_seed_scatter.png)
- Canonical paper artifact manifest:
  - [`paper_artifact_manifest.json`](/workspace/execution-aware-portfolio-rl/paper_artifact_manifest.json)
- Frozen protocol bundle:
  - [`frozen_protocol/paper_v3/current_config.yaml`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/current_config.yaml)
  - [`frozen_protocol/paper_v3/snapshot_control.yaml`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/snapshot_control.yaml)
  - [`frozen_protocol/paper_v3/validation_eta.yaml`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/validation_eta.yaml)
  - [`frozen_protocol/paper_v3/final_eta.yaml`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/final_eta.yaml)
  - [`frozen_protocol/paper_v3/selected_signals_snapshot.json`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/selected_signals_snapshot.json)
  - [`frozen_protocol/paper_v3/split_definition.json`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/split_definition.json)
  - [`frozen_protocol/paper_v3/seed_list.txt`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/seed_list.txt)
  - [`frozen_protocol/paper_v3/metric_definition.md`](/workspace/execution-aware-portfolio-rl/frozen_protocol/paper_v3/metric_definition.md)

## Locked Definitions

- Train / validation / test split:
  - train: `2010-01-01` to `2021-12-31`
  - validation: `2022-01-01` to `2023-12-31`
  - test: `2024-01-01` to `2025-12-31`
  - realized validation window after 30-day rolling features: `2022-02-15` to `2023-12-29`
  - realized test window after 30-day rolling features: `2024-02-14` to `2025-12-31`
- Eta selection rule:
  - fixed grid: `1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02`
  - baseline arm: `eta=1.0`
  - score: mean over positive-cost kappas of seed-median `sharpe_net_lin`
  - positive-cost kappas: `0.0005, 0.001`
  - threshold: `0.95 * best_score` when the best score is positive, `best_score / 0.95` when negative, `0` when zero
  - selection: choose the largest qualifying `eta`
  - locked selected operating point: `eta=0.5`
- Kappa grid: `0.0, 0.0005, 0.001`
- Seed list: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Feature set:
  - 30-day rolling returns
  - 30-day volatility features
  - previous executed weights
  - signal state enabled
  - selected signals: `reversal_5d`, `short_term_reversal`
- Core metric definitions:
  - primary metric: `sharpe_net_lin`
  - core path: executed portfolio only
  - cost definition: `kappa * executed_turnover`
  - Sharpe annualization: `sqrt(252)`
  - risk-free rate: `0`
  - core report set: net Sharpe, CAGR, max drawdown, executed turnover, realized cost

## Reproduction

- Fast baseline reproduction from frozen policy weights:

```bash
./reproduce_main_results.sh --mode frozen-models --verify
```

- Full baseline reproduction including training:

```bash
./reproduce_main_results.sh --mode full --verify
```

The fast mode reuses the locked final policy weights and regenerates the validation frontier, selection object, held-out tables, baseline tables, diagnostics, and paper pack in one command. The full mode retrains all 10 seeds before rerunning the same downstream protocol.

After either command finishes, the run root also contains an explicit paper-facing export under `paper_artifacts/` with:

- `tables/table_1.csv` style exports for the four manuscript tables
- `figures/figure_1.png` style exports for the three manuscript figures
- `manifest.json` describing the mapping from manuscript object to source file

This makes the paper-facing artifact set auditable without having to infer table/figure provenance from the larger `paper_pack/` tree.

## Selection-Rule Defense

- Selection-rule defense outputs live under:
  - [`paper_rebuild_20260324T065755Z/paper_pack/selection_defense`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/selection_defense)
- Canonical files:
  - [`selection_rule_eta_summary.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/selection_defense/selection_rule_eta_summary.csv)
  - [`selection_rule_threshold_sensitivity.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/selection_defense/selection_rule_threshold_sensitivity.csv)
  - [`selection_rule_defense.md`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/selection_defense/selection_rule_defense.md)
- Locked sensitivity check:
  - threshold `0.90` selects `eta=1.0`
  - threshold `0.95` selects `eta=0.5`
  - threshold `0.975` also selects `eta=0.5`
- Interpretation:
  - the locked choice is not the raw-best frontier point
  - `eta=0.5` retains `97.66%` of the raw-best positive-cost validation score
  - `eta=0.5` roughly halves average positive-cost executed turnover relative to `eta=1.0`
  - the rule therefore acts as a conservative interior-point selector rather than a post-hoc smallest-eta chooser

## Separation Policy

- Canonical frozen baseline outputs must remain under the locked run root [`paper_rebuild_20260324T065755Z`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z).
- New baseline reproductions must write to `outputs/reproductions/baseline/<baseline_id>/...`.
- New extension experiments must write to `outputs/extensions/<experiment_id>/...`.
- Every extension manifest must live under [`manifests/extensions`](/workspace/execution-aware-portfolio-rl/manifests/extensions) and must declare `parent_baseline_id: paper_v3_frozen_control_eta_20260324`.
- No extension run may write into the locked baseline root or into another experiment's output root.

## Canonical Paper-Facing Artifacts

- Table 1 (`tab:validation_selection`): [`paper_rebuild_20260324T065755Z/paper_pack/tables/validation_selection.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/tables/validation_selection.csv)
- Table 2 (`tab:selected_vs_eta1`): [`paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_eta1.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_eta1.csv)
- Table 3 (`tab:selected_vs_external`): [`paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_external_baselines.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_external_baselines.csv)
- Table 4 (`tab:diagnostic_v2`): [`paper_rebuild_20260324T065755Z/paper_pack/tables/diagnostic_selected_eta.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/tables/diagnostic_selected_eta.csv)
- Figure 1 (`fig:misalign`): [`paper_rebuild_20260324T065755Z/paper_pack/figures/fig_selected_trace.png`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_selected_trace.png)
- Figure 2 (`fig:frontier`): [`paper_rebuild_20260324T065755Z/paper_pack/figures/fig_validation_frontier.png`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_validation_frontier.png)
- Figure 3 (`fig:seedscatter`): [`paper_rebuild_20260324T065755Z/paper_pack/figures/fig_seed_scatter.png`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_seed_scatter.png)

- Validation selection object: [`paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.json`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.json)
- Held-out selected-vs-baseline stats: [`paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.csv)
- Held-out selected-vs-external table: [`paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_external_baselines.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_external_baselines.csv)
- Diagnostic table: [`paper_rebuild_20260324T065755Z/paper_pack/diagnostics/diagnostic_selected_eta_v2.csv`](/workspace/execution-aware-portfolio-rl/paper_rebuild_20260324T065755Z/paper_pack/diagnostics/diagnostic_selected_eta_v2.csv)
- Figure provenance: [`figure_manifest.json`](/workspace/execution-aware-portfolio-rl/figure_manifest.json)
