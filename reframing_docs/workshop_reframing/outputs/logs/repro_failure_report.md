# Reproduction Gate Failure Report

## Outcome

The reproduction gate does not pass in the current repository state.

No experiments were run.
No configs, metrics, eta-selection logic, kappa grids, or claim wording were changed.
This report is based only on inspection of existing code, configs, scripts, and stored artifacts.

## Checks That Were Reproduced From Existing Artifacts

### 1. Validation-only eta selection

- Validation selection artifacts exist at `repro/rebuilds/paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.{md,json}`.
- The stored report locks:
  - `baseline_eta = 1.0`
  - `positive_kappas = [0.0005, 0.001]`
  - `relative_threshold = 0.95`
  - `selected_eta = 0.5`
- The selection code in `prl-dow30/scripts/select_eta_from_validation.py` implements the validation-only score on positive-cost kappas and chooses the largest qualifying eta.

### 2. Selected eta = 0.5

- Confirmed in:
  - `repro/rebuilds/paper_rebuild_20260324T065755Z/validation_eta/selection/validation_eta_selection.json`
  - `repro/rebuilds/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_stats_meta.json`
  - `repro/rebuilds/paper_rebuild_20260324T065755Z/paper_pack/protocol_lock.md`

### 3. RL selected-point held-out pattern

- `repro/rebuilds/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.csv` confirms:
  - `kappa=0.0`: median delta Sharpe `-0.0002460831` with near-flat effect
  - `kappa=0.0005`: median delta Sharpe `+0.0105028745`
  - `kappa=0.001`: median delta Sharpe `+0.0212508777`
  - selected median turnover `0.0109492402`
  - baseline median turnover `0.0219953601`

### 4. Accounting diagnostic pattern

- `repro/rebuilds/paper_rebuild_20260324T065755Z/paper_pack/diagnostics/diagnostic_selected_eta_v2.csv` confirms:
  - `TOtgt / TOexec = 2.0000000041`
  - tracking discrepancy `0.0025868409`
  - final equity gap rises with cost:
    - `kappa=0.0`: `0.0006724576`
    - `kappa=0.0005`: `0.0036891417`
    - `kappa=0.001`: `0.0073901495`

### 5. CC-TA-LBIP selected c = 3000 and kappa=0 collapse logic

- Validation selection exists at `repro/auxiliary_checks/cctalibp/validation_c_selection.{csv,json}` and selects `c = 3000`.
- Final comparison exists at `repro/auxiliary_checks/cctalibp/final_selected_vs_c0.csv` and shows:
  - `kappa=0.0`: selected `c=3000` and `c=0` are identical, `delta_sharpe = 0.0`
  - positive-cost rows improve strongly over `c=0`
- Feature/fit lock is consistent with the docs:
  - `prl-dow30/configs/exp/paper_u27_cctalibp_{validation,final}.yaml` fixes `ridge_alpha: 30.0`
  - `repro/auxiliary_checks/cctalibp/*fit_summary_seed0.json` records `obs_dim = 918`

## Failure Trigger

### 6. Dense friction sensitivity pattern could not be reproduced from repository artifacts

The repository metadata points to a dense-friction source artifact, but that source artifact is missing.

Manifest references:

- `repro/manifests/paper_artifact_manifest.json` declares Figure 5 source CSV as:
  - `outputs/extensions/v1_kappa_expansion/20260401T090500Z/analysis/kappa_expansion_summary.csv`
- `repro/manifests/figure_manifest_v1_extensions.json` repeats:
  - `/workspace/execution-aware-portfolio-rl/outputs/extensions/v1_kappa_expansion/20260401T090500Z/analysis/kappa_expansion_summary.csv`
  - `/workspace/execution-aware-portfolio-rl/paper_v1_extension_artifacts/figures/fig_kappa_benefit_curve.png`

Observed repository state:

- `outputs/extensions/v1_kappa_expansion/.../analysis/kappa_expansion_summary.csv` is missing
- `paper_v1_extension_artifacts/figures/fig_kappa_benefit_curve.png` is missing
- The extension manifest and scripts exist, but the referenced dense-friction result artifacts do not

Because the required dense-friction evidence is not present as an inspectable artifact, the following required workshop-gate check cannot be completed without running the extension experiment:

- dense friction sensitivity pattern

Under `reframing_docs/AGENTS.md`, reproduction failure requires stopping rather than patching around missing evidence.

## Stop Reason

This milestone stops here because one required reproduction check is not auditable from existing repository artifacts.

To pass the gate later, the dense-friction source artifacts referenced by the manifests must be present in the repository, after which the check can be repeated without changing protocol rules.
