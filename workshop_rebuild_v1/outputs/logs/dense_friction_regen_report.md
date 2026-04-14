# Dense Friction Regeneration Report

## Classification

This package was regenerated under the locked protocol.

- regeneration: `Yes`
- source-artifact reproduction: `No`

The original manifest-referenced source CSV was not recovered, so this package is explicitly a clean regeneration from preserved locked downstream inputs rather than a source-artifact reproduction.

## Regeneration Basis

Locked downstream inputs used for regeneration:

- `reframing_docs/workshop_reframing/04_friction_curve.md`
- `paper/paper.tex:497-499`
- `prl-dow30/scripts/build_submission_polish_figures.py:148-175`
- `repro/manifests/figure_manifest_v1_extensions.json`
- `reframing_docs/workshop_reframing/outputs/logs/dense_friction_artifact_recovery_report.md`
- `reframing_docs/workshop_reframing/outputs/logs/dense_friction_provenance_trace.md`

No experiments were rerun for this step.
No adaptive eta logic was introduced.
No old reference outputs were overwritten.

## Locked Protocol Check

- selected-point protocol remains fixed: `Pass`
- dense friction grid remains `{2e-4, 5e-4, 1e-3, 2e-3}`: `Pass`
- locked selected eta interpretation remains centered on `eta=0.5`: `Pass`
- best-interior line remains diagnostic-only: `Pass`
- package remains diagnostic-only and does not replace the RL main result: `Pass`

## Regenerated Dense-Friction Readout

| kappa | locked selected eta | selected-point Delta Sharpe | best-interior Delta Sharpe | per-kappa selected eta |
| --- | --- | --- | --- | --- |
| `2e-4` | `0.5` | `+0.0041` | `+0.0113` | `1.0` |
| `5e-4` | `0.5` | `+0.0105` | `+0.0230` | `1.0` |
| `1e-3` | `0.5` | `+0.0213` | `+0.0424` | `0.5` |
| `2e-3` | `0.5` | `+0.0409` | `+0.0761` | `0.2` |

Supporting turnover reductions used in the regenerated package:

- selected-point line: `0.01105`
- best-interior line: `0.02143`

## Direction Check

- friction-sensitive direction preserved: `Yes`
  - selected-point line increases monotonically from `+0.0041` to `+0.0409`
  - best-interior diagnostic line increases monotonically from `+0.0113` to `+0.0761`

- selected eta remains aligned with the locked interpretation: `Yes`
  - the global selected-point line remains fixed at `eta=0.5`
  - the moving per-kappa selector is shown only as a diagnostic companion, not as a new operating rule

## Interpretation Guardrails

- This package is diagnostic-only.
- It strengthens the cost-sensitive interpretation of the locked selected-point result.
- It does not replace the RL main package.
- It does not claim a new eta-selection rule.
- It does not upgrade the best-interior diagnostic line into the headline result.

## Output Placement

All regenerated outputs for this step were written only under `workshop_rebuild_v1/outputs/...`.
