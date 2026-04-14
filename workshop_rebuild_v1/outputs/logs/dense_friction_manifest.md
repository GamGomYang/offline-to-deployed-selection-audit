# Dense Friction Manifest

## Bundle Identity

- package name: `dense_friction`
- bundle role: `diagnostic-only dense-friction package for the workshop build`
- bundle lane: `workshop_rebuild_v1`
- canonical bundle status: `canonical dense-friction bundle for the workshop build`

## Classification

This dense-friction package was **regenerated under the locked protocol**.

- regeneration: `Yes`
- source-artifact reproduction: `No`

This canonicalization step does not upgrade the claim beyond that classification. The original manifest-referenced source CSV was not recovered, so this bundle is archived as the final regenerated canonical bundle for the workshop build, **not** as a source-artifact reproduction bundle.

## Locked Protocol Record

- claim wording changed: `No`
- eta selection logic changed: `No`
- kappa grid changed: `No`
- broad experiments rerun for canonicalization: `No`
- old reference outputs overwritten: `No`
- package remains diagnostic-only: `Yes`

Locked interpretation carried into this bundle:

- global selected-point interpretation remains centered on `eta=0.5`
- dense diagnostic grid remains `kappa={2e-4, 5e-4, 1e-3, 2e-3}`
- best-interior line remains `diagnostic_only`
- the bundle strengthens the cost-sensitive interpretation of the RL selected-point result without replacing that main result

## Canonical Bundle Contents

Primary archived source files for this canonical bundle:

- `workshop_rebuild_v1/outputs/checks/dense_friction_regen.csv`
- `workshop_rebuild_v1/outputs/figures/fig_kappa_curve.pdf`
- `workshop_rebuild_v1/outputs/logs/dense_friction_regen_report.md`
- `workshop_rebuild_v1/outputs/logs/friction_curve_paragraph.md`
- `workshop_rebuild_v1/outputs/logs/friction_curve_caption.md`

## Provenance Chain

The canonical regenerated bundle is based on the following provenance chain:

1. `reframing_docs/workshop_reframing/04_friction_curve.md` locked the dense-friction package specification and reference values.
2. `reframing_docs/workshop_reframing/outputs/logs/dense_friction_artifact_recovery_report.md` established that the original source CSV was unavailable.
3. `reframing_docs/workshop_reframing/outputs/logs/dense_friction_provenance_trace.md` recovered the downstream figure/manuscript provenance chain.
4. `paper/paper.tex:497-499`, `prl-dow30/scripts/build_submission_polish_figures.py:148-175`, and `repro/manifests/figure_manifest_v1_extensions.json` preserved the locked downstream values used in regeneration.
5. `workshop_rebuild_v1/outputs/checks/dense_friction_regen.csv` and `workshop_rebuild_v1/outputs/figures/fig_kappa_curve.pdf` were generated as the clean rebuild lane outputs from those preserved downstream inputs.
6. This manifest canonicalizes those regenerated outputs as the final archived source bundle for the workshop build.

## Readout Snapshot

| kappa | locked selected eta | selected-point Delta Sharpe | best-interior Delta Sharpe | line role |
| --- | --- | --- | --- | --- |
| `2e-4` | `0.5` | `+0.0041` | `+0.0113` | `diagnostic_only` |
| `5e-4` | `0.5` | `+0.0105` | `+0.0230` | `diagnostic_only` |
| `1e-3` | `0.5` | `+0.0213` | `+0.0424` | `diagnostic_only` |
| `2e-3` | `0.5` | `+0.0409` | `+0.0761` | `diagnostic_only` |

## Paper-Writing Guardrail

This canonical bundle may be described in the paper only as a **regenerated dense-friction diagnostic under the locked protocol**. It must not be described as source-artifact reproduction, and it must remain diagnostic-only rather than a replacement for the RL frozen-policy selected-point main result.
