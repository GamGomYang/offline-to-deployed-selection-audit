# Dense Friction Artifact Recovery Report

## Scope

This check performed artifact recovery and provenance tracing only.

- No experiments were rerun.
- No configs, metrics, eta-selection logic, kappa grids, or claim wording were changed.
- Only the repository, git history, manifests, scripts, tracked figures, and ignored-path conventions were inspected.

## Recovery Result

### Original source CSV

Target source artifact from the manifests:

- `outputs/extensions/v1_kappa_expansion/20260401T090500Z/analysis/kappa_expansion_summary.csv`

Status:

- Not recovered from the current working tree
- Not recovered from tracked git history

Evidence:

- `repro/manifests/figure_manifest_v1_extensions.json:4` points to that exact CSV path.
- `repro/manifests/paper_artifact_manifest.json:60-65` points Figure 5 to the same source CSV.
- `prl-dow30/scripts/analyze_v1_kappa_expansion.py:329-333` shows that the intended analysis output name is `kappa_expansion_summary.csv`.
- A tracked-object search over git history found the dense-friction scripts and committed figures, but no tracked object named `kappa_expansion_summary.csv` and no tracked object under `outputs/extensions/v1_kappa_expansion/...`.

Why this is plausible:

- `.gitignore:18-20` ignores `outputs/` and `prl-dow30/outputs/`.
- `.gitignore:54` ignores `paper_v1_extension_artifacts/`.

So the manifest can reference a real generated artifact path even if that artifact was never committed.

### Figure provenance

Figure provenance was recovered.

Recovered figure artifacts now present in the repository:

- `paper/fig_kappa_benefit_curve.png`
- `paper/legacy_figures/fig_kappa_benefit_curve.png`
- `paper/submission_figures/fig_kappa_curve_submission.pdf`

Observed consistency:

- `paper/fig_kappa_benefit_curve.png` and `paper/legacy_figures/fig_kappa_benefit_curve.png` have the same SHA-256:
  - `865270b059bd01b15ba9b083b30af05a0d503b10c3abfa21b3f6e87dae8729b0`

Recovered provenance chain:

- `prl-dow30/scripts/run_v1_kappa_expansion.sh:33` defines the run root as `outputs/extensions/v1_kappa_expansion/${JOB_TS}`.
- `prl-dow30/scripts/analyze_v1_kappa_expansion.py:329-333` writes `kappa_expansion_summary.csv`.
- `prl-dow30/scripts/build_v1_extension_figures.py:150-213` reads that summary CSV and builds `fig_kappa_benefit_curve.png`.
- `prl-dow30/scripts/build_v1_extension_figures.py:236-248` writes a figure manifest and optionally copies the figure to a legacy output directory.
- `repro/manifests/figure_manifest_v1_extensions.json:1-6` preserves the exact dense-friction source CSV path and the original figure-output path.
- `paper/paper.tex:497-503` uses the dense-friction result in the manuscript and includes `submission_figures/fig_kappa_curve_submission.pdf`.
- `prl-dow30/scripts/build_submission_polish_figures.py:148-175` shows how the submission PDF was later generated from preserved dense-friction numbers.

### Can the dense friction package be reproduced now without rerunning experiments?

Not fully.

What is recoverable now:

- The published dense-friction figure lineage
- The manuscript-level dense-friction numbers
- The downstream figure-generation logic

What is not recoverable now:

- The original manifest-referenced analysis CSV
- The original manifest-referenced figure-output directory under `paper_v1_extension_artifacts/figures/...`
- A manifest-consistent source-artifact reconstruction of the dense-friction package

Important nuance:

- The dense-friction numbers are internally consistent across:
  - `paper/paper.tex:497-499`
  - `prl-dow30/scripts/build_submission_polish_figures.py:149-153`
  - the committed figure artifacts
- This points to artifact unavailability, not to a contradiction in the reported result.

## Classification

The failure mode is:

- `artifact-unavailability`

It is not:

- `result contradiction`

Rationale:

- The repository preserves downstream dense-friction figures and the exact manuscript numbers.
- The repository does not preserve the original analysis CSV and original extension-output directory referenced by the manifests.
- Therefore the package cannot be re-audited from original source artifacts alone without rerunning the extension experiment.

## Validation Answers

- Original source CSV recovered: `No`
- Figure provenance recovered: `Yes`
- Dense friction can now be reproduced without rerunning experiments: `No`

## Output Decision

`dense_friction_repro_recovered.csv` was not produced because the original source CSV was not recovered, and creating a replacement table from downstream hard-coded figure values would not count as source-artifact recovery.
