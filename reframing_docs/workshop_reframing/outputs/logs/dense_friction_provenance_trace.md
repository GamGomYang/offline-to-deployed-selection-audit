# Dense Friction Provenance Trace

## Trace Summary

This is the recovered dense-friction provenance chain from execution script to manuscript figure.

## 1. Extension run root convention

`prl-dow30/scripts/run_v1_kappa_expansion.sh:33` sets:

- `RUN_ROOT="${REPO_ROOT}/outputs/extensions/v1_kappa_expansion/${JOB_TS}"`

This establishes that dense-friction artifacts were designed to live under ignored extension-output directories, not under tracked paper assets.

## 2. Analysis stage writes the missing source CSV

`prl-dow30/scripts/analyze_v1_kappa_expansion.py:329-333` writes:

- `kappa_expansion_summary.csv`
- `kappa_expansion_summary.md`

The same file is the one later referenced by the manifests.

Related outputs in the same script:

- `validation_per_kappa_selection.csv` at `:323-327`
- `kappa_expansion_seedwise.csv` at `:347-352`
- `kappa_expansion_summary.txt` at `:354-372`

So the missing CSV is not an inferred filename; it is the canonical analysis output name produced by the script.

## 3. Figure-builder consumes that CSV

`prl-dow30/scripts/build_v1_extension_figures.py:150-213` defines `_build_kappa_benefit_curve(summary_csv, output_path)`.

It:

- reads the summary CSV at `:151`
- sorts positive-cost kappas at `:152-155`
- plots three series from CSV columns at `:164-189`
  - `global_selected_final_median_delta_sharpe`
  - `per_kappa_final_median_delta_sharpe`
  - `best_interior_final_median_delta_sharpe`
- annotates per-kappa selected eta at `:191-200`
- saves the figure at `:212`

`prl-dow30/scripts/build_v1_extension_figures.py:221-248` then:

- receives `kappa_summary_csv` as an input at `:223-225`
- writes `fig_kappa_benefit_curve.png` at `:230-234`
- writes `figure_manifest.json` at `:236-243`
- optionally copies the figure and manifest to a legacy directory at `:245-248`

This is the direct provenance link from `kappa_expansion_summary.csv` to `fig_kappa_benefit_curve.png`.

## 4. Preserved manifest records the exact lost source path

`repro/manifests/figure_manifest_v1_extensions.json:1-6` records:

- `kappa_summary_csv = /workspace/execution-aware-portfolio-rl/outputs/extensions/v1_kappa_expansion/20260401T090500Z/analysis/kappa_expansion_summary.csv`
- `fig_kappa_benefit_curve = /workspace/execution-aware-portfolio-rl/paper_v1_extension_artifacts/figures/fig_kappa_benefit_curve.png`

`repro/manifests/paper_artifact_manifest.json:60-65` records the same Figure 5 provenance:

- `canonical_png = paper_v1_extension_artifacts/figures/fig_kappa_benefit_curve.png`
- `source_csv = outputs/extensions/v1_kappa_expansion/20260401T090500Z/analysis/kappa_expansion_summary.csv`

So the repository still knows which source CSV and which original figure-output path were used, even though those artifacts are no longer present.

## 5. Manuscript numbers match the dense-friction story

`paper/paper.tex:497-499` reports:

- selected-point gains:
  - `+0.0041`
  - `+0.0105`
  - `+0.0213`
  - `+0.0409`
- best-interior gains:
  - `+0.0113`
  - `+0.0230`
  - `+0.0424`
  - `+0.0761`
- per-kappa selector movement:
  - `1.0, 1.0, 0.5, 0.2`

`paper/paper.tex:503` includes:

- `submission_figures/fig_kappa_curve_submission.pdf`

## 6. Submission-PDF provenance is preserved downstream

`prl-dow30/scripts/build_submission_polish_figures.py:148-175` defines `build_kappa_curve()`.

That function hard-codes:

- `selected = [0.0041, 0.0105, 0.0213, 0.0409]` at `:151`
- `best = [0.0113, 0.0230, 0.0424, 0.0761]` at `:152`
- `selector_eta = [1.0, 1.0, 0.5, 0.2]` at `:153`

and writes:

- `fig_kappa_curve_submission.pdf` at `:175`

This explains why the submission PDF is still present even though the original source CSV is missing.

## 7. Current surviving dense-friction figure artifacts

The following dense-friction figures are present now:

- `paper/fig_kappa_benefit_curve.png`
- `paper/legacy_figures/fig_kappa_benefit_curve.png`
- `paper/submission_figures/fig_kappa_curve_submission.pdf`

Integrity note:

- `paper/fig_kappa_benefit_curve.png` and `paper/legacy_figures/fig_kappa_benefit_curve.png` share the same SHA-256:
  - `865270b059bd01b15ba9b083b30af05a0d503b10c3abfa21b3f6e87dae8729b0`

So the committed paper PNG and legacy PNG are byte-identical copies.

## 8. Git-history checkpoints

Commit `05707d8`:

- adds `prl-dow30/scripts/analyze_v1_kappa_expansion.py`
- adds `prl-dow30/scripts/build_v1_extension_figures.py`
- adds `paper/fig_kappa_benefit_curve.png`

Commit `bc00eab`:

- adds `paper/submission_figures/fig_kappa_curve_submission.pdf`
- adds `prl-dow30/scripts/build_submission_polish_figures.py`
- adds `paper/legacy_figures/fig_kappa_benefit_curve.png`

These commits preserve the downstream figure lineage, but not the ignored extension-output CSV itself.

## 9. Why the source CSV is absent from history

Ignored-path rules explain the gap:

- `.gitignore:18-20` ignores `outputs/`
- `.gitignore:54` ignores `paper_v1_extension_artifacts/`

That matches the manifest paths exactly:

- source CSV under `outputs/extensions/...`
- original figure under `paper_v1_extension_artifacts/...`

Because both locations are ignored, it is plausible that the original dense-friction source artifacts were generated locally, used to build figures/manifests, and never committed.

## 10. Final provenance judgment

Recovered:

- figure-generation chain
- manifest trail
- committed PNG/PDF figure artifacts
- manuscript-consistent dense-friction numbers

Not recovered:

- original `kappa_expansion_summary.csv`
- original `paper_v1_extension_artifacts/figures/fig_kappa_benefit_curve.png`

Conclusion:

- figure provenance was recovered
- original source-artifact provenance was only partially recovered
- the dense-friction package cannot be re-audited from original source artifacts alone without rerunning the extension experiment
- the correct failure label is `artifact-unavailability`, not `result contradiction`
