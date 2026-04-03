# Extension Manifest Policy

Every post-baseline experiment must declare that it extends the frozen baseline rather than silently mutating it.

## Required Rules

- Put extension manifests under [`manifests/extensions`](/workspace/execution-aware-portfolio-rl/manifests/extensions).
- Put extension outputs under `outputs/extensions/<experiment_id>/...`.
- Set `parent_baseline_id` to `paper_v3_frozen_control_eta_20260324`.
- Never write extension outputs into:
  - `paper_rebuild_20260324T065755Z`
  - `outputs/reproductions/baseline/...`
- Keep one manifest per experiment family so the exact split, seeds, kappas, and cost definition can be traced later.

## Starter Template

Use [`manifests/extensions/manifest_template.json`](/workspace/execution-aware-portfolio-rl/manifests/extensions/manifest_template.json) for new experiments.

