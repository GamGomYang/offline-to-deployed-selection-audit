# Preflight Setup Report

## Scope

This step created a clean workshop rebuild lane only.

- No experiments were run.
- No scientific outputs were generated.
- No package specifications were rewritten.
- No old confirmed artifacts were moved or deleted.

## Created Lane

- `workshop_rebuild_v1/outputs/tables/`
- `workshop_rebuild_v1/outputs/figures/`
- `workshop_rebuild_v1/outputs/checks/`
- `workshop_rebuild_v1/outputs/logs/`
- `workshop_rebuild_v1/outputs/appendix/`

## Lane Label

The new lane is explicitly labeled as a clean rebuild lane, not a source-artifact reproduction lane.

## Reference Preservation

- Existing `reframing_docs/workshop_reframing/outputs/...` remains preserved as reference-only.
- New workshop-package outputs must go only to `workshop_rebuild_v1/outputs/...`

## Validation

- New lane exists: `Yes`
- Required output subfolders exist: `Yes`
- Old workshop reference outputs were modified by this setup step: `No`
