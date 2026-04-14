# Preflight Fix Report

## Scope

Performed only instruction-path consistency repair and prompt readiness checks for the workshop reframing workspace.
No experiments were run.
No paper outputs were created.
No claim, metric hierarchy, selection logic, or evidence hierarchy was changed.

## Repaired

- Updated [reframing_docs/AGENTS.md](/workspace/execution-aware-portfolio-rl/reframing_docs/AGENTS.md) so source-of-truth project-document paths and output paths point to `reframing_docs/workshop_reframing/...`.
- Updated workshop prompt files under [reframing_docs/workshop_reframing/prompts](/workspace/execution-aware-portfolio-rl/reframing_docs/workshop_reframing/prompts) so they read `reframing_docs/AGENTS.md`, reference source-of-truth docs under `reframing_docs/workshop_reframing/...`, and write deliverables under `reframing_docs/workshop_reframing/outputs/...`.
- Updated internal cross-references in workshop spec files that still pointed to `workshop_reframing/...` so they now point to `reframing_docs/workshop_reframing/...`.

## Prompt Check

Checked all prompt files in `reframing_docs/workshop_reframing/prompts/*.md`.
All prompt files were present and non-empty, so no placeholder prompt files were created.

## Validation

- `reframing_docs/AGENTS.md` now references project docs under `reframing_docs/workshop_reframing/...`.
- No `docs/`-prefixed project-document paths remain under `reframing_docs/*.md`.
- No prompt file still starts with `Read AGENTS.md`.
- Prompt directory paths are consistent with the observed repository structure.

## Notes

- The workspace is treated as the canonical paper rebuild track and the path convention now reflects the actual repository layout.
- Existing empty log files in `reframing_docs/workshop_reframing/outputs/logs/` were left unchanged because this milestone only covered preflight path repair and prompt readiness.
