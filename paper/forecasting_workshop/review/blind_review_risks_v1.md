# Blind Review Risks v1

## Blocking Risks

### 1. Local path leakage if LaTeX build artifacts are shipped

- Status: `Needs Fix`
- File: `paper/forecasting_workshop/paper_forecasting_workshop_v1.fls:1`
- Risk: the `.fls` file records `PWD /workspace/execution-aware-portfolio-rl/paper/forecasting_workshop`, which is a direct private-path leak.
- Action: do not include `.fls`, `.log`, `.aux`, or `.out` in any submission bundle.

### 2. Internal process language remains in markdown drafts

- Status: `Needs Fix`
- Files:
  - `paper/forecasting_workshop/paper_forecasting_workshop_v1.md:31`
  - `paper/forecasting_workshop/paper_forecasting_workshop_v1.md:33`
  - `paper/forecasting_workshop/paper_forecasting_workshop_v1.md:35`
  - `paper/forecasting_workshop/paper_forecasting_workshop_v1.md:43`
  - `paper/forecasting_workshop/appendix/appendix_forecasting_workshop_v1.md:9`
  - `paper/forecasting_workshop/appendix/appendix_forecasting_workshop_v1.md:25`
  - `paper/forecasting_workshop/appendix/appendix_forecasting_workshop_v1.md:31`
  - `paper/forecasting_workshop/appendix/captions_appendix_v1.md:5`
- Risk: phrases such as `evaluation audit`, `forecast-similarity audit`, `regenerated canonical`, `locked protocol`, `workshop bundle`, `final audit`, and `guardrails` make the drafts read like internal build documents.
- Action: if these markdown files are submission-facing, rewrite those phrases into plain paper-facing wording before export.

## Non-Blocking Risks

### 1. Typesetting warnings remain in the LaTeX log

- Status: `Pass` for blind review, `Needs Polish` for formatting
- File: `paper/forecasting_workshop/paper_forecasting_workshop_v1.log`
- Risk: only `Underfull` / `Overfull` box warnings remain; these do not reveal identity.
- Action: optional cleanup only.

### 2. Provenance-heavy appendix language may feel too internal

- Status: `Needs Fix` only if appendix markdown is used directly
- Risk: the appendix text is not identifying, but it still sounds more like internal package documentation than workshop prose.
- Action: soften provenance language if the markdown appendix is exported as part of the submission workflow.

## Explicit Passes

- No author name, affiliation, email, handle, or account identifier found in the active main markdown draft.
- No author-identifying text found in the active appendix markdown draft.
- No repo URL, branch name, or local path found in the compiled workshop PDF.
- Included figure PDFs expose only generic Matplotlib metadata.
- Main/appendix separation is clear.
- The active LaTeX source uses empty author metadata and does not reveal identity.

## Recommended Safe Export Set

Use only:

- `paper/forecasting_workshop/paper_forecasting_workshop_v1.tex`
- `paper/forecasting_workshop/paper_forecasting_workshop_v1.pdf`
- `paper/forecasting_workshop/assets/figures/*`
- `paper/forecasting_workshop/assets/tables/*`

Do not export:

- `paper/forecasting_workshop/*.fls`
- `paper/forecasting_workshop/*.log`
- `paper/forecasting_workshop/*.aux`
- `paper/forecasting_workshop/*.out`
- internal markdown review/audit notes unless they are rewritten as paper-facing prose
