# Forecasting Submission Compliance Report v1

## Overall Status

`Needs Fix` before final submission packaging.

The compiled workshop PDF and LaTeX source are largely clean on blind-review requirements, but three compliance issues remain in the broader `paper/forecasting_workshop/` tree:

1. the markdown draft files still contain internal audit/build phrasing
2. the appendix caption/draft still contain provenance-heavy internal wording
3. LaTeX build artifacts such as `.fls` contain an absolute local path and must not be shipped

## Checklist

| Item | Status | Notes |
| --- | --- | --- |
| Author-identifying text in `paper/forecasting_workshop/paper_forecasting_workshop_v1.md` | `Pass` | No author name, affiliation, email, handle, or account identifier found. |
| Repository / branch / local-path leakage in `paper/forecasting_workshop/paper_forecasting_workshop_v1.md` | `Pass` | No repo URL, branch name, `/workspace`, or local filesystem path found. |
| Internal jargon in `paper/forecasting_workshop/paper_forecasting_workshop_v1.md` | `Needs Fix` | `paper_forecasting_workshop_v1.md:31`, `:33`, `:35`, and `:43` still use drafting/build-style wording such as `evaluation audit`, `forecast-similarity audit`, `regenerated canonical`, `locked protocol`, and `in this build`. |
| Self-reference wording in main draft | `Pass` | No self-citation or author-revealing wording such as `our previous work` was found. |
| Author-identifying text in `paper/forecasting_workshop/appendix/appendix_forecasting_workshop_v1.md` | `Pass` | No author name, affiliation, email, or account identifier found. |
| Repository / branch / local-path leakage in appendix draft | `Pass` | No explicit repo URL, branch name, or local path found in the appendix markdown itself. |
| Internal jargon in appendix draft | `Needs Fix` | `appendix_forecasting_workshop_v1.md:9`, `:25`, and `:31` still contain `workshop bundle`, `source-artifact reproduction`, `locked protocol`, `final audit`, and `guardrails`. |
| Main captions | `Pass` | `paper/forecasting_workshop/results/captions_main_v1.md` reads like paper-facing caption text and does not expose internal paths or author identity. |
| Appendix captions | `Needs Fix` | `paper/forecasting_workshop/appendix/captions_appendix_v1.md:5` still says `In this workshop build` and uses provenance-heavy wording that reads like internal process text. |
| Included figure metadata | `Pass` | `fig_accounting_gap.pdf` and `fig_kappa_curve.pdf` expose only generic Matplotlib `Creator` / `Producer` metadata; no usernames, repo names, or local paths were found. |
| Included table source files | `Pass` | The included `.tex` table files contain numeric/tabular content only and do not expose author identity or local paths. |
| LaTeX source safety | `Pass` | `paper/forecasting_workshop/paper_forecasting_workshop_v1.tex` has empty author metadata and no author-identifying text, repo URL, or local path leakage. |
| Compiled PDF metadata | `Pass` | `paper_forecasting_workshop_v1.pdf` has empty `Author` metadata and only generic `LaTeX with hyperref` / `pdfTeX` producer fields. |
| Build artifacts in submission tree | `Needs Fix` | `paper/forecasting_workshop/paper_forecasting_workshop_v1.fls` contains `PWD /workspace/execution-aware-portfolio-rl/paper/forecasting_workshop`. `.log`, `.aux`, and `.out` are also process artifacts and should not be included in any submission archive. |
| Appendix separation | `Pass` | Main paper and appendix remain clearly separated; appendix materials are not required to understand the core claim. |

## Required Cleanup Before Packaging

1. Remove internal audit/build language from the markdown main draft.
2. Simplify appendix wording so it stays factual without `workshop bundle`, `final audit`, `guardrails`, or `in this workshop build`.
3. Exclude `.fls`, `.log`, `.aux`, and `.out` from any submission-facing archive.
4. Use the compiled workshop PDF / LaTeX source only; do not export internal drafting notes as submission files.

## Submission-Safe Core

If the final submission package is limited to the following, the blind-review risk is low:

- `paper/forecasting_workshop/paper_forecasting_workshop_v1.tex`
- `paper/forecasting_workshop/paper_forecasting_workshop_v1.pdf`
- `paper/forecasting_workshop/assets/figures/*`
- `paper/forecasting_workshop/assets/tables/*`

