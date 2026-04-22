Minimal event-forecasting micro-benchmark

- Verdict: appendix_support_ready
- Reading: fit-oriented minimal confirmation of ranking drift under switching friction
- Scope: appendix-level support rather than headline evidence
- Canonical forecast ranking: Brier only
- Log loss: robustness-only diagnostic; never used for winner selection
- Zero-friction agreement rate: 0.75
- Strongest positive-friction mean deployed gap: 0.061 at friction 1.00
- Schema note: In the shared raw schema, `forecast_metric` is stored as `-brier` only to preserve the higher-is-better ranking convention used by the summary builder; all paper-facing tables and text should still report Brier in its standard lower-is-better interpretation.
