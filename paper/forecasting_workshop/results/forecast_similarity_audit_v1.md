# Forecast Similarity Audit v1

## Scope

This audit checks how similar forecast-side information appears across the compared decision arms (`c=0` vs `c=3000`) using the cached forecast metrics and the same-forecast table. No raw per-time-step forecast vectors were found in the build artifacts, so this audit remains metric-level rather than vector-level.

## Evidence Used

- `workshop_rebuild_v1/outputs/tables/table_same_forecast_diff_decision_v2.csv`
- `workshop_rebuild_v1/outputs/logs/forecast_metric_analysis_v2.md`
- `paper/forecasting_workshop/results/table_forecast_similarity_audit.csv`

## Summary Readout

Forecast-side changes are small relative to decision-side changes in the positive-cost rows:

- `kappa=5e-4`: forecast MSE `-0.03%`, sign accuracy `+0.41pp`, rank IC `+0.0038` vs executed turnover/cost `-90.8%` and net Sharpe `+0.9539`.
- `kappa=1e-3`: forecast MSE `+0.28%`, sign accuracy `+0.49pp`, rank IC `+0.0033` vs executed turnover/cost `-95.1%` and net Sharpe `+2.0208`.

This supports the conservative phrasing `similar forecasting information` but does not prove identity of forecasts.

## Judgment

**Weak support.**

The forecast-side metrics are consistently small relative to decision-side changes, which supports the `similar forecasting information` framing. However, without direct forecast-output vectors, this remains a metric-level audit rather than a direct identity test. The wording should stay conservative, and this evidence should remain supporting rather than main.
