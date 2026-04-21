# Forecasting Redesign Scope

## Step 0: Fixed Questions

### Q1
Can realized quality differ due to friction and interface, even when using the same forecast?

같은 forecast를 써도, friction과 interface 때문에 realized quality가 달라지는가?

### Q2
When comparing different forecasters, can forecast-metric ranking differ from realized ranking?

다른 forecaster를 비교할 때, forecast metric ranking과 realized ranking이 달라지는가?

## Execution Check

- Every code path we run must be explainable as either `Q1` or `Q2`.
- If we cannot explain which question a run belongs to, we defer it until after the deadline.

## Deliverables For Tomorrow Night

- `same forecast, different interface`
- `different forecasts, same interface`
- `synthetic benchmark`
- one-page summary judgment over the three result sets

## Step 1: Shared Evaluation Harness

- All new runs should emit the same schema under `outputs/forecast_eval/`.
- `forecast_metric`, `target_metric`, and `executed_metric` are stored as score-oriented values, so higher is better for ranking.
- Domain runners live under `scripts/forecast_eval/` and the merged table is `outputs/forecast_eval/summary/master_results.csv`.

## Step 2: Candidate Lock

- The current Step 2 main candidate is the split synthetic benchmark in `outputs/forecast_eval/synthetic/`.
- Status: `candidate_locked`
- Lock date: `2026-04-21`
- `Q1` locked config: `procar1_jumps_w5_a2.60_n0.06_eta0.20_lam2.00`
- `Q2` locked config: `procblock_levels_w5_a1.10_n0.00_eta0.25_lam1.00_bs1.00_bn0.08`
- Rationale: keep the stronger `Q1` gap result and the cleaner, seed-stable `Q2` ranking-stress result.
- We do not keep tuning Step 2 unless a later change is explicitly approved.
