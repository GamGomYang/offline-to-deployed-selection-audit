# Traffic Hourly Exceedance Snapshot 2026-04-23

## Freeze Point
- Manuscript baseline branch point: `ba201b5`
- Commit message: `fix : 테이블 A2 수정`
- Redesign branch created from that point: `feat/redesign`

## Purpose
- Preserve the failed `Continuous-to-Event Exceedance Benchmark` attempt as an internal redesign reference.
- Keep the current workshop manuscript baseline clean while retaining the execution logs, manifests, and pilot diagnostics needed for future family redesign.

## Kept Artifacts
- Canonical source lock:
  - `outputs/extensions/revision_round_20260423/new_reruns/traffic_hourly_exceedance/_cache/traffic_hourly_source_lock.json`
- Canonical run provenance:
  - `outputs/extensions/revision_round_20260423/new_reruns/traffic_hourly_exceedance/fixed_threshold_tau055_q070_rep100/run_provenance.json`
- Hardening ledger:
  - `outputs/extensions/revision_round_20260423/new_reruns/traffic_hourly_exceedance/traffic_hourly_exceedance_ledger.json`
- Story manifest:
  - `outputs/extensions/revision_round_20260423/story_revision/traffic_hourly_exceedance/traffic_hourly_exceedance_result_manifest.json`
- Claim-to-evidence map:
  - `outputs/extensions/revision_round_20260423/story_revision/traffic_hourly_exceedance/traffic_hourly_exceedance_claim_to_evidence_map.json`
- Salvage pilot table:
  - `outputs/extensions/revision_round_20260423/analysis_additions/traffic_hourly_exceedance_salvage_pilot/traffic_hourly_exceedance_salvage_pilot_table.csv`
- Salvage pilot report:
  - `outputs/extensions/revision_round_20260423/analysis_additions/traffic_hourly_exceedance_salvage_pilot/traffic_hourly_exceedance_salvage_pilot_report.json`

These artifacts remain in `outputs/extensions/...` as internal reference outputs and are not paper-facing.

## Dropped Working-Tree Code
The following implementation paths were intentionally removed from the redesign branch working tree so the next family can be rebuilt cleanly:
- `configs/traffic_hourly_exceedance/`
- `exceedance_benchmark/`
- `scripts/forecast_eval/run_traffic_hourly_exceedance.py`
- `scripts/forecast_eval/run_traffic_hourly_exceedance_hardening.py`
- `scripts/forecast_eval/run_traffic_hourly_exceedance_salvage_pilot.py`

## Why The Family Failed
The failure was not a pipeline failure. The family ran successfully, but it failed the paper-facing promotion gates.

### Canonical Failure
- Promotion status: `fail`
- Gate A: failed
  - zero mean deployed gap: `0.02584`
  - zero median deployed gap: `0.02597`
  - zero tie-involved fraction: `0.0`
- Gate B: failed
  - agreement at `0.00`: `0.0`
  - agreement at `0.50`: `0.0`
  - agreement at `1.00`: `0.0`
  - deployed-suboptimal share at `0.50`: `1.0`
  - deployed-suboptimal share at `1.00`: `1.0`
- Gate C: passed
- Gate D: passed

### Diagnosis
- The problem is a design mismatch, not insufficient reruns.
- Forecast-side winner and deployed winner were already fully separated at zero friction.
- In practice, the family behaved like:
  - zero-friction mismatch is already complete
  - friction preserves or slightly scales the mismatch
  - therefore the family does not tell the intended `friction creates deployed-selection failure` story

## Salvage Pilot Outcome
- Priority pilot: 6 combinations
- Expanded `q=0.70` grid: 20 combinations total
- Expanded `q in {0.75, 0.80}` phase also run
- Final result:
  - `candidate_count = 0`
  - `strong_candidate_count = 0`

### Best Zero-Row Repairs Failed High-Friction Signal
Representative examples:
- `q=0.70, tau=0.50, c_fp=1.25`
  - zero agreement: `0.85`
  - zero mean gap: `0.00020`
  - `1.00` deployed-suboptimal share: `0.00`
  - `1.00` mean gap: `0.00000`
- `q=0.75, tau=0.40, c_fp=1.25`
  - zero agreement: `1.00`
  - zero mean gap: `0.00000`
  - `1.00` deployed-suboptimal share: `0.00`
  - `1.00` mean gap: `0.00000`

### Strong High-Friction Divergence Failed Zero-Row Explainability
Representative examples:
- `q=0.70, tau=0.55, c_fp=0.25`
  - zero agreement: `0.00`
  - zero mean gap: `0.02655`
  - `1.00` deployed-suboptimal share: `1.00`
  - `1.00` mean gap: `0.01954`
- `q=0.70, tau=0.50, c_fp=0.25`
  - zero agreement: `0.00`
  - zero mean gap: `0.02336`
  - `1.00` deployed-suboptimal share: `1.00`
  - `1.00` mean gap: `0.01604`

## Practical Conclusion
- This family should remain internal-only.
- It should not be promoted into the workshop manuscript in its current form.
- The next redesign should start from a clean manuscript baseline and a new forecasting-native family design rather than incremental repair of this one.
