# Volatility-Spike Eta Comparator Note

This note documents the second independent non-RL comparator family `arch_vol_spike_eta`.

## Why This Arm Is Independent Non-RL

This arm does **not** replay the RL target stream.

Instead, it reuses the same shared deterministic target builder already used by the deadband family:

- `scripts/generalization/build_shared_targets.py`
- `configs/generalization/shared_target_mapping.yaml`

That means the frozen forecast source and target construction are held fixed. The only difference is
the execution rule.

## Volatility Proxy And Spike Score

The comparator first builds the same base scalar volatility proxy used by the earlier volatility arm:

1. compute per-asset rolling volatility as the `20`-day standard deviation of arithmetic returns
2. convert that cross-section into a scalar proxy using the shared target weights:
   `sigma_t = sum_i w_target_t[i] * vol_i,t`

It then builds a regime reference with a rolling median:

- `sigma_ref_t = rolling_median(sigma_t, 60)`

The relative spike score is:

- `spike_t = sigma_t / (sigma_ref_t + eps)`

This redesign is deliberate. The earlier `arch_vol_scaled_eta` arm reacted to absolute volatility at
every step, and the first `arch_vol_spike_eta` version still used a smooth beta schedule that stayed
too close to an always-slightly-slower execution rule. The current version is a triggered fixed-eta
gate: it stays at full rebalance in normal conditions and intervenes only when volatility is elevated
relative to its own recent baseline.

## Execution Rule

Let:

- `w_prev` be previous executed weights
- `w_target_t` be the shared deterministic target
- `sigma_t` be the target-weighted rolling volatility proxy
- `sigma_ref_t` be the rolling median reference
- `spike_t = sigma_t / (sigma_ref_t + eps)`

Then:

- if `spike_t <= trigger`, set `eta_t = 1.0`
- else set `eta_t = eta_low`
- execute `w_exec_t = (1 - eta_t) * w_prev + eta_t * w_target_t`

The reference behavior is the same full-rebalance baseline used by the deadband package:

- `w_exec_t = w_target_t`

## Why Volatility-Spike Execution May Matter Under Friction

Under friction, a relative volatility spike can be more informative than absolute volatility for
execution control. If volatility is elevated versus its own recent baseline, a smaller `eta_t` can
slow target chasing just in those stressed periods. If the regime is normal, `eta_t` stays at `1.0`
and the executed path remains aligned with the target path.

So this arm tests a genuinely different heuristic family from deadband:

- deadband: trade-gating based on weight gap
- volatility-spike eta: state-dependent partial execution based on regime change

## Default Parameter Grid

The redesigned validation grid is:

- `trigger in {1.10, 1.15, 1.20, 1.25}`
- `eta_low in {0.985, 0.987, 0.988, 0.989, 0.990, 0.992, 0.994, 0.996}`
- `lookback_sigma = 20`
- `lookback_ref = 60`
- `eps = 1e-8`

## Intended Selection Rule

Champion selection is validation-based only and should preserve the runner-up.

The intended rule is:

1. keep only configs that satisfy the deadband-style validation eligibility rule
2. among configs within a documented score band of the best positive-cost `ΔSharpe_exec`, prefer:
   - smaller zero-cost `|ΔSharpe_exec|`
   - smaller average intervention away from full rebalance
   - smaller activation rate
   - larger disagreement strength
   - simpler parameters

Both the champion and runner-up should still be reported on test. The champion is for emphasis only,
not for hiding the runner-up.
