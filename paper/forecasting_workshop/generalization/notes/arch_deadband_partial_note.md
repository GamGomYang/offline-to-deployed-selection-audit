# Deadband Partial Comparator Note

This note documents the independent non-RL comparator `arch_deadband_partial`.

## Why This Arm Is Independent Non-RL

This arm does **not** replay the RL target stream.

Instead, it imports the shared deterministic target builder from:

- `scripts/generalization/build_shared_targets.py`
- `configs/generalization/shared_target_mapping.yaml`

That shared module fixes:

- the frozen forecast source
- the cross-sectional z-score processing
- the long-only fully-invested target mapping

As a result, this comparator is independent from RL replay and the only new element is the execution layer.

## Execution Rule

Let:

- `w_prev` be previous executed weights
- `w_target_t` be the shared deterministic target at time `t`
- `gap_t = ||w_target_t - w_prev||_1`

Then the rule is:

1. if `gap_t <= delta`, do nothing and keep `w_exec_t = w_prev`
2. if `gap_t > delta`, execute only a partial move:
   `w_exec_t = (1 - eta_db) * w_prev + eta_db * w_target_t`

The reference behavior is the full-rebalance baseline:

- `w_exec_t = w_target_t`

## Why Deadband Can Create Target-vs-Executed Divergence

Deadband logic can create divergence in two ways:

1. small target moves are ignored entirely when the L1 gap does not clear `delta`
2. larger target moves are only partially implemented when the rule applies `eta_db < 1`

So the executed path can lag the target path even though both come from the same frozen deterministic target mapping. That is exactly the intended Step 8 probe: keep the forecast source and target construction fixed, then ask whether the execution layer alone creates a separate target-versus-executed reading.

## Parameter Grid

The runner evaluates the documented grid:

- `delta in {0.00, 0.02, 0.05, 0.10, 0.15}`
- `eta_db in {0.995, 0.999, 0.9995, 0.9999}`
- reference baseline: full rebalance

This revised grid is intentionally conservative. The earlier coarse grid used small deadbands and much lower
`eta_db`, which made the comparator behave like aggressive smoothing and failed the zero-cost near-flat screen.
The new grid keeps the candidate close to full rebalance while still allowing a measurable execution-side gap.

## Intended Selection Rule

Champion selection is **not** applied in this step.

The intended later rule is validation-based only:

- evaluate the full candidate grid on validation
- primary score: mean executed-path delta Sharpe across positive-cost rows
- guardrail: require near-flat zero-cost behavior
- if several candidates are close, prefer the smallest positive intervention within `95%` of the best validation score

Even after that later selection step, both validation and test should continue to report the full candidate set rather than hiding the runner-up.
