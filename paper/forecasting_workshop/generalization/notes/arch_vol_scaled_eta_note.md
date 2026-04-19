# Volatility-Scaled Eta Comparator Note

This note documents the independent non-RL comparator `arch_vol_scaled_eta`.

## Why This Arm Is Independent Non-RL

This arm does **not** replay the RL target stream.

Instead, it reuses the same shared deterministic target builder as the deadband comparator:

- `scripts/generalization/build_shared_targets.py`
- `configs/generalization/shared_target_mapping.yaml`

That means the frozen forecast source and target construction are held fixed. The only difference is
the execution rule.

## Volatility Proxy

The comparator builds a scalar volatility proxy in two steps:

1. compute per-asset rolling volatility as the `20`-day standard deviation of arithmetic returns
2. convert that cross-section into a scalar proxy using the shared target weights:
   `sigma_t = sum_i w_target_t[i] * vol_i,t`

For the first rows of the evaluation window, the rolling-volatility frame is backfilled from the
first fully observed window so that the execution rule remains deterministic for the entire period.

## Execution Rule

Let:

- `w_prev` be previous executed weights
- `w_target_t` be the shared deterministic target
- `sigma_t` be the target-weighted rolling volatility proxy

Then the rule is:

- `eta_t = clip(alpha / (sigma_t + eps), eta_min, 1.0)`
- `w_exec_t = (1 - eta_t) * w_prev + eta_t * w_target_t`

The reference behavior is the same full-rebalance baseline used by the deadband package:

- `w_exec_t = w_target_t`

## Why Volatility-Aware Execution May Matter Under Friction

Under friction, a volatility-aware execution rule can matter even when the target path is fixed.
If the current target points toward a more volatile cross-section, the rule lowers `eta_t` and
moves more cautiously. If the proxy is lower, `eta_t` rises toward `1.0` and the executed path
stays closer to the target path.

So this arm is useful for asking a different question from deadband:

- deadband asks whether a thresholded no-trade region creates target-versus-executed divergence
- volatility-scaled eta asks whether a continuous state-dependent execution rule creates that divergence

## Parameter Grid

The current validation-focused grid is:

- `alpha in {0.024, 0.025, 0.026, 0.027, 0.028}`
- `eta_min in {0.90, 0.95, 0.98, 0.99}`
- `lookback = 20`
- `eps = 1e-8`

This narrowed grid is deliberate. The earlier coarse grid either moved too slowly and violated the
zero-cost near-flat guardrail, or collapsed back to the full-rebalance baseline and lost
target-versus-executed disagreement. The revised grid concentrates on the boundary region where
the zero-cost row stays near-flat while positive-cost rows can still show an informative executed-path
advantage and disagreement.

## Intended Selection Rule

Champion selection is **not** applied in this step.

The intended later rule is validation-based only:

- evaluate the full candidate grid on validation
- primary score: mean executed-path delta Sharpe across positive-cost rows
- guardrail: require near-flat zero-cost behavior
- if multiple candidates are close, prefer the candidate closer to full rebalance
  by favoring larger `eta_min` and then larger `alpha`

Both validation and test should continue to report the runner-up rather than hiding it.
