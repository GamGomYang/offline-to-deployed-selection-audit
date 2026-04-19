# Architecture Construction Note

These architecture specifications define a narrow support package for decision-architecture robustness.

The central rule is that the **forecast source is held fixed inside each architecture-family comparison**. For RL-based arms, this means the trained policy or target stream is frozen before the execution-interface comparison is run. For linear or convex comparator arms, this means the ridge forecast map is fit once and then frozen before the decision-layer comparison begins.

This package is therefore **not** about training different predictive models against each other in an unconstrained way. The comparison is about the execution interface or decision translation layer: how proposed targets become realized executed portfolios, and whether the target-versus-executed evaluation discrepancy survives outside the exact current RL interface.

## Purpose

The purpose of these specs is to reduce the `RL-only artifact` concern.

They exist to test whether:

- the executed-path interpretation survives beyond the current main RL interface
- target-versus-executed disagreement recurs across multiple execution-layer designs
- the same qualitative reading appears in both RL-based and linear or convex support architectures

## Architecture Roles

- `arch_rl_selected.yaml`: exact current main RL selected interface
- `arch_rule_eta_fixed.yaml`: minimal deterministic partial-execution smoothing rule
- `arch_linear_prox.yaml`: linear or convex proximal support architecture under a fixed linear forecast map
- `arch_threshold_rebalance.yaml`: appendix-only threshold or no-trade diagnostic arm

## Writing Constraint

Paper-facing wording should say that the forecast source is held fixed and that the comparison is about the execution interface, not about training different predictive models. The purpose is to reduce `RL-only artifact` concerns, not to claim universal architectural robustness.
