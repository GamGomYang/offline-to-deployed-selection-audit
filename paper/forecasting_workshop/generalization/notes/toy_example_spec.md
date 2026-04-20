# Toy Example Spec

This toy example is appendix-only support for the generalization package. It is not intended to become a new main empirical result.

## Purpose

The purpose is narrow: show that a target-versus-executed evaluation mismatch can arise in a generic decision process with friction, not only in the main finance pipeline.

## Process

The toy process is one-dimensional.

1. A latent desired action level evolves over time.
2. A frozen forecast signal is formed by adding small noise to that latent level.
3. Two deterministic proposal rules map the same forecast to proposed target actions.
4. A frictional execution rule limits how quickly the realized action can move.
5. The same pair of arms is evaluated in two ways:
   - `target-based`: score the proposed action as if it were realized directly
   - `executed-based`: score the realized action after the frictional execution rule

## Shared Forecast Source

Both arms use the same frozen forecast signal. The difference is only in the proposed target construction and the realized execution path.

- `responsive`: use the forecast directly
- `tempered`: shrink the same forecast toward the neutral action by a fixed gain

This keeps the example easy to read for a general ML audience and makes the evaluation-object issue the main moving part.

## Execution Rule

The realized action is updated by a simple actuator-style step cap:

`x_t = x_{t-1} + clip(a_t - x_{t-1}, -c(lambda), c(lambda))`

where:

- `a_t` is the proposed target action
- `x_t` is the realized executed action
- `c(0) = 1.0`, so zero friction reproduces the proposal directly
- `c(lambda)` shrinks as friction increases

## Utility

Per-step utility is generic squared-error tracking to the latent desired action:

`u(a_t) = -(a_t - y_t)^2`

This is intentionally plain. The example is meant to illustrate an evaluation-object issue, not domain-specific reward engineering.

## Intended Reading

The intended qualitative pattern is:

- near-agreement at zero friction
- little or no disagreement at very low friction
- a larger proposal-versus-executed gap as friction rises
- at least one positive-friction row where target-based and executed-based evaluation imply different action preference

## Limitations

- This is a one-dimensional toy process, not a realistic deployment environment.
- The friction rule is stylized and deterministic.
- The utility is deliberately simple and should not be read as a domain model.
- The example only supports intuition; it does not add a new main claim.
