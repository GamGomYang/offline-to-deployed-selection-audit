# Forecasting Claim Freeze v1

## Primary Claim

Holding predictive information fixed, forecast-to-execution interfaces can materially change realized decision quality under frictions, so evaluation should follow realized actions rather than targets.

## Allowed Claims

- The paper is a forecasting-to-decision evaluation paper.
- Forecast quality and realized decision quality can diverge once frictions intervene between proposed targets and realized actions.
- Executed-path evaluation is primary, while target-level quantities are diagnostic only.
- The main evidence is a narrow cost-sensitive portfolio case study with fixed predictive information.
- The observed gains are implementation-side rather than evidence of a stronger predictive signal.

## Forbidden Claims

- Do not claim a new RL algorithm.
- Do not claim a stronger predictive signal.
- Do not claim a better forecasting method.
- Do not claim superior alpha generation.
- Do not claim a universal forecasting theorem.
- Do not claim benchmark dominance.
- Do not claim production readiness.
- Do not present target-level quantities as headline evaluation objects.
- Do not present dense-friction evidence as source-artifact reproduction.
- Do not present auxiliary comparator evidence as a second main identification result.
- Do not present the same-forecast package as proof of exact forecast identity.

## Main Evidence

- RL frozen-policy selected-point comparison with `eta=0.5` versus `eta=1.0`, evaluated on executed-path net Sharpe with positive-cost improvement and a near-flat zero-cost row.
- Accounting diagnostic showing why executed-path evaluation is primary and why target-level quantities remain diagnostic only.

## Supporting Evidence

- `Similar forecasting information, different realized decision quality` as conservative forecasting-relevant support when space allows.

## Appendix-only Evidence

- Dense-friction sensitivity, described only as a regenerated canonical diagnostic bundle under the locked protocol.
- CC-TA-LBIP auxiliary comparator and any local `c`-ablation.
- Provenance, rebuild, audit, and robustness details beyond the compact workshop story.

## Reviewer-facing One-line Framing

This paper evaluates how forecasting outputs are translated into realized decisions under frictions, using portfolio decisions only as a narrow case study rather than as the basis for a new RL or forecasting-method claim.

## Locked Core Sentences

- Changing only the forecast-to-execution interface changes realized outcomes.
- Target-level quantities are diagnostic, not headline evaluation objects.
- The result is implementation-side, not evidence of a stronger predictor.
