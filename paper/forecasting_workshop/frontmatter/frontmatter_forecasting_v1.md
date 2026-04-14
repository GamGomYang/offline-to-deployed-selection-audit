# Front Matter Forecasting v1

## Recommended Title

From Forecasts to Realized Decisions: Why Forecast-to-Execution Interfaces Matter Under Frictions

## Alternative Titles

1. From Forecasts to Realized Decisions: Why Forecast-to-Execution Interfaces Matter Under Frictions
2. Forecasts Are Not Decisions: Evaluating Forecast-to-Execution Interfaces Under Costs
3. Similar Forecasting Information, Different Realized Decision Quality

## Abstract

Forecasting-driven decision systems are often assessed as if proposed targets were immediately realized. Under frictions, proposed targets and realized actions can diverge, so prediction-side metrics alone can misstate realized system quality. We study a forecast-to-execution interface that separates proposed targets from realized executed decisions and attaches primary evaluation to the executed path while holding predictive information fixed. In a narrow cost-sensitive portfolio case study with a frozen learned policy, the validation-selected interior operating point `eta=0.5` improves held-out executed-path net Sharpe relative to `eta=1.0` in positive-cost regimes while roughly halving executed turnover, and the zero-cost row remains near-flat. Accounting diagnostics and conservative supporting analyses support the same conclusion: this is an implementation-side evaluation result rather than evidence of a stronger predictor.

## Introduction Opening

Forecasting-driven systems are often evaluated at the level of proposed outputs, as if those outputs were directly realized by downstream decision layers. Under frictions or implementation constraints, however, proposed targets need not coincide with realized actions. This creates an evaluation problem: forecast quality and realized decision quality are not the same object once a forecast-to-decision interface intervenes. The central question is therefore not only whether forecasts look good before implementation, but also whether the system is evaluated on the realized path it actually induces.

We study this general evaluation issue in a concrete cost-sensitive portfolio case study. The portfolio setting is the evidence source rather than the paper's identity: predictive information is held fixed, target-level quantities remain diagnostic only, and primary evaluation is attached to executed-path outcomes as the forecast-to-execution interface is varied. In the central selected-point comparison, the validation-selected interior operating point `eta=0.5` is evaluated against the immediate-execution baseline `eta=1.0`; the positive-cost rows improve, while the zero-cost row remains near-flat. The contribution is therefore an implementation-side evaluation argument for forecasting-driven decision systems, not a stronger-prediction claim and not a new RL-method claim.

## Contributions

- We frame forecast-to-decision evaluation around the distinction between proposed targets and realized actions under frictions, and argue that executed-path evaluation should be primary while target-level quantities remain diagnostic.
- We provide narrow fixed-signal evidence from a cost-sensitive portfolio case study showing that the validation-selected interior operating point `eta=0.5` improves positive-cost held-out executed-path performance relative to `eta=1.0`, while the zero-cost row remains near-flat.
- We provide supporting evidence from accounting diagnostics and a conservative same-forecast analysis showing that forecast-side movement can stay small even when realized decision-side consequences become large.
