# Main Text Captions v1

## Table 1. RL Selected-Point Main Result

Held-out executed-path comparison for the frozen-policy selected operating point. With predictive information held fixed, the validation-selected interior point `eta=0.5` is compared against the immediate-execution baseline `eta=1.0` on `kappa={0, 5e-4, 1e-3}`, showing positive-cost gains with a large reduction in executed turnover while the zero-cost row remains near-flat. This is the paper's only main empirical result, because it isolates how the forecast-to-execution interface changes realized decision quality under frictions.

## Figure 1. Why Target-Level Evaluation Can Misstate Realized Decision Quality

Diagnostic accounting summaries compare target-level and executed-path quantities at the validation-selected operating point `eta=0.5`. Target turnover remains about twice executed turnover, tracking is small but nonzero, and the final path gap widens with cost, showing why the realized executed path rather than the target path is the primary evaluation object in this paper.

## Conceptual Figure. From Forecasts to Realized Decisions Under Frictions

The conceptual diagram shows how predictive outputs become proposed targets and then realized actions once a forecast-to-execution interface and trading frictions intervene. Its role is to make the workshop framing explicit: forecast quality and realized decision quality are related but not identical, so evaluation should follow realized actions rather than targets alone.
