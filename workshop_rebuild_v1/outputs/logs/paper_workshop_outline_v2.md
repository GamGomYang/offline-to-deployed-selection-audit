Title: Forecast-to-Execution Interfaces Matter in Cost-Sensitive Portfolio Decisions

Framing sentence:
In cost-sensitive portfolio decision systems, holding the predictive signal fixed, the forecast-to-execution interface materially changes realized decision quality.

Section 1. Introduction
- Forecasting-driven systems often evaluate proposed targets as if they were immediately realized.
- Under frictions, realized positions differ from targets, so realized-path evaluation should be primary.
- The paper studies this question in a narrow portfolio decision setting with fixed predictive signals and without claiming a stronger predictor.

Section 2. Forecast-to-Execution Interface
- Define target portfolio `w_t^tgt` and executed portfolio `w_t^exec`.
- Describe the partial-execution update briefly.
- State that target-path quantities are diagnostic only and executed-path evaluation is primary.

Section 3. Experimental Setup
- Canonical U27 frozen split and frozen-policy RL protocol.
- Validation-selected `eta=0.5` versus baseline `eta=1.0`.
- Locked `kappa` rows for the main RL table and dense diagnostic grid for friction sensitivity.
- Auxiliary CC-TA-LBIP comparator with same state, fixed ridge forecast map, validation-selected `c=3000`, and a narrow local robustness check around that selected point.

Section 4. Results
- Main result: RL frozen-policy selected-point comparison.
- Diagnostic support: target-versus-executed accounting gap.
- Diagnostic support: dense-friction regenerated canonical diagnostic bundle, described explicitly as regenerated under the locked protocol and archived as the workshop build's canonical dense-friction bundle.
- Auxiliary support: CC-TA-LBIP comparator, with any `c`-ablation framed only as a narrow robustness check.
- Optional supporting analysis in main text: `Similar forecasting information, different realized decision quality`, because the v2 analysis note still judges it strong enough under a conservative interpretation.

Section 5. Conclusion
- Evaluation must follow realized executed decisions.
- The gain is implementation-side under costs, not necessarily alpha-side.
- Evidence is narrow, case-study-based, and does not claim universality.

Contribution list:
1. A forecast-to-execution interface for cost-sensitive portfolio decisions that separates target and executed portfolios and attaches primary evaluation to realized paths.
2. Frozen-policy evidence that the validation-selected interior point improves positive-cost held-out decision quality while sharply reducing executed turnover.
3. Supporting evidence from accounting diagnostics, a regenerated canonical dense-friction diagnostic bundle, an auxiliary linear forecast-plus-convex-optimization comparator with a narrow reviewer-safe robustness check, and a conservative same-forecast table.

Out-of-scope items for the main paper:
- heuristic baseline zoo
- buy-and-hold framing
- rolling-window detail
- U36 detail
- retraining detail
- long implementation appendix in the main text
