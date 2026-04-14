Recommended title: Forecast-to-Execution Interfaces Matter in Cost-Sensitive Portfolio Decisions

Main text order:
1. `Main Table 1`: RL frozen-policy selected-point result
2. `Main Figure 1`: target-vs-executed accounting gap diagnostic
3. `Main Figure 2`: dense-friction regenerated canonical diagnostic bundle
4. `Main Table 2`: CC-TA-LBIP auxiliary comparator
5. `Optional Main Table 3`: Similar forecasting information, different realized decision quality

Ordering note:
- The RL selected-point table stays first and central.
- The accounting gap follows immediately to explain why executed-path evaluation is primary and why target-path quantities remain diagnostic only.
- The dense-friction figure follows as the regenerated canonical diagnostic bundle for the workshop build, showing that the locked selected-point gain becomes more consequential as costs rise without changing the selected-point protocol.
- The CC-TA-LBIP table stays auxiliary and can mention the narrow `c`-ablation only as reviewer-facing robustness context, not as a new main experiment.
- The same-forecast table can remain in the main paper because the v2 analysis note still classifies it as strong enough for main-text supporting use, but it must stay last among the results items and must keep the conservative `Similar forecasting information` title.

Appendix candidates:
- additional diagnostics
- external baselines / context only
- rolling-window robustness
- U36 and retraining checks
- implementation details
