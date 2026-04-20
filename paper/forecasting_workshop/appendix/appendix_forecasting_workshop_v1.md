# Appendix for the Forecasting Workshop Submission

This appendix supports the main workshop paper without widening its claim or changing its evidence hierarchy. The central empirical result remains the frozen-policy RL selected-point comparison in the main text, and the accounting package remains the main diagnostic support for why executed-path evaluation is primary. Everything collected here is subordinate to that story.

## Appendix A. Cost Sweep Support

The denser cost-grid figure is included here as appendix support rather than as a second result. Its role is to show the friction-sensitive evaluation pattern more directly on a finer `kappa` grid. On the executed path, the selected-point delta Sharpe is near-flat at `-0.000246` when `kappa=0`, then stays positive at `+0.003052`, `+0.005549`, `+0.010503`, `+0.021251`, and `+0.048707` as the cost grid moves through `1e-4`, `2e-4`, `5e-4`, `1e-3`, and `2e-3`. In the companion target-based panel, the same comparison remains approximately zero or negative across the grid.

This block remains appendix support only. It should not be used to claim temporal robustness, and it should not be used to promote the cost sweep into a second main finding.

## Appendix B. Multi-Universe Robustness

The fixed-universe package asks a narrow question: whether the same evaluation-object discrepancy remains visible outside the exact current U27 basket. It does not claim survivorship-free historical coverage and it does not claim broad market representativeness.

The current support is directionally compatible but still narrow. In `u27_current`, the positive-cost rows remain `+0.0119` and `+0.0223` on the executed path while target-based deltas stay near zero to negative. In `u27_sector_balanced`, the same pattern repeats at `+0.0110` and `+0.0192`. The large-cap alternative remains mixed because its zero-cost row is slightly above the documented near-flat threshold at `+0.0054`, even though its positive-cost rows remain directionally compatible at `+0.0142` and `+0.0230`.

This block should therefore be read as support-only fixed-universe recurrence rather than as a broader robustness claim.

## Appendix C. Decision-Architecture Robustness

The architecture package remains support-only. Beyond the main RL row, two independent non-RL families are now included: a deadband family and a volatility-scaled family. Each is shown with champion and runner-up configurations selected conservatively, and both families contribute additional positive-cost `ranking_mismatch` rows. The linear/prox family remains mixed because target-based and executed-based evaluation stay numerically aligned there.

This block therefore reduces the specific RL-only artifact concern more directly than the earlier architecture draft, while remaining support-only.

## Appendix D. Target-vs-Executed Master Audit

The master audit pools cost rows, fixed universes, architecture families, and the optional canonical split reference into one conservative disagreement map. The aggregate remains dominated by `ranking_mismatch` rows, with a smaller number of `sign_flip` and `magnitude_only` cases, reinforcing the same narrow point: under positive frictions, the evaluation object can change the interpretation.

The point of this appendix block is not to inflate the claim. It is only to show, in one place, that the evaluation-object discrepancy recurs across several support settings and is not confined to the single main selected-point table.

## Appendix E. Toy Example

The toy example is appendix-only and illustrative. It is not a new main empirical result. Its purpose is to show that the target-versus-executed distinction is not purely a finance-accounting curiosity.

At zero friction, the target-based and executed-based readings coincide in the toy process. At positive friction, the executed-based reading diverges gradually: the very low-friction row remains effectively a tie, while higher-friction rows become `ranking_mismatch`. This is only an intuition aid and should remain secondary to the empirical evidence.

## Appendix F. Meta Figures

The meta figures summarize the package at a glance. Their role is to keep the visual emphasis on fixed-universe recurrence and execution-layer recurrence rather than on the compact temporal pilot.

The compact temporal pilot was mixed and should be reported that way. In the compact split file, `3` of the `4` rows do not reproduce the full positive-cost direction-plus-disagreement pattern. The stronger added support instead comes from the fixed-universe package and the decision-architecture package, which now recur in a more stable way across the tested support settings.

## Appendix G. Similar Forecasting Information, Different Realized Decision Quality

This appendix section keeps the same-forecast package available without letting it overtake the main story. The title stays `Similar forecasting information, different realized decision quality`, and the interpretation stays conservative. With the fitted ridge forecast map held fixed, forecast-side metrics move only slightly while realized decision-side metrics move much more strongly in the positive-cost rows.

At `kappa=5e-4` and `kappa=1e-3`, forecast MSE shifts by only `-0.03%` and `+0.28%`, whereas executed turnover and realized cost fall by `90.8%` and `95.1%`, with net Sharpe rising by `+0.9539` and `+2.0208`. These support-package deltas are not intended to be compared numerically with the selected-point RL deltas in Table 1; they come from a different auxiliary comparison setting and are included only to show the asymmetry between small forecast-side movement and large decision-side consequences. This is useful support for forecasting relevance because it makes the forecast-side versus decision-side contrast visible. It remains weak support, however, because only metric-level similarity is established here; raw forecast-vector identity across arms is not directly shown.
