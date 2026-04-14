# Appendix for the Forecasting Workshop Submission

This appendix supports the main workshop paper without widening its claim or changing its evidence hierarchy. The central empirical result remains the frozen-policy RL selected-point comparison in the main text, and the accounting package remains the main diagnostic support for why executed-path evaluation is primary. Everything collected here is subordinate to that story.

## Appendix A. Denser Cost-Grid Support for the Friction-Sensitive Evaluation Pattern

The denser cost-grid figure is included here as appendix support rather than as a second result. Its role is to show the friction-sensitive evaluation pattern more directly on a finer `kappa` grid. On the executed path, the selected-point delta Sharpe is near-flat at `-0.000246` when `kappa=0`, then rises to `+0.003052`, `+0.005549`, `+0.010503`, `+0.021251`, and `+0.048707` as the cost grid moves through `1e-4`, `2e-4`, `5e-4`, `1e-3`, and `2e-3`. In the companion target-based panel, the same comparison remains approximately zero or negative across the grid. This appendix figure therefore supports the main evaluation reading without becoming a new main result.

The correct caption is the appendix caption for Figure A1: denser cost-grid support for the friction-sensitive evaluation pattern. The figure remains appendix support only. It should not be used to claim temporal robustness, and it should not be used to promote the appendix cost sweep into a second main finding.

## Appendix B. CC-TA-LBIP Auxiliary Evidence

CC-TA-LBIP remains auxiliary same-state evidence only. Its purpose is to show that the implementation-side pattern is not confined to the RL package under a fixed ridge forecast map and the same executed-path accounting. It is not a second identification result. The `kappa=0` row preserves the documented collapse logic, while the positive-cost rows move in the same qualitative direction as the RL main result.

The correct caption is the appendix caption for Table A1: CC-TA-LBIP auxiliary comparator under the same state. The appendix should keep that auxiliary framing throughout. If space pressure arises, this package should shrink before any main-text broadening is allowed.

## Appendix C. Narrow `c`-Ablation Robustness

The narrow `c`-ablation over `c in {0, 2000, 3000, 4000}` is a reviewer-facing local robustness check around the documented operating point `c=3000`. It is included to show that the auxiliary comparator is not knife-edge in the positive-cost regimes, not to reopen model selection or to argue for a uniquely optimal `c`. Under the locked validation rule, `c=3000` remains the selected operating point; `c=4000` qualifies but is not selected, and `c=2000` falls below the threshold.

The correct caption is the appendix caption for Table A2: narrow `c`-ablation around the selected comparator setting. This appendix section should therefore remain explicitly local and auxiliary.

## Appendix D. Cost-Grid Evaluation-Object Disagreement

This appendix section records the full cost-grid support table for the main selected-point RL comparison. Its role is simple: to show that the choice of evaluation object changes the visible positive-cost conclusion across the appendix cost sweep rather than only on the three locked rows. On the executed path, the delta Sharpe is near-flat at `-0.000246` when `kappa=0`, then rises to `+0.003052`, `+0.005549`, `+0.010503`, `+0.021251`, and `+0.048707` as the grid moves through `1e-4`, `2e-4`, `5e-4`, `1e-3`, and `2e-3`. Under target-based evaluation, the same comparison remains approximately zero or negative at `-0.000000`, `-0.000084`, `-0.000186`, `-0.000150`, `-0.000308`, and `-0.002033`.

This is not a new result class. It is a support table for the paper's evaluation argument. Its purpose is to make the interpretation mismatch visible at a glance: away from zero cost, target-based evaluation weakens or erases the selected-point improvement that remains visible on the realized executed path.

## Appendix E. Similar Forecasting Information, Different Realized Decision Quality

This appendix section keeps the same-forecast package available without letting it overtake the main story. The title stays `Similar forecasting information, different realized decision quality`, and the interpretation stays conservative. With the fitted ridge forecast map held fixed, forecast-side metrics move only slightly while realized decision-side metrics move much more strongly in the positive-cost rows.

At `kappa=5e-4` and `kappa=1e-3`, forecast MSE shifts by only `-0.03%` and `+0.28%`, whereas executed turnover and realized cost fall by `90.8%` and `95.1%`, with net Sharpe rising by `+0.9539` and `+2.0208`. This is useful support for forecasting relevance because it makes the forecast-side versus decision-side contrast visible. It remains weak support, however, because only metric-level similarity is established here; raw forecast-vector identity across arms is not directly shown.

## Appendix F. Provenance Note

The provenance note is included for factual clarity only. For the workshop submission, the relevant statement is short: the current appendix cost-grid figure is generated directly from the recorded cost-sweep results under the documented selected-point comparison. Historical sensitivity-support materials remain archival and are not needed for the submission-facing appendix.

This provenance note should stay concise. It exists to support honest wording, not to become part of the main narrative.

## Appendix G. Scope Summary

The appendix closes with the minimum scope summary needed to preserve the paper's narrow reading. The governing points are simple:

- the RL frozen-policy selected-point result remains central
- target-path quantities remain diagnostic only
- the denser cost-grid sweep remains appendix support
- CC-TA-LBIP remains auxiliary
- same-forecast wording remains at the `similar forecasting information` level

This summary is included to document scope, not to add new evidence. The appendix therefore supports the main paper's narrow forecasting-to-decision evaluation claim without displacing it.
