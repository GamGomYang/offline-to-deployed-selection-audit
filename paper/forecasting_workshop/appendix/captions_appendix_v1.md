# Appendix Captions v1

## Figure A1. Denser Cost-Grid Support for the Friction-Sensitive Evaluation Pattern

The appendix cost sweep extends the selected-point comparison to `kappa in {0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3}`. The zero-cost row remains near-flat, while the executed-path advantage becomes more visible across positive-cost rows. In contrast, target-based deltas remain approximately zero or negative, reinforcing the paper's evaluation-object interpretation without introducing a new main result class.

## Table A1. CC-TA-LBIP Auxiliary Comparator Under the Same State

This table reports auxiliary supporting evidence from CC-TA-LBIP while preserving the same 918-dimensional state and the fixed ridge forecast map used in the forecasting-support analysis. It matters because the `kappa=0` row preserves the collapse logic while the positive-cost rows move in the same implementation-side direction as the RL main result, so the comparator supports the story without becoming a second main finding.

## Table A2. Narrow `c`-Ablation Around the Selected Comparator Setting

This table reports a local robustness check over `c in {0, 2000, 3000, 4000}` in the positive-cost regimes for the fixed forecast-map comparator. It matters because it shows that the documented `c=3000` operating point remains reasonable in a narrow neighborhood while keeping the comparator auxiliary and avoiding a new model-selection claim.

## Table A3. Cost-Grid Target-Based Versus Executed-Path Evaluation Disagreement

This table extends the evaluation-object comparison to the full appendix cost grid. It matters because the executed-path delta stays near-flat at zero cost but becomes increasingly positive across the positive-cost rows, while the target-based delta remains approximately zero or negative and the mismatch flag turns on away from zero cost. As in the main table, the delta columns summarize aligned paired comparisons rather than simple subtraction of displayed marginal quantities. The table is support for the evaluation argument, not a new main result class.

## Table A4. Similar Forecasting Information, Different Realized Decision Quality

This table compares forecast-side and decision-side movement for the fixed forecast-map comparator while keeping the title conservative. It matters because forecast-side metrics move only slightly while realized decision-side metrics move much more strongly in the positive-cost rows, but only metric-level similarity is established here, so the table supports the interface-side reading without claiming exact forecast identity.
