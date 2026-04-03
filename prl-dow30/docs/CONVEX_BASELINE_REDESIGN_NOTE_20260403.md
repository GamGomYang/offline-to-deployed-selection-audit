# Convex Baseline Redesign Note (2026-04-03)

## Bottom line
The PEC-swept LBIP family is not the cleanest main-text convex comparator.

After implementing an anchored target variant (`target_mode: anchored_mean_variance`) and running both validation and held-out grids, two recurring patterns appeared:

1. When the zero-cost gap became very small, the validation selector often moved to very small execution rates such as `eta=0.02`.
2. When the selected interior point stayed at more interpretable values like `eta=0.2` or `eta=0.5`, the zero-cost gap was usually still non-negligible.

That means a second PEC-style convex baseline would likely blur the paper's main logic rather than strengthen it.

## Recommended pivot
Use a single integrated turnover-aware convex comparator instead of another PEC frontier.

Suggested name:
- `Turnover-Aware Linear Information-Parity Baseline (TA-LBIP)`

Suggested objective:
\[
\max_{w \in \Delta}
\hat\mu_t^\top w
- \frac{\gamma}{2} w^\top \hat\Sigma_t w
- \frac{\tau}{2} \lVert w - w^{exec}_{t-1} \rVert_2^2.
\]

Interpretation:
- same information set as RL and LBIP
- linear expected-return mapping
- explicit convex turnover regularization inside the target optimizer
- no extra PEC layer in the main comparator

## Why this is cleaner
- It answers the "compare against classical cost-aware optimization" critique directly.
- It avoids a second "why does `kappa=0` improve?" thread in the main text.
- It keeps the RL story clean: RL + PEC is the execution/accounting intervention; TA-LBIP is the classical external comparator.

## What to keep from the current exploration
The anchored LBIP implementation is still useful:
- as an appendix diagnostic
- or as a development scaffold for TA-LBIP

Relevant files:
- `prl/linear_information_parity.py`
- `scripts/run_information_parity_baselines.py`
- `scripts/run_lbip_tuning_grid.py`
- `outputs/v2_u27_albip_anchor_grid/final_consistency_summary.csv`

## Practical recommendation
Main text:
1. keep RL immediate vs RL PEC as the identification result
2. add TA-LBIP as the classical cost-aware external comparator
3. move PEC-swept convex variants to appendix or drop them

Appendix only, if kept:
- one short note that PEC can also regularize over-reactive linear targets, but this is not the primary identification result
