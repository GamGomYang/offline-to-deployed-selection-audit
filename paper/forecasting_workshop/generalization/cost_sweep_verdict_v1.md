# Cost Sweep Verdict v1

## Classification

`Green`

## Why This Is Green

The pre-registered Green conditions are satisfied.

- the zero-cost row remains near-flat relative to the positive-cost rows: `ΔSharpe_exec = -0.000246` at `kappa=0`
- every positive-cost row preserves the executed-path advantage for the selected interface: `+0.003052`, `+0.005549`, `+0.010503`, `+0.021251`, `+0.048707`
- target-versus-executed disagreement remains visible at every positive-cost row, with `disagreement_flag = yes` throughout the positive-cost grid
- turnover reduction remains directionally consistent with the implementation-side reading: executed turnover stays near `0.0218-0.0220` for `eta=1.0` and near `0.0109` for `eta=0.5`

The sweep is not perfectly monotone in every diagnostic quantity. In particular, the final path gap rises strongly through `1e-3` and then softens at `2e-3`. That does not overturn the Green classification, but it does mean the paper should describe the result as friction-sensitive strengthening rather than strict monotonicity.

## Three-Sentence Interpretation

The denser cost sweep preserves the same qualitative reading as the locked main result. The zero-cost row remains near-flat, while the positive-cost rows show a clearer executed-path advantage for `eta=0.5` and a persistent target-versus-executed evaluation disagreement. This supports the paper's narrow implementation-side interpretation under frictions without creating a new empirical centerpiece or authorizing any broader theorem-like claim.

## Paper Placement

Use as:

- `appendix support` with figures/tables
- `one short main-text or discussion sentence` if desired

Do not use as:

- a second main empirical result
- abstract- or title-level strengthening
- a broad robustness claim across arbitrary regimes

## Main-Text Decision

`Appendix support plus one compact main-text sentence` is allowed under the registered rules.

## Safe Paper-Facing Sentence

`A denser cost sweep preserves the same friction-sensitive direction: the zero-cost row remains near-flat, while the positive-cost rows continue to favor the selected interface on the realized executed path and maintain the target-versus-executed evaluation disagreement.`

## Too-Strong Sentence to Avoid

`The interface effect increases monotonically with trading frictions and therefore establishes a general rule for forecasting systems under costs.`

## Writing Constraint

Even with a Green outcome, this sweep should remain a robustness check around the documented selected-point case study. The paper must continue to treat executed-path quantities as primary, target-path quantities as diagnostic, and the original RL selected-point comparison as the only main empirical result.
