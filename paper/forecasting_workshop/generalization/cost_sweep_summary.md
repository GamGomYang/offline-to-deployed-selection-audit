# Cost Sweep Summary v1

This appendix-only robustness check evaluates the locked selected-point comparison `eta=1.0` versus `eta=0.5` on a denser cost grid. It does not replace the paper's main result. Its role is to test whether the same implementation-side reading remains visible when friction levels are varied more finely under the same evaluation pipeline.

The zero-cost row remains near-flat. At `kappa=0`, the executed-path paired-median delta Sharpe is `-0.000246`, which is consistent with the paper's existing zero-cost interpretation rather than with a broad gain that appears even without frictions.

Across the positive-cost rows, the executed-path advantage becomes more visible. The paired-median executed-path delta Sharpe increases from `+0.003052` at `1e-4` to `+0.005549` at `2e-4`, `+0.010503` at `5e-4`, `+0.021251` at `1e-3`, and `+0.048707` at `2e-3`. This is the main pattern that matters for the workshop paper: once frictions intervene, the selected forecast-to-execution interface produces a clearer realized-path difference.

The target-versus-executed disagreement also remains visible across the positive-cost rows. Target-based paired-median delta Sharpe is approximately flat to slightly negative throughout the same grid, from `-0.000084` at `1e-4` to `-0.002033` at `2e-3`, while the executed-path delta is positive at every positive-cost point. The disagreement flag is therefore `yes` for every positive-cost row in the sweep, and the evaluation-object contrast becomes easier to see as costs rise: the gap between executed-path and target-based delta Sharpe is about `0.0031` at `1e-4` and about `0.0507` at `2e-3`.

The diagnostic gaps support the same narrow reading but should not be overclaimed. The target-versus-executed turnover gap is nearly constant at about `0.0109`, while the final path gap is small at very low costs and larger at the middle and higher positive-cost rows. That path-gap series is not perfectly monotone: it rises strongly through `1e-3` and then softens slightly at `2e-3`. The correct interpretation is therefore friction-sensitive strengthening rather than strict monotonicity.

Taken together, the cost sweep is aligned with the paper's current claim. It supports the view that realized-path evaluation matters more once frictions separate proposed targets from realized actions, while keeping the paper narrow: this is a robustness check around the documented selected-point result, not a new empirical centerpiece and not evidence of a stronger predictor.
