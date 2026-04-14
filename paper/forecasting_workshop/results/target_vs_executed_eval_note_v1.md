# Target vs Executed Evaluation Note v1

This support analysis compares target-based and executed-based evaluation for the locked RL main comparison `eta=1.0` versus `eta=0.5` on `kappa={0, 5e-4, 1e-3}`. The result is diagnostic and does not replace the main executed-path selected-point result.

The main pattern is a positive-cost interpretation mismatch. On the executed path, the validation-selected interface `eta=0.5` improves paired-median net Sharpe by `+0.010503` at `kappa=5e-4` and `+0.021251` at `kappa=1e-3`. Under target-based evaluation, however, the same comparison is approximately flat to slightly negative at `-0.000150` and `-0.000308`. The positive-cost rows therefore show a ranking disagreement: executed-path evaluation favors the selected interface, while target-based evaluation would largely erase that gain.

The zero-cost row behaves differently. Both evaluation views are near-flat at the displayed precision, which is consistent with the locked zero-cost reading and does not suggest a broad target-versus-executed contradiction in the absence of frictions.

The interpretation is narrow. This table does not argue that target-path quantities are useless. It shows that, for this locked main comparison, target-based evaluation answers a different question and can miss the realized positive-cost benefit that appears on the executed path. That is why the paper keeps executed-path evaluation primary and treats target-level quantities as diagnostic support only.
