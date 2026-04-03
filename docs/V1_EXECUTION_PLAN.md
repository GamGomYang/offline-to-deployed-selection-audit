# V1 Execution Plan

This file records the working order for strengthening the frozen-policy v1 paper without expanding the scope into retraining comparisons.

## Order

1. Step 1: scope lock
2. Step 2: protocol freeze and regeneration hardening
3. Step 6: mechanism decomposition
4. Step 7: selection-rule defense
5. Step 3: rolling-origin windows
6. Step 5: kappa-grid expansion
7. Step 4: stronger classical baselines (decision check only; exclude from mainline if it weakens scope)
8. Step 8: figure redesign
9. Step 9: related-work tightening
10. Step 10: final paper reconstruction

## Why This Order

- Lock the v1 identity first so later additions cannot blur the claim.
- Freeze the baseline protocol and regeneration path before adding new evidence.
- Extract as much value as possible from existing traces before launching heavier runs.
- Delay compute-heavy rolling-window and baseline expansions until the paper story and canonical baseline are stable.

## Main-Text Compression Rule

Keep the main text centered on four blocks:

- frozen-policy core result
- rolling robustness summary
- mechanism summary
- dense-friction diagnostic

Send detailed split-wise tables, sensitivity checks, and larger robustness dumps to the appendix.

## Scope Guard

The v1 mainline paper should keep the following statements true everywhere:

- the same trained policy is reused across the internal execution arms
- only the execution mapping changes in the main causal comparison
- the paper does not claim retraining superiority
- training-time alignment belongs to future work, not to the main contribution

## V2 Material

The Step 2 and Step 3 training-comparison artifacts are useful research material, but they are outside the v1 mainline paper story and should be treated as v2 material unless explicitly revived for a later paper.

The stronger-classical-baseline study from Step 4 should also stay out of the v1 mainline if it shifts the paper from an execution-control study into a comparator-dominance paper. In the current draft, those results are treated only as secondary context rather than as core evidence.
