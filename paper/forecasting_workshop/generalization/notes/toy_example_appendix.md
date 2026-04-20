# Toy Example Appendix Note

This toy example is appendix-only support. It should be described conservatively and should not be elevated into a new main empirical block.

## Setup

The example compares two proposals built from the same frozen forecast signal:

- `responsive`: uses the forecast directly
- `tempered`: shrinks that same forecast toward the neutral action

The realized action is then passed through a frictional actuator rule that can only move by a limited amount per step. Utility is generic squared-error tracking to a latent desired action level.

## What It Shows

At zero friction, the target-based and executed-based readings agree exactly in this toy setup because the execution cap is inactive. In the generated results, the arm gap is `-0.0008` under both views at friction `0.0`, so both evaluations are effectively near-agreement and only weakly favor the responsive proposal.

The more useful feature of the current toy is that the discrepancy grows gradually rather than appearing immediately at full strength. The target-based gap stays fixed at `-0.0008`, which keeps the proposal-level reading slightly tilted toward the responsive arm, while the executed-based gap moves as friction increases:

- friction `0.10`: executed gap `+0.0000`, effectively still a tie
- friction `0.25`: executed gap `+0.0017`
- friction `0.50`: executed gap `+0.0042`
- friction `1.00`: executed gap `+0.0040`

Under the same conservative disagreement labels used elsewhere in the package, friction `0.10` is still `none`, while the higher-friction rows become `ranking_mismatch`. That is the point of the toy example: once constrained execution separates realized actions from proposed targets, the evaluation object can change the interpretation even in a minimal generic process, and that change can emerge gradually rather than only as an immediate sign flip.

The second panel in [fig_toy_example.pdf](/workspace/execution-aware-portfolio-rl/paper/forecasting_workshop/generalization/figures/fig_toy_example.pdf:1) shows why that mismatch becomes more plausible as friction rises. The responsive proposal generates a larger mean proposal-versus-execution gap than the tempered proposal. At friction `0.50`, for example, the mean gap is `0.0926` for the responsive arm and `0.0664` for the tempered arm.

## Safe Reading

The safe reading is narrow:

- the evaluation-object issue is not purely a finance-accounting curiosity
- a generic proposal-versus-realization mismatch can already create a target-versus-executed disagreement
- the disagreement can strengthen with friction instead of appearing all at once
- this example is only an intuition aid and should remain appendix-only

## Limitations

- The process is one-dimensional and intentionally stylized.
- The friction rule is chosen for clarity, not realism.
- The scores are not meant to be compared to the main paper's empirical magnitudes.
- The example does not justify any claim of universality.
