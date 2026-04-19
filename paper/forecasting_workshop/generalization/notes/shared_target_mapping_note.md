# Shared Target Mapping Note

This note fixes the common deterministic target construction for the new independent non-RL comparators.

The key requirement is simple: after this step, comparator differences should come from the **execution rule only**, not from different target construction pipelines.

## Forecast Source Used

The shared forecast source is the frozen paper-control signal package referenced by:

- `frozen_protocol/paper_v3/selected_signals_snapshot.json`

That snapshot currently fixes the selected signal set to:

- `reversal_5d`
- `short_term_reversal`

The signal-state features are rebuilt deterministically from cached market data using the existing repo signal-generation code under the fixed `paper_control_frozen` selection policy. This reuses the existing frozen signal artifact rather than introducing a new learned forecast family.

## Deterministic Mapping Rule

The shared deterministic target builder applies the following rule:

1. start from the frozen forecast signal package
2. compute the per-asset frozen score vector `s_t` as the equal-weight mean across the selected signal-state values
3. apply a cross-sectional z-score transform to `s_t`
4. clip the processed z-scores conservatively at the configured absolute threshold
5. map the processed cross-section into long-only fully-invested target weights using `stable_softmax`

This produces target weights that are:

- deterministic
- long-only
- fully-invested
- reusable across multiple execution comparators

## Assumptions

- The frozen signal package is treated as the forecast source for the new non-RL comparator family.
- The shared score aggregation is an equal-weight mean across the selected frozen signals.
- Cross-sectional z-scoring is applied again at the shared-score stage so that both new comparators start from the same normalized target-construction input.
- `stable_softmax` with a fixed scale is used as the long-only fully-invested target map.
- If an entire score row is invalid, the fallback is equal weight.

## Why This Is Independent From RL Target Replay

This module does **not** replay the RL policy target stream.

Instead, it constructs targets directly from the frozen signal package using a deterministic mapping rule. The resulting targets are therefore independent from:

- RL action logits
- RL policy replay
- RL selected-point target traces

That independence matters for Step 8 follow-up work. Once `arch_deadband_partial` and `arch_vol_scaled_eta` both import this shared target builder, the clean comparison statement becomes straightforward:

`the new comparator pair starts from the same frozen forecast signal and the same deterministic target construction, so any difference between them is attributable to the execution rule rather than to RL replay or to a different target builder.`
