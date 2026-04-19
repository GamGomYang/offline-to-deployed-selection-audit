# Forecasting Workshop Generalization Package

This directory is the paper-facing home for the forecasting workshop paper's support-only generalization bundle.

The package is intentionally narrow. It does not attempt to revive broad temporal robustness as the main generalization axis, and it does not change the identity of the workshop paper. The goal is to organize additional support around recurrence of the same implementation-side reading across fixed universes, decision architectures, and evaluation objects.

## Current Primary Axes

The default generalization package is organized around four support tracks:

1. `multi-universe robustness`
2. `decision-architecture robustness`
3. `target-vs-executed disagreement audit`
4. `minimal domain-general toy example`

These tracks are support-only unless explicitly promoted later. Executed-path quantities remain primary. Target-level quantities remain diagnostic.

## Legacy Material Status

This folder already contains earlier temporal-support and cost-sweep materials, including `multi_split_*`, `cost_sweep_*`, and related notes/results. Those files remain in place as historical or legacy support context.

They should not be treated as the default generalization axis for the current package, and they should not be reused as the basis for a broad temporal-robustness claim. In particular, the compact multi-split pilot is not a license to claim broad temporal robustness for the workshop paper.

## Directory Map

- `universe_specs/`: specifications for fixed-universe robustness checks
- `architecture_specs/`: specifications for RL and linear-comparator architecture robustness checks
- `tables/`: paper-facing tables reserved for generalization support artifacts
- `figures/`: paper-facing figures reserved for generalization support artifacts
- `notes/`: short interpretation notes, verdict memos, and writing guards
- `outputs/`: reserved axis-first output root for generated support artifacts
- [`scripts/generalization/`](/workspace/execution-aware-portfolio-rl/scripts/generalization): repo-root namespace for wrapper and orchestration entrypoints
- [`configs/generalization/`](/workspace/execution-aware-portfolio-rl/configs/generalization): repo-root namespace for generalization experiment configs

## Future Naming Convention

Use the following package conventions as new support experiments are added:

- universe-level studies belong under `universe_specs/`
- architecture comparisons belong under `architecture_specs/`
- paper-facing assets belong under `tables/`, `figures/`, and `notes/`
- generated outputs should be grouped under axis-first paths below `outputs/`
- runnable wrappers belong under repo-root `scripts/generalization/`
- config files belong under repo-root `configs/generalization/`

Reserved axis-first output paths:

- `paper/forecasting_workshop/generalization/outputs/multi_universe/`
- `paper/forecasting_workshop/generalization/outputs/architecture/`
- `paper/forecasting_workshop/generalization/outputs/target_vs_executed/`
- `paper/forecasting_workshop/generalization/outputs/toy_example/`

## Boundary Note

Do not modify the main paper yet. This package is for support structure, planning, and future experiment organization only.
