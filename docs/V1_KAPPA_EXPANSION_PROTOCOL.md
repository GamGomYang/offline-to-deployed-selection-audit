# V1 Kappa Expansion Protocol

## Goal

Densify the friction axis inside the existing frozen-policy paper scope.

This extension does **not** retrain any policy. It reuses the canonical frozen-policy control models from the paper baseline and evaluates a denser transaction-cost grid:

- `kappa = 0`
- `kappa = 2e-4`
- `kappa = 5e-4`
- `kappa = 1e-3`
- `kappa = 2e-3`

## Scope lock

- same trained policy reused for every arm
- same feature set
- same validation window: `2022-01-01 ~ 2023-12-31`
- same final window: `2024-01-01 ~ 2025-12-31`
- same eta grid: `{1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02}`
- same validation rule: largest qualifying eta under the locked relative threshold

## Outputs

- validation full-grid frontier on the expanded kappa set
- final full-grid frontier on the expanded kappa set
- global validation-selected eta using positive kappas `{2e-4, 5e-4, 1e-3, 2e-3}`
- per-kappa qualifying eta diagnostics
- per-kappa frontier comparison against immediate execution `eta=1.0`
- kappa-benefit summary for Sharpe, turnover, realized-cost proxy, and tracking

## Interpretation rule

This extension is a frozen-policy frontier diagnostic. It is intended to answer:

> As proportional friction increases, does the held-out execution frontier become more favorable to interior eta values relative to immediate execution?

It is **not** a retraining experiment and it does **not** replace the canonical main-text selection result.
