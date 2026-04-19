# Deadband Partial Validation Note

This note summarizes the validation-only grid search for `arch_deadband_partial`.

The validation eligibility rule uses the following documented near-flat threshold:

- `|ΔSharpe_exec(kappa=0)| <= 0.01`

The remaining eligibility checks are:

- both positive-cost rows (`kappa in {5e-4, 1e-3}`) must have `ΔSharpe_exec > 0`
- at least one positive-cost row must show `disagreement_type in {ranking_mismatch, sign_flip}`

This stage uses the same Step 8 pair-audit logic for ranking, sign, and disagreement labels, but it keeps the validation qualification threshold at `0.01` for the zero-cost near-flat screen. Test has not been run in this step.

Eligible configurations:
- `delta_0.10__eta_0.999`: mean positive-cost `ΔSharpe_exec=0.0062`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.999`: mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.02__eta_0.999`: mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.00__eta_0.999`: mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.10__eta_0.9995`: mean positive-cost `ΔSharpe_exec=0.0035`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.9995`: mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.02__eta_0.9995`: mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.00__eta_0.9995`: mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.10__eta_0.9999`: mean positive-cost `ΔSharpe_exec=0.0013`, sum disagreement strength `4`, mean turnover reduction `0.0%`

The recommended deadband configuration for later test evaluation is `delta_0.10__eta_0.999`. It is selected by largest mean positive-cost `ΔSharpe_exec`, then largest summed disagreement strength, then largest mean positive-cost turnover reduction, and finally by the simpler parameter preference (wider deadband, smaller partial step).
