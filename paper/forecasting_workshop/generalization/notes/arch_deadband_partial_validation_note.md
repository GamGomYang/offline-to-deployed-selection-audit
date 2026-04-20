# Deadband Partial Validation Note

This note summarizes the validation-only grid search for `arch_deadband_partial`.

The validation eligibility rule uses the following documented near-flat threshold:

- `|ΔSharpe_exec(kappa=0)| <= 0.01`

The remaining eligibility checks are:

- both positive-cost rows (`kappa in {5e-4, 1e-3}`) must have `ΔSharpe_exec > 0`
- at least one positive-cost row must show `disagreement_type in {ranking_mismatch, sign_flip}`

This stage uses the same Step 8 pair-audit logic for ranking, sign, and disagreement labels, but it keeps the validation qualification threshold at `0.01` for the zero-cost near-flat screen. Test has not been run in this step.

Eligible configurations:
- `delta_0.08__eta_0.9988`: zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0065`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.08__eta_0.9994`: zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0033`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.08__eta_0.9997`: zero-cost `|ΔSharpe_exec|=0.0016`, mean positive-cost `ΔSharpe_exec=0.0016`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.07__eta_0.9997`: zero-cost `|ΔSharpe_exec|=0.0016`, mean positive-cost `ΔSharpe_exec=0.0016`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.06__eta_0.9997`: zero-cost `|ΔSharpe_exec|=0.0016`, mean positive-cost `ΔSharpe_exec=0.0016`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.05__eta_0.9997`: zero-cost `|ΔSharpe_exec|=0.0016`, mean positive-cost `ΔSharpe_exec=0.0016`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.04__eta_0.9997`: zero-cost `|ΔSharpe_exec|=0.0016`, mean positive-cost `ΔSharpe_exec=0.0016`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.03__eta_0.9997`: zero-cost `|ΔSharpe_exec|=0.0016`, mean positive-cost `ΔSharpe_exec=0.0016`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.08__eta_0.9996`: zero-cost `|ΔSharpe_exec|=0.0021`, mean positive-cost `ΔSharpe_exec=0.0022`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.07__eta_0.9996`: zero-cost `|ΔSharpe_exec|=0.0021`, mean positive-cost `ΔSharpe_exec=0.0022`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.06__eta_0.9996`: zero-cost `|ΔSharpe_exec|=0.0021`, mean positive-cost `ΔSharpe_exec=0.0022`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.05__eta_0.9996`: zero-cost `|ΔSharpe_exec|=0.0021`, mean positive-cost `ΔSharpe_exec=0.0022`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.04__eta_0.9996`: zero-cost `|ΔSharpe_exec|=0.0021`, mean positive-cost `ΔSharpe_exec=0.0022`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.03__eta_0.9996`: zero-cost `|ΔSharpe_exec|=0.0021`, mean positive-cost `ΔSharpe_exec=0.0022`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.08__eta_0.9995`: zero-cost `|ΔSharpe_exec|=0.0026`, mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.07__eta_0.9995`: zero-cost `|ΔSharpe_exec|=0.0026`, mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.06__eta_0.9995`: zero-cost `|ΔSharpe_exec|=0.0026`, mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.05__eta_0.9995`: zero-cost `|ΔSharpe_exec|=0.0026`, mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.04__eta_0.9995`: zero-cost `|ΔSharpe_exec|=0.0026`, mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.03__eta_0.9995`: zero-cost `|ΔSharpe_exec|=0.0026`, mean positive-cost `ΔSharpe_exec=0.0027`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `delta_0.09__eta_0.9997`: zero-cost `|ΔSharpe_exec|=0.0026`, mean positive-cost `ΔSharpe_exec=0.0028`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.07__eta_0.9994`: zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0033`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.06__eta_0.9994`: zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0033`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.9994`: zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0033`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.04__eta_0.9994`: zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0033`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.03__eta_0.9994`: zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0033`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9996`: zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0034`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.08__eta_0.9993`: zero-cost `|ΔSharpe_exec|=0.0036`, mean positive-cost `ΔSharpe_exec=0.0038`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.07__eta_0.9993`: zero-cost `|ΔSharpe_exec|=0.0036`, mean positive-cost `ΔSharpe_exec=0.0038`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.06__eta_0.9993`: zero-cost `|ΔSharpe_exec|=0.0036`, mean positive-cost `ΔSharpe_exec=0.0038`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.9993`: zero-cost `|ΔSharpe_exec|=0.0036`, mean positive-cost `ΔSharpe_exec=0.0038`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.04__eta_0.9993`: zero-cost `|ΔSharpe_exec|=0.0036`, mean positive-cost `ΔSharpe_exec=0.0038`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.03__eta_0.9993`: zero-cost `|ΔSharpe_exec|=0.0036`, mean positive-cost `ΔSharpe_exec=0.0038`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9995`: zero-cost `|ΔSharpe_exec|=0.0036`, mean positive-cost `ΔSharpe_exec=0.0039`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.08__eta_0.9992`: zero-cost `|ΔSharpe_exec|=0.0042`, mean positive-cost `ΔSharpe_exec=0.0043`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.07__eta_0.9992`: zero-cost `|ΔSharpe_exec|=0.0042`, mean positive-cost `ΔSharpe_exec=0.0043`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.06__eta_0.9992`: zero-cost `|ΔSharpe_exec|=0.0042`, mean positive-cost `ΔSharpe_exec=0.0043`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.9992`: zero-cost `|ΔSharpe_exec|=0.0042`, mean positive-cost `ΔSharpe_exec=0.0043`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.04__eta_0.9992`: zero-cost `|ΔSharpe_exec|=0.0042`, mean positive-cost `ΔSharpe_exec=0.0043`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.03__eta_0.9992`: zero-cost `|ΔSharpe_exec|=0.0042`, mean positive-cost `ΔSharpe_exec=0.0043`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9994`: zero-cost `|ΔSharpe_exec|=0.0042`, mean positive-cost `ΔSharpe_exec=0.0044`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.08__eta_0.9991`: zero-cost `|ΔSharpe_exec|=0.0047`, mean positive-cost `ΔSharpe_exec=0.0049`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.07__eta_0.9991`: zero-cost `|ΔSharpe_exec|=0.0047`, mean positive-cost `ΔSharpe_exec=0.0049`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.06__eta_0.9991`: zero-cost `|ΔSharpe_exec|=0.0047`, mean positive-cost `ΔSharpe_exec=0.0049`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.9991`: zero-cost `|ΔSharpe_exec|=0.0047`, mean positive-cost `ΔSharpe_exec=0.0049`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.04__eta_0.9991`: zero-cost `|ΔSharpe_exec|=0.0047`, mean positive-cost `ΔSharpe_exec=0.0049`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.03__eta_0.9991`: zero-cost `|ΔSharpe_exec|=0.0047`, mean positive-cost `ΔSharpe_exec=0.0049`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9993`: zero-cost `|ΔSharpe_exec|=0.0047`, mean positive-cost `ΔSharpe_exec=0.0050`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.08__eta_0.999`: zero-cost `|ΔSharpe_exec|=0.0052`, mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.07__eta_0.999`: zero-cost `|ΔSharpe_exec|=0.0052`, mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.06__eta_0.999`: zero-cost `|ΔSharpe_exec|=0.0052`, mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.999`: zero-cost `|ΔSharpe_exec|=0.0052`, mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.04__eta_0.999`: zero-cost `|ΔSharpe_exec|=0.0052`, mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.03__eta_0.999`: zero-cost `|ΔSharpe_exec|=0.0052`, mean positive-cost `ΔSharpe_exec=0.0054`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9992`: zero-cost `|ΔSharpe_exec|=0.0052`, mean positive-cost `ΔSharpe_exec=0.0055`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.08__eta_0.9989`: zero-cost `|ΔSharpe_exec|=0.0057`, mean positive-cost `ΔSharpe_exec=0.0060`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.07__eta_0.9989`: zero-cost `|ΔSharpe_exec|=0.0057`, mean positive-cost `ΔSharpe_exec=0.0060`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.06__eta_0.9989`: zero-cost `|ΔSharpe_exec|=0.0057`, mean positive-cost `ΔSharpe_exec=0.0060`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.9989`: zero-cost `|ΔSharpe_exec|=0.0057`, mean positive-cost `ΔSharpe_exec=0.0060`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.04__eta_0.9989`: zero-cost `|ΔSharpe_exec|=0.0057`, mean positive-cost `ΔSharpe_exec=0.0060`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.03__eta_0.9989`: zero-cost `|ΔSharpe_exec|=0.0057`, mean positive-cost `ΔSharpe_exec=0.0060`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9991`: zero-cost `|ΔSharpe_exec|=0.0057`, mean positive-cost `ΔSharpe_exec=0.0061`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.07__eta_0.9988`: zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0065`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.06__eta_0.9988`: zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0065`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.05__eta_0.9988`: zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0065`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.04__eta_0.9988`: zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0065`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.03__eta_0.9988`: zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0065`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.999`: zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0066`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9989`: zero-cost `|ΔSharpe_exec|=0.0068`, mean positive-cost `ΔSharpe_exec=0.0072`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `delta_0.09__eta_0.9988`: zero-cost `|ΔSharpe_exec|=0.0073`, mean positive-cost `ΔSharpe_exec=0.0077`, sum disagreement strength `4`, mean turnover reduction `0.1%`

The redesigned stability-first selection rule first keeps only eligible configurations, then forms a champion band with mean positive-cost `ΔSharpe_exec >= 80%` of the best eligible score. Inside that band, it prefers smaller zero-cost `|ΔSharpe_exec|`, then smaller mean turnover reduction, then wider deadband, and finally larger `eta_db`.

The runner-up is selected from the remaining eligible configurations using a secondary band with mean positive-cost `ΔSharpe_exec >= 40%` of the best eligible score, then the same stability-first tie-break.

Selected pair for test rerun:
- Champion: `delta_0.08__eta_0.9988` with zero-cost `|ΔSharpe_exec|=0.0062`, mean positive-cost `ΔSharpe_exec=0.0065`, mean turnover reduction `0.1%`
- Runner-up: `delta_0.08__eta_0.9994` with zero-cost `|ΔSharpe_exec|=0.0031`, mean positive-cost `ΔSharpe_exec=0.0033`, mean turnover reduction `0.1%`
