# Volatility-Spike Eta Validation Note

This note summarizes the redesigned validation-only grid search for `arch_vol_spike_eta`.

The validation eligibility rule uses the same documented near-flat threshold as the deadband comparator:

- `|ΔSharpe_exec(kappa=0)| <= 0.01`

The remaining eligibility checks are also the same:

- both positive-cost rows (`kappa in {5e-4, 1e-3}`) must have `ΔSharpe_exec > 0`
- at least one positive-cost row must show `disagreement_type in {ranking_mismatch, sign_flip}`

Selection is validation-based only. Within the eligible set, the champion is chosen from a high-score band and then filtered by stability-first tie-breaks:

- smaller zero-cost `|ΔSharpe_exec|`
- smaller mean intervention away from full rebalance
- smaller activation rate
- larger disagreement strength
- simpler parameters

Eligible configurations:
- `trigger_1.10__etaLow_0.992__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0099`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.203%`, activation `0.253`
- `trigger_1.15__etaLow_0.987__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0093`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.226%`, activation `0.174`
- `trigger_1.15__etaLow_0.988__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0086`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.208%`, activation `0.174`
- `trigger_1.15__etaLow_0.989__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0079`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.191%`, activation `0.174`
- `trigger_1.10__etaLow_0.994__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0074`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.152%`, activation `0.253`
- `trigger_1.15__etaLow_0.990__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0072`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.174%`, activation `0.174`
- `trigger_1.20__etaLow_0.985__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0067`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.192%`, activation `0.128`
- `trigger_1.20__etaLow_0.987__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0058`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.166%`, activation `0.128`
- `trigger_1.15__etaLow_0.992__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0057`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.139%`, activation `0.174`
- `trigger_1.20__etaLow_0.988__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0053`, sum disagreement strength `4`, mean turnover reduction `0.2%`, mean intervention `0.153%`, activation `0.128`
- `trigger_1.10__etaLow_0.996__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0050`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.101%`, activation `0.253`
- `trigger_1.20__etaLow_0.989__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0049`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.141%`, activation `0.128`
- `trigger_1.20__etaLow_0.990__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0044`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.128%`, activation `0.128`
- `trigger_1.15__etaLow_0.994__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0043`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.104%`, activation `0.174`
- `trigger_1.25__etaLow_0.985__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0040`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.138%`, activation `0.092`
- `trigger_1.20__etaLow_0.992__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0035`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.102%`, activation `0.128`
- `trigger_1.25__etaLow_0.987__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0034`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.119%`, activation `0.092`
- `trigger_1.25__etaLow_0.988__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0032`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.110%`, activation `0.092`
- `trigger_1.25__etaLow_0.989__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0029`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.101%`, activation `0.092`
- `trigger_1.15__etaLow_0.996__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0029`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.069%`, activation `0.174`
- `trigger_1.20__etaLow_0.994__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0026`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.077%`, activation `0.128`
- `trigger_1.25__etaLow_0.990__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0026`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.092%`, activation `0.092`
- `trigger_1.25__etaLow_0.992__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0021`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.073%`, activation `0.092`
- `trigger_1.20__etaLow_0.996__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0018`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.051%`, activation `0.128`
- `trigger_1.25__etaLow_0.994__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0016`, sum disagreement strength `4`, mean turnover reduction `0.1%`, mean intervention `0.055%`, activation `0.092`
- `trigger_1.25__etaLow_0.996__lb_20__ref_60`: mean positive-cost `ΔSharpe_exec=0.0010`, sum disagreement strength `4`, mean turnover reduction `0.0%`, mean intervention `0.037%`, activation `0.092`

Champion recommendation: `trigger_1.15__etaLow_0.987__lb_20__ref_60` with mean positive-cost `ΔSharpe_exec=0.0093`, zero-cost `|ΔSharpe_exec|=0.0095`, mean intervention `0.226%`, activation `0.174`
Runner-up recommendation: `trigger_1.15__etaLow_0.990__lb_20__ref_60` with mean positive-cost `ΔSharpe_exec=0.0072`, zero-cost `|ΔSharpe_exec|=0.0073`, mean intervention `0.174%`, activation `0.174`
