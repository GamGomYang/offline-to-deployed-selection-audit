# Volatility-Spike Eta Test Note

This note reports final/test evaluation for both selected `arch_vol_spike_eta` configurations.

Selection policy:
- No post-hoc reselection is performed on test.
- The candidate pair is fixed from the redesigned validation-only volatility-spike note and CSV.
- Zero-cost near-flat on test uses the same threshold as validation: `|ΔSharpe_exec(kappa=0)| <= 0.01`.

Champion:
- Architecture: `arch_vol_spike_eta`
- Validation source: [arch_vol_spike_eta_validation_note.md](/workspace/execution-aware-portfolio-rl/paper/forecasting_workshop/generalization/notes/arch_vol_spike_eta_validation_note.md:1)
- Fixed config: `trigger_1.15__etaLow_0.987__lb_20__ref_60`
- Test verdict: `Green`
Per-kappa summary:
- `kappa=0`: `ΔSharpe_exec=0.0046`, `ΔSharpe_tgt=0.0000`, `turnover_reduction=0.338%`, `mean_eta=0.9963`, `activation=0.287`, `disagreement=none`
- `kappa=5e-4`: `ΔSharpe_exec=0.0042`, `ΔSharpe_tgt=-0.0000`, `turnover_reduction=0.338%`, `mean_eta=0.9963`, `activation=0.287`, `disagreement=ranking_mismatch`
- `kappa=1e-3`: `ΔSharpe_exec=0.0039`, `ΔSharpe_tgt=-0.0000`, `turnover_reduction=0.338%`, `mean_eta=0.9963`, `activation=0.287`, `disagreement=ranking_mismatch`

Runner-up:
- Architecture: `arch_vol_spike_eta`
- Validation source: [arch_vol_spike_eta_validation_note.md](/workspace/execution-aware-portfolio-rl/paper/forecasting_workshop/generalization/notes/arch_vol_spike_eta_validation_note.md:1)
- Fixed config: `trigger_1.15__etaLow_0.990__lb_20__ref_60`
- Test verdict: `Green`
Per-kappa summary:
- `kappa=0`: `ΔSharpe_exec=0.0035`, `ΔSharpe_tgt=0.0000`, `turnover_reduction=0.262%`, `mean_eta=0.9971`, `activation=0.287`, `disagreement=none`
- `kappa=5e-4`: `ΔSharpe_exec=0.0033`, `ΔSharpe_tgt=0.0000`, `turnover_reduction=0.262%`, `mean_eta=0.9971`, `activation=0.287`, `disagreement=ranking_mismatch`
- `kappa=1e-3`: `ΔSharpe_exec=0.0030`, `ΔSharpe_tgt=0.0000`, `turnover_reduction=0.262%`, `mean_eta=0.9971`, `activation=0.287`, `disagreement=ranking_mismatch`

Interpretation:
- Both rows are reported because champion status is for emphasis only, not for hiding the runner-up.
- The test readout should be interpreted conservatively and only promoted into Step 8 if the broader architecture audit is updated in a later step.
