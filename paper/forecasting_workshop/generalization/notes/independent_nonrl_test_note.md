# Independent Non-RL Test Note

This note reports final/test evaluation for both selected independent non-RL comparator configurations.

Selection policy:
- No post-hoc reselection is performed on test.
- The candidate pair is fixed from the existing validation notes.
- The overall champion and runner-up are both taken from the redesigned deadband validation selection rule.
- Zero-cost near-flat on test uses the same threshold as validation: `|ΔSharpe_exec(kappa=0)| <= 0.01`.

Champion:
- Architecture: `arch_deadband_partial`
- Validation source: [arch_deadband_partial_validation_note.md](/workspace/execution-aware-portfolio-rl/paper/forecasting_workshop/generalization/notes/arch_deadband_partial_validation_note.md:1)
- Fixed config: `delta_0.08__eta_0.9988` with `delta=0.08`, `eta_db=0.9988`
- Test verdict: `Green`
Per-kappa summary:
- `kappa=0`: `ΔSharpe_exec=0.0078`, `ΔSharpe_tgt=0.0000`, `turnover_reduction=0.197%`, `disagreement=none`
- `kappa=5e-4`: `ΔSharpe_exec=0.0080`, `ΔSharpe_tgt=-0.0001`, `turnover_reduction=0.197%`, `disagreement=ranking_mismatch`
- `kappa=1e-3`: `ΔSharpe_exec=0.0082`, `ΔSharpe_tgt=-0.0003`, `turnover_reduction=0.197%`, `disagreement=ranking_mismatch`

Runner-up:
- Architecture: `arch_deadband_partial`
- Validation source: [arch_deadband_partial_validation_note.md](/workspace/execution-aware-portfolio-rl/paper/forecasting_workshop/generalization/notes/arch_deadband_partial_validation_note.md:1)
- Fixed config: `delta_0.08__eta_0.9994` with `delta=0.08`, `eta_db=0.9994`
- Test verdict: `Green`
Per-kappa summary:
- `kappa=0`: `ΔSharpe_exec=0.0055`, `ΔSharpe_tgt=0.0000`, `turnover_reduction=0.136%`, `disagreement=none`
- `kappa=5e-4`: `ΔSharpe_exec=0.0056`, `ΔSharpe_tgt=-0.0001`, `turnover_reduction=0.136%`, `disagreement=ranking_mismatch`
- `kappa=1e-3`: `ΔSharpe_exec=0.0058`, `ΔSharpe_tgt=-0.0003`, `turnover_reduction=0.136%`, `disagreement=ranking_mismatch`

Interpretation:
- Both rows are reported because champion status is for emphasis only, not for hiding the runner-up.
- The test readout should be interpreted conservatively and fed into Step 8 only after the broader audit is updated in a later step.
