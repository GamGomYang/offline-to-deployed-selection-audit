# Volatility-Scaled Eta Validation Note

This note summarizes the validation-only grid search for `arch_vol_scaled_eta`.

The validation eligibility rule uses the same documented near-flat threshold as the deadband comparator:

- `|ΔSharpe_exec(kappa=0)| <= 0.01`

The remaining eligibility checks are also the same:

- both positive-cost rows (`kappa in {5e-4, 1e-3}`) must have `ΔSharpe_exec > 0`
- at least one positive-cost row must show `disagreement_type in {ranking_mismatch, sign_flip}`

This stage uses the same Step 8 pair-audit logic and the same deadband-style validation eligibility rule. Test has not been run in this step, and there is no comparator-versus-deadband comparison here.

Eligible configurations:
- `alpha_0.024__etaMin_0.95__lb_20`: mean positive-cost `ΔSharpe_exec=0.0090`, sum disagreement strength `4`, mean turnover reduction `0.3%`
- `alpha_0.025__etaMin_0.90__lb_20`: mean positive-cost `ΔSharpe_exec=0.0082`, sum disagreement strength `4`, mean turnover reduction `0.3%`
- `alpha_0.026__etaMin_0.90__lb_20`: mean positive-cost `ΔSharpe_exec=0.0058`, sum disagreement strength `4`, mean turnover reduction `0.2%`
- `alpha_0.025__etaMin_0.95__lb_20`: mean positive-cost `ΔSharpe_exec=0.0051`, sum disagreement strength `4`, mean turnover reduction `0.2%`
- `alpha_0.024__etaMin_0.98__lb_20`: mean positive-cost `ΔSharpe_exec=0.0046`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `alpha_0.027__etaMin_0.90__lb_20`: mean positive-cost `ΔSharpe_exec=0.0045`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `alpha_0.026__etaMin_0.95__lb_20`: mean positive-cost `ΔSharpe_exec=0.0034`, sum disagreement strength `4`, mean turnover reduction `0.2%`
- `alpha_0.027__etaMin_0.95__lb_20`: mean positive-cost `ΔSharpe_exec=0.0030`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `alpha_0.028__etaMin_0.90__lb_20`: mean positive-cost `ΔSharpe_exec=0.0028`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `alpha_0.024__etaMin_0.99__lb_20`: mean positive-cost `ΔSharpe_exec=0.0024`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `alpha_0.028__etaMin_0.95__lb_20`: mean positive-cost `ΔSharpe_exec=0.0019`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `alpha_0.025__etaMin_0.98__lb_20`: mean positive-cost `ΔSharpe_exec=0.0017`, sum disagreement strength `4`, mean turnover reduction `0.1%`
- `alpha_0.027__etaMin_0.98__lb_20`: mean positive-cost `ΔSharpe_exec=0.0014`, sum disagreement strength `4`, mean turnover reduction `0.0%`
- `alpha_0.026__etaMin_0.98__lb_20`: mean positive-cost `ΔSharpe_exec=0.0013`, sum disagreement strength `4`, mean turnover reduction `0.1%`

The recommended volatility-scaled configuration for later test evaluation is `alpha_0.024__etaMin_0.95__lb_20`. It is selected by largest mean positive-cost `ΔSharpe_exec`, then largest summed disagreement strength, then largest mean positive-cost turnover reduction, and finally by the simpler parameter preference that favors larger `eta_min` and then larger `alpha`.
