# Inventory Q2 Selection Drift

Table 1 reports the operational selection consequence in the required inventory domain under one fixed responsive replenishment interface.

Winner columns report the most frequent seed-level best model at each friction level.
Positive mean deployed gap means the forecast-selected model underperforms the deployed-selected model by that amount.
Deployed-suboptimal seeds / total reports the number of seeds in which the forecast-selected model is not a deployed best model.

## Friction Rows
- friction=0.00: forecast_winner=Small MLP, deployed_winner=Small MLP, agreement_rate=0.70, mean_deployed_gap=0.022, deployed_suboptimal=3/10
- friction=0.25: forecast_winner=Small MLP, deployed_winner=Linear AR, agreement_rate=0.70, mean_deployed_gap=0.053, deployed_suboptimal=3/10
- friction=0.50: forecast_winner=Small MLP, deployed_winner=Moving average (7), agreement_rate=0.10, mean_deployed_gap=0.308, deployed_suboptimal=9/10
- friction=1.00: forecast_winner=Small MLP, deployed_winner=Moving average (7), agreement_rate=0.00, mean_deployed_gap=1.187, deployed_suboptimal=10/10
