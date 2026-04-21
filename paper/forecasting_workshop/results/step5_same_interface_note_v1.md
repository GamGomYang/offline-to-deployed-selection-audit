# Step 5: Same-Interface Q2 Summary

Step 5 is a Q2 evidence package rather than a new experiment. Its claim is that under a fixed deployed interface, forecast-metric ranking does not reliably determine deployed operational ranking as frictions increase. This is distinct from the Q1 target-versus-executed separation result.

## Domain Rows
- Synthetic: status=pass, zero_flip_rate=0.000, zero_spearman=1.000, strongest_flip_pair=linear_ar|noisy_overreactive
- Inventory: status=pass, zero_flip_rate=0.083, zero_spearman=0.880, strongest_flip_pair=mlp_small|moving_average_7
- Portfolio: status=excluded, zero_flip_rate=0.611, zero_spearman=-0.200, strongest_flip_pair=ewma_20|rolling_mean_20

## Count-Based Verdict
- 2/2 required domains show stronger positive-friction ranking mismatch than at zero friction.
- 2/2 required domains show lower rank correlation as friction increases.
- Portfolio omitted by stretch gate.
