# Step 5: Same-Interface Q2 Summary

Step 5 is a descriptive Q2 package rather than a new experiment. Its role is to summarize the same-interface ranking evidence after the fifth inventory baseline is added, while the main text now centers the inventory selection-drift consequence directly.

## Domain Rows
- Synthetic: status=pass, zero_flip_rate=0.000, zero_spearman=1.000, strongest_flip_pair=Linear AR / Reactive extrapolation heuristic
- Inventory: status=fail, zero_flip_rate=0.120, zero_spearman=0.840, strongest_flip_pair=Small MLP / Moving average (7)
- Portfolio: status=excluded, zero_flip_rate=0.611, zero_spearman=-0.200, strongest_flip_pair=EWMA (20) / Rolling mean (20)

## Count-Based Verdict
- 2/2 required domains show stronger positive-friction ranking mismatch than at zero friction.
- 2/2 required domains show lower rank correlation as friction increases.
- Portfolio omitted by stretch gate.
