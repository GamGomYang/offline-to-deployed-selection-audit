Load-following support-domain reading

- Verdict: appendix_support_only
- Resolution: 60min
- Q1: mixed support
- Q2: secondary corroboration only
- Balance: balance_warn
- Evaluation uses disjoint client groups rather than overlapping rolling windows.
- Zero-row mismatch is not a tie-policy artifact; zero-row winning sets are singletons in all seeds.
- Locked zero row: 3/10 mismatches, mean deployed gap 0.000226, median deployed gap 0.0.
- A grouping-only balance repair reduced imbalance and lowered the held-out Q1 target-clip rate while preserving the Q2 drift structure.
- Balance-repair zero row: 3/10 mismatches, mean deployed gap 0.000117, median deployed gap 0.0.
- Best paper-facing interpretation: near-zero seed instability / grouping granularity, not a clean zero-friction story.
- A final restricted cap/margin search found no jointly feasible configuration that preserves the Q2 gate while satisfying the remaining Q1 requirement.
- This domain remains appendix-only secondary corroboration in the current submission round.
