# Step 2 Candidate Lock

This file freezes the current Step 2 synthetic benchmark as the main candidate for the workshop reframing push.

## Locked Version

- Lock date: `2026-04-21`
- Variant: `split_q1_q2_v1`
- Output directory: `outputs/forecast_eval/synthetic/`
- Snapshot directory: `outputs/forecast_eval/synthetic_step2_candidate_lock/`

## Locked Configs

- `Q1` config: `procar1_jumps_w5_a2.60_n0.06_eta0.20_lam2.00`
- `Q2` config: `procblock_levels_w5_a1.10_n0.00_eta0.25_lam1.00_bs1.00_bn0.08`

## Why This Is The Main Candidate

- `Q1` is stronger than the earlier single-benchmark versions.
- `Q1` keeps exact zero-friction agreement and shows a clean increasing gap with friction.
- `Q2` gives a cleaner ranking-stress benchmark than the single-benchmark versions.
- `Q2` keeps exact zero-friction agreement and shows stable positive-friction ranking mismatch across seeds.
- The remaining limitation is a `0.5` positive-friction plateau rather than a strict increase, but low-risk local tuning around the locked `Q2` config did not produce a strict monotone alternative.

## Practical Rule

- Treat this as the Step 2 main version for paper assembly and summary writing.
- Do not overwrite it conceptually with later exploratory runs unless we explicitly unlock and replace it.
- If more synthetic tuning is proposed later, compare against this lock rather than against the older single-benchmark runs.
