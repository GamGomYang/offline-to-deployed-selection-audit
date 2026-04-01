# V1 Rolling-Origin Windows Protocol

This protocol strengthens the frozen-policy v1 paper without expanding the scope into retraining comparisons.

## Scope Guard

- Within each split, the same trained policy is reused across the internal eta arms.
- Only the execution mapping changes inside the main causal comparison.
- This study does not test retraining superiority.

## Split Set

- Split A
  - train: `2010-01-01` to `2017-12-31`
  - validation: `2018-01-01` to `2019-12-31`
  - test: `2020-01-01` to `2021-12-31`
- Split B
  - train: `2010-01-01` to `2019-12-31`
  - validation: `2020-01-01` to `2021-12-31`
  - test: `2022-01-01` to `2023-12-31`
- Split C
  - train: `2010-01-01` to `2021-12-31`
  - validation: `2022-01-01` to `2023-12-31`
  - test: `2024-01-01` to `2025-12-31`
  - reused from the canonical frozen baseline root

The machine-readable source of truth is [`split_definitions.json`](/workspace/execution-aware-portfolio-rl/frozen_protocol/rolling_windows_v1/split_definitions.json).

## Locked Experimental Objects

- parent baseline: `paper_v3_frozen_control_eta_20260324`
- eta grid: `1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02`
- kappa grid: `0.0, 0.0005, 0.001`
- seeds: `0..9`
- selection rule: same as the frozen baseline
- core metrics: same executed-path definitions as the frozen baseline

## Run Layout

- experiment root:
  - `outputs/extensions/v1_rolling_origin_windows/<timestamp>/`
- launched splits:
  - `split_a`
  - `split_b`
- canonical reference split:
  - `split_c`

## Commands

- Launch A and B in detached tmux sessions:

```bash
prl-dow30/scripts/launch_v1_rolling_origin_tmux.sh
```

- Check status:

```bash
python3 prl-dow30/scripts/rolling_origin_status.py --experiment-root <experiment-root>
```

- Aggregate completed split results:

```bash
python3 prl-dow30/scripts/analyze_v1_rolling_origin_windows.py \
  --experiment-root <experiment-root> \
  --output-dir <experiment-root>/analysis
```

## Output Targets

- per-split selected eta summary
- per-split kappa summary
- split-level positive-cost summary
- split-median summary across completed splits
- verdict JSON for the two-of-three directional rule

## Success Criterion

At least two of the three splits should preserve the positive-cost directional result, meaning the selected operating point keeps `median_delta_sharpe_net_lin > 0` against `eta=1.0` for both positive-cost kappas.
