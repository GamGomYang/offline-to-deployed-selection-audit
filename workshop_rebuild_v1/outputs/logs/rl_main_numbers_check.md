# RL Main Numbers Check

## Source

- Package lane: `workshop_rebuild_v1`
- Canonical stats source: `repro/rebuilds/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.csv`
- Build mode: clean rebuild packaging from locked canonical artifacts only
- Experiments rerun for this step: `No`

## Locked Setup Check

- baseline eta = `1.0`: `Pass`
- selected operating point eta = `0.5`: `Pass`
- kappa rows = `{0, 5e-4, 1e-3}`: `Pass`
- headline metric = executed-path net Sharpe: `Pass`
- comparison scope = `eta=1.0` vs `eta=0.5` only: `Pass`
- target-path quantities used as headline metrics: `No`

## Main Readout

| kappa | Net Sharpe (`eta=1.0`) | Net Sharpe (`eta=0.5`) | Paired-median Delta Sharpe | TOexec (`eta=1.0`) | TOexec (`eta=0.5`) |
| --- | --- | --- | --- | --- | --- |
| `0` | `1.1542` | `1.1554` | `-0.0002` | `0.02200` | `0.01095` |
| `5e-4` | `1.1340` | `1.1447` | `+0.0105` | `0.02200` | `0.01095` |
| `1e-3` | `1.1127` | `1.1340` | `+0.0213` | `0.02200` | `0.01095` |

## Validation Against Package Criteria

- positive-cost gains present: `Pass`
  - `kappa=5e-4`: `+0.0105`
  - `kappa=1e-3`: `+0.0213`
- `kappa=0` remains negligible or near-flat: `Pass`
  - observed paired-median delta Sharpe: `-0.0002`
- turnover reduction remains large: `Pass`
  - baseline `TOexec`: `0.02200`
  - selected `TOexec`: `0.01095`
  - absolute reduction: `0.01105`
  - relative reduction: about `50.2%`
- wording discipline remains implementation-side, not alpha-side: `Pass`

## Deviations

- None relative to the locked setup used for this package build.
