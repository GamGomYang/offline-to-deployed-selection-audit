# Multi-Split Plan v1

This document fixes a compact multi-split temporal robustness plan for the forecasting workshop paper. The goal is to weaken the single-split artifact criticism without changing the paper's identity, selection logic, or evaluation hierarchy. The resulting package is intended for appendix robustness support rather than for a new main-result class.

## Purpose

The multi-split check asks whether the documented selected-point interpretation remains directionally compatible across additional chronological windows. It is not a claim-expansion exercise. It is a narrow robustness check around the current case-study result.

## Total Number of Splits

Use exactly `4` temporal splits:

- `split_a`
- `split_b`
- `split_c`
- `split_d`

`split_d` is the current canonical paper split.

## Split Definitions

Use expanding training windows with fixed-length validation and test windows.

| Split | Train window | Validation window | Test window | Role |
| --- | --- | --- | --- | --- |
| `split_a` | `2010-01-01` to `2015-12-31` | `2016-01-01` to `2017-12-31` | `2018-01-01` to `2019-12-31` | earliest robustness split |
| `split_b` | `2010-01-01` to `2017-12-31` | `2018-01-01` to `2019-12-31` | `2020-01-01` to `2021-12-31` | mid-history robustness split |
| `split_c` | `2010-01-01` to `2019-12-31` | `2020-01-01` to `2021-12-31` | `2022-01-01` to `2023-12-31` | pre-canonical robustness split |
| `split_d` | `2010-01-01` to `2021-12-31` | `2022-01-01` to `2023-12-31` | `2024-01-01` to `2025-12-31` | current canonical reference split |

## Why These Splits

- chronological order is preserved throughout
- each split changes both the validation and test windows in a meaningful way
- the latest split remains aligned with the current paper
- the windows are long enough to keep the robustness package interpretable for a workshop appendix
- the plan avoids a short and potentially unstable `2026 YTD` test window

## What Remains Fixed Across Splits

The following must remain fixed across all four splits:

- paper identity: forecasting-to-decision evaluation paper
- policy family: the same PRL frozen-policy family used in the current paper
- selected-point comparison: `eta=1.0` versus `eta=0.5`
- evaluation hierarchy: executed-path quantities primary, target-path quantities diagnostic only
- evaluation pipeline: same backtest code path, same accounting conventions, same metric definitions
- cost rows: the locked main rows `kappa in {0, 5e-4, 1e-3}`
- signal-state setup: same frozen signal-state design and same feature construction rule
- model selection logic: validation-first, then held-out test
- interpretation rule: implementation-side reading only, not stronger-predictor wording

## What Changes Across Splits

The following are allowed to change across splits:

- the train end date
- the validation window
- the test window
- the split-specific trained model instance produced by the same model family and protocol

These changes are the point of the robustness check. They do not authorize any change in the claim wording or in the main paper's evidence hierarchy.

## Overlap Policy

- `No overlap` is allowed between validation and test within a split.
- `No overlap` is planned between the test window of one split and the test window of another split.
- `Adjacency` is allowed: the test window of one split may become the validation window of the next split.
- `Training overlap` is allowed and expected because the training design is expanding-window rather than disjoint-window.

This overlap policy is intentional. It keeps the plan chronological and realistic while avoiding an unnecessarily fragmented appendix experiment.

## Core Recorded Metrics

For each split and each locked cost row, record:

- executed-path net Sharpe for `eta=1.0`
- executed-path net Sharpe for `eta=0.5`
- paired or median executed-path delta Sharpe
- executed turnover for both arms
- target turnover for `eta=0.5`, if available
- tracking discrepancy for `eta=0.5`
- final path gap for `eta=0.5`
- target-based delta Sharpe, if available
- positive-cost direction flag
- zero-cost near-flat flag

Primary reading:

- executed-path net Sharpe on the positive-cost rows

Diagnostic-only reading:

- target-based quantities
- target-versus-executed gap quantities

## Reporting Shape

The planned robustness package should be compact:

- one split-summary table
- one appendix-only figure or small panel if needed
- one short markdown verdict note tied to the pre-registered rules

The package should not become a second empirical center of gravity.

## Success / Failure Logic Reference

Interpretation must follow:

- [generalization_decision_rules_v1.md](/workspace/execution-aware-portfolio-rl/paper/forecasting_workshop/generalization/generalization_decision_rules_v1.md)

In particular for multi-split temporal robustness:

- `Green`: at least `3 of 4` splits preserve the positive-cost executed-path direction, turnover reduction is mostly preserved, and the zero-cost row stays flat or mixed/noisy rather than systematically positive
- `Yellow`: only `2 of 4` splits preserve the positive-cost direction, or the positive-cost advantage becomes unstable across splits
- `Red`: fewer than `2 of 4` splits preserve the positive-cost direction, or the zero-cost row often becomes strongly positive in a way that weakens the friction-specific interpretation

## Paper-Writing Consequences

- `Green`: one short sentence may enter the main paper, with details kept in the appendix
- `Yellow`: appendix only
- `Red`: do not use as supporting evidence for a broader robustness statement

## Final Constraint

Even if this plan later yields a Green result, the workshop paper remains a narrow forecasting-to-decision case study. The multi-split package is a robustness layer around the existing selected-point result, not a license to make universal or benchmark-dominance claims.
