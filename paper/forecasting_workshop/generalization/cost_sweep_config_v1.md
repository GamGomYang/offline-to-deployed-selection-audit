# Cost Sweep Configuration v1

This document fixes the experiment configuration for the cost-regime sweep used as the highest-priority generalization check in the forecasting workshop paper. It is a robustness/generalization exercise under the paper's existing selected-point interpretation, not a new main result class.

## Purpose

The cost sweep asks whether the paper's implementation-side reading remains aligned when proportional trading costs are evaluated on a denser grid. The goal is not to replace the main selected-point result. The goal is to test whether the same qualitative pattern remains visible when friction levels are varied more finely under the same evaluation pipeline.

## Locked Comparison

- compared arms: `eta=1.0` versus `eta=0.5`
- interpretation: selected-point comparison only
- policy status: same learned policy as the documented paper result
- predictive information: fixed
- evaluation pipeline: same as the main paper
- accounting basis: same executed-path accounting and same target-versus-executed diagnostics as the main paper

## Fixed Cost Grid

Evaluate the selected-point comparison on the following cost grid only:

- `0`
- `1e-4`
- `2e-4`
- `5e-4`
- `1e-3`
- `2e-3`

This grid is fixed for the sweep. Adding extra `kappa` values later would count as a scope change rather than a continuation of this registered check.

## Primary Evaluation

Executed-path quantities remain primary.

### Primary Metric

- executed-path net Sharpe

### Primary Comparison

For each `kappa` in the fixed grid, report the executed-path difference between:

- baseline arm: `eta=1.0`
- selected arm: `eta=0.5`

### Supporting Executed-Path Metrics

- executed turnover
- realized transaction cost, if available in the same evaluation records

## Target-Path Diagnostics

Target-path quantities remain diagnostic only and must not be promoted to headline evaluation.

Collect the following target-versus-executed diagnostics where available:

- target turnover
- executed turnover
- `TOtgt / TOexec`
- tracking discrepancy between target and executed allocations
- final path gap or final equity gap
- target-based net Sharpe, if available from the same cached evaluation pipeline

These diagnostics are collected only to test whether the interpretation changes when the evaluation object changes.

## Required Outputs

The sweep should produce enough material to support either appendix reporting or an internal decision under the pre-registered Green/Yellow/Red rules.

Required outputs:

- one row per `kappa` for the executed-path selected-point comparison
- one row per `kappa` for the target-versus-executed diagnostic comparison, where available
- one compact markdown note interpreting the sweep under the pre-registered decision rules
- one plotting-ready table for a cost-sensitivity figure, if figure generation is needed later

Recommended file pattern:

- `paper/forecasting_workshop/generalization/cost_sweep_results_v1.csv`
- `paper/forecasting_workshop/generalization/cost_sweep_diagnostics_v1.csv`
- `paper/forecasting_workshop/generalization/cost_sweep_note_v1.md`
- `paper/forecasting_workshop/generalization/cost_sweep_plot_data_v1.csv`

These filenames are recommended for consistency, but the core requirement is that the outputs cleanly separate:

- executed-path primary results
- target-path diagnostic quantities
- interpretation note

## What Must Stay Fixed

- same two `eta` arms: `1.0` and `0.5`
- same learned policy
- same predictive information
- same evaluation code path used for the current paper result
- same executed-path accounting conventions
- same role assignment: executed-path primary, target-path diagnostic

## What This Sweep Is Not

- not a new model-selection exercise
- not an adaptive-`eta` study
- not a retraining experiment
- not a new main empirical result class
- not evidence for a stronger predictive signal

## Reporting Rule

If the sweep is later summarized in the paper, it should be described as:

- a denser cost-regime robustness check under the same selected-point comparison

It should not be described as:

- a new main result
- a new method comparison
- a broader theorem

## Link to Decision Rules

Interpretation of the sweep must follow:

- `paper/forecasting_workshop/generalization/generalization_decision_rules_v1.md`

In particular:

- Green permits a brief main-text sentence plus appendix support
- Yellow stays appendix-only
- Red does not support a broader robustness statement
