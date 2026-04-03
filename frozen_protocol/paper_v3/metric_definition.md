# Metric Definition

This file freezes the baseline metric definitions used by the paper baseline `paper_v3_frozen_control_eta_20260324`.

## Core Path

- `w_tgt,t`: target portfolio produced by the policy
- `w_exec,t`: executed portfolio used for realized dynamics
- `w_exec,t = (1 - eta_t) * w_exec,t-1 + eta_t * w_tgt,t`

## Turnover and Cost

- target turnover:
  - `TO_tgt,t = || w_tgt,t - w_exec,t-1 ||_1`
  - diagnostic only
- executed turnover:
  - `TO_exec,t = || w_exec,t - w_exec,t-1 ||_1`
  - core accounting quantity
- realized cost:
  - `cost_t = kappa * TO_exec,t`

## Realized Return

- executed-path net linear return:
  - `R_t+1 = dot(w_exec,t, arithmetic_return_t+1) - cost_t`

## Primary Evaluation Metric

- `sharpe_net_lin`
- annualization: `sqrt(252)`
- risk-free rate: `0`
- computed from executed-path net linear returns only

## Core Report Set

- net Sharpe
- CAGR
- max drawdown
- average executed turnover
- realized cost

## Diagnostics

- average target turnover
- tracking error / misalignment summaries
- mean absolute return gap
- final equity gap
- maximum daily absolute gap

Core metrics are the only metrics allowed to drive the baseline validation score and main held-out selection claim.
