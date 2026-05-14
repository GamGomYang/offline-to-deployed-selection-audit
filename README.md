# Offline-to-Deployed Selection Transfer Audit

This repository contains the manuscript artifacts and audit code for:

**Auditing Offline-to-Deployed Selection Transfer under Fixed Decision Interfaces**

The paper studies a narrow pre-deployment model-selection question: when an offline validation score selects a model, does that selected model remain the deployed-utility winner after every candidate is passed through the same fixed decision interface and friction model?

The repository name used by this README is:

```text
offline-to-deployed-selection-transfer-audit
```

## Core Idea

Offline pipelines often rank candidate models by a validation score computed on collected data. In deployment, those model outputs are usually consumed by a fixed interface, for example:

- threshold alerts
- hysteresis rules
- budgeted top-k actions
- residual-warning screens
- replenishment rules

The audit checks whether the offline-selected model remains best after that fixed interface and deployment friction are applied. It does not argue that offline validation metrics are invalid for prediction, and it does not propose a universal deployed metric. It adds a report-card check for cases where offline validation is being used as deployment-facing model-selection advice.

## Report Card

For each task, interface, and friction level, the audit records:

- the offline-selected model
- the deployed-utility winner
- transfer or agreement rate
- deployed-suboptimal share/count
- paired deployed-utility shortfall
- tie and uncertainty diagnostics

Selection transfer is preserved when the offline-selected model and deployed-utility winner agree. A positive deployed shortfall means the offline-selected model is deployed-suboptimal under the specified interface and friction model.

## Main Findings From The PDF

| Task | Fixed interface | Friction | Offline-selected | Deployed winner | Transfer | Suboptimal cases | Mean shortfall |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| Synthetic | zero-friction anchor | 0.00 | Naive last | Naive last | 1.00 | 0/20 | 0.000 |
| Event warning | threshold tau=0.55 | 0.50 | Reactive sharp | Calibrated | 0.31 | 69/100 | 0.011 |
| Event warning | threshold tau=0.55 | 1.00 | Reactive sharp | Smoother | 0.01 | 99/100 | 0.057 |
| Budgeted traffic alert | budget k=249 | 0.50 | Reactive short | Smoother | 0.00 | 100/100 | 7.02 |
| Budgeted traffic alert | budget k=249 | 1.00 | Reactive short | Smoother | 0.00 | 100/100 | 24.65 |
| PM2.5 warning | residual warning | 1.00 | Reactive lag-1 | Long smoother | 0.00 | 720/825 | 17.94 |
| Inventory replenishment | replenishment | 1.00 | Small MLP | MA(7) | 0.01 | 99/100 | 1.03 |

The qualitative result is that offline-selected candidates can become deployed-suboptimal once switching or adjustment costs are applied through the fixed interface. Event warning and Traffic-Hourly are the main prediction-to-decision evidence; PM2.5 and inventory provide residual or operational support.

## What Is In This Repository

```text
121_Auditing_Offline_to_Deploy.pdf   submitted/reference PDF used for this README
paper/forecasting_workshop/          manuscript source, compiled PDF, tables, and figures
scripts/forecast_eval/               paper-facing audit, uncertainty, and robustness builders
scripts/actionabilitybench_lite/      lightweight actionability/selection experiment scripts
scripts/deployed_selection_tsfm/      deployed-selection time-series forecasting scripts
scripts/reporting/                   report-card interval utilities
outputs/                             generated experiment outputs and frozen result snapshots
```

## Rebuilding Paper Artifacts

The generated manuscript assets are already present under `paper/forecasting_workshop/`. To rebuild selected paper-facing figures and tables:

```bash
python paper/forecasting_workshop/results/build_v2_main_figures.py
python scripts/forecast_eval/build_workshop_freeze_and_uncertainty.py
```

To compile the manuscript source:

```bash
cd paper/forecasting_workshop
latexmk -pdf paper_forecasting_workshop_v2.tex
```

The scripts assume a Python environment with the usual scientific stack used by the builders, including `pandas` and `matplotlib`, and a LaTeX installation for PDF figure/manuscript generation.

## Interpretation Boundaries

The audit is a reporting diagnostic, not a new benchmark suite or a new forecasting model. Deployed utility is interface-specific, simulator-specific, and friction-specific. The intended use is to make model-selection transfer visible before deployment, online evaluation, or adaptation.
