# Reproducibility Assets

This directory collects paper-facing support material that does not need to sit at the repository root.

Layout:
- `BASELINE_PROTOCOL.md`: frozen baseline protocol notes
- `manifests/`: baseline and extension manifests plus figure/paper artifact manifests
- `rebuilds/`: archived paper rebuilds used for manuscript statistics and tables
- `checks/`: smoke-check and guard-check runs
- `outputs/`: root-level legacy outputs and reproduction runs
- `reproduce_main_results.sh`: convenience entry point for replaying the frozen paper baseline

The main manuscript still points to `frozen_protocol/` for locked protocol definitions and to `prl-dow30/` for the active training/evaluation code.
