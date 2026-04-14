#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 SPLIT_ID CURRENT_CONFIG RUN_ROOT VALIDATION_START VALIDATION_END FINAL_START FINAL_END" >&2
  exit 2
fi

SPLIT_ID="$1"
CURRENT_CONFIG="$2"
RUN_ROOT="$3"
VALIDATION_START="$4"
VALIDATION_END="$5"
FINAL_START="$6"
FINAL_END="$7"

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN="/workspace/execution-aware-portfolio-rl/.venv/bin/python"

cd "$ROOT"
export PYTHON_BIN
export CURRENT_CONFIG
export RUN_ROOT
export VALIDATION_START
export VALIDATION_END
export FINAL_START
export FINAL_END
export RUN_TRAIN=0
export RUN_VALIDATION=1
export RUN_SELECT=1
export RUN_FINAL=1
export RUN_BASELINES=1
export RUN_PACK=1
export FINAL_MODE=selected_plus_baseline
export SEEDS="0 1 2"
export SAC_TOTAL_TIMESTEPS=25000

echo "[EVALFIX-START] split=${SPLIT_ID} seeds=${SEEDS} run_root=${RUN_ROOT}"
scripts/run_u27_control_eta_validation_first.sh
echo "[EVALFIX-END] split=${SPLIT_ID}"
