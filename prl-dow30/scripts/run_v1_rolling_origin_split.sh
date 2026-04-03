#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_ROOT=""
SPLIT_ID=""
CURRENT_CONFIG=""
VALIDATION_START=""
VALIDATION_END=""
FINAL_START=""
FINAL_END=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --split-id)
      SPLIT_ID="$2"
      shift 2
      ;;
    --current-config)
      CURRENT_CONFIG="$2"
      shift 2
      ;;
    --validation-start)
      VALIDATION_START="$2"
      shift 2
      ;;
    --validation-end)
      VALIDATION_END="$2"
      shift 2
      ;;
    --final-start)
      FINAL_START="$2"
      shift 2
      ;;
    --final-end)
      FINAL_END="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ROOT" || -z "$SPLIT_ID" || -z "$CURRENT_CONFIG" || -z "$VALIDATION_START" || -z "$VALIDATION_END" || -z "$FINAL_START" || -z "$FINAL_END" ]]; then
  echo "Missing required arguments." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

export PYTHON_CMD="${PYTHON_CMD:-${PYTHON_BIN}}"
export CURRENT_CONFIG="$CURRENT_CONFIG"
export RUN_ROOT="$RUN_ROOT"
export VALIDATION_START="$VALIDATION_START"
export VALIDATION_END="$VALIDATION_END"
export FINAL_START="$FINAL_START"
export FINAL_END="$FINAL_END"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_VALIDATION="${RUN_VALIDATION:-1}"
export RUN_SELECT="${RUN_SELECT:-1}"
export RUN_FINAL="${RUN_FINAL:-1}"
export RUN_BASELINES="${RUN_BASELINES:-1}"
export RUN_PACK="${RUN_PACK:-1}"
export FINAL_MODE="${FINAL_MODE:-selected_plus_baseline}"
export SAC_TOTAL_TIMESTEPS="${SAC_TOTAL_TIMESTEPS:-0}"

cd "$ROOT"
scripts/run_u27_control_eta_validation_first.sh

"${PYTHON_BIN}" scripts/build_mechanism_decomposition.py \
  --validation-root "$RUN_ROOT/validation_eta" \
  --final-root "$RUN_ROOT/final_eta" \
  --selection-json "$RUN_ROOT/validation_eta/selection/validation_eta_selection.json" \
  --output-dir "$RUN_ROOT/paper_pack/mechanism"

"${PYTHON_BIN}" scripts/build_selection_rule_defense.py \
  --selection-csv "$RUN_ROOT/validation_eta/selection/validation_eta_selection.csv" \
  --selection-json "$RUN_ROOT/validation_eta/selection/validation_eta_selection.json" \
  --mechanism-frontier-csv "$RUN_ROOT/paper_pack/mechanism/mechanism_frontier_summary.csv" \
  --mechanism-pair-summary-csv "$RUN_ROOT/paper_pack/mechanism/selected_vs_eta1_mechanism_summary.csv" \
  --output-dir "$RUN_ROOT/paper_pack/selection_defense"

echo "ROLLING_SPLIT_ID=${SPLIT_ID}"
echo "ROLLING_RUN_ROOT=${RUN_ROOT}"
