#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT_ID=""
RUN_ROOT=""
CURRENT_CONFIG=""
VALIDATION_START=""
VALIDATION_END=""
FINAL_START=""
FINAL_END=""
SEEDS_RAW=""
EXPECTED_SEEDS_RAW="0 1 2 3 4 5 6 7 8 9"
FINALIZER="0"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split-id)
      SPLIT_ID="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
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
    --seeds)
      SEEDS_RAW="$2"
      shift 2
      ;;
    --expected-seeds)
      EXPECTED_SEEDS_RAW="$2"
      shift 2
      ;;
    --finalizer)
      FINALIZER="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SPLIT_ID" || -z "$RUN_ROOT" || -z "$CURRENT_CONFIG" || -z "$VALIDATION_START" || -z "$VALIDATION_END" || -z "$FINAL_START" || -z "$FINAL_END" || -z "$SEEDS_RAW" ]]; then
  echo "Missing required arguments." >&2
  exit 2
fi

read -r -a SEEDS <<< "$SEEDS_RAW"
read -r -a EXPECTED_SEEDS <<< "$EXPECTED_SEEDS_RAW"

SNAPSHOT_CONFIG="${RUN_ROOT}/configs/snapshot_control.yaml"
TRAIN_ROOT="${RUN_ROOT}/train_control"
MODEL_ROOT="${TRAIN_ROOT}/models"

mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}"

log() {
  echo "$1"
}

wait_for_all_models() {
  while true; do
    local missing=()
    local seed
    for seed in "${EXPECTED_SEEDS[@]}"; do
      if [[ -f "${MODEL_ROOT}/prl_seed${seed}_final.zip" ]]; then
        continue
      fi
      if ls "${MODEL_ROOT}"/*_seed"${seed}"_prl_*_final.zip >/dev/null 2>&1; then
        continue
      fi
      if ls "${MODEL_ROOT}"/*_seed"${seed}"_*_final.zip >/dev/null 2>&1; then
        continue
      fi
      if [[ ! -f "${MODEL_ROOT}/prl_seed${seed}_final.zip" ]]; then
        missing+=("${seed}")
      fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
      log "[WAIT] all expected models are present for ${SPLIT_ID}"
      return 0
    fi
    log "[WAIT] ${SPLIT_ID} waiting for remaining seeds: ${missing[*]}"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

cd "${ROOT}"
export PYTHONPATH="."

log "[BUCKET-START] split=${SPLIT_ID} seeds=${SEEDS[*]} finalizer=${FINALIZER}"

for seed in "${SEEDS[@]}"; do
  log "[TRAIN-START] split=${SPLIT_ID} seed=${seed}"
  "${PYTHON_BIN}" scripts/run_train.py \
    --config "${SNAPSHOT_CONFIG}" \
    --model-type prl \
    --seed "${seed}" \
    --offline \
    --output-root "${TRAIN_ROOT}"
  log "[TRAIN-END] split=${SPLIT_ID} seed=${seed}"
done

if [[ "${FINALIZER}" != "1" ]]; then
  log "[BUCKET-END] split=${SPLIT_ID} seeds=${SEEDS[*]} finalizer=0"
  exit 0
fi

wait_for_all_models

export PYTHON_CMD="${PYTHON_CMD:-${PYTHON_BIN}}"
export CURRENT_CONFIG="${CURRENT_CONFIG}"
export RUN_ROOT="${RUN_ROOT}"
export VALIDATION_START="${VALIDATION_START}"
export VALIDATION_END="${VALIDATION_END}"
export FINAL_START="${FINAL_START}"
export FINAL_END="${FINAL_END}"
export RUN_TRAIN="0"
export RUN_VALIDATION="1"
export RUN_SELECT="1"
export RUN_FINAL="1"
export RUN_BASELINES="1"
export RUN_PACK="1"
export FINAL_MODE="${FINAL_MODE:-selected_plus_baseline}"
export SAC_TOTAL_TIMESTEPS="${SAC_TOTAL_TIMESTEPS:-0}"

log "[FINALIZER-START] split=${SPLIT_ID}"
scripts/run_u27_control_eta_validation_first.sh

"${PYTHON_BIN}" scripts/build_mechanism_decomposition.py \
  --validation-root "${RUN_ROOT}/validation_eta" \
  --final-root "${RUN_ROOT}/final_eta" \
  --selection-json "${RUN_ROOT}/validation_eta/selection/validation_eta_selection.json" \
  --output-dir "${RUN_ROOT}/paper_pack/mechanism"

"${PYTHON_BIN}" scripts/build_selection_rule_defense.py \
  --selection-csv "${RUN_ROOT}/validation_eta/selection/validation_eta_selection.csv" \
  --selection-json "${RUN_ROOT}/validation_eta/selection/validation_eta_selection.json" \
  --mechanism-frontier-csv "${RUN_ROOT}/paper_pack/mechanism/mechanism_frontier_summary.csv" \
  --mechanism-pair-summary-csv "${RUN_ROOT}/paper_pack/mechanism/selected_vs_eta1_mechanism_summary.csv" \
  --output-dir "${RUN_ROOT}/paper_pack/selection_defense"

log "[FINALIZER-END] split=${SPLIT_ID}"
