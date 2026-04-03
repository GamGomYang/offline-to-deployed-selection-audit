#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
REPO_ROOT="/workspace/execution-aware-portfolio-rl"

if [[ -n "${PYTHON_CMD:-}" ]]; then
  read -r -a PYTHON <<< "$PYTHON_CMD"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON=("${REPO_ROOT}/.venv/bin/python")
else
  PYTHON=("python3")
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON=("$PYTHON_BIN")
fi

CURRENT_CONFIG="${CURRENT_CONFIG:-${REPO_ROOT}/paper_rebuild_20260324T065755Z/configs/snapshot_control.yaml}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/paper_rebuild_20260324T065755Z/train_control}"
SEEDS_RAW="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ETAS_RAW="${ETAS:-1.0 0.5 0.2 0.1 0.082 0.05 0.02}"
KAPPAS_RAW="${KAPPAS:-0.0 0.0002 0.0005 0.001 0.002}"
POSITIVE_KAPPAS="${POSITIVE_KAPPAS:-0.0002,0.0005,0.001,0.002}"
RELATIVE_THRESHOLD="${RELATIVE_THRESHOLD:-0.95}"
VALIDATION_START="${VALIDATION_START:-2022-01-01}"
VALIDATION_END="${VALIDATION_END:-2023-12-31}"
FINAL_START="${FINAL_START:-2024-01-01}"
FINAL_END="${FINAL_END:-2025-12-31}"
BASELINE_ETA="${BASELINE_ETA:-1.0}"
MAX_STEPS="${MAX_STEPS:-0}"
JOB_TS="${JOB_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/extensions/v1_kappa_expansion/${JOB_TS}}"

VALIDATION_ROOT="${RUN_ROOT}/validation_eta_full_grid"
FINAL_ROOT="${RUN_ROOT}/final_eta_full_grid"
CONFIG_ROOT="${RUN_ROOT}/configs"
ANALYSIS_ROOT="${RUN_ROOT}/analysis"
LOG_DIR="${RUN_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/master.log"
SNAPSHOT_CONFIG="${CONFIG_ROOT}/snapshot_control.yaml"
SIGNAL_SNAPSHOT="${CONFIG_ROOT}/selected_signals_snapshot.json"
VALIDATION_CONFIG="${CONFIG_ROOT}/validation_eta.yaml"
FINAL_CONFIG="${CONFIG_ROOT}/final_eta.yaml"
MATERIALIZATION_META="${CONFIG_ROOT}/materialization_meta.json"
SELECTION_DIR="${VALIDATION_ROOT}/selection"

mkdir -p "${VALIDATION_ROOT}" "${FINAL_ROOT}" "${CONFIG_ROOT}" "${ANALYSIS_ROOT}" "${LOG_DIR}"

cd "${ROOT}"
export PYTHONPATH="."

read -r -a SEEDS <<< "${SEEDS_RAW}"
read -r -a ETAS <<< "${ETAS_RAW}"
read -r -a KAPPAS <<< "${KAPPAS_RAW}"

log() {
  echo "$1" | tee -a "${MASTER_LOG}"
}

run_cmd() {
  local name="$1"
  shift
  log "[STEP-START] ${name} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  "$@" 2>&1 | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  log "[STEP-END] ${name} rc=${rc} :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "${rc}"
}

log "[INFO] phase=v1_kappa_expansion"
log "[INFO] current_config=${CURRENT_CONFIG}"
log "[INFO] model_root=${MODEL_ROOT}"
log "[INFO] seeds=${SEEDS[*]}"
log "[INFO] etas=${ETAS[*]}"
log "[INFO] kappas=${KAPPAS[*]}"
log "[INFO] positive_kappas=${POSITIVE_KAPPAS}"
log "[INFO] validation_window=${VALIDATION_START}~${VALIDATION_END}"
log "[INFO] final_window=${FINAL_START}~${FINAL_END}"
log "[INFO] run_root=${RUN_ROOT}"
log "[INFO] python_cmd=${PYTHON[*]}"

run_cmd "materialize_configs" \
  "${PYTHON[@]}" scripts/materialize_u27_control_eta_paper_configs.py \
    --current-config "${CURRENT_CONFIG}" \
    --snapshot-config-out "${SNAPSHOT_CONFIG}" \
    --signal-snapshot-out "${SIGNAL_SNAPSHOT}" \
    --validation-config-out "${VALIDATION_CONFIG}" \
    --final-config-out "${FINAL_CONFIG}" \
    --meta-out "${MATERIALIZATION_META}" \
    --job-ts "${JOB_TS}" \
    --validation-start "${VALIDATION_START}" \
    --validation-end "${VALIDATION_END}" \
    --final-start "${FINAL_START}" \
    --final-end "${FINAL_END}" \
    --train-output-root "${MODEL_ROOT}"

run_cmd "validation_full_grid" \
  "${PYTHON[@]}" scripts/step6_run_matrix.py \
    --config "${VALIDATION_CONFIG}" \
    --model-type prl \
    --model-root "${MODEL_ROOT}" \
    --seed-model-mode independent \
    --seeds "${SEEDS[@]}" \
    --kappas "${KAPPAS[@]}" \
    --etas "${ETAS[@]}" \
    --out "${VALIDATION_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --offline

run_cmd "validation_build_reports" \
  "${PYTHON[@]}" scripts/step6_build_reports.py \
    --root "${VALIDATION_ROOT}"

run_cmd "validation_select_eta" \
  "${PYTHON[@]}" scripts/select_eta_from_validation.py \
    --root "${VALIDATION_ROOT}" \
    --output-dir "${SELECTION_DIR}" \
    --baseline-eta "${BASELINE_ETA}" \
    --positive-kappas "${POSITIVE_KAPPAS}" \
    --relative-threshold "${RELATIVE_THRESHOLD}"

run_cmd "final_full_grid" \
  "${PYTHON[@]}" scripts/step6_run_matrix.py \
    --config "${FINAL_CONFIG}" \
    --model-type prl \
    --model-root "${MODEL_ROOT}" \
    --seed-model-mode independent \
    --seeds "${SEEDS[@]}" \
    --kappas "${KAPPAS[@]}" \
    --etas "${ETAS[@]}" \
    --out "${FINAL_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --offline

run_cmd "final_build_reports" \
  "${PYTHON[@]}" scripts/step6_build_reports.py \
    --root "${FINAL_ROOT}"

run_cmd "mechanism_decomposition" \
  "${PYTHON[@]}" scripts/build_mechanism_decomposition.py \
    --validation-root "${VALIDATION_ROOT}" \
    --final-root "${FINAL_ROOT}" \
    --selection-json "${SELECTION_DIR}/validation_eta_selection.json" \
    --output-dir "${ANALYSIS_ROOT}/mechanism"

run_cmd "analyze_kappa_expansion" \
  "${PYTHON[@]}" scripts/analyze_v1_kappa_expansion.py \
    --validation-root "${VALIDATION_ROOT}" \
    --final-root "${FINAL_ROOT}" \
    --selection-json "${SELECTION_DIR}/validation_eta_selection.json" \
    --selection-csv "${SELECTION_DIR}/validation_eta_selection.csv" \
    --output-dir "${ANALYSIS_ROOT}"

log "[DONE] v1_kappa_expansion completed"
