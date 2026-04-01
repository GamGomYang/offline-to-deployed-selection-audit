#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl"
PRL_ROOT="${ROOT}/prl-dow30"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_ROOT="${ROOT}/outputs/extensions/v1_rolling_origin_windows/${STAMP}"
SPLITS_JSON="${ROOT}/frozen_protocol/rolling_windows_v1/split_definitions.json"
SEED_BUCKETS=("0 1 2 3" "4 5 6" "7 8 9")
EXPECTED_SEEDS="0 1 2 3 4 5 6 7 8 9"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-root)
      EXPERIMENT_ROOT="$2"
      shift 2
      ;;
    --splits-json)
      SPLITS_JSON="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${EXPERIMENT_ROOT}/logs"

"${PYTHON_BIN}" "${PRL_ROOT}/scripts/prepare_v1_rolling_origin_windows.py" \
  --splits-json "${SPLITS_JSON}" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  > "${EXPERIMENT_ROOT}/logs/prepare_manifest.json"

MANIFEST_PATH="${EXPERIMENT_ROOT}/prepared/manifest.json"
SESSIONS_TSV="${EXPERIMENT_ROOT}/logs/tmux_sessions.tsv"
printf "split_id\tbucket_id\tseeds\tsession_name\trun_root\n" > "${SESSIONS_TSV}"

readarray -t LAUNCH_ROWS < <("${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for split_id, split in manifest["splits"].items():
    if split["status"] != "launch":
        continue
    print("\t".join([
        split_id,
        split["current_config"],
        split["validation"]["start"],
        split["validation"]["end"],
        split["test"]["start"],
        split["test"]["end"],
        split["run_root"],
    ]))
PY
)

SHORT_STAMP="$(basename "${EXPERIMENT_ROOT}" | cut -c1-15)"

for row in "${LAUNCH_ROWS[@]}"; do
  IFS=$'\t' read -r SPLIT_ID CURRENT_CONFIG VALIDATION_START VALIDATION_END FINAL_START FINAL_END RUN_ROOT <<< "$row"

  mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/configs" "${RUN_ROOT}/train_control" "${RUN_ROOT}/validation_eta" "${RUN_ROOT}/final_eta" "${RUN_ROOT}/external_baselines" "${RUN_ROOT}/paper_pack" "${RUN_ROOT}/logs"

  "${PYTHON_BIN}" "${PRL_ROOT}/scripts/materialize_u27_control_eta_paper_configs.py" \
    --current-config "${CURRENT_CONFIG}" \
    --snapshot-config-out "${RUN_ROOT}/configs/snapshot_control.yaml" \
    --signal-snapshot-out "${RUN_ROOT}/configs/selected_signals_snapshot.json" \
    --validation-config-out "${RUN_ROOT}/configs/validation_eta.yaml" \
    --final-config-out "${RUN_ROOT}/configs/final_eta.yaml" \
    --meta-out "${RUN_ROOT}/configs/materialization_meta.json" \
    --job-ts "${STAMP}" \
    --validation-start "${VALIDATION_START}" \
    --validation-end "${VALIDATION_END}" \
    --final-start "${FINAL_START}" \
    --final-end "${FINAL_END}" \
    --train-output-root "${RUN_ROOT}/train_control" \
    --validation-output-root "${RUN_ROOT}/validation_eta" \
    --final-output-root "${RUN_ROOT}/final_eta" \
    > "${EXPERIMENT_ROOT}/logs/${SPLIT_ID}_materialize.json"

  (
    cd "${PRL_ROOT}"
    "${PYTHON_BIN}" scripts/warm_volatility_stats.py \
      --config "${RUN_ROOT}/configs/snapshot_control.yaml"
  ) > "${EXPERIMENT_ROOT}/logs/${SPLIT_ID}_warm_stats.txt"

  idx=0
  for seeds in "${SEED_BUCKETS[@]}"; do
    SESSION_NAME="v1roll6_${SHORT_STAMP}_${SPLIT_ID}_b${idx}"
    LOG_PATH="${EXPERIMENT_ROOT}/logs/${SPLIT_ID}_bucket${idx}.log"
    FINALIZER=0
    if [[ "${idx}" == "0" ]]; then
      FINALIZER=1
    fi
    CMD="cd '${ROOT}' && PYTHON_BIN='${PYTHON_BIN}' '${PRL_ROOT}/scripts/run_v1_rolling_origin_bucket.sh' --split-id '${SPLIT_ID}' --run-root '${RUN_ROOT}' --current-config '${CURRENT_CONFIG}' --validation-start '${VALIDATION_START}' --validation-end '${VALIDATION_END}' --final-start '${FINAL_START}' --final-end '${FINAL_END}' --seeds '${seeds}' --expected-seeds '${EXPECTED_SEEDS}' --finalizer '${FINALIZER}' 2>&1 | tee '${LOG_PATH}'"
    tmux new-session -d -s "${SESSION_NAME}" "bash -lc ${CMD@Q}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${SPLIT_ID}" "${idx}" "${seeds}" "${SESSION_NAME}" "${RUN_ROOT}" >> "${SESSIONS_TSV}"
    idx=$((idx + 1))
  done
done

echo "ROLLING_EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
echo "ROLLING_MANIFEST=${MANIFEST_PATH}"
echo "ROLLING_SESSIONS=${SESSIONS_TSV}"
