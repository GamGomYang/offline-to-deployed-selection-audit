#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl"
PRL_ROOT="${ROOT}/prl-dow30"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT_ROOT="${ROOT}/outputs/extensions/v1_rolling_origin_windows/${STAMP}"
SPLITS_JSON="${ROOT}/frozen_protocol/rolling_windows_v1/split_definitions.json"

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
printf "split_id\tsession_name\trun_root\n" > "${SESSIONS_TSV}"

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
  SESSION_NAME="v1roll_${SHORT_STAMP}_${SPLIT_ID}"
  LOG_PATH="${EXPERIMENT_ROOT}/logs/${SPLIT_ID}.log"
  CMD="cd '${ROOT}' && PYTHON_BIN='${PYTHON_BIN}' '${PRL_ROOT}/scripts/run_v1_rolling_origin_split.sh' --run-root '${RUN_ROOT}' --split-id '${SPLIT_ID}' --current-config '${CURRENT_CONFIG}' --validation-start '${VALIDATION_START}' --validation-end '${VALIDATION_END}' --final-start '${FINAL_START}' --final-end '${FINAL_END}' 2>&1 | tee '${LOG_PATH}'"
  tmux new-session -d -s "${SESSION_NAME}" "bash -lc ${CMD@Q}"
  printf "%s\t%s\t%s\n" "${SPLIT_ID}" "${SESSION_NAME}" "${RUN_ROOT}" >> "${SESSIONS_TSV}"
done

echo "ROLLING_EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
echo "ROLLING_MANIFEST=${MANIFEST_PATH}"
echo "ROLLING_SESSIONS=${SESSIONS_TSV}"
