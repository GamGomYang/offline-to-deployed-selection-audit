#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${ROOT}/manifests/baselines/paper_v3_frozen.json"
MODE="frozen-models"
VERIFY="1"
RUN_ROOT=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --verify)
      VERIFY="1"
      shift
      ;;
    --no-verify)
      VERIFY="0"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Baseline manifest not found: ${MANIFEST}" >&2
  exit 2
fi

readarray -t MANIFEST_VALUES < <("${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest["baseline_id"])
print(manifest["canonical_run_root"])
print(manifest["frozen_source_config"])
PY
)

BASELINE_ID="${MANIFEST_VALUES[0]}"
CANONICAL_RUN_ROOT="${ROOT}/${MANIFEST_VALUES[1]}"
FROZEN_SOURCE_CONFIG="${ROOT}/${MANIFEST_VALUES[2]}"

if [[ -z "${RUN_ROOT}" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ROOT="${ROOT}/outputs/reproductions/baseline/${BASELINE_ID}/${STAMP}"
fi

mkdir -p "${RUN_ROOT}"

if [[ "${MODE}" == "frozen-models" ]]; then
  mkdir -p "${RUN_ROOT}/train_control/models"
  for src in "${CANONICAL_RUN_ROOT}"/train_control/models/*_final.zip; do
    seed="$(basename "${src}" | sed -E 's/.*_seed([0-9]+)_.*_final\.zip/\1/')"
    cp "${src}" "${RUN_ROOT}/train_control/models/prl_seed${seed}_final.zip"
  done
  RUN_TRAIN="0"
elif [[ "${MODE}" == "full" ]]; then
  RUN_TRAIN="1"
else
  echo "Unsupported mode: ${MODE}. Use frozen-models or full." >&2
  exit 2
fi

export PYTHON_CMD="${PYTHON_CMD:-${PYTHON_BIN}}"
export CURRENT_CONFIG="${FROZEN_SOURCE_CONFIG}"
export RUN_ROOT="${RUN_ROOT}"
export RUN_TRAIN
export RUN_VALIDATION="1"
export RUN_SELECT="1"
export RUN_FINAL="1"
export RUN_BASELINES="1"
export RUN_PACK="1"
export FINAL_MODE="selected_plus_baseline"
export SAC_TOTAL_TIMESTEPS="${SAC_TOTAL_TIMESTEPS:-0}"

(cd "${ROOT}" && prl-dow30/scripts/run_u27_control_eta_validation_first.sh)

"${PYTHON_BIN}" "${ROOT}/prl-dow30/scripts/export_paper_artifacts.py" \
  --run-root "${RUN_ROOT}"

if [[ "${VERIFY}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/prl-dow30/scripts/verify_baseline_reproduction.py" \
    --manifest "${MANIFEST}" \
    --run-root "${RUN_ROOT}"
fi

echo "REPRO_BASELINE_ID=${BASELINE_ID}"
echo "REPRO_MODE=${MODE}"
echo "REPRO_RUN_ROOT=${RUN_ROOT}"
