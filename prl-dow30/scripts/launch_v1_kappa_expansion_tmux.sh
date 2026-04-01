#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/execution-aware-portfolio-rl/prl-dow30"
REPO_ROOT="/workspace/execution-aware-portfolio-rl"
JOB_TS="${JOB_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/extensions/v1_kappa_expansion/${JOB_TS}}"
SESSION="${SESSION:-v1kappa_${JOB_TS}}"

mkdir -p "${RUN_ROOT}/logs"

tmux new-session -d -s "${SESSION}" \
  "cd ${ROOT} && JOB_TS='${JOB_TS}' RUN_ROOT='${RUN_ROOT}' bash scripts/run_v1_kappa_expansion.sh"

printf 'SESSION\t%s\nRUN_ROOT\t%s\n' "${SESSION}" "${RUN_ROOT}"
