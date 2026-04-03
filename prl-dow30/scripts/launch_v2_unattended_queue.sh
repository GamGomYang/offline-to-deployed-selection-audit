#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/v2_unattended}"
SUPERVISOR_LOG="$RUN_ROOT/supervisor.log"
SUPERVISOR_STDOUT_LOG="$RUN_ROOT/supervisor.stdout.log"
SUPERVISOR_PID_FILE="$RUN_ROOT/supervisor.pid"
mkdir -p "$RUN_ROOT"

if [[ -f "$SUPERVISOR_PID_FILE" ]]; then
  OLD_PID="$(cat "$SUPERVISOR_PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[INFO] unattended supervisor already running with pid=$OLD_PID"
    exit 0
  fi
fi

EXISTING_PID="$(pgrep -f 'bash scripts/run_v2_unattended_supervisor.sh' | head -n 1 || true)"
if [[ -n "${EXISTING_PID:-}" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
  echo "$EXISTING_PID" > "$SUPERVISOR_PID_FILE"
  echo "[INFO] adopted existing unattended supervisor pid=$EXISTING_PID"
  exit 0
fi

setsid -f bash scripts/run_v2_unattended_supervisor.sh >>"$SUPERVISOR_STDOUT_LOG" 2>&1
sleep 1
PID="$(pgrep -f 'bash scripts/run_v2_unattended_supervisor.sh' | head -n 1 || true)"
if [[ -z "${PID:-}" ]]; then
  echo "[ERROR] supervisor failed to start"
  exit 1
fi
echo "$PID" > "$SUPERVISOR_PID_FILE"

echo "[STARTED] supervisor pid=$PID"
echo "[LOG] $SUPERVISOR_LOG"
echo "[STDOUT LOG] $SUPERVISOR_STDOUT_LOG"
echo "[PID] $SUPERVISOR_PID_FILE"
