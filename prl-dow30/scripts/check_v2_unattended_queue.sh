#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/v2_unattended}"
SUPERVISOR_PID_FILE="$RUN_ROOT/supervisor.pid"
QUEUE_PID_FILE="$RUN_ROOT/master.pid"
SUPERVISOR_LOG="$RUN_ROOT/supervisor.log"
MASTER_LOG="$RUN_ROOT/master.log"
STATE_DIR="$RUN_ROOT/state"

find_live_pid() {
  local pattern="$1"
  pgrep -f "$pattern" | head -n 1 || true
}

SUP_PID="${SUP_PID:-}"
if [[ -f "$SUPERVISOR_PID_FILE" ]]; then
  SUP_PID="$(cat "$SUPERVISOR_PID_FILE" 2>/dev/null || true)"
fi
if [[ -z "${SUP_PID:-}" ]] || ! kill -0 "$SUP_PID" 2>/dev/null; then
  ADOPT_SUP="$(find_live_pid 'bash scripts/run_v2_unattended_supervisor.sh')"
  if [[ -n "${ADOPT_SUP:-}" ]] && kill -0 "$ADOPT_SUP" 2>/dev/null; then
    SUP_PID="$ADOPT_SUP"
    echo "$SUP_PID" > "$SUPERVISOR_PID_FILE"
  fi
fi

QUEUE_PID="${QUEUE_PID:-}"
if [[ -f "$QUEUE_PID_FILE" ]]; then
  QUEUE_PID="$(cat "$QUEUE_PID_FILE" 2>/dev/null || true)"
fi
if [[ -z "${QUEUE_PID:-}" ]] || ! kill -0 "$QUEUE_PID" 2>/dev/null; then
  ADOPT_QUEUE="$(find_live_pid 'bash scripts/run_v2_unattended_queue.sh')"
  if [[ -n "${ADOPT_QUEUE:-}" ]] && kill -0 "$ADOPT_QUEUE" 2>/dev/null; then
    QUEUE_PID="$ADOPT_QUEUE"
    echo "$QUEUE_PID" > "$QUEUE_PID_FILE"
  fi
fi

if [[ -n "${SUP_PID:-}" ]] && kill -0 "$SUP_PID" 2>/dev/null; then
  echo "[SUPERVISOR] running pid=$SUP_PID"
else
  echo "[SUPERVISOR] not running"
fi

if [[ -n "${QUEUE_PID:-}" ]] && kill -0 "$QUEUE_PID" 2>/dev/null; then
  echo "[QUEUE] running pid=$QUEUE_PID"
else
  echo "[QUEUE] not running"
fi

echo
echo "[STATE FILES]"
find "$STATE_DIR" -maxdepth 1 -type f 2>/dev/null | sort || true

echo
echo "[SUPERVISOR LOG TAIL]"
tail -n 30 "$SUPERVISOR_LOG" 2>/dev/null || true

echo
echo "[MASTER LOG TAIL]"
tail -n 40 "$MASTER_LOG" 2>/dev/null || true

LATEST_STEP="$(find "$RUN_ROOT/logs" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
if [[ -n "${LATEST_STEP:-}" ]]; then
  echo
  echo "[LATEST STEP LOG] $LATEST_STEP"
  tail -n 40 "$LATEST_STEP" 2>/dev/null || true
fi
