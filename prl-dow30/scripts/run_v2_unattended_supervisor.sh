#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/v2_unattended}"
LOG_FILE="$RUN_ROOT/master.log"
SUPERVISOR_LOG="$RUN_ROOT/supervisor.log"
SUPERVISOR_PID_FILE="$RUN_ROOT/supervisor.pid"
QUEUE_PID_FILE="$RUN_ROOT/master.pid"
STATE_DIR="$RUN_ROOT/state"
DEADLINE_FILE="$RUN_ROOT/deadline.env"
mkdir -p "$RUN_ROOT" "$STATE_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" >> "$SUPERVISOR_LOG"
}

if [[ -f "$DEADLINE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEADLINE_FILE"
fi

MAX_HOURS="${MAX_HOURS:-7}"
if [[ -z "${DEADLINE_EPOCH:-}" ]]; then
  START_EPOCH="$(date +%s)"
  DEADLINE_EPOCH="$((START_EPOCH + MAX_HOURS * 3600))"
  cat > "$DEADLINE_FILE" <<EODEAD
START_EPOCH=$START_EPOCH
MAX_HOURS=$MAX_HOURS
DEADLINE_EPOCH=$DEADLINE_EPOCH
EODEAD
fi

echo $$ > "$SUPERVISOR_PID_FILE"
log "[START] supervisor deadline=$(date -u -d "@$DEADLINE_EPOCH" '+%Y-%m-%dT%H:%M:%SZ')"

find_queue_pid() {
  local pid
  pid="$(pgrep -f 'bash scripts/run_v2_unattended_queue.sh' | head -n 1 || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid"
  fi
}

launch_queue() {
  log "[LAUNCH] queue"
  DEADLINE_EPOCH="$DEADLINE_EPOCH" MAX_HOURS="$MAX_HOURS" \
    nohup bash scripts/run_v2_unattended_queue.sh >>"$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$QUEUE_PID_FILE"
  log "[QUEUE_PID] $pid"
}

while true; do
  now="$(date +%s)"
  if (( now >= DEADLINE_EPOCH )); then
    log "[STOP] deadline reached"
    break
  fi

  if [[ -f "$STATE_DIR/master_queue.done" ]]; then
    log "[DONE] queue completed all planned steps"
    break
  fi

  queue_pid="$(find_queue_pid || true)"
  if [[ -z "$queue_pid" ]]; then
    launch_queue
    sleep 5
    queue_pid="$(find_queue_pid || true)"
    if [[ -z "$queue_pid" ]]; then
      log "[WARN] queue launch did not produce a live pid; retrying after backoff"
      sleep 30
      continue
    fi
  else
    echo "$queue_pid" > "$QUEUE_PID_FILE"
    log "[ADOPT] queue pid=$queue_pid"
  fi

  while kill -0 "$queue_pid" 2>/dev/null; do
    sleep 30
    now="$(date +%s)"
    if (( now >= DEADLINE_EPOCH )); then
      log "[STOP] deadline reached while queue pid=$queue_pid is running"
      break
    fi
  done

  if [[ -f "$STATE_DIR/master_queue.done" ]]; then
    log "[DONE] queue completed all planned steps"
    break
  fi

  now="$(date +%s)"
  if (( now >= DEADLINE_EPOCH )); then
    break
  fi

  log "[RESTART] queue exited before completion; restarting after short backoff"
  sleep 20
 done

log "[EXIT] supervisor finished"
