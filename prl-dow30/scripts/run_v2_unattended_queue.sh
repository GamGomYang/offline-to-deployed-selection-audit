#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FATAL] Missing Python environment at $PYTHON_BIN" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/v2_unattended}"
LOG_DIR="$RUN_ROOT/logs"
STATE_DIR="$RUN_ROOT/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

MASTER_STATE="$RUN_ROOT/master_state.env"
START_EPOCH="$(date +%s)"
MAX_HOURS="${MAX_HOURS:-7}"
if [[ -n "${DEADLINE_EPOCH:-}" ]]; then
  DEADLINE_EPOCH="$DEADLINE_EPOCH"
else
  DEADLINE_EPOCH="$((START_EPOCH + MAX_HOURS * 3600))"
fi

SEEDS_PILOT_STR="${SEEDS_PILOT:-0 1 2}"
SEEDS_EXPAND_STR="${SEEDS_EXPAND:-3 4}"
KAPPAS_STR="${KAPPAS:-0.0 0.0005 0.001}"
ETAS_STR="${ETAS:-1.0 0.5 0.2 0.1 0.082 0.05 0.02}"

read -r -a SEEDS_PILOT <<< "$SEEDS_PILOT_STR"
read -r -a SEEDS_EXPAND <<< "$SEEDS_EXPAND_STR"
read -r -a KAPPAS <<< "$KAPPAS_STR"
read -r -a ETAS <<< "$ETAS_STR"

MIN_SUCCESS_SEEDS="${MIN_SUCCESS_SEEDS:-2}"
RETRIES_CACHE="${RETRIES_CACHE:-3}"
RETRIES_TRAIN="${RETRIES_TRAIN:-2}"
RETRIES_EVAL="${RETRIES_EVAL:-2}"
AUTO_EXPAND_TO_5="${AUTO_EXPAND_TO_5:-1}"

U27_TRAIN_ROOT="outputs/v2_u27_eta05_retrain_pilot/train_control"
U27_VAL_OUT="outputs/v2_u27_eta05_retrain_pilot/validation_main_vs_baseline"
U27_TEST_OUT="outputs/v2_u27_eta05_retrain_pilot/final_main_vs_baseline"
U36_TRAIN_ROOT="outputs/v2_u36_sector_frozen_pilot/train_control"
U36_VAL_OUT="outputs/v2_u36_sector_frozen_pilot/validation_eta"
U36_TEST_OUT="outputs/v2_u36_sector_frozen_pilot/final_eta"

U27_TRAIN_CFG="configs/exp/paper_u27_eta05_snapshot_control.yaml"
U27_VAL_CFG="configs/exp/paper_u27_eta05_validation_main_vs_baseline.yaml"
U27_TEST_CFG="configs/exp/paper_u27_eta05_final_main_vs_baseline.yaml"
U36_TRAIN_CFG="configs/exp/paper_u36_sector_snapshot_control.yaml"
U36_VAL_CFG="configs/exp/paper_u36_sector_validation_eta.yaml"
U36_TEST_CFG="configs/exp/paper_u36_sector_final_eta.yaml"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

time_left_seconds() {
  echo "$((DEADLINE_EPOCH - $(date +%s)))"
}

write_master_state() {
  cat >"$MASTER_STATE" <<EOF
START_EPOCH=$START_EPOCH
MAX_HOURS=$MAX_HOURS
DEADLINE_EPOCH=$DEADLINE_EPOCH
TIME_LEFT_SECONDS=$(time_left_seconds)
EOF
}

mark_state() {
  local name="$1"
  local suffix="$2"
  : > "$STATE_DIR/${name}.${suffix}"
}

run_step() {
  local step_id="$1"
  local timeout_s="$2"
  local retries="$3"
  shift 3

  local done_file="$STATE_DIR/${step_id}.done"
  local fail_file="$STATE_DIR/${step_id}.fail"
  local log_file="$LOG_DIR/${step_id}.log"
  local attempt rc
  local remaining_s effective_timeout

  remaining_s="$(time_left_seconds)"
  if (( remaining_s <= 0 )); then
    log "[STOP] deadline reached before $step_id"
    return 2
  fi
  effective_timeout="$timeout_s"
  if (( remaining_s < effective_timeout )); then
    effective_timeout="$remaining_s"
  fi

  if [[ -f "$done_file" ]]; then
    log "[SKIP] $step_id already done"
    return 0
  fi

  rm -f "$fail_file"

  for attempt in $(seq 1 "$retries"); do
    log "[RUN] $step_id attempt ${attempt}/${retries}"
    {
      printf '[%s] START attempt %s/%s\n' "$(timestamp)" "$attempt" "$retries"
      printf '[%s] CMD ' "$(timestamp)"
      printf '%q ' "$@"
      printf '\n'
    } >>"$log_file"

    timeout --signal=TERM --kill-after=60 "$effective_timeout" "$@" >>"$log_file" 2>&1
    rc=$?
    printf '[%s] EXIT rc=%s\n' "$(timestamp)" "$rc" >>"$log_file"

    if [[ "$rc" -eq 0 ]]; then
      mark_state "$step_id" "done"
      log "[OK] $step_id"
      return 0
    fi

    log "[WARN] $step_id failed with rc=$rc"
    sleep $((30 * attempt))
  done

  mark_state "$step_id" "fail"
  log "[FAIL] $step_id exhausted retries"
  return 1
}

gather_done_seeds() {
  local prefix="$1"
  shift
  local seed
  local out=()
  for seed in "$@"; do
    if [[ -f "$STATE_DIR/${prefix}_train_seed${seed}.done" ]]; then
      out+=("$seed")
    fi
  done
  printf '%s\n' "${out[*]}"
}

count_seed_words() {
  local raw="$1"
  if [[ -z "${raw// }" ]]; then
    echo 0
  else
    wc -w <<<"$raw" | tr -d ' '
  fi
}

have_cache() {
  local processed_dir="$1"
  [[ -f "$processed_dir/data_manifest.json" && -f "$processed_dir/prices.parquet" && -f "$processed_dir/returns.parquet" ]]
}

run_u27_block() {
  local seed
  local good_seeds
  local good_count

  log "[BLOCK] U27 execution-aligned retraining pilot"
  for seed in "${SEEDS_PILOT[@]}"; do
    run_step "u27_train_seed${seed}" 7200 "$RETRIES_TRAIN" \
      "$PYTHON_BIN" scripts/run_train.py \
      --config "$U27_TRAIN_CFG" \
      --model-type prl \
      --seed "$seed" \
      --offline \
      --output-root "$U27_TRAIN_ROOT" || true
  done

  good_seeds="$(gather_done_seeds u27 "${SEEDS_PILOT[@]}")"
  good_count="$(count_seed_words "$good_seeds")"
  log "[INFO] U27 successful pilot seeds: ${good_seeds:-<none>}"

  if (( good_count >= MIN_SUCCESS_SEEDS )); then
    # shellcheck disable=SC2206
    local seed_args=( $good_seeds )
    run_step "u27_validation_compare_pilot" 10800 "$RETRIES_EVAL" \
      "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$U27_VAL_CFG" \
      --kappas "${KAPPAS[@]}" \
      --seeds "${seed_args[@]}" \
      --out "$U27_VAL_OUT" \
      --model-type prl \
      --model-root "$U27_TRAIN_ROOT" \
      --seed-model-mode independent \
      --max-steps 0 \
      --offline || true

    run_step "u27_final_compare_pilot" 10800 "$RETRIES_EVAL" \
      "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$U27_TEST_CFG" \
      --kappas "${KAPPAS[@]}" \
      --seeds "${seed_args[@]}" \
      --out "$U27_TEST_OUT" \
      --model-type prl \
      --model-root "$U27_TRAIN_ROOT" \
      --seed-model-mode independent \
      --max-steps 0 \
      --offline || true
  else
    log "[WARN] U27 skipped eval because only ${good_count} seed(s) succeeded"
  fi

  if [[ "$AUTO_EXPAND_TO_5" == "1" ]]; then
    log "[BLOCK] Optional U27 expansion to 5 seeds"
    for seed in "${SEEDS_EXPAND[@]}"; do
      run_step "u27_train_seed${seed}" 7200 "$RETRIES_TRAIN" \
        "$PYTHON_BIN" scripts/run_train.py \
        --config "$U27_TRAIN_CFG" \
        --model-type prl \
        --seed "$seed" \
        --offline \
        --output-root "$U27_TRAIN_ROOT" || true
    done

    good_seeds="$(gather_done_seeds u27 "${SEEDS_PILOT[@]}" "${SEEDS_EXPAND[@]}")"
    good_count="$(count_seed_words "$good_seeds")"
    log "[INFO] U27 successful expanded seeds: ${good_seeds:-<none>}"

    if (( good_count >= MIN_SUCCESS_SEEDS )); then
      # shellcheck disable=SC2206
      local seed_args_expanded=( $good_seeds )
      run_step "u27_validation_compare_expanded" 14400 "$RETRIES_EVAL" \
        "$PYTHON_BIN" scripts/step6_run_matrix.py \
        --config "$U27_VAL_CFG" \
        --kappas "${KAPPAS[@]}" \
        --seeds "${seed_args_expanded[@]}" \
        --out "$U27_VAL_OUT" \
        --model-type prl \
        --model-root "$U27_TRAIN_ROOT" \
        --seed-model-mode independent \
        --max-steps 0 \
        --offline || true

      run_step "u27_final_compare_expanded" 14400 "$RETRIES_EVAL" \
        "$PYTHON_BIN" scripts/step6_run_matrix.py \
        --config "$U27_TEST_CFG" \
        --kappas "${KAPPAS[@]}" \
        --seeds "${seed_args_expanded[@]}" \
        --out "$U27_TEST_OUT" \
        --model-type prl \
        --model-root "$U27_TRAIN_ROOT" \
        --seed-model-mode independent \
        --max-steps 0 \
        --offline || true
    fi
  fi
}

run_u36_block() {
  local seed
  local good_seeds
  local good_count

  log "[BLOCK] U36 second-universe frozen pilot"
  if have_cache "data/processed_u36_sector"; then
    mark_state "u36_build_cache" "done"
    log "[SKIP] U36 cache already present"
  else
    run_step "u36_build_cache" 5400 "$RETRIES_CACHE" \
      "$PYTHON_BIN" scripts/build_cache.py --config "$U36_TRAIN_CFG" || true
  fi

  for seed in "${SEEDS_PILOT[@]}"; do
    run_step "u36_train_seed${seed}" 7200 "$RETRIES_TRAIN" \
      "$PYTHON_BIN" scripts/run_train.py \
      --config "$U36_TRAIN_CFG" \
      --model-type prl \
      --seed "$seed" \
      --offline \
      --output-root "$U36_TRAIN_ROOT" || true
  done

  good_seeds="$(gather_done_seeds u36 "${SEEDS_PILOT[@]}")"
  good_count="$(count_seed_words "$good_seeds")"
  log "[INFO] U36 successful pilot seeds: ${good_seeds:-<none>}"

  if (( good_count >= MIN_SUCCESS_SEEDS )); then
    # shellcheck disable=SC2206
    local seed_args=( $good_seeds )
    run_step "u36_validation_eta_pilot" 14400 "$RETRIES_EVAL" \
      "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$U36_VAL_CFG" \
      --kappas "${KAPPAS[@]}" \
      --etas "${ETAS[@]}" \
      --seeds "${seed_args[@]}" \
      --out "$U36_VAL_OUT" \
      --model-type prl \
      --model-root "$U36_TRAIN_ROOT" \
      --seed-model-mode independent \
      --max-steps 0 \
      --offline || true

    run_step "u36_final_eta_pilot" 14400 "$RETRIES_EVAL" \
      "$PYTHON_BIN" scripts/step6_run_matrix.py \
      --config "$U36_TEST_CFG" \
      --kappas "${KAPPAS[@]}" \
      --etas "${ETAS[@]}" \
      --seeds "${seed_args[@]}" \
      --out "$U36_TEST_OUT" \
      --model-type prl \
      --model-root "$U36_TRAIN_ROOT" \
      --seed-model-mode independent \
      --max-steps 0 \
      --offline || true
  else
    log "[WARN] U36 skipped eval because only ${good_count} seed(s) succeeded"
  fi

  if [[ "$AUTO_EXPAND_TO_5" == "1" ]]; then
    log "[BLOCK] Optional U36 expansion to 5 seeds"
    for seed in "${SEEDS_EXPAND[@]}"; do
      run_step "u36_train_seed${seed}" 7200 "$RETRIES_TRAIN" \
        "$PYTHON_BIN" scripts/run_train.py \
        --config "$U36_TRAIN_CFG" \
        --model-type prl \
        --seed "$seed" \
        --offline \
        --output-root "$U36_TRAIN_ROOT" || true
    done

    good_seeds="$(gather_done_seeds u36 "${SEEDS_PILOT[@]}" "${SEEDS_EXPAND[@]}")"
    good_count="$(count_seed_words "$good_seeds")"
    log "[INFO] U36 successful expanded seeds: ${good_seeds:-<none>}"

    if (( good_count >= MIN_SUCCESS_SEEDS )); then
      # shellcheck disable=SC2206
      local seed_args_expanded=( $good_seeds )
      run_step "u36_validation_eta_expanded" 18000 "$RETRIES_EVAL" \
        "$PYTHON_BIN" scripts/step6_run_matrix.py \
        --config "$U36_VAL_CFG" \
        --kappas "${KAPPAS[@]}" \
        --etas "${ETAS[@]}" \
        --seeds "${seed_args_expanded[@]}" \
        --out "$U36_VAL_OUT" \
        --model-type prl \
        --model-root "$U36_TRAIN_ROOT" \
        --seed-model-mode independent \
        --max-steps 0 \
        --offline || true

      run_step "u36_final_eta_expanded" 18000 "$RETRIES_EVAL" \
        "$PYTHON_BIN" scripts/step6_run_matrix.py \
        --config "$U36_TEST_CFG" \
        --kappas "${KAPPAS[@]}" \
        --etas "${ETAS[@]}" \
        --seeds "${seed_args_expanded[@]}" \
        --out "$U36_TEST_OUT" \
        --model-type prl \
        --model-root "$U36_TRAIN_ROOT" \
        --seed-model-mode independent \
        --max-steps 0 \
        --offline || true
    fi
  fi
}

main() {
  write_master_state
  log "[START] v2 unattended queue"
  log "[INFO] deadline=$(date -u -d "@$DEADLINE_EPOCH" '+%Y-%m-%dT%H:%M:%SZ')"
  log "[INFO] pilot seeds=${SEEDS_PILOT[*]} expand seeds=${SEEDS_EXPAND[*]} kappas=${KAPPAS[*]}"

  run_u27_block
  run_u36_block

  write_master_state
  mark_state "master_queue" "done"
  log "[DONE] v2 unattended queue finished"
}

main "$@"
