#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_TRAIN="configs/exp/paper_u27_eta05_snapshot_control.yaml"
CONFIG_VAL="configs/exp/paper_u27_eta05_validation_main_vs_baseline.yaml"
CONFIG_TEST="configs/exp/paper_u27_eta05_final_main_vs_baseline.yaml"

SEEDS_STR="${SEEDS:-0 1 2}"
KAPPAS_STR="${KAPPAS:-0.0 0.0005 0.001}"
TRAIN_ROOT="${TRAIN_ROOT:-outputs/v2_u27_eta05_retrain_pilot/train_control}"
VAL_OUT="${VAL_OUT:-outputs/v2_u27_eta05_retrain_pilot/validation_main_vs_baseline}"
TEST_OUT="${TEST_OUT:-outputs/v2_u27_eta05_retrain_pilot/final_main_vs_baseline}"

echo "[1/3] Train execution-aligned eta=0.5 checkpoints on U27"
for seed in $SEEDS_STR; do
  python scripts/run_train.py \
    --config "$CONFIG_TRAIN" \
    --model-type prl \
    --seed "$seed" \
    --offline \
    --output-root "$TRAIN_ROOT"
done

echo "[2/3] Validation comparison: retrained eta=0.5 main arm vs eta=1.0 baseline arm"
python scripts/step6_run_matrix.py \
  --config "$CONFIG_VAL" \
  --kappas $KAPPAS_STR \
  --seeds $SEEDS_STR \
  --out "$VAL_OUT" \
  --model-type prl \
  --model-root "$TRAIN_ROOT" \
  --seed-model-mode independent \
  --max-steps 0 \
  --offline

echo "[3/3] Held-out comparison: retrained eta=0.5 main arm vs eta=1.0 baseline arm"
python scripts/step6_run_matrix.py \
  --config "$CONFIG_TEST" \
  --kappas $KAPPAS_STR \
  --seeds $SEEDS_STR \
  --out "$TEST_OUT" \
  --model-type prl \
  --model-root "$TRAIN_ROOT" \
  --seed-model-mode independent \
  --max-steps 0 \
  --offline

echo "[DONE] U27 eta=0.5 execution-aligned retraining pilot complete."
