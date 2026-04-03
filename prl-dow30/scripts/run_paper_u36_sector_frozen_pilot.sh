#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_TRAIN="configs/exp/paper_u36_sector_snapshot_control.yaml"
CONFIG_VAL="configs/exp/paper_u36_sector_validation_eta.yaml"
CONFIG_TEST="configs/exp/paper_u36_sector_final_eta.yaml"

SEEDS_STR="${SEEDS:-0 1 2}"
KAPPAS_STR="${KAPPAS:-0.0 0.0005 0.001}"
ETAS_STR="${ETAS:-1.0 0.5 0.2 0.1 0.082 0.05 0.02}"
TRAIN_ROOT="${TRAIN_ROOT:-outputs/v2_u36_sector_frozen_pilot/train_control}"
VAL_OUT="${VAL_OUT:-outputs/v2_u36_sector_frozen_pilot/validation_eta}"
TEST_OUT="${TEST_OUT:-outputs/v2_u36_sector_frozen_pilot/final_eta}"

echo "[1/4] Build cache for the second universe"
python scripts/build_cache.py --config "$CONFIG_TRAIN"

echo "[2/4] Train one frozen control checkpoint per seed"
for seed in $SEEDS_STR; do
  python scripts/run_train.py \
    --config "$CONFIG_TRAIN" \
    --model-type prl \
    --seed "$seed" \
    --offline \
    --output-root "$TRAIN_ROOT"
done

echo "[3/4] Validation eta sweep on the second universe"
python scripts/step6_run_matrix.py \
  --config "$CONFIG_VAL" \
  --kappas $KAPPAS_STR \
  --etas $ETAS_STR \
  --seeds $SEEDS_STR \
  --out "$VAL_OUT" \
  --model-type prl \
  --model-root "$TRAIN_ROOT" \
  --seed-model-mode independent \
  --max-steps 0 \
  --offline

echo "[4/4] Held-out eta sweep on the second universe"
python scripts/step6_run_matrix.py \
  --config "$CONFIG_TEST" \
  --kappas $KAPPAS_STR \
  --etas $ETAS_STR \
  --seeds $SEEDS_STR \
  --out "$TEST_OUT" \
  --model-type prl \
  --model-root "$TRAIN_ROOT" \
  --seed-model-mode independent \
  --max-steps 0 \
  --offline

echo "[DONE] U36 sector-balanced frozen pilot complete."
