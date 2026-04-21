#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import save_results
import run_inventory as inventory_v2


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "inventory_step4_seed_stability_locked"
DEFAULT_CALIBRATION_LOG = REPO_ROOT / "outputs" / "forecast_eval" / "inventory" / "inventory_v2_calibration_log.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 4 inventory seed-stability check with the locked config.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calibration-log", default=str(DEFAULT_CALIBRATION_LOG))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--horizon", type=int, default=inventory_v2.DEFAULT_HORIZON)
    parser.add_argument("--train-end", type=int, default=inventory_v2.TRAIN_END)
    return parser.parse_args()


def _load_selected_config(calibration_log_path: Path) -> tuple[float, float, float]:
    calibration_df = pd.read_csv(calibration_log_path)
    selected = calibration_df[calibration_df["selected_flag"] == True]  # noqa: E712
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one selected config in {calibration_log_path}, found {len(selected)}.")
    row = selected.iloc[0]
    return float(row["burst_amp"]), float(row["safety_stock"]), float(row["stockout_w"])


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    burst_amp, safety_stock, stockout_w = _load_selected_config(Path(args.calibration_log))
    seeds = [int(seed) for seed in args.seeds]
    horizon = int(args.horizon)
    train_end = int(args.train_end)

    data_cache: dict[tuple[int, float], dict[str, Any]] = {}
    for seed in seeds:
        data_cache[(seed, burst_amp)] = inventory_v2._build_forecast_cache(
            seed=seed,
            burst_amp=burst_amp,
            horizon=horizon,
            train_end=train_end,
        )

    candidate = inventory_v2._evaluate_candidate(
        seeds=seeds,
        data_cache=data_cache,
        burst_amp=burst_amp,
        safety_stock=safety_stock,
        stockout_w=stockout_w,
    )

    q1_path = output_dir / "inventory_v2_seed_stability_q1.csv"
    q2_path = output_dir / "inventory_v2_seed_stability_q2.csv"
    diagnostics_path = output_dir / "inventory_v2_seed_stability_diagnostics.csv"
    freeze_path = output_dir / "inventory_v2_seed_stability_freeze_check.csv"
    summary_path = output_dir / "inventory_v2_seed_stability_summary.csv"

    q1_df = save_results(candidate["q1_df"], q1_path)
    q2_df = save_results(candidate["q2_df"], q2_path)
    diagnostics_df = candidate["diagnostics_df"].copy()
    freeze_df = candidate["freeze_df"].copy()
    diagnostics_df.to_csv(diagnostics_path, index=False)
    freeze_df.to_csv(freeze_path, index=False)

    summary_payload = {
        "seed_count": len(seeds),
        "burst_amp": burst_amp,
        "safety_stock": safety_stock,
        "stockout_w": stockout_w,
        "all_freeze_checks_pass": bool(
            freeze_df["forecast_hash_identical_flag"].all()
            and freeze_df["proposal_hash_identical_flag"].all()
            and freeze_df["initial_inventory_match_flag"].all()
            and freeze_df["initial_prev_order_match_flag"].all()
            and (freeze_df["pairing_failure_count"] == 0).all()
        ),
        **candidate["metrics"],
    }
    pd.DataFrame([summary_payload]).to_csv(summary_path, index=False)

    print(
        "[inventory-seed-stability] "
        f"config=(burst_amp={burst_amp}, safety_stock={safety_stock}, stockout_w={stockout_w}) seeds={len(seeds)}"
    )
    print(f"[inventory-seed-stability] wrote {len(q1_df)} Q1 rows to {q1_path}")
    print(f"[inventory-seed-stability] wrote {len(q2_df)} Q2 rows to {q2_path}")
    print(f"[inventory-seed-stability] wrote diagnostics to {diagnostics_path}")
    print(f"[inventory-seed-stability] wrote freeze check to {freeze_path}")
    print(f"[inventory-seed-stability] wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
