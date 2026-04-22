#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import save_results
import run_load_dispatch as load_dispatch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_dispatch_support_locked"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked load-following support-domain window-stability check.")
    parser.add_argument("--raw-path", default=str(load_dispatch.DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--train-hours", type=int, default=load_dispatch.DEFAULT_TRAIN_HOURS)
    parser.add_argument("--eval-hours", type=int, default=load_dispatch.DEFAULT_EVAL_HOURS)
    parser.add_argument("--window-step-hours", type=int, default=load_dispatch.DEFAULT_WINDOW_STEP_HOURS)
    parser.add_argument("--max-lag-hours", type=int, default=load_dispatch.MAX_LAG_HOURS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_dispatch.run_experiment(
        raw_path=Path(args.raw_path),
        window_ids=[int(window_id) for window_id in args.window_ids],
        train_hours=int(args.train_hours),
        eval_hours=int(args.eval_hours),
        window_step_hours=int(args.window_step_hours),
        max_lag_hours=int(args.max_lag_hours),
    )

    q1_path = output_dir / "load_dispatch_seed_stability_q1.csv"
    q2_path = output_dir / "load_dispatch_seed_stability_q2.csv"
    diagnostics_path = output_dir / "load_dispatch_seed_stability_diagnostics.csv"
    freeze_path = output_dir / "load_dispatch_seed_stability_freeze_check.csv"
    failure_path = output_dir / "load_dispatch_seed_stability_model_failures.csv"
    schedule_path = output_dir / "load_dispatch_seed_stability_window_schedule.csv"
    processed_path = output_dir / "processed_hourly_load.csv"
    metadata_path = output_dir / "load_dispatch_seed_stability_summary.csv"

    q1_df = save_results(results["q1_df"], q1_path)
    q2_df = save_results(results["q2_df"], q2_path)
    results["diagnostics_df"].to_csv(diagnostics_path, index=False)
    results["freeze_df"].to_csv(freeze_path, index=False)
    results["model_failures_df"].to_csv(failure_path, index=False)
    results["window_schedule_df"].to_csv(schedule_path, index=False)
    results["processed_hourly_df"].to_csv(processed_path, index=False)

    metadata = {
        **results["metadata"],
        "raw_path": str(results["raw_path"]),
        "q1_rows": int(len(q1_df)),
        "q2_rows": int(len(q2_df)),
        "model_failure_count": int(len(results["model_failures_df"])),
        "min_forecasters_per_window_friction": int(
            q2_df.groupby(["seed", "friction_level"], dropna=False)["forecaster_id"].nunique().min()
        ),
    }
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)

    print(
        "[load-dispatch-seed-stability] "
        f"windows={len(results['metadata']['window_ids'])} train_hours={results['metadata']['train_hours']} "
        f"eval_hours={results['metadata']['eval_hours']} failures={len(results['model_failures_df'])}"
    )
    print(f"[load-dispatch-seed-stability] wrote {len(q1_df)} Q1 rows to {q1_path}")
    print(f"[load-dispatch-seed-stability] wrote {len(q2_df)} Q2 rows to {q2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
