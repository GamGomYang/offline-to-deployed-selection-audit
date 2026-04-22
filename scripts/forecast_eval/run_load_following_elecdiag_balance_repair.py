#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_load_following_elecdiag_groups as groups  # noqa: E402


DEFAULT_BASELINE_WORK_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_balance_repair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grouping-only balance-repair rerun for the elecdiag load-following domain.")
    parser.add_argument("--raw-path", default=str(groups.DEFAULT_RAW_PATH))
    parser.add_argument("--baseline-work-dir", default=str(DEFAULT_BASELINE_WORK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _load_baseline_inputs(baseline_work_dir: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    metadata_df = pd.read_csv(baseline_work_dir / "run_metadata.csv")
    selected_config_df = pd.read_csv(baseline_work_dir / "load_following_selected_config.csv")
    assignments_df = pd.read_csv(baseline_work_dir / "group_assignments.csv")
    calibration_log_df = pd.read_csv(baseline_work_dir / "load_following_calibration_log.csv")
    return metadata_df.iloc[0], selected_config_df.iloc[0], assignments_df, calibration_log_df


def main() -> int:
    args = parse_args()
    baseline_work_dir = Path(args.baseline_work_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, selected_config, baseline_assignments_df, calibration_log_df = _load_baseline_inputs(baseline_work_dir)
    if int(round(float(selected_config["resolution_minutes"]))) != 60:
        raise RuntimeError("Balance-repair rerun is frozen to resolution_minutes=60.")
    if abs(float(selected_config["reserve_margin_multiplier"]) - 0.10) > 1e-12:
        raise RuntimeError("Balance-repair rerun is frozen to reserve_margin_multiplier=0.10.")

    timestamps, values = groups.load_raw_electricity_panel(Path(args.raw_path))
    baseline_eligible_client_ids = tuple(str(value) for value in baseline_assignments_df["client_id"].tolist())
    shared_block = groups.resolve_shared_block_from_metadata(
        timestamps,
        metadata,
        eligible_client_ids=baseline_eligible_client_ids,
    )

    block_values = values.iloc[shared_block.start_idx : shared_block.end_idx_exclusive].to_numpy(dtype="float32")
    eligible_stats_df = groups.compute_eligible_client_stats(block_values, values.columns.to_numpy(dtype=str))
    recomputed_eligible = tuple(sorted(str(value) for value in eligible_stats_df["client_id"].tolist()))
    baseline_eligible = tuple(sorted(baseline_eligible_client_ids))
    eligible_client_set_matches_baseline = recomputed_eligible == baseline_eligible
    if not eligible_client_set_matches_baseline:
        raise RuntimeError("Eligible-client set on the fixed shared block does not match the frozen baseline lineage.")

    assignments_df, dropped_clients_df, retention_diagnostics_df = groups.build_balance_repair_assignments(eligible_stats_df)
    retention_diagnostics_df["eligible_client_set_matches_baseline_flag"] = bool(eligible_client_set_matches_baseline)
    retention_diagnostics_df["baseline_eligible_client_count"] = int(len(baseline_eligible))
    retention_diagnostics_df["rerun_eligible_client_count"] = int(len(recomputed_eligible))

    results = groups.run_config_from_assignments(
        values=values,
        timestamps=timestamps,
        shared_block=shared_block,
        assignments=assignments_df,
        resolution_minutes=60,
        reserve_margin_multiplier=0.10,
        group_ids=list(range(10)),
    )
    results.update(
        {
            "grouping_strategy": groups.GROUPING_STRATEGY_BALANCE_REPAIR_V1,
            "baseline_work_dir": str(baseline_work_dir),
            "eligible_client_set_matches_baseline_flag": bool(eligible_client_set_matches_baseline),
            "baseline_eligible_client_count": int(len(baseline_eligible)),
            "rerun_eligible_client_count": int(len(recomputed_eligible)),
            "retained_client_count": int(retention_diagnostics_df.iloc[0]["retained_client_count"]),
            "dropped_client_count": int(retention_diagnostics_df.iloc[0]["dropped_client_count"]),
            "retained_client_fraction": float(retention_diagnostics_df.iloc[0]["retained_client_fraction"]),
            "retention_diagnostics_df": retention_diagnostics_df,
            "dropped_clients_df": dropped_clients_df,
        }
    )
    groups.write_group_config_outputs(results, output_dir)

    calibration_log_df.to_csv(output_dir / "load_following_calibration_log.csv", index=False)
    pd.DataFrame([selected_config]).to_csv(output_dir / "load_following_selected_config.csv", index=False)

    print(f"[load-following-elecdiag-balance-repair] wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
