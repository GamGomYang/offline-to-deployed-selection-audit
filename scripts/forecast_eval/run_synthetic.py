#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_synthetic_summary import build_synthetic_outputs  # noqa: E402
from synthetic_core import (  # noqa: E402
    CALIBRATION_SEEDS,
    DEFAULT_CALIBRATION_REPORT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SELECTED_CONFIG_PATH,
    REPORT_SEEDS,
    build_q1_frame,
    build_q2_frame,
    config_from_dict,
    config_id,
    load_selected_config,
    run_split_calibration,
    save_selected_config,
    update_calibration_report_failure,
    validate_report_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and report the synthetic benchmark for Q1/Q2.")
    parser.add_argument(
        "--mode",
        choices=["calibrate", "report", "all"],
        default="all",
        help="Whether to only calibrate, only report with a selected config, or do both.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for synthetic artifacts.")
    return parser.parse_args()


def _write_results(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _refresh_master_summary() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "forecast_eval" / "build_summary.py")],
        cwd=str(ROOT.parent),
        check=True,
    )


def run_report(output_dir: Path, selected_payload: dict[str, object] | None = None) -> dict[str, object]:
    payload = selected_payload or load_selected_config(output_dir / DEFAULT_SELECTED_CONFIG_PATH.name)
    if "q1_selected_config" in payload and "q2_selected_config" in payload:
        q1_config = config_from_dict(payload["q1_selected_config"])
        q2_config = config_from_dict(payload["q2_selected_config"])
        q1_df = build_q1_frame(q1_config, REPORT_SEEDS)
        q2_df = build_q2_frame(q2_config, REPORT_SEEDS)
        selected_config_ids = [config_id(q1_config), config_id(q2_config)]
    else:
        config = config_from_dict(payload["selected_config"])
        q1_df = build_q1_frame(config, REPORT_SEEDS)
        q2_df = build_q2_frame(config, REPORT_SEEDS)
        selected_config_ids = [config_id(config)]

    q1_path = output_dir / "q1_same_forecast_diff_interface.csv"
    q2_path = output_dir / "q2_diff_forecasts_same_interface.csv"
    _write_results(q1_df, q1_path)
    _write_results(q2_df, q2_path)

    summary_payload = build_synthetic_outputs(output_dir, q1_df=q1_df, q2_df=q2_df)
    validation_payload = validate_report_payload(
        q1_df,
        q2_df,
        summary_payload["q1_gap_by_friction"],
        summary_payload["q2_ranking_disagreement_by_friction"],
        summary_payload["q2_rank_correlation_by_friction"],
        summary_payload["q2_pairwise_flips"],
        report_seeds=REPORT_SEEDS,
    )

    payload["post_report_checks"] = {
        "report_seeds": REPORT_SEEDS,
        **validation_payload,
    }
    save_selected_config(output_dir / DEFAULT_SELECTED_CONFIG_PATH.name, payload)
    update_calibration_report_failure(
        output_dir / DEFAULT_CALIBRATION_REPORT_PATH.name,
        selected_config_ids,
        validation_payload,
    )

    _refresh_master_summary()

    print(f"[synthetic] wrote {len(q1_df)} Q1 rows to {q1_path}")
    print(f"[synthetic] wrote {len(q2_df)} Q2 rows to {q2_path}")
    print(
        "[synthetic] validation "
        f"failed={validation_payload['failed_benchmark']} "
        f"reasons={validation_payload['failure_reasons']}"
    )
    return payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_payload: dict[str, object] | None = None
    if args.mode in {"calibrate", "all"}:
        report_df, selected_payload = run_split_calibration(output_dir, CALIBRATION_SEEDS)
        print(
            "[synthetic] calibration complete "
            f"rows={len(report_df)} "
            f"q1_selected={selected_payload['q1_selected_config_id']} "
            f"q1_warning={selected_payload['q1_selection_warning']} "
            f"q2_selected={selected_payload['q2_selected_config_id']} "
            f"q2_warning={selected_payload['q2_selection_warning']}"
        )

    if args.mode in {"report", "all"}:
        run_report(output_dir, selected_payload=selected_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
