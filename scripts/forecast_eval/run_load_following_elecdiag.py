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

import run_load_following_elecdiag_calibration as calibration  # noqa: E402
import run_load_following_elecdiag_groups as groups  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the promotion-track elecdiag load-following experiment.")
    parser.add_argument("--raw-path", default=str(groups.DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_log_df, selected = calibration.run_calibration(Path(args.raw_path))
    calibration_log_df.to_csv(output_dir / "load_following_calibration_log.csv", index=False)
    pd.DataFrame([selected]).to_csv(output_dir / "load_following_selected_config.csv", index=False)

    results = groups.run_group_config(
        raw_path=Path(args.raw_path),
        resolution_minutes=int(selected["resolution_minutes"]),
        reserve_margin_multiplier=float(selected["reserve_margin_multiplier"]),
        group_ids=list(range(10)),
    )
    groups.write_group_config_outputs(results, output_dir)

    metadata_path = output_dir / "run_metadata.csv"
    metadata_df = pd.read_csv(metadata_path)
    metadata_df["selected_resolution_minutes"] = int(selected["resolution_minutes"])
    metadata_df["selected_reserve_margin_multiplier"] = float(selected["reserve_margin_multiplier"])
    metadata_df["calibration_groups"] = "|".join(str(v) for v in groups.CALIBRATION_GROUP_IDS)
    metadata_df["evaluation_groups"] = "|".join(str(v) for v in groups.EVALUATION_GROUP_IDS)
    metadata_df.to_csv(metadata_path, index=False)

    print(f"[load-following-elecdiag] wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
