#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from synthetic_core import DEFAULT_OUTPUT_DIR, build_q1_gap_summary, build_q2_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic-benchmark figure-data CSVs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Synthetic output directory.")
    parser.add_argument("--q1-csv", default=None, help="Optional explicit path to the synthetic Q1 CSV.")
    parser.add_argument("--q2-csv", default=None, help="Optional explicit path to the synthetic Q2 CSV.")
    return parser.parse_args()


def build_synthetic_outputs(
    output_dir: str | Path,
    *,
    q1_df: pd.DataFrame | None = None,
    q2_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    base_dir = Path(output_dir)
    q1_frame = q1_df if q1_df is not None else pd.read_csv(base_dir / "q1_same_forecast_diff_interface.csv")
    q2_frame = q2_df if q2_df is not None else pd.read_csv(base_dir / "q2_diff_forecasts_same_interface.csv")

    _, q1_gap_by_friction = build_q1_gap_summary(q1_frame)
    _, q2_ranking_disagreement_by_friction, _, q2_rank_correlation_by_friction, q2_pairwise_flips = build_q2_diagnostics(
        q2_frame
    )

    outputs = {
        "q1_gap_by_friction": q1_gap_by_friction,
        "q2_ranking_disagreement_by_friction": q2_ranking_disagreement_by_friction,
        "q2_rank_correlation_by_friction": q2_rank_correlation_by_friction,
        "q2_pairwise_flips": q2_pairwise_flips,
    }

    for name, frame in outputs.items():
        path = base_dir / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    return outputs


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    q1_df = pd.read_csv(args.q1_csv) if args.q1_csv else None
    q2_df = pd.read_csv(args.q2_csv) if args.q2_csv else None
    outputs = build_synthetic_outputs(output_dir, q1_df=q1_df, q2_df=q2_df)

    for name in outputs:
        print(f"[synthetic-summary] wrote {output_dir / f'{name}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
