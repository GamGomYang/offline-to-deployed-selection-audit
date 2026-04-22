#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_load_following_elecdiag_groups as groups  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run restricted calibration for the elecdiag load-following domain.")
    parser.add_argument("--raw-path", default=str(groups.DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def select_best_config(calibration_log_df: pd.DataFrame) -> pd.Series:
    valid = calibration_log_df[~calibration_log_df["discarded"]].copy()
    if valid.empty:
        raise RuntimeError("All calibration configs were discarded.")
    resolution_preference = {60: 0, 30: 1}
    margin_preference = {0.10: 0, 0.15: 1, 0.05: 2}
    valid["resolution_pref"] = valid["resolution_minutes"].map(resolution_preference)
    valid["margin_pref"] = valid["reserve_margin_multiplier"].map(margin_preference)
    valid = valid.sort_values(
        [
            "n_positive_drift_frictions",
            "first_drift_friction_filled",
            "resolution_pref",
            "margin_pref",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return valid.iloc[0]


def run_calibration(raw_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    timestamps, values = groups.load_raw_electricity_panel(raw_path)
    shared_block, assignments = groups.choose_shared_block_and_assign_groups(timestamps, values)
    block_timestamps, group_series_15m = groups.build_group_aggregates(values, timestamps, shared_block, assignments)

    log_rows: list[dict[str, Any]] = []
    for resolution_minutes in groups.RESOLUTION_CHOICES:
        bundles, group_summary_df, balance_audit_df = groups.prepare_resolution_bundles(
            block_timestamps,
            group_series_15m,
            assignments,
            resolution_minutes=resolution_minutes,
        )
        for reserve_margin_multiplier in (0.05, 0.10, 0.15):
            results = groups.run_config(
                bundles,
                reserve_margin_multiplier=float(reserve_margin_multiplier),
                group_ids=list(groups.CALIBRATION_GROUP_IDS),
            )
            q1_assessment = groups.assess_q1(
                results["q1_df"],
                results["freeze_df"],
                results["diagnostics_df"],
                eval_group_ids=groups.CALIBRATION_GROUP_IDS,
            )
            q2_assessment = groups.assess_q2(
                results["q2_df"],
                eval_group_ids=groups.CALIBRATION_GROUP_IDS,
            )
            diagnostics_df = results["diagnostics_df"]
            calibration_diag = diagnostics_df[diagnostics_df["seed"].isin(groups.CALIBRATION_GROUP_IDS)]
            target_clip_rate = float(calibration_diag["dispatch_target_clip_rate"].mean()) if not calibration_diag.empty else 0.0
            exec_clip_rate = float(calibration_diag["dispatch_exec_clip_rate"].mean()) if not calibration_diag.empty else 0.0

            discard_reasons: list[str] = []
            if target_clip_rate > 0.02:
                discard_reasons.append("target_clip_rate_above_0.02")
            if exec_clip_rate > 0.02:
                discard_reasons.append("exec_clip_rate_above_0.02")
            if not q1_assessment["zero_gap_ok"]:
                discard_reasons.append("q1_zero_friction_exact_control_failed")
            if q2_assessment["zero_friction_mean_flip_rate"] > 0.10:
                discard_reasons.append("q2_zero_friction_flip_rate_above_0.10")

            drift_levels = q2_assessment["drift_positive_frictions"]
            first_drift_friction = q2_assessment["first_drift_friction"]
            first_drift_friction_filled = 99.0 if first_drift_friction is None else float(first_drift_friction)
            log_rows.append(
                {
                    "resolution_minutes": int(resolution_minutes),
                    "reserve_margin_multiplier": float(reserve_margin_multiplier),
                    "target_clip_rate": target_clip_rate,
                    "exec_clip_rate": exec_clip_rate,
                    "q1_zero_gap_ok": bool(q1_assessment["zero_gap_ok"]),
                    "q2_zero_friction_mean_flip_rate": float(q2_assessment["zero_friction_mean_flip_rate"]),
                    "n_positive_drift_frictions": int(len(drift_levels)),
                    "first_drift_friction": first_drift_friction,
                    "first_drift_friction_filled": first_drift_friction_filled,
                    "drift_positive_frictions": "|".join(str(v) for v in drift_levels),
                    "balance_status": str(balance_audit_df.iloc[0]["balance_status"]),
                    "discarded": bool(discard_reasons),
                    "discard_reason": "|".join(discard_reasons),
                    "selection_reason": "",
                }
            )

    calibration_log_df = pd.DataFrame(log_rows).sort_values(
        ["resolution_minutes", "reserve_margin_multiplier"]
    ).reset_index(drop=True)
    selected = select_best_config(calibration_log_df)
    selected_mask = (
        (calibration_log_df["resolution_minutes"] == int(selected["resolution_minutes"]))
        & (calibration_log_df["reserve_margin_multiplier"] == float(selected["reserve_margin_multiplier"]))
    )
    valid = calibration_log_df[~calibration_log_df["discarded"]].copy()
    tied_by_drift = valid[
        (valid["n_positive_drift_frictions"] == int(selected["n_positive_drift_frictions"]))
        & np.isclose(valid["first_drift_friction_filled"], float(selected["first_drift_friction_filled"]), atol=1e-12)
    ].copy()
    if int(selected["resolution_minutes"]) == 60 and len(tied_by_drift) > 1:
        selection_reason = (
            f"selected by predeclared objective order with {int(selected['n_positive_drift_frictions'])} drift-positive frictions "
            f"and first_drift_friction={float(selected['first_drift_friction_filled']):.2f}; 60min wins by the simpler-operational-grid tie-break."
        )
    else:
        selection_reason = (
            f"selected by predeclared objective order with {int(selected['n_positive_drift_frictions'])} drift-positive frictions "
            f"and first_drift_friction={float(selected['first_drift_friction_filled']):.2f}."
        )
    calibration_log_df.loc[selected_mask, "selection_reason"] = selection_reason
    return calibration_log_df, selected


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_log_df, selected = run_calibration(Path(args.raw_path))
    calibration_log_path = output_dir / "load_following_calibration_log.csv"
    selected_config_path = output_dir / "load_following_selected_config.csv"
    calibration_log_df.to_csv(calibration_log_path, index=False)
    pd.DataFrame([selected]).to_csv(selected_config_path, index=False)
    print(f"[load-following-elecdiag-calibration] wrote {calibration_log_path}")
    print(f"[load-following-elecdiag-calibration] wrote {selected_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
