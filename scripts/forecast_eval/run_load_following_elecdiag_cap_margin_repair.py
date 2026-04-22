#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_load_following_elecdiag_groups as groups  # noqa: E402


DEFAULT_BASELINE_WORK_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_balance_repair"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_cap_margin_repair"

SEARCH_CONFIGS: tuple[tuple[float, float], ...] = (
    (0.08, 0.99),
    (0.08, 0.995),
    (0.10, 0.99),
    (0.10, 0.995),
    (0.12, 0.99),
    (0.12, 0.995),
)
BASELINE_MARGIN = 0.10
BASELINE_CAP_QUANTILE = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cap/margin repair rerun for the elecdiag load-following domain.")
    parser.add_argument("--raw-path", default=str(groups.DEFAULT_RAW_PATH))
    parser.add_argument("--baseline-work-dir", default=str(DEFAULT_BASELINE_WORK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _load_balance_repair_inputs(
    baseline_work_dir: Path,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_df = pd.read_csv(baseline_work_dir / "run_metadata.csv")
    assignments_df = pd.read_csv(baseline_work_dir / "group_assignments.csv")
    dropped_clients_df = pd.read_csv(baseline_work_dir / "dropped_client_ids.csv")
    retention_diagnostics_df = pd.read_csv(baseline_work_dir / "retention_diagnostics.csv")
    baseline_selected_config_df = pd.read_csv(baseline_work_dir / "load_following_selected_config.csv")
    return (
        metadata_df.iloc[0],
        assignments_df,
        dropped_clients_df,
        retention_diagnostics_df,
        baseline_selected_config_df,
    )


def _q1_calibration_target_clip_mean(diagnostics_df: pd.DataFrame) -> float:
    q1_diag = diagnostics_df[
        (diagnostics_df["question_id"] == "Q1")
        & (diagnostics_df["seed"].isin(groups.CALIBRATION_GROUP_IDS))
    ][["group_id", "dispatch_target_clip_rate"]].drop_duplicates(subset=["group_id"])
    if q1_diag.empty:
        return float("nan")
    return float(q1_diag["dispatch_target_clip_rate"].mean())


def _selection_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        -float(row["q1_tempered_win_rate_10"]),
        float(row["q1_calibration_mean_target_clip_rate"]),
        abs(float(row["reserve_margin_multiplier"]) - BASELINE_MARGIN),
        float(row["dispatch_cap_quantile"]),
        float(row["reserve_margin_multiplier"]),
    )


def main() -> int:
    args = parse_args()
    baseline_work_dir = Path(args.baseline_work_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        metadata,
        assignments_df,
        dropped_clients_df,
        retention_diagnostics_df,
        baseline_selected_config_df,
    ) = _load_balance_repair_inputs(baseline_work_dir)
    baseline_selected_config = baseline_selected_config_df.iloc[0]

    if int(round(float(metadata["resolution_minutes"]))) != 60:
        raise RuntimeError("Cap/margin repair rerun is frozen to resolution_minutes=60.")
    if abs(float(metadata["reserve_margin_multiplier"]) - BASELINE_MARGIN) > 1e-12:
        raise RuntimeError("Balance-repair baseline must have reserve_margin_multiplier=0.10.")
    if str(metadata["grouping_strategy"]) != groups.GROUPING_STRATEGY_BALANCE_REPAIR_V1:
        raise RuntimeError("Cap/margin repair rerun must start from the balance_repair_v1 grouping lineage.")

    timestamps, values = groups.load_raw_electricity_panel(Path(args.raw_path))
    frozen_eligible_client_ids = tuple(
        sorted(
            list(assignments_df["client_id"].astype(str).tolist())
            + list(dropped_clients_df["client_id"].astype(str).tolist())
        )
    )
    shared_block = groups.resolve_shared_block_from_metadata(
        timestamps,
        metadata,
        eligible_client_ids=frozen_eligible_client_ids,
    )

    block_values = values.iloc[shared_block.start_idx : shared_block.end_idx_exclusive].to_numpy(dtype="float32")
    eligible_stats_df = groups.compute_eligible_client_stats(block_values, values.columns.to_numpy(dtype=str))
    recomputed_eligible_client_ids = tuple(sorted(str(value) for value in eligible_stats_df["client_id"].tolist()))
    eligible_client_set_matches_balance_repair = recomputed_eligible_client_ids == frozen_eligible_client_ids
    if not eligible_client_set_matches_balance_repair:
        raise RuntimeError("Eligible-client set on the fixed shared block no longer matches the balance-repair lineage.")

    block_timestamps, group_series_15m = groups.build_group_aggregates(values, timestamps, shared_block, assignments_df)
    bundles, group_summary_df, balance_audit_df = groups.prepare_resolution_bundles(
        block_timestamps,
        group_series_15m,
        assignments_df,
        resolution_minutes=60,
    )

    calibration_rows: list[dict[str, Any]] = []
    for reserve_margin_multiplier, dispatch_cap_quantile in SEARCH_CONFIGS:
        calibration_results = groups.run_config(
            bundles,
            reserve_margin_multiplier=float(reserve_margin_multiplier),
            dispatch_cap_quantile=float(dispatch_cap_quantile),
            group_ids=list(groups.CALIBRATION_GROUP_IDS),
        )
        q1_assessment = groups.assess_q1(
            calibration_results["q1_df"],
            calibration_results["freeze_df"],
            calibration_results["diagnostics_df"],
            eval_group_ids=groups.CALIBRATION_GROUP_IDS,
        )
        q2_assessment = groups.assess_q2(
            calibration_results["q2_df"],
            eval_group_ids=groups.CALIBRATION_GROUP_IDS,
        )
        q1_target_clip_mean = _q1_calibration_target_clip_mean(calibration_results["diagnostics_df"])
        invalid_slice_flag = not bool(q2_assessment["paper_facing_valid"])
        q2_promotion_pass = bool(q2_assessment["promotion_gate_pass"])

        discard_reasons: list[str] = []
        if invalid_slice_flag:
            discard_reasons.append("invalid_slice")
        if not q2_promotion_pass:
            discard_reasons.append("q2_promotion_pass_failed")
        if not q1_target_clip_mean <= 0.02:
            discard_reasons.append("q1_target_clip_above_0_02")
        feasible = not discard_reasons

        calibration_rows.append(
            {
                "resolution_minutes": 60,
                "reserve_margin_multiplier": float(reserve_margin_multiplier),
                "dispatch_cap_quantile": float(dispatch_cap_quantile),
                "q1_calibration_mean_target_clip_rate": float(q1_target_clip_mean),
                "q1_tempered_win_rate_05": float(q1_assessment["high_friction_tempered_win_rate_05"]),
                "q1_tempered_win_rate_10": float(q1_assessment["high_friction_tempered_win_rate_10"]),
                "q1_zero_gap_ok": bool(q1_assessment["zero_gap_ok"]),
                "q2_zero_friction_mean_flip_rate": float(q2_assessment["zero_friction_mean_flip_rate"]),
                "n_positive_drift_frictions": int(len(q2_assessment["drift_positive_frictions"])),
                "drift_positive_frictions": "|".join(str(v) for v in q2_assessment["drift_positive_frictions"]),
                "first_drift_friction": q2_assessment["first_drift_friction"],
                "q2_promotion_pass": bool(q2_promotion_pass),
                "invalid_slice_flag": bool(invalid_slice_flag),
                "matches_balance_repair_baseline_flag": bool(
                    abs(float(reserve_margin_multiplier) - BASELINE_MARGIN) <= 1e-12
                    and abs(float(dispatch_cap_quantile) - BASELINE_CAP_QUANTILE) <= 1e-12
                ),
                "feasible": bool(feasible),
                "discard_reason": ";".join(discard_reasons),
                "selection_reason": "",
            }
        )

    calibration_log_df = pd.DataFrame(calibration_rows).sort_values(
        ["reserve_margin_multiplier", "dispatch_cap_quantile"]
    ).reset_index(drop=True)

    feasible_rows = [row for row in calibration_rows if bool(row["feasible"])]
    no_viable_cap_margin_config_flag = not feasible_rows

    if no_viable_cap_margin_config_flag:
        retention_out_df = retention_diagnostics_df.copy()
        retention_out_df["eligible_client_set_matches_balance_repair_flag"] = bool(eligible_client_set_matches_balance_repair)
        retention_out_df["frozen_group_assignments_reused_flag"] = True
        retention_out_df["frozen_dropped_client_set_reused_flag"] = True

        calibration_log_df.to_csv(output_dir / "cap_margin_repair_calibration_log.csv", index=False)
        calibration_log_df.to_csv(output_dir / "load_following_calibration_log.csv", index=False)
        assignments_df.to_csv(output_dir / "group_assignments.csv", index=False)
        dropped_clients_df.to_csv(output_dir / "dropped_client_ids.csv", index=False)
        retention_out_df.to_csv(output_dir / "retention_diagnostics.csv", index=False)
        group_summary_df.to_csv(output_dir / "group_summary.csv", index=False)
        balance_audit_df.to_csv(output_dir / "group_balance_audit.csv", index=False)
        pd.DataFrame(
            [
                {
                    "domain": groups.DOMAIN_ID,
                    "resolution_minutes": 60,
                    "reserve_margin_multiplier": BASELINE_MARGIN,
                    "dispatch_cap_quantile": BASELINE_CAP_QUANTILE,
                    "block_start_timestamp": shared_block.start_timestamp,
                    "block_end_timestamp": shared_block.end_timestamp,
                    "n_raw_steps": int(shared_block.n_raw_steps),
                    "calibration_group_ids": "|".join(str(v) for v in groups.CALIBRATION_GROUP_IDS),
                    "evaluation_group_ids": "|".join(str(v) for v in groups.EVALUATION_GROUP_IDS),
                    "grouping_strategy": str(metadata["grouping_strategy"]),
                    "baseline_work_dir": str(baseline_work_dir),
                    "eligible_client_set_matches_balance_repair_flag": bool(eligible_client_set_matches_balance_repair),
                    "retained_client_count": int(retention_out_df.iloc[0]["retained_client_count"]),
                    "dropped_client_count": int(retention_out_df.iloc[0]["dropped_client_count"]),
                    "retained_client_fraction": float(retention_out_df.iloc[0]["retained_client_fraction"]),
                }
            ]
        ).to_csv(output_dir / "run_metadata.csv", index=False)
        selected_config_df = pd.DataFrame(
            [
                {
                    "resolution_minutes": 60,
                    "reserve_margin_multiplier": float("nan"),
                    "dispatch_cap_quantile": float("nan"),
                    "q1_calibration_mean_target_clip_rate": float("nan"),
                    "q1_tempered_win_rate_10": float("nan"),
                    "q2_promotion_pass": False,
                    "invalid_slice_flag": False,
                    "matches_balance_repair_baseline_flag": False,
                    "no_effective_parameter_repair_flag": False,
                    "no_viable_cap_margin_config_flag": True,
                    "selection_reason": "no feasible cap/margin config after declared feasibility filters",
                    "discard_reason": "no_feasible_config_after_filters",
                }
            ]
        )
        selected_config_df.to_csv(output_dir / "cap_margin_selected_config.csv", index=False)
        selected_config_df.to_csv(output_dir / "load_following_selected_config.csv", index=False)
        print(f"[load-following-elecdiag-cap-margin-repair] no viable config found; wrote calibration-only outputs to {output_dir}")
        return 0

    selected_row = min(feasible_rows, key=_selection_sort_key).copy()
    selection_reason = (
        "selected by calibration Q1 friction-1.0 win-rate, then lower Q1 clip, "
        "then margin closeness to 0.10, then lower cap quantile, then lower margin"
    )
    calibration_log_df.loc[
        (calibration_log_df["reserve_margin_multiplier"] == float(selected_row["reserve_margin_multiplier"]))
        & (calibration_log_df["dispatch_cap_quantile"] == float(selected_row["dispatch_cap_quantile"])),
        "selection_reason",
    ] = selection_reason
    selected_row["selection_reason"] = selection_reason
    selected_row["no_viable_cap_margin_config_flag"] = False
    selected_row["no_effective_parameter_repair_flag"] = bool(selected_row["matches_balance_repair_baseline_flag"])

    final_results = groups.run_config(
        bundles,
        reserve_margin_multiplier=float(selected_row["reserve_margin_multiplier"]),
        dispatch_cap_quantile=float(selected_row["dispatch_cap_quantile"]),
        group_ids=list(range(10)),
    )

    retention_out_df = retention_diagnostics_df.copy()
    retention_out_df["eligible_client_set_matches_balance_repair_flag"] = bool(eligible_client_set_matches_balance_repair)
    retention_out_df["frozen_group_assignments_reused_flag"] = True
    retention_out_df["frozen_dropped_client_set_reused_flag"] = True

    final_results.update(
        {
            "shared_block": shared_block,
            "assignments_df": assignments_df,
            "group_summary_df": group_summary_df,
            "group_balance_audit_df": balance_audit_df,
            "resolution_minutes": 60,
            "reserve_margin_multiplier": float(selected_row["reserve_margin_multiplier"]),
            "dispatch_cap_quantile": float(selected_row["dispatch_cap_quantile"]),
            "grouping_strategy": str(metadata["grouping_strategy"]),
            "baseline_work_dir": str(baseline_work_dir),
            "eligible_client_set_matches_balance_repair_flag": bool(eligible_client_set_matches_balance_repair),
            "retention_diagnostics_df": retention_out_df,
            "dropped_clients_df": dropped_clients_df,
            "selected_dispatch_cap_quantile": float(selected_row["dispatch_cap_quantile"]),
        }
    )
    groups.write_group_config_outputs(final_results, output_dir)

    calibration_log_df.to_csv(output_dir / "cap_margin_repair_calibration_log.csv", index=False)
    calibration_log_df.to_csv(output_dir / "load_following_calibration_log.csv", index=False)
    selected_config_df = pd.DataFrame([selected_row])
    selected_config_df.to_csv(output_dir / "cap_margin_selected_config.csv", index=False)
    selected_config_df.to_csv(output_dir / "load_following_selected_config.csv", index=False)

    print(f"[load-following-elecdiag-cap-margin-repair] wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
