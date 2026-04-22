#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_load_following_elecdiag_groups as groups  # noqa: E402


DEFAULT_WORK_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag"
DEFAULT_LOCK_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_promotion_locked"
DEFAULT_PAPER_RESULTS_DIR = REPO_ROOT / "paper" / "forecasting_workshop" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build promotion-track summaries for the elecdiag load-following domain.")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_LOCK_DIR))
    parser.add_argument("--paper-results-dir", default=str(DEFAULT_PAPER_RESULTS_DIR))
    parser.add_argument("--baseline-work-dir", default=None)
    parser.add_argument("--skip-paper-results", action="store_true")
    return parser.parse_args()


def _safe_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "--"
    text = str(value).strip()
    return text if text else "--"


def _latex_escape(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\textbackslash ").replace("_", "\\_")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _copy_raw_artifacts(work_dir: Path, output_dir: Path) -> None:
    for name in [
        "q1_same_forecast_diff_interface.csv",
        "q2_diff_forecasts_same_interface.csv",
        "load_following_elecdiag_diagnostics.csv",
        "load_following_elecdiag_q1_freeze_check.csv",
        "load_following_elecdiag_model_failures.csv",
        "load_following_calibration_log.csv",
        "load_following_selected_config.csv",
        "group_assignments.csv",
        "group_summary.csv",
        "group_balance_audit.csv",
        "retention_diagnostics.csv",
        "dropped_client_ids.csv",
        "cap_margin_repair_calibration_log.csv",
        "cap_margin_selected_config.csv",
        "run_metadata.csv",
    ]:
        src = work_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _support_status(q1: dict[str, Any], q2: dict[str, Any], balance_status: str) -> tuple[str, str]:
    promotion_ready = bool(q1["promotion_gate_pass"] and q2["promotion_gate_pass"] and q2["paper_facing_valid"])
    if promotion_ready:
        if balance_status == "balance_warn":
            return "promotion_ready", "promotion-ready Q2 candidate with imbalance caveat"
        return "promotion_ready", "promotion-ready Q2 candidate"
    if q2["paper_facing_valid"] and (q1["minimum_support"] or q2["qualitative_minimum"]):
        if q2["first_drift_friction"] is not None and float(q2["first_drift_friction"]) >= 1.0 - 1e-12:
            return "appendix_support_only", "high-friction qualitative compatibility"
        return "appendix_support_only", "appendix support only"
    return "appendix_archival_only", "neutral archival reporting only"


def _balance_metrics(group_summary_df: pd.DataFrame, *, group_ids: tuple[int, ...] | None = None) -> dict[str, float]:
    frame = group_summary_df.copy()
    if group_ids is not None:
        frame = frame[frame["group_id"].isin(group_ids)].copy()
    return {
        "mean_ratio": float(frame["train_mean_load"].max() / max(frame["train_mean_load"].min(), 1e-12)),
        "std_ratio": float(frame["train_std_load"].max() / max(frame["train_std_load"].min(), 1e-12)),
        "count_ratio": float(frame["client_count"].max() / max(frame["client_count"].min(), 1)),
    }


def _q1_mean_target_clip_rate(diagnostics_df: pd.DataFrame) -> float:
    q1_diag = diagnostics_df[
        (diagnostics_df["question_id"] == "Q1")
        & (diagnostics_df["seed"].isin(groups.EVALUATION_GROUP_IDS))
    ].copy()
    return float(q1_diag["dispatch_target_clip_rate"].mean()) if not q1_diag.empty else float("nan")


def _calibration_q1_mean_target_clip_rate(diagnostics_df: pd.DataFrame) -> float:
    q1_diag = diagnostics_df[
        (diagnostics_df["question_id"] == "Q1")
        & (diagnostics_df["seed"].isin(groups.CALIBRATION_GROUP_IDS))
    ].copy()
    return float(q1_diag["dispatch_target_clip_rate"].mean()) if not q1_diag.empty else float("nan")


def _build_balance_repair_vs_baseline_summary(
    *,
    baseline_work_dir: Path,
    current_work_dir: Path,
    current_q1_assessment: dict[str, Any],
    current_q2_assessment: dict[str, Any],
    current_group_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_q1_df = _load_csv(baseline_work_dir / "q1_same_forecast_diff_interface.csv")
    baseline_q2_df = _load_csv(baseline_work_dir / "q2_diff_forecasts_same_interface.csv")
    baseline_diagnostics_df = _load_csv(baseline_work_dir / "load_following_elecdiag_diagnostics.csv")
    baseline_freeze_df = _load_csv(baseline_work_dir / "load_following_elecdiag_q1_freeze_check.csv")
    baseline_group_summary_df = _load_csv(baseline_work_dir / "group_summary.csv")
    retention_path = current_work_dir / "retention_diagnostics.csv"
    current_retention_df = _load_csv(retention_path) if retention_path.exists() else pd.DataFrame()

    baseline_q1_assessment = groups.assess_q1(baseline_q1_df, baseline_freeze_df, baseline_diagnostics_df)
    baseline_q2_assessment = groups.assess_q2(baseline_q2_df)

    baseline_overall = _balance_metrics(baseline_group_summary_df)
    baseline_eval = _balance_metrics(baseline_group_summary_df, group_ids=groups.EVALUATION_GROUP_IDS)
    current_overall = _balance_metrics(current_group_summary_df)
    current_eval = _balance_metrics(current_group_summary_df, group_ids=groups.EVALUATION_GROUP_IDS)

    baseline_q1_clip = _q1_mean_target_clip_rate(baseline_diagnostics_df)
    current_diagnostics_df = _load_csv(current_work_dir / "load_following_elecdiag_diagnostics.csv")
    current_q1_clip = _q1_mean_target_clip_rate(current_diagnostics_df)

    balance_materially_improved = bool(
        current_eval["mean_ratio"] <= baseline_eval["mean_ratio"] * 0.80
        and abs(current_eval["count_ratio"] - 1.0) <= 1e-12
    )
    q1_clip_materially_improved = bool(
        current_q1_clip <= 0.02
        or current_q1_clip <= baseline_q1_clip * 0.75
    )
    q1_winrate_materially_improved = bool(current_q1_assessment["high_friction_tempered_win_rate_10"] >= 0.75)
    baseline_drift_count = int(len(baseline_q2_assessment["drift_positive_frictions"]))
    current_drift_count = int(len(current_q2_assessment["drift_positive_frictions"]))
    strongest_pair_same = current_q2_assessment["strongest_flip_pair"] == baseline_q2_assessment["strongest_flip_pair"]
    strongest_pair_stable = bool(strongest_pair_same or current_q2_assessment["strongest_flip_share"] >= 0.5)
    q2_stable = bool(
        current_q2_assessment["promotion_gate_pass"]
        and current_q2_assessment["paper_facing_valid"]
        and current_drift_count >= baseline_drift_count
        and strongest_pair_stable
    )
    later_cap_margin_rerun_justified = bool(
        balance_materially_improved
        and q2_stable
        and (q1_clip_materially_improved or q1_winrate_materially_improved)
    )
    next_action = "justify_small_cap_margin_rerun" if later_cap_margin_rerun_justified else "stop_keep_current_verdict"

    retained_client_count = None
    dropped_client_count = None
    retained_client_fraction = None
    eligible_match_flag = None
    if not current_retention_df.empty:
        retained_client_count = int(current_retention_df.iloc[0]["retained_client_count"])
        dropped_client_count = int(current_retention_df.iloc[0]["dropped_client_count"])
        retained_client_fraction = float(current_retention_df.iloc[0]["retained_client_fraction"])
        eligible_match_flag = bool(current_retention_df.iloc[0]["eligible_client_set_matches_baseline_flag"])

    return pd.DataFrame(
        [
            {
                "baseline_overall_mean_ratio": baseline_overall["mean_ratio"],
                "rerun_overall_mean_ratio": current_overall["mean_ratio"],
                "baseline_overall_std_ratio": baseline_overall["std_ratio"],
                "rerun_overall_std_ratio": current_overall["std_ratio"],
                "baseline_overall_count_ratio": baseline_overall["count_ratio"],
                "rerun_overall_count_ratio": current_overall["count_ratio"],
                "baseline_eval_mean_ratio": baseline_eval["mean_ratio"],
                "rerun_eval_mean_ratio": current_eval["mean_ratio"],
                "baseline_eval_std_ratio": baseline_eval["std_ratio"],
                "rerun_eval_std_ratio": current_eval["std_ratio"],
                "baseline_eval_count_ratio": baseline_eval["count_ratio"],
                "rerun_eval_count_ratio": current_eval["count_ratio"],
                "baseline_q1_zero_gap": baseline_q1_assessment["zero_friction_mean_group_abs_gap"],
                "rerun_q1_zero_gap": current_q1_assessment["zero_friction_mean_group_abs_gap"],
                "baseline_q1_mean_target_clip_rate": baseline_q1_clip,
                "rerun_q1_mean_target_clip_rate": current_q1_clip,
                "baseline_q1_tempered_win_rate_05": baseline_q1_assessment["high_friction_tempered_win_rate_05"],
                "rerun_q1_tempered_win_rate_05": current_q1_assessment["high_friction_tempered_win_rate_05"],
                "baseline_q1_tempered_win_rate_10": baseline_q1_assessment["high_friction_tempered_win_rate_10"],
                "rerun_q1_tempered_win_rate_10": current_q1_assessment["high_friction_tempered_win_rate_10"],
                "baseline_q2_drift_positive_frictions": "|".join(str(v) for v in baseline_q2_assessment["drift_positive_frictions"]),
                "rerun_q2_drift_positive_frictions": "|".join(str(v) for v in current_q2_assessment["drift_positive_frictions"]),
                "baseline_q2_strongest_flip_pair": baseline_q2_assessment["strongest_flip_pair"],
                "rerun_q2_strongest_flip_pair": current_q2_assessment["strongest_flip_pair"],
                "baseline_q2_strongest_flip_share": baseline_q2_assessment["strongest_flip_share"],
                "rerun_q2_strongest_flip_share": current_q2_assessment["strongest_flip_share"],
                "baseline_q2_invalid_slice_flag": not baseline_q2_assessment["paper_facing_valid"],
                "rerun_q2_invalid_slice_flag": not current_q2_assessment["paper_facing_valid"],
                "eligible_client_set_matches_baseline_flag": eligible_match_flag,
                "retained_client_count": retained_client_count,
                "dropped_client_count": dropped_client_count,
                "retained_client_fraction": retained_client_fraction,
                "balance_materially_improved_flag": balance_materially_improved,
                "q1_clip_materially_improved_flag": q1_clip_materially_improved,
                "q1_winrate_materially_improved_flag": q1_winrate_materially_improved,
                "q2_stable_flag": q2_stable,
                "later_cap_margin_rerun_justified_flag": later_cap_margin_rerun_justified,
                "next_action": next_action,
            }
        ]
    )


def _build_cap_margin_repair_vs_balance_repair_summary(
    *,
    baseline_work_dir: Path,
    current_work_dir: Path,
    current_q1_assessment: dict[str, Any],
    current_q2_assessment: dict[str, Any],
) -> pd.DataFrame:
    baseline_q1_df = _load_csv(baseline_work_dir / "q1_same_forecast_diff_interface.csv")
    baseline_q2_df = _load_csv(baseline_work_dir / "q2_diff_forecasts_same_interface.csv")
    baseline_diagnostics_df = _load_csv(baseline_work_dir / "load_following_elecdiag_diagnostics.csv")
    baseline_freeze_df = _load_csv(baseline_work_dir / "load_following_elecdiag_q1_freeze_check.csv")
    current_diagnostics_df = _load_csv(current_work_dir / "load_following_elecdiag_diagnostics.csv")

    baseline_q1_assessment = groups.assess_q1(baseline_q1_df, baseline_freeze_df, baseline_diagnostics_df)
    baseline_q2_assessment = groups.assess_q2(baseline_q2_df)

    baseline_q1_clip = _q1_mean_target_clip_rate(baseline_diagnostics_df)
    current_q1_clip = _q1_mean_target_clip_rate(current_diagnostics_df)
    baseline_q1_clip_calibration = _calibration_q1_mean_target_clip_rate(baseline_diagnostics_df)
    current_q1_clip_calibration = _calibration_q1_mean_target_clip_rate(current_diagnostics_df)

    baseline_drift_count = int(len(baseline_q2_assessment["drift_positive_frictions"]))
    current_drift_count = int(len(current_q2_assessment["drift_positive_frictions"]))
    strongest_pair_same = current_q2_assessment["strongest_flip_pair"] == baseline_q2_assessment["strongest_flip_pair"]
    strongest_pair_stable = bool(strongest_pair_same or current_q2_assessment["strongest_flip_share"] >= 0.5)
    q2_stable = bool(
        current_q2_assessment["promotion_gate_pass"]
        and current_q2_assessment["paper_facing_valid"]
        and current_drift_count >= baseline_drift_count
        and strongest_pair_stable
    )
    q1_clip_kept = bool(current_q1_clip <= 0.02)
    q1_winrate_materially_improved = bool(current_q1_assessment["high_friction_tempered_win_rate_10"] >= 0.75)
    promotion_reconsideration_justified = bool(
        q2_stable and q1_clip_kept and q1_winrate_materially_improved
    )
    next_action = (
        "reconsider_promotion_with_balance_caveat"
        if promotion_reconsideration_justified
        else "stop_keep_current_verdict"
    )

    return pd.DataFrame(
        [
            {
                "baseline_q1_mean_target_clip_rate": baseline_q1_clip,
                "rerun_q1_mean_target_clip_rate": current_q1_clip,
                "baseline_calibration_q1_mean_target_clip_rate": baseline_q1_clip_calibration,
                "rerun_calibration_q1_mean_target_clip_rate": current_q1_clip_calibration,
                "baseline_q1_tempered_win_rate_05": baseline_q1_assessment["high_friction_tempered_win_rate_05"],
                "rerun_q1_tempered_win_rate_05": current_q1_assessment["high_friction_tempered_win_rate_05"],
                "baseline_q1_tempered_win_rate_10": baseline_q1_assessment["high_friction_tempered_win_rate_10"],
                "rerun_q1_tempered_win_rate_10": current_q1_assessment["high_friction_tempered_win_rate_10"],
                "baseline_q2_drift_positive_frictions": "|".join(str(v) for v in baseline_q2_assessment["drift_positive_frictions"]),
                "rerun_q2_drift_positive_frictions": "|".join(str(v) for v in current_q2_assessment["drift_positive_frictions"]),
                "baseline_q2_strongest_flip_pair": baseline_q2_assessment["strongest_flip_pair"],
                "rerun_q2_strongest_flip_pair": current_q2_assessment["strongest_flip_pair"],
                "baseline_q2_strongest_flip_share": baseline_q2_assessment["strongest_flip_share"],
                "rerun_q2_strongest_flip_share": current_q2_assessment["strongest_flip_share"],
                "baseline_q2_invalid_slice_flag": not baseline_q2_assessment["paper_facing_valid"],
                "rerun_q2_invalid_slice_flag": not current_q2_assessment["paper_facing_valid"],
                "q1_clip_kept_at_or_below_0_02_flag": q1_clip_kept,
                "q1_winrate_materially_improved_flag": q1_winrate_materially_improved,
                "q2_stable_flag": q2_stable,
                "promotion_reconsideration_justified_flag": promotion_reconsideration_justified,
                "next_action": next_action,
            }
        ]
    )


def _build_no_viable_cap_margin_summary(*, baseline_work_dir: Path) -> pd.DataFrame:
    baseline_q1_df = _load_csv(baseline_work_dir / "q1_same_forecast_diff_interface.csv")
    baseline_q2_df = _load_csv(baseline_work_dir / "q2_diff_forecasts_same_interface.csv")
    baseline_diagnostics_df = _load_csv(baseline_work_dir / "load_following_elecdiag_diagnostics.csv")
    baseline_freeze_df = _load_csv(baseline_work_dir / "load_following_elecdiag_q1_freeze_check.csv")

    baseline_q1_assessment = groups.assess_q1(baseline_q1_df, baseline_freeze_df, baseline_diagnostics_df)
    baseline_q2_assessment = groups.assess_q2(baseline_q2_df)

    return pd.DataFrame(
        [
            {
                "baseline_q1_mean_target_clip_rate": _q1_mean_target_clip_rate(baseline_diagnostics_df),
                "rerun_q1_mean_target_clip_rate": float("nan"),
                "baseline_q1_tempered_win_rate_05": baseline_q1_assessment["high_friction_tempered_win_rate_05"],
                "rerun_q1_tempered_win_rate_05": float("nan"),
                "baseline_q1_tempered_win_rate_10": baseline_q1_assessment["high_friction_tempered_win_rate_10"],
                "rerun_q1_tempered_win_rate_10": float("nan"),
                "baseline_q2_drift_positive_frictions": "|".join(str(v) for v in baseline_q2_assessment["drift_positive_frictions"]),
                "rerun_q2_drift_positive_frictions": "",
                "baseline_q2_strongest_flip_pair": baseline_q2_assessment["strongest_flip_pair"],
                "rerun_q2_strongest_flip_pair": "",
                "baseline_q2_strongest_flip_share": baseline_q2_assessment["strongest_flip_share"],
                "rerun_q2_strongest_flip_share": float("nan"),
                "baseline_q2_invalid_slice_flag": not baseline_q2_assessment["paper_facing_valid"],
                "rerun_q2_invalid_slice_flag": float("nan"),
                "no_viable_cap_margin_config_flag": True,
                "q1_clip_kept_at_or_below_0_02_flag": False,
                "q1_winrate_materially_improved_flag": False,
                "q2_stable_flag": False,
                "promotion_reconsideration_justified_flag": False,
                "next_action": "stop_keep_current_verdict",
            }
        ]
    )


def _write_no_viable_cap_margin_note(
    *,
    output_dir: Path,
    selected_config: pd.Series,
    calibration_log_df: pd.DataFrame,
    balance_status: str,
    metadata: pd.Series,
) -> None:
    verdict_path = output_dir / "load_following_elecdiag_verdict.md"
    lines = [
        "# Load-Following Operational Domain Verdict",
        "",
        "- Verdict: appendix_support_only.",
        "- Interpretation note: no viable cap/margin config was found under the declared feasibility filters.",
        "- This cap/margin repair rerun did not execute a held-out full rerun because calibration groups 0|1 yielded no feasible config.",
        "- The paper-facing verdict therefore remains the current appendix_support_only branch.",
        f"- Selected operational resolution: {int(selected_config['resolution_minutes'])} minutes.",
        f"- Balance status inherited from the balance-repair lineage: {balance_status}.",
        "",
        "## Calibration stop condition",
        f"- selection_reason: {_safe_text(selected_config.get('selection_reason'))}.",
        f"- discard_reason: {_safe_text(selected_config.get('discard_reason'))}.",
        "- No further reruns are authorized beyond this cap/margin repair round.",
        "",
        "## Paper-facing use",
        "- reconsider_promotion_with_balance_caveat is not authorized.",
        "- Positive appendix wording remains governed by the current appendix_support_only verdict from the prior lineage.",
        f"- Shared block start: {metadata['block_start_timestamp']}.",
        f"- Shared block end: {metadata['block_end_timestamp']}.",
        "",
    ]
    verdict_path.write_text("\n".join(lines) + "\n")


def write_paper_table(
    paper_results_dir: Path,
    *,
    verdict: str,
    note: str,
    selected_config: pd.Series,
    q1_assessment: dict[str, Any],
    q2_assessment: dict[str, Any],
    balance_status: str,
) -> pd.DataFrame:
    if verdict == "promotion_ready":
        role = "promotion-ready"
    elif verdict == "appendix_support_only":
        role = "support"
    else:
        role = "archival"

    table_df = pd.DataFrame(
        [
            {
                "Domain": "Aggregate electricity-load proxy",
                "Role": role,
                "Groups": len(groups.EVALUATION_GROUP_IDS),
                "Resolution": f"{int(selected_config['resolution_minutes'])}min",
                "Q1": "promotion pass" if q1_assessment["promotion_gate_pass"] else ("mixed support" if q1_assessment["minimum_support"] else "archival"),
                "Q2": "promotion pass" if q2_assessment["promotion_gate_pass"] else ("qualitative compatibility" if q2_assessment["qualitative_minimum"] else "archival"),
                "Balance": balance_status,
                "Note": note,
            }
        ]
    )
    csv_path = paper_results_dir / "table_load_following_support_summary.csv"
    tex_path = paper_results_dir / "table_load_following_support_summary.tex"
    table_df.to_csv(csv_path, index=False)

    row = table_df.iloc[0]
    tex_lines = [
        r"\begin{tabular}{llrlllll}",
        r"\toprule",
        r"Domain & Role & Groups & Res. & Q1 & Q2 & Balance & Note \\",
        r"\midrule",
        (
            f"{_latex_escape(row['Domain'])} & "
            f"{_latex_escape(row['Role'])} & "
            f"{int(row['Groups'])} & "
            f"{_latex_escape(row['Resolution'])} & "
            f"{_latex_escape(row['Q1'])} & "
            f"{_latex_escape(row['Q2'])} & "
            f"{_latex_escape(row['Balance'])} & "
            f"{_latex_escape(row['Note'])} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex_path.write_text("\n".join(tex_lines) + "\n")
    return table_df


def write_verdict_note(
    output_dir: Path,
    paper_results_dir: Path,
    *,
    verdict: str,
    note: str,
    q1_assessment: dict[str, Any],
    q2_assessment: dict[str, Any],
    selected_config: pd.Series,
    calibration_log_df: pd.DataFrame,
    balance_status: str,
    metadata: pd.Series,
) -> None:
    verdict_path = output_dir / "load_following_elecdiag_verdict.md"
    paper_note_path = paper_results_dir / "load_following_support_note.md"

    match_mask = (
        (calibration_log_df["resolution_minutes"] == int(selected_config["resolution_minutes"]))
        & (calibration_log_df["reserve_margin_multiplier"] == float(selected_config["reserve_margin_multiplier"]))
    )
    if "dispatch_cap_quantile" in calibration_log_df.columns and "dispatch_cap_quantile" in selected_config.index:
        match_mask = match_mask & (
            calibration_log_df["dispatch_cap_quantile"] == float(selected_config["dispatch_cap_quantile"])
        )
    selected_match = calibration_log_df[match_mask].iloc[0]

    if verdict == "promotion_ready" and balance_status == "balance_warn":
        imbalance_caveat = (
            "Promotion criteria are met, but group-balance diagnostics are flagged as balance_warn; "
            "this promoted evidence should therefore be read with that imbalance caveat."
        )
    else:
        imbalance_caveat = ""

    positive_wording_allowed = verdict != "appendix_archival_only"
    main_text_mention_eligible = bool(
        verdict == "appendix_support_only"
        and q2_assessment["qualitative_minimum"]
        and balance_status != "balance_warn"
    )

    lines = [
        "# Load-Following Operational Domain Verdict",
        "",
        f"- Verdict: {verdict}.",
        f"- Interpretation note: {note}.",
        "- Seed definition: disjoint client-group evaluation units, with calibration groups 0|1 and evaluation groups 2|3|4|5|6|7|8|9.",
        "- Promotion is Q2-only in this submission round; no symmetric Q1 promotion is allowed.",
        f"- Selected operational resolution: {int(selected_config['resolution_minutes'])} minutes.",
        f"- Selected reserve margin multiplier: {_safe_float(selected_config['reserve_margin_multiplier'], digits=2)}.",
        (
            f"- Selected dispatch cap quantile: {_safe_float(selected_config['dispatch_cap_quantile'], digits=3)}."
            if "dispatch_cap_quantile" in selected_config.index
            else "- Selected dispatch cap quantile: 0.990."
        ),
        f"- Selected config selection reason: {_safe_text(selected_match['selection_reason']).rstrip('.')}.",
        f"- Balance status: {balance_status}.",
    ]
    if imbalance_caveat:
        lines.append(f"- Imbalance caveat: {imbalance_caveat}")
    if "matches_balance_repair_baseline_flag" in selected_config.index and bool(selected_config["matches_balance_repair_baseline_flag"]):
        lines.append("- Parameter-repair note: selected cap/margin config matches the balance-repair baseline, so no effective parameter repair was found.")
    if "no_effective_parameter_repair_flag" in selected_config.index and bool(selected_config["no_effective_parameter_repair_flag"]):
        lines.append("- Parameter-repair note: selected cap/margin config matches the balance-repair baseline, so no effective parameter repair was found.")

    lines.extend(
        [
            "",
            "## Q1",
            f"- Promotion gate pass: {'yes' if q1_assessment['promotion_gate_pass'] else 'no'}.",
            f"- Minimum support met: {'yes' if q1_assessment['minimum_support'] else 'no'}.",
            f"- Zero-friction mean abs gap: {_safe_float(q1_assessment['zero_friction_mean_group_abs_gap'])}.",
            f"- Pass friction summary: {q1_assessment['pass_friction_summary']}.",
            "",
            "## Q2",
            f"- Promotion gate pass: {'yes' if q2_assessment['promotion_gate_pass'] else 'no'}.",
            f"- Qualitative compatibility minimum met: {'yes' if q2_assessment['qualitative_minimum'] else 'no'}.",
            f"- Zero-friction mean flip rate: {_safe_float(q2_assessment['zero_friction_mean_flip_rate'])}.",
            f"- Zero-friction mean Spearman rho: {_safe_float(q2_assessment['zero_friction_mean_spearman_rho'])}.",
            f"- First drift friction: {_safe_float(q2_assessment['first_drift_friction'])}.",
            f"- Drift-positive frictions: {'|'.join(str(v) for v in q2_assessment['drift_positive_frictions']) or '--'}.",
            f"- Strongest flip pair: {q2_assessment['strongest_flip_pair'] or '--'} ({_safe_float(q2_assessment['strongest_flip_share'])}).",
            "",
            "## Promotion blocks",
            (
                f"- calibration_selected_config: {int(selected_config['resolution_minutes'])}min / "
                f"margin={_safe_float(selected_config['reserve_margin_multiplier'], digits=2)} / "
                f"cap_q={_safe_float(selected_config['dispatch_cap_quantile'], digits=3)}."
                if "dispatch_cap_quantile" in selected_config.index
                else f"- calibration_selected_config: {int(selected_config['resolution_minutes'])}min / margin={_safe_float(selected_config['reserve_margin_multiplier'], digits=2)}."
            ),
            f"- calibration_discard_reason: {_safe_text(selected_match['discard_reason'])}.",
            f"- evaluation_invalid_slice_flag: {'yes' if not q2_assessment['paper_facing_valid'] else 'no'}.",
            f"- promotion_block_reason: {'--' if verdict == 'promotion_ready' else note}.",
            "",
            "## Paper-facing use",
            f"- Positive appendix wording allowed: {'yes' if positive_wording_allowed else 'no'}.",
            (
                "- Invalid paper-facing use still allows archival of raw outputs, but it blocks both the optional "
                "main-text sentence and any positive appendix-support wording."
            ),
            f"- Optional main-text sentence eligible: {'yes' if main_text_mention_eligible else 'no'}.",
            f"- Shared block start: {metadata['block_start_timestamp']}.",
            f"- Shared block end: {metadata['block_end_timestamp']}.",
            "",
        ]
    )
    verdict_path.write_text("\n".join(lines) + "\n")

    paper_note_lines = [
        "Load-following support-domain reading",
        "",
        f"- Verdict: {verdict}",
        f"- Resolution: {int(selected_config['resolution_minutes'])}min",
        (
            f"- Cap quantile: {_safe_float(selected_config['dispatch_cap_quantile'], digits=3)}"
            if "dispatch_cap_quantile" in selected_config.index
            else "- Cap quantile: 0.990"
        ),
        f"- Q1: {'promotion pass' if q1_assessment['promotion_gate_pass'] else ('mixed support' if q1_assessment['minimum_support'] else 'archival')}",
        f"- Q2: {'promotion pass' if q2_assessment['promotion_gate_pass'] else ('qualitative compatibility' if q2_assessment['qualitative_minimum'] else 'archival')}",
        f"- Balance: {balance_status}",
        "- Evaluation uses disjoint client groups rather than overlapping rolling windows.",
    ]
    if imbalance_caveat:
        paper_note_lines.append(f"- Caveat: {imbalance_caveat}")
    if verdict == "appendix_archival_only":
        paper_note_lines.append("- This branch permits only neutral reporting of configuration and failure status.")
    elif verdict == "appendix_support_only":
        paper_note_lines.append("- This domain remains outside the main evidence hierarchy in the current submission round.")
    else:
        paper_note_lines.append("- If promoted into the manuscript, this domain is promoted for Q2 only.")
    if "matches_balance_repair_baseline_flag" in selected_config.index and bool(selected_config["matches_balance_repair_baseline_flag"]):
        paper_note_lines.append("- No effective cap/margin parameter repair was found relative to the balance-repair baseline.")
    paper_note_path.write_text("\n".join(paper_note_lines) + "\n")


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    paper_results_dir = Path(args.paper_results_dir)
    baseline_work_dir = Path(args.baseline_work_dir) if args.baseline_work_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_paper_results:
        paper_results_dir.mkdir(parents=True, exist_ok=True)

    selected_config_path = work_dir / "cap_margin_selected_config.csv"
    if not selected_config_path.exists():
        selected_config_path = work_dir / "load_following_selected_config.csv"
    selected_config_df = _load_csv(selected_config_path)
    selected_config = selected_config_df.iloc[0]
    no_viable_cap_margin = bool(selected_config.get("no_viable_cap_margin_config_flag", False))

    if no_viable_cap_margin:
        if baseline_work_dir is None:
            raise RuntimeError("cap/margin no-viable summary requires --baseline-work-dir.")
        calibration_log_df = _load_csv(work_dir / "load_following_calibration_log.csv")
        baseline_metadata_df = _load_csv(baseline_work_dir / "run_metadata.csv")
        baseline_balance_audit_df = _load_csv(baseline_work_dir / "group_balance_audit.csv")
        metadata = baseline_metadata_df.iloc[0]
        balance_status = str(baseline_balance_audit_df.iloc[0]["balance_status"])
        _copy_raw_artifacts(work_dir, output_dir)
        comparison_df = _build_no_viable_cap_margin_summary(baseline_work_dir=baseline_work_dir)
        comparison_df.to_csv(output_dir / "cap_margin_repair_vs_balance_repair_summary.csv", index=False)
        _write_no_viable_cap_margin_note(
            output_dir=output_dir,
            selected_config=selected_config,
            calibration_log_df=calibration_log_df,
            balance_status=balance_status,
            metadata=metadata,
        )
        pd.DataFrame(
            [
                {
                    "verdict": "appendix_support_only",
                    "note": "no viable cap/margin config under declared feasibility filters",
                    "balance_status": balance_status,
                }
            ]
        ).to_csv(output_dir / "verdict_status.csv", index=False)
        print(f"[load-following-elecdiag-summary] wrote {output_dir / 'cap_margin_repair_vs_balance_repair_summary.csv'}")
        print(f"[load-following-elecdiag-summary] wrote {output_dir / 'load_following_elecdiag_verdict.md'}")
        return 0

    q1_df = _load_csv(work_dir / "q1_same_forecast_diff_interface.csv")
    q2_df = _load_csv(work_dir / "q2_diff_forecasts_same_interface.csv")
    diagnostics_df = _load_csv(work_dir / "load_following_elecdiag_diagnostics.csv")
    freeze_df = _load_csv(work_dir / "load_following_elecdiag_q1_freeze_check.csv")
    calibration_log_df = _load_csv(work_dir / "load_following_calibration_log.csv")
    group_summary_df = _load_csv(work_dir / "group_summary.csv")
    group_balance_audit_df = _load_csv(work_dir / "group_balance_audit.csv")
    metadata_df = _load_csv(work_dir / "run_metadata.csv")

    metadata = metadata_df.iloc[0]
    balance_status = str(group_balance_audit_df.iloc[0]["balance_status"])

    q1_assessment = groups.assess_q1(q1_df, freeze_df, diagnostics_df)
    q2_assessment = groups.assess_q2(q2_df)
    verdict, note = _support_status(q1_assessment, q2_assessment, balance_status)

    q1_summary_path = output_dir / "q1_friction_threshold_summary.csv"
    q2_summary_path = output_dir / "q2_forecast_vs_deployed_summary.csv"
    rank_corr_path = output_dir / "q2_rank_correlation_by_friction.csv"
    pairwise_path = output_dir / "q2_pairwise_flips_by_friction.csv"
    diagnostics_path = output_dir / "diagnostics_share_by_friction.csv"

    q1_assessment["summary_df"].to_csv(q1_summary_path, index=False)
    q2_forecast_vs_deployed = groups.build_q2_forecast_vs_deployed_summary(
        q2_assessment["outputs"]["rank_correlation_by_friction"],
        q2_assessment["outputs"]["pairwise_flips_by_friction"],
    )
    q2_forecast_vs_deployed.to_csv(q2_summary_path, index=False)
    q2_assessment["outputs"]["rank_correlation_by_friction"].to_csv(rank_corr_path, index=False)
    q2_assessment["outputs"]["pairwise_flips_by_friction"].to_csv(pairwise_path, index=False)
    groups.build_diagnostics_share_summary(diagnostics_df).to_csv(diagnostics_path, index=False)

    _copy_raw_artifacts(work_dir, output_dir)
    if baseline_work_dir is not None:
        if (work_dir / "cap_margin_selected_config.csv").exists():
            comparison_df = _build_cap_margin_repair_vs_balance_repair_summary(
                baseline_work_dir=baseline_work_dir,
                current_work_dir=work_dir,
                current_q1_assessment=q1_assessment,
                current_q2_assessment=q2_assessment,
            )
            comparison_name = "cap_margin_repair_vs_balance_repair_summary.csv"
        else:
            comparison_df = _build_balance_repair_vs_baseline_summary(
                baseline_work_dir=baseline_work_dir,
                current_work_dir=work_dir,
                current_q1_assessment=q1_assessment,
                current_q2_assessment=q2_assessment,
                current_group_summary_df=group_summary_df,
            )
            comparison_name = "balance_repair_vs_baseline_summary.csv"
        comparison_df.to_csv(output_dir / comparison_name, index=False)
    if not args.skip_paper_results:
        write_paper_table(
            paper_results_dir,
            verdict=verdict,
            note=note,
            selected_config=selected_config,
            q1_assessment=q1_assessment,
            q2_assessment=q2_assessment,
            balance_status=balance_status,
        )
        write_verdict_note(
            output_dir,
            paper_results_dir,
            verdict=verdict,
            note=note,
            q1_assessment=q1_assessment,
            q2_assessment=q2_assessment,
            selected_config=selected_config,
            calibration_log_df=calibration_log_df,
            balance_status=balance_status,
            metadata=metadata,
        )
    else:
        write_verdict_note(
            output_dir,
            output_dir,
            verdict=verdict,
            note=note,
            q1_assessment=q1_assessment,
            q2_assessment=q2_assessment,
            selected_config=selected_config,
            calibration_log_df=calibration_log_df,
            balance_status=balance_status,
            metadata=metadata,
        )
        note_path = output_dir / "load_following_support_note.md"
        if note_path.exists():
            note_path.unlink()

    pd.DataFrame([{"verdict": verdict, "note": note, "balance_status": balance_status}]).to_csv(
        output_dir / "verdict_status.csv",
        index=False,
    )

    print(f"[load-following-elecdiag-summary] wrote {q1_summary_path}")
    print(f"[load-following-elecdiag-summary] wrote {q2_summary_path}")
    print(f"[load-following-elecdiag-summary] wrote {rank_corr_path}")
    print(f"[load-following-elecdiag-summary] wrote {pairwise_path}")
    print(f"[load-following-elecdiag-summary] wrote {diagnostics_path}")
    if baseline_work_dir is not None:
        if (work_dir / "cap_margin_selected_config.csv").exists():
            print(f"[load-following-elecdiag-summary] wrote {output_dir / 'cap_margin_repair_vs_balance_repair_summary.csv'}")
        else:
            print(f"[load-following-elecdiag-summary] wrote {output_dir / 'balance_repair_vs_baseline_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
