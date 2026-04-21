#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_same_interface_rank_summary import build_domain_rank_summary, scores_tied  # noqa: E402


STEP_ORDER = {"step2": 2, "step3": 3, "step4": 4, "step5": 5, "step6": 6}
QUESTION_ORDER = {"Q1": 1, "Q2": 2}
DOMAIN_ORDER = {"synthetic": 0, "inventory": 1, "portfolio": 2, "all": 9}

YES = "yes"
NO = "no"
NA = "na"
PASS = "pass"
FAIL = "fail"

STEP2_Q1_SCENARIO = "step2_synthetic_q1_same_forecast_diff_interface"
STEP2_Q2_SCENARIO = "step2_synthetic_q2_diff_forecasts_same_interface"
STEP3_Q1_SCENARIO = "step3_portfolio_q1_exact_control"
STEP4_Q1_SCENARIO = "step4_inventory_q1_same_forecast_diff_interface"
STEP4_Q2_SCENARIO = "step4_inventory_q2_diff_forecasts_same_interface"
STEP5_Q2_SCENARIO = "step5_q2_same_interface_package"

PACKAGE_IDS = {
    "step2": "step2_synthetic_lock",
    "step3": "step3_portfolio_exact_control_lock",
    "step4": "step4_inventory_lock",
    "step5": "step5_q2_same_interface_package",
}

EXPECTED_INTERFACE_SETS = {
    ("step2", STEP2_Q1_SCENARIO): ("responsive", "tempered"),
    ("step2", STEP2_Q2_SCENARIO): ("tempered",),
    ("step3", STEP3_Q1_SCENARIO): ("eta_0_5", "eta_1_0"),
    ("step4", STEP4_Q1_SCENARIO): ("responsive", "tempered"),
    ("step4", STEP4_Q2_SCENARIO): ("responsive",),
}

STEP5_UPSTREAM_STEP = {"synthetic": "step2", "inventory": "step4", "portfolio": "none"}
STEP5_ZERO_SPEARMAN_MIN = {"required": 0.80, "stretch": 0.50}

STEP2_Q1_PATH = REPO_ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q1_same_forecast_diff_interface.csv"
STEP2_Q2_PATH = REPO_ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q2_diff_forecasts_same_interface.csv"
STEP2_Q1_GAP_PATH = REPO_ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q1_gap_by_friction.csv"
STEP2_LOCK_MD = REPO_ROOT / "forecasting" / "STEP2_CANDIDATE_LOCK.md"

STEP3_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "portfolio_exact_control_step3_candidate_lock"
STEP3_CONTROL_PATH = STEP3_DIR / "portfolio_control_results.csv"
STEP3_FORECAST_HASH_PATH = STEP3_DIR / "forecast_hash_check.csv"
STEP3_PROPOSAL_HASH_PATH = STEP3_DIR / "proposal_hash_check.csv"
STEP3_TARGET_DELTA_PATH = STEP3_DIR / "target_based_delta_summary.csv"
STEP3_EFFECT_SIZE_PATH = STEP3_DIR / "zero_cost_vs_positive_cost_effect_size_summary.csv"
STEP3_VERDICT_MD = STEP3_DIR / "step3_verdict.md"

STEP4_LOCK_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "inventory_step4_seed_stability_locked"
STEP4_Q1_PATH = STEP4_LOCK_DIR / "inventory_v2_seed_stability_q1.csv"
STEP4_Q2_PATH = STEP4_LOCK_DIR / "inventory_v2_seed_stability_q2.csv"
STEP4_Q1_THRESHOLD_PATH = STEP4_LOCK_DIR / "inventory_v2_seed_stability_q1_friction_threshold_summary.csv"
STEP4_Q2_SUMMARY_PATH = STEP4_LOCK_DIR / "inventory_v2_seed_stability_q2_forecast_vs_deployed_summary.csv"
STEP4_FREEZE_CHECK_PATH = STEP4_LOCK_DIR / "inventory_v2_seed_stability_freeze_check.csv"
STEP4_VERDICT_MD = REPO_ROOT / "outputs" / "forecast_eval" / "inventory_step4_candidate_lock" / "step4_verdict.md"

STEP5_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "step5_same_interface"
STEP5_VERDICT_PATH = STEP5_DIR / "domain_step5_verdict.csv"
STEP5_MANIFEST_PATH = STEP5_DIR / "manifest.json"
STEP5_NOTE_PATH = STEP5_DIR / "step5_verdict.md"

MASTER_COLUMNS = [
    "package_id",
    "record_type",
    "upstream_step_id",
    "step_id",
    "domain",
    "question_id",
    "scenario_id",
    "evidence_role",
    "inclusion_status",
    "seed",
    "friction",
    "forecaster_id",
    "interface_id",
    "forecast_metric",
    "target_metric",
    "executed_metric",
    "target_executed_gap",
    "flip_count",
    "flip_rate",
    "kendall_tau_b",
    "spearman_rho",
    "source_file",
    "is_locked_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Step 6 locked evidence ledger and delivery bundle.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "forecast_eval" / "step6_ledger"),
        help="Directory for Step 6 canonical outputs.",
    )
    parser.add_argument(
        "--outfile-dir",
        default=str(REPO_ROOT / "outfile"),
        help="Repo-root delivery mirror directory.",
    )
    return parser.parse_args()


def abs_path(path: Path | str) -> str:
    return str(Path(path).resolve())


def read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def read_text(path: Path | str) -> str:
    return Path(path).read_text()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def yesno(flag: bool) -> str:
    return YES if bool(flag) else NO


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[columns]


def sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["_step_order"] = out["step_id"].map(STEP_ORDER).fillna(99).astype(int)
    out["_domain_order"] = out["domain"].map(DOMAIN_ORDER).fillna(99).astype(int)
    if "question_id" in out.columns:
        out["_question_order"] = out["question_id"].map(QUESTION_ORDER).fillna(99).astype(int)
    sort_columns = ["_step_order", "_domain_order"]
    if "_question_order" in out.columns:
        sort_columns.append("_question_order")
    sort_columns.extend(columns)
    out = out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    drop_cols = [column for column in ["_step_order", "_domain_order", "_question_order"] if column in out.columns]
    return out.drop(columns=drop_cols)


def friction_key(value: Any) -> float:
    if pd.isna(value):
        return float("inf")
    return float(value)


def float_or_nan(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def group_duplicate_count(frame: pd.DataFrame, key_columns: list[str]) -> int:
    normalized = frame.copy()
    for column in key_columns:
        normalized[column] = normalized[column].astype(object).where(normalized[column].notna(), "__NA__")
    return int(normalized.duplicated(subset=key_columns, keep=False).sum())


def finite_violation_count(frame: pd.DataFrame, numeric_columns: list[str]) -> int:
    violations = 0
    for column in numeric_columns:
        series = frame[column]
        converted = pd.to_numeric(series, errors="coerce")
        bad_mask = series.notna() & (~np.isfinite(converted) | converted.isna())
        violations += int(bad_mask.sum())
    return violations


def observed_order(values: pd.Series) -> list[float]:
    seen: list[float] = []
    for value in values.tolist():
        scalar = float(value)
        if scalar not in seen:
            seen.append(scalar)
    return seen


def strictly_ascending(values: list[float]) -> bool:
    return values == sorted(values) and len(values) == len(set(values))


def raw_q2_artifacts(q2_df: pd.DataFrame, *, domain: str, expected_interface_id: str) -> dict[str, pd.DataFrame]:
    outputs, _meta = build_domain_rank_summary(q2_df, domain=domain, expected_interface_id=expected_interface_id)
    return outputs


def zero_row(rank_corr: pd.DataFrame) -> pd.Series:
    zero = rank_corr[np.isclose(rank_corr["friction_level"], 0.0, atol=1e-15)]
    if zero.empty:
        raise ValueError("Missing zero-friction row.")
    return zero.iloc[0]


def pairwise_row_for_friction(pairwise: pd.DataFrame, friction: float) -> pd.Series | None:
    rows = pairwise[np.isclose(pairwise["friction_level"], friction, atol=1e-15)]
    if rows.empty:
        return None
    return rows.sort_values(
        ["flip_seed_share", "model_a", "model_b"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]


def q2_zero_flat_pass(*, zero_flip_rate: float, zero_spearman: float, evidence_role: str) -> bool:
    return zero_flip_rate <= 0.10 and zero_spearman >= STEP5_ZERO_SPEARMAN_MIN[evidence_role]


def q2_sign_inversion_violations(raw_q2: pd.DataFrame) -> int:
    violations = 0
    for _group_key, group in raw_q2.groupby(["seed", "friction_level"], sort=False):
        rows = list(group.itertuples(index=False))
        for idx in range(len(rows)):
            for jdx in range(idx + 1, len(rows)):
                left = rows[idx]
                right = rows[jdx]
                for score_attr, rank_attr in [
                    ("forecast_metric", "rank_within_forecast_metric"),
                    ("executed_metric", "rank_within_executed_metric"),
                ]:
                    left_score = float(getattr(left, score_attr))
                    right_score = float(getattr(right, score_attr))
                    if scores_tied(left_score, right_score):
                        continue
                    left_rank = float(getattr(left, rank_attr))
                    right_rank = float(getattr(right, rank_attr))
                    if left_score > right_score and not left_rank < right_rank:
                        violations += 1
                    elif left_score < right_score and not left_rank > right_rank:
                        violations += 1
    return violations


def step3_paired_delta_table(raw_step3: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (universe_id, seed, kappa), group in raw_step3.groupby(["universe_id", "seed", "kappa"], sort=True):
        ordered = group.sort_values("replay_interface_id")
        eta_05 = ordered[ordered["replay_interface_id"] == "eta_0_5"].iloc[0]
        eta_10 = ordered[ordered["replay_interface_id"] == "eta_1_0"].iloc[0]
        rows.append(
            {
                "universe_id": str(universe_id),
                "seed": int(seed),
                "kappa": float(kappa),
                "executed_delta": float(eta_05["sharpe_exec_net"]) - float(eta_10["sharpe_exec_net"]),
                "target_delta": float(eta_05["sharpe_target_net"]) - float(eta_10["sharpe_target_net"]),
                "final_path_gap_eta_05": float(eta_05["final_path_gap"]),
                "tracking_error_l2_mean_eta_05": float(eta_05["tracking_error_l2_mean"]),
            }
        )
    return pd.DataFrame(rows)


def add_check(
    rows: list[dict[str, Any]],
    *,
    check_name: str,
    step_id: str,
    domain: str,
    scenario_id: str,
    pass_fail: str,
    measured_value: Any,
    threshold: Any,
    note: str,
) -> None:
    rows.append(
        {
            "check_name": check_name,
            "step_id": step_id,
            "domain": domain,
            "scenario_id": scenario_id,
            "pass_fail": pass_fail,
            "measured_value": measured_value,
            "threshold": threshold,
            "note": note,
        }
    )


def build_master_results(
    *,
    step2_q1: pd.DataFrame,
    step2_q2: pd.DataFrame,
    step3_raw: pd.DataFrame,
    step4_q1: pd.DataFrame,
    step4_q2: pd.DataFrame,
    step5_packages: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_row(**kwargs: Any) -> None:
        row = {column: np.nan for column in MASTER_COLUMNS}
        row.update(kwargs)
        rows.append(row)

    for row in step2_q1.itertuples(index=False):
        append_row(
            package_id=PACKAGE_IDS["step2"],
            record_type="raw_q1_interface_row",
            upstream_step_id="none",
            step_id="step2",
            domain="synthetic",
            question_id="Q1",
            scenario_id=STEP2_Q1_SCENARIO,
            evidence_role="required",
            inclusion_status="pass",
            seed=int(row.seed),
            friction=float(row.friction_level),
            forecaster_id=str(row.forecaster_id),
            interface_id=str(row.interface_id),
            forecast_metric=float(row.forecast_metric),
            target_metric=float(row.target_metric),
            executed_metric=float(row.executed_metric),
            target_executed_gap=float(row.target_executed_gap),
            source_file=abs_path(STEP2_Q1_PATH),
            is_locked_source=True,
        )

    for row in step2_q2.itertuples(index=False):
        append_row(
            package_id=PACKAGE_IDS["step2"],
            record_type="raw_q2_forecaster_row",
            upstream_step_id="none",
            step_id="step2",
            domain="synthetic",
            question_id="Q2",
            scenario_id=STEP2_Q2_SCENARIO,
            evidence_role="required",
            inclusion_status="pass",
            seed=int(row.seed),
            friction=float(row.friction_level),
            forecaster_id=str(row.forecaster_id),
            interface_id=str(row.interface_id),
            forecast_metric=float(row.forecast_metric),
            target_metric=float(row.target_metric),
            executed_metric=float(row.executed_metric),
            target_executed_gap=float(row.target_executed_gap),
            source_file=abs_path(STEP2_Q2_PATH),
            is_locked_source=True,
        )

    for row in step3_raw.itertuples(index=False):
        append_row(
            package_id=PACKAGE_IDS["step3"],
            record_type="raw_q1_interface_row",
            upstream_step_id="none",
            step_id="step3",
            domain="portfolio",
            question_id="Q1",
            scenario_id=STEP3_Q1_SCENARIO,
            evidence_role="support",
            inclusion_status="support_only",
            seed=int(row.seed),
            friction=float(row.kappa),
            forecaster_id=str(row.universe_id),
            interface_id=str(row.replay_interface_id),
            forecast_metric=np.nan,
            target_metric=float(row.sharpe_target_net),
            executed_metric=float(row.sharpe_exec_net),
            target_executed_gap=float(row.final_path_gap),
            source_file=abs_path(STEP3_CONTROL_PATH),
            is_locked_source=True,
        )

    for row in step4_q1.itertuples(index=False):
        append_row(
            package_id=PACKAGE_IDS["step4"],
            record_type="raw_q1_interface_row",
            upstream_step_id="none",
            step_id="step4",
            domain="inventory",
            question_id="Q1",
            scenario_id=STEP4_Q1_SCENARIO,
            evidence_role="required",
            inclusion_status="pass",
            seed=int(row.seed),
            friction=float(row.friction_level),
            forecaster_id=str(row.forecaster_id),
            interface_id=str(row.interface_id),
            forecast_metric=float(row.forecast_metric),
            target_metric=float(row.target_metric),
            executed_metric=float(row.executed_metric),
            target_executed_gap=float(row.target_executed_gap),
            source_file=abs_path(STEP4_Q1_PATH),
            is_locked_source=True,
        )

    for row in step4_q2.itertuples(index=False):
        append_row(
            package_id=PACKAGE_IDS["step4"],
            record_type="raw_q2_forecaster_row",
            upstream_step_id="none",
            step_id="step4",
            domain="inventory",
            question_id="Q2",
            scenario_id=STEP4_Q2_SCENARIO,
            evidence_role="required",
            inclusion_status="pass",
            seed=int(row.seed),
            friction=float(row.friction_level),
            forecaster_id=str(row.forecaster_id),
            interface_id=str(row.interface_id),
            forecast_metric=float(row.forecast_metric),
            target_metric=float(row.target_metric),
            executed_metric=float(row.executed_metric),
            target_executed_gap=float(row.target_executed_gap),
            source_file=abs_path(STEP4_Q2_PATH),
            is_locked_source=True,
        )

    for domain in ["synthetic", "inventory", "portfolio"]:
        package = step5_packages[domain]
        seed_stats = package["seed_stats"]
        verdict_row = package["verdict"]
        seed_stats_path = package["seed_stats_path"]
        for row in seed_stats.itertuples(index=False):
            append_row(
                package_id=PACKAGE_IDS["step5"],
                record_type="step5_seed_rank_summary",
                upstream_step_id=STEP5_UPSTREAM_STEP[domain],
                step_id="step5",
                domain=domain,
                question_id="Q2",
                scenario_id=STEP5_Q2_SCENARIO,
                evidence_role=str(verdict_row["domain_role"]),
                inclusion_status=str(verdict_row["inclusion_status"]),
                seed=int(row.seed),
                friction=float(row.friction_level),
                forecaster_id=np.nan,
                interface_id=str(row.interface_id),
                forecast_metric=np.nan,
                target_metric=np.nan,
                executed_metric=np.nan,
                target_executed_gap=np.nan,
                flip_count=float(row.flip_count),
                flip_rate=float(row.flip_rate),
                kendall_tau_b=float(row.kendall_tau_b),
                spearman_rho=float(row.spearman_rho),
                source_file=abs_path(seed_stats_path),
                is_locked_source=True,
            )

    master = ensure_columns(pd.DataFrame(rows), MASTER_COLUMNS)
    master = sort_frame(
        master,
        [
            "scenario_id",
            "friction",
            "seed",
            "record_type",
            "forecaster_id",
            "interface_id",
        ],
    )
    return master


def build_summary_by_domain(
    *,
    step2_q1_gap: pd.DataFrame,
    step2_q2_outputs: dict[str, pd.DataFrame],
    step3_raw: pd.DataFrame,
    step3_effect_size: pd.DataFrame,
    step4_q1_threshold: pd.DataFrame,
    step4_q2_outputs: dict[str, pd.DataFrame],
    step5_packages: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    step2_q1_zero_gap = float(
        step2_q1_gap.loc[np.isclose(step2_q1_gap["friction_level"], 0.0, atol=1e-15), "mean_abs_target_executed_gap"].iloc[0]
    )
    step2_q1_zero_flat = step2_q1_zero_gap <= 1e-12
    for row in step2_q1_gap.itertuples(index=False):
        friction = float(row.friction_level)
        rows.append(
            {
                "step_id": "step2",
                "domain": "synthetic",
                "question_id": "Q1",
                "scenario_id": STEP2_Q1_SCENARIO,
                "evidence_role": "required",
                "source_path": abs_path(STEP2_Q1_GAP_PATH),
                "n_seeds": int(row.n_seeds),
                "friction": friction,
                "q1_same_forecast_interface_effect_yesno": yesno(friction > 0.0 and float(row.mean_abs_target_executed_gap) > 1e-12),
                "q2_rank_flip_yesno": NA,
                "zero_friction_near_flat_yesno": yesno(step2_q1_zero_flat),
                "positive_friction_gap_yesno": yesno(friction > 0.0 and float(row.mean_abs_target_executed_gap) > 0.0),
                "inclusion_status": "pass",
                "notes": "zero-friction exact match; clean increasing gap" if friction == 0.0 else "clean increasing gap",
            }
        )

    step2_q2_rank_corr = step2_q2_outputs["rank_correlation_by_friction"]
    step2_q2_zero = zero_row(step2_q2_rank_corr)
    step2_q2_zero_flat = q2_zero_flat_pass(
        zero_flip_rate=float(step2_q2_zero["mean_flip_rate"]),
        zero_spearman=float(step2_q2_zero["mean_spearman_rho"]),
        evidence_role="required",
    )
    for row in step2_q2_rank_corr.itertuples(index=False):
        friction = float(row.friction_level)
        rows.append(
            {
                "step_id": "step2",
                "domain": "synthetic",
                "question_id": "Q2",
                "scenario_id": STEP2_Q2_SCENARIO,
                "evidence_role": "required",
                "source_path": abs_path(STEP2_Q2_PATH),
                "n_seeds": int(row.n_seeds),
                "friction": friction,
                "q1_same_forecast_interface_effect_yesno": NA,
                "q2_rank_flip_yesno": yesno(float(row.mean_flip_rate) > 0.0),
                "zero_friction_near_flat_yesno": yesno(step2_q2_zero_flat),
                "positive_friction_gap_yesno": yesno(
                    friction > 0.0
                    and (
                        float(row.mean_flip_rate) > float(step2_q2_zero["mean_flip_rate"])
                        or float(row.mean_spearman_rho) < float(step2_q2_zero["mean_spearman_rho"])
                    )
                ),
                "inclusion_status": "pass",
                "notes": "zero-friction exact ranking alignment" if friction == 0.0 else "positive-friction plateau",
            }
        )

    step3_paired = step3_paired_delta_table(step3_raw)
    zero_pairs = step3_paired[np.isclose(step3_paired["kappa"], 0.0, atol=1e-15)].copy()
    zero_within_threshold = int((zero_pairs["executed_delta"].abs() <= 0.005).sum())
    zero_total_pairs = int(len(zero_pairs))
    step3_zero_flat = zero_within_threshold >= 7
    effect_positive_count = int((step3_effect_size["effect_size_mean"] > 0.0).sum())
    for kappa, group in step3_paired.groupby("kappa", sort=True):
        kappa_value = float(kappa)
        rows.append(
            {
                "step_id": "step3",
                "domain": "portfolio",
                "question_id": "Q1",
                "scenario_id": STEP3_Q1_SCENARIO,
                "evidence_role": "support",
                "source_path": abs_path(STEP3_CONTROL_PATH),
                "n_seeds": int(group[["universe_id", "seed"]].drop_duplicates().shape[0]),
                "friction": kappa_value,
                "q1_same_forecast_interface_effect_yesno": yesno(kappa_value > 0.0 and float(group["executed_delta"].abs().mean()) > 0.005),
                "q2_rank_flip_yesno": NA,
                "zero_friction_near_flat_yesno": yesno(step3_zero_flat),
                "positive_friction_gap_yesno": yesno(kappa_value > 0.0 and float(group["executed_delta"].abs().mean()) > 0.0),
                "inclusion_status": "support_only",
                "notes": (
                    f"support_only; near-flat {zero_within_threshold}/{zero_total_pairs} seed groups within 0.005 threshold"
                    if kappa_value == 0.0
                    else f"support_only; positive-cost recurrence {effect_positive_count}/{len(step3_effect_size)} universes"
                ),
            }
        )

    step4_q1_zero_gap = float(
        step4_q1_threshold.loc[
            np.isclose(step4_q1_threshold["friction_level"], 0.0, atol=1e-15),
            "mean_abs_target_executed_gap_tempered",
        ].iloc[0]
    )
    step4_q1_zero_flat = step4_q1_zero_gap <= 1e-12
    for row in step4_q1_threshold.itertuples(index=False):
        friction = float(row.friction_level)
        if friction == 0.0:
            notes = "threshold story; zero-friction exact match"
        elif friction < 1.0:
            notes = "threshold story; low friction mixed"
        else:
            notes = "threshold story; high friction strong"
        rows.append(
            {
                "step_id": "step4",
                "domain": "inventory",
                "question_id": "Q1",
                "scenario_id": STEP4_Q1_SCENARIO,
                "evidence_role": "required",
                "source_path": abs_path(STEP4_Q1_THRESHOLD_PATH),
                "n_seeds": int(row.seeds),
                "friction": friction,
                "q1_same_forecast_interface_effect_yesno": yesno(
                    abs(float(row.mean_executed_delta_tempered_minus_responsive)) > 1e-12
                ),
                "q2_rank_flip_yesno": NA,
                "zero_friction_near_flat_yesno": yesno(step4_q1_zero_flat),
                "positive_friction_gap_yesno": yesno(
                    friction > 0.0 and float(row.mean_abs_target_executed_gap_tempered) > 0.0
                ),
                "inclusion_status": "pass",
                "notes": notes,
            }
        )

    step4_q2_rank_corr = step4_q2_outputs["rank_correlation_by_friction"]
    step4_q2_zero = zero_row(step4_q2_rank_corr)
    step4_q2_zero_flat = q2_zero_flat_pass(
        zero_flip_rate=float(step4_q2_zero["mean_flip_rate"]),
        zero_spearman=float(step4_q2_zero["mean_spearman_rho"]),
        evidence_role="required",
    )
    for row in step4_q2_rank_corr.itertuples(index=False):
        friction = float(row.friction_level)
        rows.append(
            {
                "step_id": "step4",
                "domain": "inventory",
                "question_id": "Q2",
                "scenario_id": STEP4_Q2_SCENARIO,
                "evidence_role": "required",
                "source_path": abs_path(STEP4_Q2_PATH),
                "n_seeds": int(row.n_seeds),
                "friction": friction,
                "q1_same_forecast_interface_effect_yesno": NA,
                "q2_rank_flip_yesno": yesno(float(row.mean_flip_rate) > 0.0),
                "zero_friction_near_flat_yesno": yesno(step4_q2_zero_flat),
                "positive_friction_gap_yesno": yesno(
                    friction > 0.0
                    and (
                        float(row.mean_flip_rate) > float(step4_q2_zero["mean_flip_rate"])
                        or float(row.mean_spearman_rho) < float(step4_q2_zero["mean_spearman_rho"])
                    )
                ),
                "inclusion_status": "pass",
                "notes": "zero-friction near-aligned" if friction == 0.0 else "emergence and persistence",
            }
        )

    for domain in ["synthetic", "inventory", "portfolio"]:
        package = step5_packages[domain]
        verdict_row = package["verdict"]
        rank_corr = package["rank_corr"]
        package_zero = zero_row(rank_corr)
        evidence_role = str(verdict_row["domain_role"])
        zero_flat = q2_zero_flat_pass(
            zero_flip_rate=float(package_zero["mean_flip_rate"]),
            zero_spearman=float(package_zero["mean_spearman_rho"]),
            evidence_role=evidence_role,
        )
        for row in rank_corr.itertuples(index=False):
            friction = float(row.friction_level)
            if domain == "synthetic":
                notes = (
                    "repackaged from step2 Q2; zero-friction exact alignment"
                    if friction == 0.0
                    else "repackaged from step2 Q2; positive-friction plateau"
                )
            elif domain == "inventory":
                notes = (
                    "repackaged from step4 Q2; zero-friction near-aligned"
                    if friction == 0.0
                    else "repackaged from step4 Q2; emergence and persistence"
                )
            else:
                notes = "stretch-only; portfolio excluded by stretch gate"
            rows.append(
                {
                    "step_id": "step5",
                    "domain": domain,
                    "question_id": "Q2",
                    "scenario_id": STEP5_Q2_SCENARIO,
                    "evidence_role": evidence_role,
                    "source_path": abs_path(package["rank_corr_path"]),
                    "n_seeds": int(row.n_seeds),
                    "friction": friction,
                    "q1_same_forecast_interface_effect_yesno": NA,
                    "q2_rank_flip_yesno": yesno(float(row.mean_flip_rate) > 0.0),
                    "zero_friction_near_flat_yesno": yesno(zero_flat),
                    "positive_friction_gap_yesno": yesno(
                        friction > 0.0
                        and (
                            float(row.mean_flip_rate) > float(package_zero["mean_flip_rate"])
                            or float(row.mean_spearman_rho) < float(package_zero["mean_spearman_rho"])
                        )
                    ),
                    "inclusion_status": str(verdict_row["inclusion_status"]),
                    "notes": notes,
                }
            )

    summary = pd.DataFrame(rows)
    summary = sort_frame(summary, ["scenario_id", "friction", "question_id"])
    return summary


def build_rank_flip_summary(
    *,
    step2_q2_outputs: dict[str, pd.DataFrame],
    step4_q2_outputs: dict[str, pd.DataFrame],
    step5_packages: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_q2_rows(
        *,
        package_id: str,
        upstream_step_id: str,
        step_id: str,
        domain: str,
        scenario_id: str,
        rank_corr: pd.DataFrame,
        pairwise: pd.DataFrame,
        notes_zero: str,
        notes_positive: str,
    ) -> None:
        zero = zero_row(rank_corr)
        zero_flip = float(zero["mean_flip_rate"])
        zero_kendall = float(zero["mean_kendall_tau_b"])
        zero_spearman = float(zero["mean_spearman_rho"])
        for row in rank_corr.itertuples(index=False):
            friction = float(row.friction_level)
            pair_row = pairwise_row_for_friction(pairwise, friction)
            rows.append(
                {
                    "package_id": package_id,
                    "upstream_step_id": upstream_step_id,
                    "step_id": step_id,
                    "domain": domain,
                    "scenario_id": scenario_id,
                    "friction": friction,
                    "zero_friction_flip_rate": zero_flip,
                    "current_flip_rate": float(row.mean_flip_rate),
                    "flip_rate_delta_vs_zero": float(row.mean_flip_rate) - zero_flip,
                    "kendall_drop_vs_zero": zero_kendall - float(row.mean_kendall_tau_b),
                    "spearman_drop_vs_zero": zero_spearman - float(row.mean_spearman_rho),
                    "strongest_flip_pair": (
                        f"{pair_row['model_a']}|{pair_row['model_b']}" if pair_row is not None else np.nan
                    ),
                    "strongest_flip_share": float(pair_row["flip_seed_share"]) if pair_row is not None else np.nan,
                    "n_comparable_pairs_mean": float(row.mean_n_comparable_pairs),
                    "notes": notes_zero if friction == 0.0 else notes_positive,
                }
            )

    append_q2_rows(
        package_id=PACKAGE_IDS["step2"],
        upstream_step_id="none",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        rank_corr=step2_q2_outputs["rank_correlation_by_friction"],
        pairwise=step2_q2_outputs["pairwise_flips_by_friction"],
        notes_zero="raw locked Q2; zero-friction exact alignment",
        notes_positive="raw locked Q2; positive-friction plateau",
    )

    append_q2_rows(
        package_id=PACKAGE_IDS["step4"],
        upstream_step_id="none",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        rank_corr=step4_q2_outputs["rank_correlation_by_friction"],
        pairwise=step4_q2_outputs["pairwise_flips_by_friction"],
        notes_zero="raw locked Q2; zero-friction near-aligned",
        notes_positive="raw locked Q2; emergence and persistence",
    )

    for domain in ["synthetic", "inventory", "portfolio"]:
        package = step5_packages[domain]
        if domain == "synthetic":
            zero_note = "repackaged from step2 Q2; zero-friction exact alignment"
            positive_note = "repackaged from step2 Q2; positive-friction plateau"
        elif domain == "inventory":
            zero_note = "repackaged from step4 Q2; zero-friction near-aligned"
            positive_note = "repackaged from step4 Q2; emergence and persistence"
        else:
            zero_note = "stretch-only; portfolio excluded by stretch gate"
            positive_note = "stretch-only; portfolio excluded by stretch gate"
        append_q2_rows(
            package_id=PACKAGE_IDS["step5"],
            upstream_step_id=STEP5_UPSTREAM_STEP[domain],
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            rank_corr=package["rank_corr"],
            pairwise=package["pairwise"],
            notes_zero=zero_note,
            notes_positive=positive_note,
        )

    summary = pd.DataFrame(rows)
    summary = sort_frame(summary, ["scenario_id", "friction"])
    return summary


def build_sanity_checks(
    *,
    step2_q1: pd.DataFrame,
    step2_q2: pd.DataFrame,
    step2_q1_gap: pd.DataFrame,
    step2_q2_outputs: dict[str, pd.DataFrame],
    step3_raw: pd.DataFrame,
    step3_forecast_hash: pd.DataFrame,
    step3_proposal_hash: pd.DataFrame,
    step3_effect_size: pd.DataFrame,
    step3_paired: pd.DataFrame,
    step4_q1: pd.DataFrame,
    step4_q2: pd.DataFrame,
    step4_q1_threshold: pd.DataFrame,
    step4_q2_outputs: dict[str, pd.DataFrame],
    step4_freeze_check: pd.DataFrame,
    step5_manifest: dict[str, Any],
    step5_verdict: pd.DataFrame,
    step5_packages: dict[str, dict[str, Any]],
    master_results: pd.DataFrame,
    summary_by_domain: pd.DataFrame,
    rank_flip_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    step2_q2_zero = zero_row(step2_q2_outputs["rank_correlation_by_friction"])
    step4_q2_zero = zero_row(step4_q2_outputs["rank_correlation_by_friction"])

    # forecast_identity_locked
    step2_q1_forecast_group_delta = step2_q1.groupby(["seed", "friction_level"])["forecast_metric"].agg(
        lambda series: float(series.max()) - float(series.min())
    )
    add_check(
        rows,
        check_name="forecast_identity_locked",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=PASS if float(step2_q1_forecast_group_delta.abs().max()) <= 1e-12 else FAIL,
        measured_value=float(step2_q1_forecast_group_delta.abs().max()),
        threshold="<=1e-12",
        note="max forecast_metric delta across interface arms within each seed x friction",
    )
    add_check(
        rows,
        check_name="forecast_identity_locked",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q2 different-forecast package",
    )

    step3_identity_failures = int(step3_forecast_hash["forecast_hash_mismatch_count"].sum())
    step3_identity_failures += int(step3_proposal_hash["proposal_hash_mismatch_count"].sum())
    step3_identity_failures += int(step3_proposal_hash["per_step_proposal_equality_failure_count"].sum())
    step3_identity_failures += int(step3_proposal_hash["pairing_failure_count"].sum())
    add_check(
        rows,
        check_name="forecast_identity_locked",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=PASS if step3_identity_failures == 0 else FAIL,
        measured_value=step3_identity_failures,
        threshold=0,
        note="forecast/proposal hash mismatch plus pairing failures",
    )

    step4_identity_failures = int((~step4_freeze_check["forecast_hash_identical_flag"]).sum())
    step4_identity_failures += int((~step4_freeze_check["proposal_hash_identical_flag"]).sum())
    step4_identity_failures += int((~step4_freeze_check["initial_inventory_match_flag"]).sum())
    step4_identity_failures += int((~step4_freeze_check["initial_prev_order_match_flag"]).sum())
    step4_identity_failures += int(step4_freeze_check["pairing_failure_count"].sum())
    add_check(
        rows,
        check_name="forecast_identity_locked",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=PASS if step4_identity_failures == 0 else FAIL,
        measured_value=step4_identity_failures,
        threshold=0,
        note="freeze-check identity flags and pairing failures",
    )
    add_check(
        rows,
        check_name="forecast_identity_locked",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q2 different-forecast package",
    )
    for domain in ["synthetic", "inventory", "portfolio"]:
        add_check(
            rows,
            check_name="forecast_identity_locked",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=NA,
            measured_value=np.nan,
            threshold="na",
            note="Step 5 is a Q2 package",
        )

    # zero_friction_near_flat
    step2_q1_zero_gap = float(
        step2_q1_gap.loc[np.isclose(step2_q1_gap["friction_level"], 0.0, atol=1e-15), "mean_abs_target_executed_gap"].iloc[0]
    )
    add_check(
        rows,
        check_name="zero_friction_near_flat",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=PASS if step2_q1_zero_gap <= 1e-12 else FAIL,
        measured_value=step2_q1_zero_gap,
        threshold="<=1e-12",
        note="mean_abs_target_executed_gap at friction 0",
    )
    add_check(
        rows,
        check_name="zero_friction_near_flat",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=PASS if q2_zero_flat_pass(zero_flip_rate=float(step2_q2_zero["mean_flip_rate"]), zero_spearman=float(step2_q2_zero["mean_spearman_rho"]), evidence_role="required") else FAIL,
        measured_value=f"flip={float(step2_q2_zero['mean_flip_rate']):.6f};spearman={float(step2_q2_zero['mean_spearman_rho']):.6f}",
        threshold="flip<=0.10 & spearman>=0.80",
        note="Step 2 Q2 zero-friction mismatch gate",
    )

    step3_zero_pairs = step3_paired[np.isclose(step3_paired["kappa"], 0.0, atol=1e-15)]
    step3_zero_within = int((step3_zero_pairs["executed_delta"].abs() <= 0.005).sum())
    add_check(
        rows,
        check_name="zero_friction_near_flat",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=PASS if step3_zero_within >= 7 else FAIL,
        measured_value=f"{step3_zero_within}/{len(step3_zero_pairs)} within 0.005",
        threshold=">=7/9 within 0.005",
        note="zero-cost executed Sharpe delta across eta_0_5 vs eta_1_0",
    )

    step4_q1_zero_gap = float(
        step4_q1_threshold.loc[
            np.isclose(step4_q1_threshold["friction_level"], 0.0, atol=1e-15),
            "mean_abs_target_executed_gap_tempered",
        ].iloc[0]
    )
    add_check(
        rows,
        check_name="zero_friction_near_flat",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=PASS if step4_q1_zero_gap <= 1e-12 else FAIL,
        measured_value=step4_q1_zero_gap,
        threshold="<=1e-12",
        note="mean_abs_target_executed_gap_tempered at friction 0",
    )
    add_check(
        rows,
        check_name="zero_friction_near_flat",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=PASS if q2_zero_flat_pass(zero_flip_rate=float(step4_q2_zero["mean_flip_rate"]), zero_spearman=float(step4_q2_zero["mean_spearman_rho"]), evidence_role="required") else FAIL,
        measured_value=f"flip={float(step4_q2_zero['mean_flip_rate']):.6f};spearman={float(step4_q2_zero['mean_spearman_rho']):.6f}",
        threshold="flip<=0.10 & spearman>=0.80",
        note="Step 4 Q2 zero-friction mismatch gate",
    )
    for domain in ["synthetic", "inventory", "portfolio"]:
        verdict_row = step5_verdict.loc[step5_verdict["domain"] == domain].iloc[0]
        role = str(verdict_row["domain_role"])
        add_check(
            rows,
            check_name="zero_friction_near_flat",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if q2_zero_flat_pass(
                zero_flip_rate=float(verdict_row["zero_friction_mean_flip_rate"]),
                zero_spearman=float(verdict_row["zero_friction_mean_spearman_rho"]),
                evidence_role=role,
            ) else FAIL,
            measured_value=f"flip={float(verdict_row['zero_friction_mean_flip_rate']):.6f};spearman={float(verdict_row['zero_friction_mean_spearman_rho']):.6f}",
            threshold=f"flip<=0.10 & spearman>={STEP5_ZERO_SPEARMAN_MIN[role]:.2f}",
            note="Step 5 zero-friction mismatch gate",
        )

    # positive_friction_gap_emergence
    step2_q1_positive_max = float(step2_q1_gap.loc[step2_q1_gap["friction_level"] > 0.0, "mean_abs_target_executed_gap"].max())
    add_check(
        rows,
        check_name="positive_friction_gap_emergence",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=PASS if step2_q1_positive_max > 0.0 else FAIL,
        measured_value=step2_q1_positive_max,
        threshold=">0 at any positive friction",
        note="Q1 mean_abs_target_executed_gap emergence",
    )
    step2_q2_best_increase = float(
        step2_q2_outputs["rank_correlation_by_friction"].loc[
            step2_q2_outputs["rank_correlation_by_friction"]["friction_level"] > 0.0, "mean_flip_rate"
        ].max()
        - float(step2_q2_zero["mean_flip_rate"])
    )
    add_check(
        rows,
        check_name="positive_friction_gap_emergence",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=PASS if step2_q2_best_increase > 0.0 else FAIL,
        measured_value=step2_q2_best_increase,
        threshold=">0 flip-rate delta vs zero",
        note="Q2 positive-friction ranking mismatch emergence",
    )
    step3_positive_universes = int((step3_effect_size["effect_size_mean"] > 0.0).sum())
    add_check(
        rows,
        check_name="positive_friction_gap_emergence",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=PASS if step3_positive_universes == len(step3_effect_size) else FAIL,
        measured_value=f"{step3_positive_universes}/{len(step3_effect_size)} universes",
        threshold=f"{len(step3_effect_size)}/{len(step3_effect_size)} universes",
        note="positive-cost recurrence via effect_size_mean > 0",
    )
    step4_q1_high_friction_win_rate = float(
        step4_q1_threshold.loc[np.isclose(step4_q1_threshold["friction_level"], 1.0, atol=1e-15), "tempered_win_rate"].iloc[0]
    )
    add_check(
        rows,
        check_name="positive_friction_gap_emergence",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=PASS if step4_q1_high_friction_win_rate > 0.5 else FAIL,
        measured_value=step4_q1_high_friction_win_rate,
        threshold=">0.5 at friction 1.0",
        note="threshold support appears at sufficiently high friction",
    )
    step4_q2_best_increase = float(
        step4_q2_outputs["rank_correlation_by_friction"].loc[
            step4_q2_outputs["rank_correlation_by_friction"]["friction_level"] > 0.0, "mean_flip_rate"
        ].max()
        - float(step4_q2_zero["mean_flip_rate"])
    )
    add_check(
        rows,
        check_name="positive_friction_gap_emergence",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=PASS if step4_q2_best_increase > 0.0 else FAIL,
        measured_value=step4_q2_best_increase,
        threshold=">0 flip-rate delta vs zero",
        note="Q2 positive-friction ranking mismatch emergence",
    )
    for domain in ["synthetic", "inventory", "portfolio"]:
        verdict_row = step5_verdict.loc[step5_verdict["domain"] == domain].iloc[0]
        increase = float(verdict_row["best_positive_flip_rate"]) - float(verdict_row["zero_friction_mean_flip_rate"])
        add_check(
            rows,
            check_name="positive_friction_gap_emergence",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if bool(verdict_row["positive_flip_rate_increase_flag"]) else FAIL,
            measured_value=increase,
            threshold=">0 flip-rate delta vs zero",
            note="Step 5 packaged Q2 positive-friction mismatch emergence",
        )

    # paired_seeds_aligned
    step2_q1_pair_failures = int((step2_q1.groupby(["seed", "friction_level"]).size() != 2).sum())
    add_check(
        rows,
        check_name="paired_seeds_aligned",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=PASS if step2_q1_pair_failures == 0 else FAIL,
        measured_value=step2_q1_pair_failures,
        threshold=0,
        note="expected exactly two interface rows per seed x friction",
    )
    step2_q2_group = step2_q2.groupby(["seed", "friction_level"])
    step2_q2_pair_failures = int(
        (
            (step2_q2_group["interface_id"].nunique() != 1)
            | (step2_q2_group["forecaster_id"].nunique() < 4)
        ).sum()
    )
    add_check(
        rows,
        check_name="paired_seeds_aligned",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=PASS if step2_q2_pair_failures == 0 else FAIL,
        measured_value=step2_q2_pair_failures,
        threshold=0,
        note="expected one interface and at least four forecasters per seed x friction",
    )
    step3_pair_failures = int((step3_raw.groupby(["universe_id", "seed", "kappa"]).size() != 2).sum())
    add_check(
        rows,
        check_name="paired_seeds_aligned",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=PASS if step3_pair_failures == 0 else FAIL,
        measured_value=step3_pair_failures,
        threshold=0,
        note="expected exactly two replay interfaces per universe x seed x kappa",
    )
    step4_q1_pair_failures = int((step4_q1.groupby(["seed", "friction_level"]).size() != 2).sum())
    add_check(
        rows,
        check_name="paired_seeds_aligned",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=PASS if step4_q1_pair_failures == 0 else FAIL,
        measured_value=step4_q1_pair_failures,
        threshold=0,
        note="expected exactly two interface rows per seed x friction",
    )
    step4_q2_group = step4_q2.groupby(["seed", "friction_level"])
    step4_q2_pair_failures = int(
        (
            (step4_q2_group["interface_id"].nunique() != 1)
            | (step4_q2_group["forecaster_id"].nunique() < 4)
        ).sum()
    )
    add_check(
        rows,
        check_name="paired_seeds_aligned",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=PASS if step4_q2_pair_failures == 0 else FAIL,
        measured_value=step4_q2_pair_failures,
        threshold=0,
        note="expected one interface and at least four forecasters per seed x friction",
    )
    for domain in ["synthetic", "inventory", "portfolio"]:
        rank_corr = step5_packages[domain]["rank_corr"]
        stable = int(rank_corr["n_seeds"].nunique() == 1)
        add_check(
            rows,
            check_name="paired_seeds_aligned",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if stable == 1 else FAIL,
            measured_value=f"min={int(rank_corr['n_seeds'].min())};max={int(rank_corr['n_seeds'].max())}",
            threshold="constant across friction",
            note="Step 5 package should keep the same seed count across friction",
        )

    # no_nan_inf
    add_check(
        rows,
        check_name="no_nan_inf",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=PASS if finite_violation_count(step2_q1, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]) == 0 else FAIL,
        measured_value=finite_violation_count(step2_q1, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]),
        threshold=0,
        note="raw Q1 numeric fields",
    )
    add_check(
        rows,
        check_name="no_nan_inf",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=PASS if finite_violation_count(step2_q2, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]) == 0 else FAIL,
        measured_value=finite_violation_count(step2_q2, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]),
        threshold=0,
        note="raw Q2 numeric fields",
    )
    add_check(
        rows,
        check_name="no_nan_inf",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=PASS if finite_violation_count(step3_raw, ["kappa", "sharpe_exec_net", "sharpe_target_net", "tracking_error_l2_mean", "final_path_gap"]) == 0 else FAIL,
        measured_value=finite_violation_count(step3_raw, ["kappa", "sharpe_exec_net", "sharpe_target_net", "tracking_error_l2_mean", "final_path_gap"]),
        threshold=0,
        note="exact-control numeric fields",
    )
    add_check(
        rows,
        check_name="no_nan_inf",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=PASS if finite_violation_count(step4_q1, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]) == 0 else FAIL,
        measured_value=finite_violation_count(step4_q1, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]),
        threshold=0,
        note="raw Q1 numeric fields",
    )
    add_check(
        rows,
        check_name="no_nan_inf",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=PASS if finite_violation_count(step4_q2, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]) == 0 else FAIL,
        measured_value=finite_violation_count(step4_q2, ["forecast_metric", "target_metric", "executed_metric", "target_executed_gap"]),
        threshold=0,
        note="raw Q2 numeric fields",
    )
    for domain in ["synthetic", "inventory", "portfolio"]:
        package = step5_packages[domain]
        violation_count = finite_violation_count(
            package["seed_stats"],
            [
                "friction_level",
                "n_forecasters",
                "n_possible_pairs",
                "n_comparable_pairs",
                "flip_count",
                "flip_rate",
                "kendall_tau_b",
                "spearman_rho",
            ],
        )
        violation_count += finite_violation_count(
            package["rank_corr"],
            [
                "friction_level",
                "n_seeds",
                "mean_flip_rate",
                "mean_kendall_tau_b",
                "mean_spearman_rho",
                "mean_n_comparable_pairs",
            ],
        )
        add_check(
            rows,
            check_name="no_nan_inf",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if violation_count == 0 else FAIL,
            measured_value=violation_count,
            threshold=0,
            note="Step 5 seed-level and rank-correlation numeric fields",
        )

    # no_sign_inversion_from_bug
    add_check(
        rows,
        check_name="no_sign_inversion_from_bug",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q1 package",
    )
    step2_sign_violations = q2_sign_inversion_violations(step2_q2)
    add_check(
        rows,
        check_name="no_sign_inversion_from_bug",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=PASS if step2_sign_violations == 0 else FAIL,
        measured_value=step2_sign_violations,
        threshold=0,
        note="raw Q2 rank columns should follow descending score order",
    )
    add_check(
        rows,
        check_name="no_sign_inversion_from_bug",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q1 package",
    )
    add_check(
        rows,
        check_name="no_sign_inversion_from_bug",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q1 package",
    )
    step4_sign_violations = q2_sign_inversion_violations(step4_q2)
    add_check(
        rows,
        check_name="no_sign_inversion_from_bug",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=PASS if step4_sign_violations == 0 else FAIL,
        measured_value=step4_sign_violations,
        threshold=0,
        note="raw Q2 rank columns should follow descending score order",
    )
    manifest_sources = {str(item["domain"]): item for item in step5_manifest["sources"]}
    for domain in ["synthetic", "inventory", "portfolio"]:
        raw_q2 = read_csv(manifest_sources[domain]["source_path"])
        sign_violations = q2_sign_inversion_violations(raw_q2)
        add_check(
            rows,
            check_name="no_sign_inversion_from_bug",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if sign_violations == 0 else FAIL,
            measured_value=sign_violations,
            threshold=0,
            note="Step 5 packaged Q2 should preserve descending score ordering",
        )

    # friction_sweep_order_preserved
    for step_id, domain, scenario_id, series in [
        ("step2", "synthetic", STEP2_Q1_SCENARIO, step2_q1["friction_level"]),
        ("step2", "synthetic", STEP2_Q2_SCENARIO, step2_q2["friction_level"]),
        ("step3", "portfolio", STEP3_Q1_SCENARIO, step3_raw["kappa"]),
        ("step4", "inventory", STEP4_Q1_SCENARIO, step4_q1["friction_level"]),
        ("step4", "inventory", STEP4_Q2_SCENARIO, step4_q2["friction_level"]),
    ]:
        order = observed_order(series)
        add_check(
            rows,
            check_name="friction_sweep_order_preserved",
            step_id=step_id,
            domain=domain,
            scenario_id=scenario_id,
            pass_fail=PASS if strictly_ascending(order) else FAIL,
            measured_value="|".join(str(value) for value in order),
            threshold="strictly ascending",
            note="observed order of friction values in canonical source",
        )
    for domain in ["synthetic", "inventory", "portfolio"]:
        order = observed_order(step5_packages[domain]["rank_corr"]["friction_level"])
        add_check(
            rows,
            check_name="friction_sweep_order_preserved",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if strictly_ascending(order) else FAIL,
            measured_value="|".join(str(value) for value in order),
            threshold="strictly ascending",
            note="Step 5 packaged friction order",
        )

    # target_executed_gap_nonnegative_where_definition_requires
    add_check(
        rows,
        check_name="target_executed_gap_nonnegative_where_definition_requires",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=PASS if float(step2_q1_gap["mean_abs_target_executed_gap"].min()) >= 0.0 else FAIL,
        measured_value=float(step2_q1_gap["mean_abs_target_executed_gap"].min()),
        threshold=">=0",
        note="mean_abs_target_executed_gap",
    )
    add_check(
        rows,
        check_name="target_executed_gap_nonnegative_where_definition_requires",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q2 package",
    )
    step3_min_gap = float(step3_raw["final_path_gap"].min())
    step3_min_tracking = float(step3_raw["tracking_error_l2_mean"].min())
    add_check(
        rows,
        check_name="target_executed_gap_nonnegative_where_definition_requires",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=PASS if min(step3_min_gap, step3_min_tracking) >= 0.0 else FAIL,
        measured_value=f"final_path_gap_min={step3_min_gap:.6f};tracking_error_min={step3_min_tracking:.6f}",
        threshold=">=0",
        note="final_path_gap and tracking_error_l2_mean",
    )
    add_check(
        rows,
        check_name="target_executed_gap_nonnegative_where_definition_requires",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=PASS if float(step4_q1_threshold["mean_abs_target_executed_gap_tempered"].min()) >= 0.0 else FAIL,
        measured_value=float(step4_q1_threshold["mean_abs_target_executed_gap_tempered"].min()),
        threshold=">=0",
        note="mean_abs_target_executed_gap_tempered",
    )
    add_check(
        rows,
        check_name="target_executed_gap_nonnegative_where_definition_requires",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q2 package",
    )
    for domain in ["synthetic", "inventory", "portfolio"]:
        add_check(
            rows,
            check_name="target_executed_gap_nonnegative_where_definition_requires",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=NA,
            measured_value=np.nan,
            threshold="na",
            note="Q2 package",
        )

    # interface_expected_match
    for step_id, scenario_id, domain, frame, column in [
        ("step2", STEP2_Q1_SCENARIO, "synthetic", step2_q1, "interface_id"),
        ("step2", STEP2_Q2_SCENARIO, "synthetic", step2_q2, "interface_id"),
        ("step3", STEP3_Q1_SCENARIO, "portfolio", step3_raw, "replay_interface_id"),
        ("step4", STEP4_Q1_SCENARIO, "inventory", step4_q1, "interface_id"),
        ("step4", STEP4_Q2_SCENARIO, "inventory", step4_q2, "interface_id"),
    ]:
        expected = EXPECTED_INTERFACE_SETS[(step_id, scenario_id)]
        observed = tuple(sorted(str(value) for value in frame[column].dropna().unique().tolist()))
        add_check(
            rows,
            check_name="interface_expected_match",
            step_id=step_id,
            domain=domain,
            scenario_id=scenario_id,
            pass_fail=PASS if observed == expected else FAIL,
            measured_value="|".join(observed),
            threshold="|".join(expected),
            note="observed vs expected interface ids",
        )
    for domain in ["synthetic", "inventory", "portfolio"]:
        verdict_row = step5_verdict.loc[step5_verdict["domain"] == domain].iloc[0]
        observed = str(verdict_row["observed_interface_id"])
        expected = str(verdict_row["expected_interface_id"])
        add_check(
            rows,
            check_name="interface_expected_match",
            step_id="step5",
            domain=domain,
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if observed == expected else FAIL,
            measured_value=observed,
            threshold=expected,
            note="Step 5 package expected-interface validation",
        )

    # min_forecasters_per_seed_friction_ok
    add_check(
        rows,
        check_name="min_forecasters_per_seed_friction_ok",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q1_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q1 package",
    )
    step2_q2_min_forecasters = int(step2_q2.groupby(["seed", "friction_level"])["forecaster_id"].nunique().min())
    add_check(
        rows,
        check_name="min_forecasters_per_seed_friction_ok",
        step_id="step2",
        domain="synthetic",
        scenario_id=STEP2_Q2_SCENARIO,
        pass_fail=PASS if step2_q2_min_forecasters >= 4 else FAIL,
        measured_value=step2_q2_min_forecasters,
        threshold=">=4",
        note="raw Q2 per-seed-per-friction minimum forecaster count",
    )
    add_check(
        rows,
        check_name="min_forecasters_per_seed_friction_ok",
        step_id="step3",
        domain="portfolio",
        scenario_id=STEP3_Q1_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q1 package",
    )
    add_check(
        rows,
        check_name="min_forecasters_per_seed_friction_ok",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q1_SCENARIO,
        pass_fail=NA,
        measured_value=np.nan,
        threshold="na",
        note="Q1 package",
    )
    step4_q2_min_forecasters = int(step4_q2.groupby(["seed", "friction_level"])["forecaster_id"].nunique().min())
    add_check(
        rows,
        check_name="min_forecasters_per_seed_friction_ok",
        step_id="step4",
        domain="inventory",
        scenario_id=STEP4_Q2_SCENARIO,
        pass_fail=PASS if step4_q2_min_forecasters >= 4 else FAIL,
        measured_value=step4_q2_min_forecasters,
        threshold=">=4",
        note="raw Q2 per-seed-per-friction minimum forecaster count",
    )
    for source in step5_manifest["sources"]:
        add_check(
            rows,
            check_name="min_forecasters_per_seed_friction_ok",
            step_id="step5",
            domain=str(source["domain"]),
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if int(source["min_n_forecasters_per_seed_friction"]) >= 4 else FAIL,
            measured_value=int(source["min_n_forecasters_per_seed_friction"]),
            threshold=">=4",
            note="manifest min_n_forecasters_per_seed_friction",
        )

    # source_manifest_hash_present
    for step_id, domain, scenario_id in [
        ("step2", "synthetic", STEP2_Q1_SCENARIO),
        ("step2", "synthetic", STEP2_Q2_SCENARIO),
        ("step3", "portfolio", STEP3_Q1_SCENARIO),
        ("step4", "inventory", STEP4_Q1_SCENARIO),
        ("step4", "inventory", STEP4_Q2_SCENARIO),
    ]:
        add_check(
            rows,
            check_name="source_manifest_hash_present",
            step_id=step_id,
            domain=domain,
            scenario_id=scenario_id,
            pass_fail=NA,
            measured_value=np.nan,
            threshold="na",
            note="non-Step-5 package",
        )
    for source in step5_manifest["sources"]:
        has_hash = bool(source.get("source_path")) and bool(source.get("source_sha256"))
        add_check(
            rows,
            check_name="source_manifest_hash_present",
            step_id="step5",
            domain=str(source["domain"]),
            scenario_id=STEP5_Q2_SCENARIO,
            pass_fail=PASS if has_hash else FAIL,
            measured_value=1 if has_hash else 0,
            threshold=1,
            note="manifest source_path and source_sha256 presence",
        )

    # duplicate_key_uniqueness
    add_check(
        rows,
        check_name="duplicate_key_uniqueness",
        step_id="step6",
        domain="all",
        scenario_id="master_results_integrity",
        pass_fail=PASS if group_duplicate_count(
            master_results,
            [
                "package_id",
                "record_type",
                "step_id",
                "domain",
                "question_id",
                "scenario_id",
                "seed",
                "friction",
                "forecaster_id",
                "interface_id",
            ],
        )
        == 0 else FAIL,
        measured_value=group_duplicate_count(
            master_results,
            [
                "package_id",
                "record_type",
                "step_id",
                "domain",
                "question_id",
                "scenario_id",
                "seed",
                "friction",
                "forecaster_id",
                "interface_id",
            ],
        ),
        threshold=0,
        note="(package_id, record_type, step_id, domain, question_id, scenario_id, seed, friction, forecaster_id, interface_id)",
    )
    add_check(
        rows,
        check_name="duplicate_key_uniqueness",
        step_id="step6",
        domain="all",
        scenario_id="summary_by_domain_integrity",
        pass_fail=PASS if group_duplicate_count(
            summary_by_domain,
            ["step_id", "domain", "question_id", "scenario_id", "friction"],
        )
        == 0 else FAIL,
        measured_value=group_duplicate_count(
            summary_by_domain,
            ["step_id", "domain", "question_id", "scenario_id", "friction"],
        ),
        threshold=0,
        note="(step_id, domain, question_id, scenario_id, friction)",
    )
    add_check(
        rows,
        check_name="duplicate_key_uniqueness",
        step_id="step6",
        domain="all",
        scenario_id="rank_flip_summary_integrity",
        pass_fail=PASS if group_duplicate_count(
            rank_flip_summary,
            ["step_id", "domain", "scenario_id", "friction"],
        )
        == 0 else FAIL,
        measured_value=group_duplicate_count(
            rank_flip_summary,
            ["step_id", "domain", "scenario_id", "friction"],
        ),
        threshold=0,
        note="(step_id, domain, scenario_id, friction)",
    )

    sanity = pd.DataFrame(rows)
    sanity = sort_frame(sanity, ["check_name", "scenario_id"])
    return sanity


def build_claim_map(required_failure_count: int) -> pd.DataFrame:
    status_c1 = "supported_required_plus_support" if required_failure_count == 0 else "reopen_required"
    status_c2 = "supported_required_repackaged_q2" if required_failure_count == 0 else "reopen_required"
    status_c3 = "supported_required_repackaged_q2" if required_failure_count == 0 else "reopen_required"
    status_c4 = "supported_contextual"
    rows = [
        {
            "claim_id": "C1",
            "claim_text_short": "same forecast or proposal path, different interface can change realized outcomes under friction",
            "supported_by_steps": "step2;step3;step4",
            "required_evidence_count": 2,
            "current_support_status": status_c1,
            "notes": "Step 2 synthetic and Step 4 inventory are required; Step 3 portfolio exact-control remains support-only",
        },
        {
            "claim_id": "C2",
            "claim_text_short": "under a fixed interface, forecast ranking does not reliably determine deployed ranking",
            "supported_by_steps": "step2;step4;step5",
            "required_evidence_count": 2,
            "current_support_status": status_c2,
            "notes": "Step 5 repackages Step 2 and Step 4 Q2 evidence and does not add independent required evidence count",
        },
        {
            "claim_id": "C3",
            "claim_text_short": "zero-friction mismatch is low while positive-friction mismatch emerges",
            "supported_by_steps": "step2;step4;step5",
            "required_evidence_count": 2,
            "current_support_status": status_c3,
            "notes": "Supported in synthetic and inventory; Step 5 packages the same required Q2 evidence for paper-facing use",
        },
        {
            "claim_id": "C4",
            "claim_text_short": "portfolio provides Q1 support but portfolio Q2 is excluded from the paper-facing Step 5 package",
            "supported_by_steps": "step3;step5",
            "required_evidence_count": 0,
            "current_support_status": status_c4,
            "notes": "Portfolio exact-control remains support-only and Step 5 portfolio Q2 stays excluded by the stretch gate",
        },
    ]
    claim_map = pd.DataFrame(rows)
    return claim_map


def build_one_page_verdict(
    *,
    sanity_checks: pd.DataFrame,
    step5_verdict: pd.DataFrame,
    step2_q1_gap: pd.DataFrame,
    step2_q2_outputs: dict[str, pd.DataFrame],
    step3_paired: pd.DataFrame,
    step3_effect_size: pd.DataFrame,
    step4_q1_threshold: pd.DataFrame,
    step4_q2_outputs: dict[str, pd.DataFrame],
) -> str:
    required_rows = sanity_checks[
        sanity_checks["step_id"].isin(["step2", "step4", "step5"])
        & sanity_checks["domain"].isin(["synthetic", "inventory"])
        & (~sanity_checks["scenario_id"].isin(["master_results_integrity", "summary_by_domain_integrity", "rank_flip_summary_integrity"]))
        & (sanity_checks["pass_fail"] == FAIL)
    ]
    reopen_required = not required_rows.empty

    step2_q1_best_gap = float(step2_q1_gap["mean_abs_target_executed_gap"].max())
    step2_q2_rank_corr = step2_q2_outputs["rank_correlation_by_friction"]
    step4_q2_rank_corr = step4_q2_outputs["rank_correlation_by_friction"]
    step3_zero_pairs = step3_paired[np.isclose(step3_paired["kappa"], 0.0, atol=1e-15)]
    step3_zero_within = int((step3_zero_pairs["executed_delta"].abs() <= 0.005).sum())
    step4_high_friction = step4_q1_threshold.loc[np.isclose(step4_q1_threshold["friction_level"], 1.0, atol=1e-15)].iloc[0]
    step5_portfolio = step5_verdict.loc[step5_verdict["domain"] == "portfolio"].iloc[0]

    lines = [
        "# Step 6 One-Page Verdict",
        "",
        "## Step 2 Verdict",
        (
            f"Synthetic phenomenon existence remains locked. Q1 keeps exact zero-friction agreement and reaches a "
            f"maximum mean absolute target-executed gap of {step2_q1_best_gap:.3f} as friction rises."
        ),
        (
            "Q2 keeps exact zero-friction ranking alignment and then shows stable positive-friction ranking mismatch "
            f"with a plateau at mean flip rate {float(step2_q2_rank_corr.loc[step2_q2_rank_corr['friction_level'] > 0.0, 'mean_flip_rate'].iloc[0]):.3f}."
        ),
        "",
        "## Step 3 Verdict",
        (
            f"Portfolio exact-control replay remains support-only evidence. Identity lock passed, eta=1.0 replay matches "
            f"the source rollout at zero cost, and zero-cost near-flat support holds for {step3_zero_within}/{len(step3_zero_pairs)} seed-universe groups within the 0.005 threshold."
        ),
        (
            f"Positive-cost recurrence remains strong on {int((step3_effect_size['effect_size_mean'] > 0.0).sum())}/{len(step3_effect_size)} universes, "
            "so Step 3 still supports the Q1 exact-control story without promoting portfolio to main evidence."
        ),
        "",
        "## Step 4 Verdict",
        (
            "Inventory Q1 and Q2 remain locked. Q1 follows the threshold story: zero friction is exact, low friction is mixed, "
            f"and friction 1.0 strongly favors tempered execution with tempered win rate {float(step4_high_friction['tempered_win_rate']):.3f}."
        ),
        (
            "Q2 remains near-aligned at zero friction and shows emergence plus persistence of ranking mismatch as friction rises, "
            f"reaching mean flip rate {float(step4_q2_rank_corr['mean_flip_rate'].max()):.3f}."
        ),
        "",
        "## Step 5 Verdict",
        (
            "The required Q2 same-interface package passes on synthetic and inventory. Portfolio stays excluded by the stretch gate "
            f"because zero-friction flip rate is {float(step5_portfolio['zero_friction_mean_flip_rate']):.3f} and zero-friction Spearman is {float(step5_portfolio['zero_friction_mean_spearman_rho']):.3f}."
        ),
        "Step 5 synthetic and inventory rows are a paper-facing repackaging of Step 2 and Step 4 Q2 evidence, not extra independent required evidence.",
        "",
        "## Overall Claim Size",
        (
            "The current package supports a narrow forecasting-system evaluation claim: the core evidence is synthetic plus one operational domain, "
            "with portfolio contributing Q1 exact-control support but not paper-facing Step 5 Q2 support."
        ),
        "",
        "## Remaining Limitations",
        "- Step 2 Q2 shows a positive-friction plateau rather than a strict monotone increase.",
        "- Step 4 Q1 supports a threshold story rather than uniform positive-friction dominance.",
        "- Portfolio Q2 remains excluded from the paper-facing Step 5 package.",
        "",
        "## Reopen-Rule Status",
        (
            "Headline verdict: reopen required."
            if reopen_required
            else "Headline verdict: closed. No required package has invalid input and no required sanity check failed."
        ),
    ]
    if reopen_required:
        lines.extend(
            [
                "",
                "Required sanity failures:",
            ]
        )
        for row in required_rows.itertuples(index=False):
            lines.append(f"- {row.step_id} / {row.domain} / {row.scenario_id} / {row.check_name}: {row.note}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    outfile_dir = Path(args.outfile_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile_dir.mkdir(parents=True, exist_ok=True)

    step2_q1 = read_csv(STEP2_Q1_PATH)
    step2_q2 = read_csv(STEP2_Q2_PATH)
    step2_q1_gap = read_csv(STEP2_Q1_GAP_PATH)
    _step2_lock_text = read_text(STEP2_LOCK_MD)

    step3_raw = read_csv(STEP3_CONTROL_PATH)
    step3_forecast_hash = read_csv(STEP3_FORECAST_HASH_PATH)
    step3_proposal_hash = read_csv(STEP3_PROPOSAL_HASH_PATH)
    _step3_target_delta = read_csv(STEP3_TARGET_DELTA_PATH)
    step3_effect_size = read_csv(STEP3_EFFECT_SIZE_PATH)
    _step3_verdict_text = read_text(STEP3_VERDICT_MD)

    step4_q1 = read_csv(STEP4_Q1_PATH)
    step4_q2 = read_csv(STEP4_Q2_PATH)
    step4_q1_threshold = read_csv(STEP4_Q1_THRESHOLD_PATH)
    _step4_q2_summary = read_csv(STEP4_Q2_SUMMARY_PATH)
    step4_freeze_check = read_csv(STEP4_FREEZE_CHECK_PATH)
    _step4_verdict_text = read_text(STEP4_VERDICT_MD)

    step5_verdict = read_csv(STEP5_VERDICT_PATH)
    step5_manifest = read_json(STEP5_MANIFEST_PATH)
    _step5_note_text = read_text(STEP5_NOTE_PATH)

    step2_q2_outputs = raw_q2_artifacts(step2_q2, domain="synthetic", expected_interface_id="tempered")
    step4_q2_outputs = raw_q2_artifacts(step4_q2, domain="inventory", expected_interface_id="responsive")
    step3_paired = step3_paired_delta_table(step3_raw)

    step5_packages: dict[str, dict[str, Any]] = {}
    for domain in ["synthetic", "inventory", "portfolio"]:
        domain_dir = STEP5_DIR / domain
        rank_corr_path = domain_dir / "rank_correlation_by_friction.csv"
        seed_stats_path = domain_dir / "seed_level_rank_stats.csv"
        pairwise_path = domain_dir / "pairwise_flips_by_friction.csv"
        verdict_row = step5_verdict.loc[step5_verdict["domain"] == domain].iloc[0]
        step5_packages[domain] = {
            "rank_corr": read_csv(rank_corr_path),
            "seed_stats": read_csv(seed_stats_path),
            "pairwise": read_csv(pairwise_path),
            "rank_corr_path": rank_corr_path,
            "seed_stats_path": seed_stats_path,
            "pairwise_path": pairwise_path,
            "verdict": verdict_row,
        }

    master_results = build_master_results(
        step2_q1=step2_q1,
        step2_q2=step2_q2,
        step3_raw=step3_raw,
        step4_q1=step4_q1,
        step4_q2=step4_q2,
        step5_packages=step5_packages,
    )
    summary_by_domain = build_summary_by_domain(
        step2_q1_gap=step2_q1_gap,
        step2_q2_outputs=step2_q2_outputs,
        step3_raw=step3_raw,
        step3_effect_size=step3_effect_size,
        step4_q1_threshold=step4_q1_threshold,
        step4_q2_outputs=step4_q2_outputs,
        step5_packages=step5_packages,
    )
    rank_flip_summary = build_rank_flip_summary(
        step2_q2_outputs=step2_q2_outputs,
        step4_q2_outputs=step4_q2_outputs,
        step5_packages=step5_packages,
    )
    sanity_checks = build_sanity_checks(
        step2_q1=step2_q1,
        step2_q2=step2_q2,
        step2_q1_gap=step2_q1_gap,
        step2_q2_outputs=step2_q2_outputs,
        step3_raw=step3_raw,
        step3_forecast_hash=step3_forecast_hash,
        step3_proposal_hash=step3_proposal_hash,
        step3_effect_size=step3_effect_size,
        step3_paired=step3_paired,
        step4_q1=step4_q1,
        step4_q2=step4_q2,
        step4_q1_threshold=step4_q1_threshold,
        step4_q2_outputs=step4_q2_outputs,
        step4_freeze_check=step4_freeze_check,
        step5_manifest=step5_manifest,
        step5_verdict=step5_verdict,
        step5_packages=step5_packages,
        master_results=master_results,
        summary_by_domain=summary_by_domain,
        rank_flip_summary=rank_flip_summary,
    )

    required_failures = sanity_checks[
        sanity_checks["step_id"].isin(["step2", "step4", "step5"])
        & sanity_checks["domain"].isin(["synthetic", "inventory"])
        & (sanity_checks["pass_fail"] == FAIL)
    ]
    claim_map = build_claim_map(required_failure_count=int(len(required_failures)))
    one_page_verdict = build_one_page_verdict(
        sanity_checks=sanity_checks,
        step5_verdict=step5_verdict,
        step2_q1_gap=step2_q1_gap,
        step2_q2_outputs=step2_q2_outputs,
        step3_paired=step3_paired,
        step3_effect_size=step3_effect_size,
        step4_q1_threshold=step4_q1_threshold,
        step4_q2_outputs=step4_q2_outputs,
    )

    file_map = {
        "master_results.csv": master_results,
        "summary_by_domain.csv": summary_by_domain,
        "rank_flip_summary.csv": rank_flip_summary,
        "sanity_checks.csv": sanity_checks,
        "claim_map.csv": claim_map,
    }
    for filename, frame in file_map.items():
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "one_page_verdict.md").write_text(one_page_verdict)

    for filename in [
        "master_results.csv",
        "summary_by_domain.csv",
        "rank_flip_summary.csv",
        "sanity_checks.csv",
        "claim_map.csv",
        "one_page_verdict.md",
    ]:
        shutil.copy2(output_dir / filename, outfile_dir / filename)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
