#!/usr/bin/env python3
"""
Run final/test evaluation for the two independent non-RL family winners.

Assumptions documented here on purpose:

1. We do not perform post-hoc reselection on test. The inputs are fixed by the redesigned
   deadband validation note and CSV, which now provide a stability-first champion and runner-up.
2. This rerun is intentionally scoped to the deadband family because the previous volatility-scaled
   family winner failed on test. The goal here is to check whether a more conservative deadband
   selection rule can produce a cleaner independent non-RL support pair.
3. The zero-cost near-flat screen on test uses the same threshold as the validation eligibility
   notes: |ΔSharpe_exec(kappa=0)| <= 0.01.
4. Both selected configurations are evaluated on the same shared deterministic target mapping and
   the same Step 8-compatible audit logic. The comparison remains about execution rules only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_arch_deadband_partial as deadband
from target_exec_audit_utils import classify_pair, format_float, kappa_label, kappa_sort_key, zero_cost_near_flat_override


REPO_ROOT = Path(__file__).resolve().parents[2]
DEADBAND_CONFIG = REPO_ROOT / "configs" / "generalization" / "arch_deadband_partial.yaml"
DEADBAND_VALIDATION_CSV = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "arch_deadband_partial"
    / "validation_results.csv"
)
DEADBAND_VALIDATION_NOTE = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "notes"
    / "arch_deadband_partial_validation_note.md"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "arch_independent_nonrl"
)
OUTPUT_CHAMPION_CSV = OUTPUT_DIR / "test_results_champion.csv"
OUTPUT_RUNNERUP_CSV = OUTPUT_DIR / "test_results_runnerup.csv"
OUTPUT_NOTE = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "notes"
    / "independent_nonrl_test_note.md"
)

TEST_NEAR_FLAT_THRESHOLD = 0.01


@dataclass(frozen=True)
class SelectedConfig:
    family: str
    architecture: str
    config_key: str
    validation_note_path: Path
    validation_csv_path: Path
    family_rank_score: tuple[float, int, float]
    params: dict[str, float | int]
    reference_arm: str
    period: str = "final"
    seed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final/test evaluation for champion and runner-up independent non-RL configs.")
    parser.add_argument("--output-champion-csv", default=str(OUTPUT_CHAMPION_CSV))
    parser.add_argument("--output-runnerup-csv", default=str(OUTPUT_RUNNERUP_CSV))
    parser.add_argument("--output-note", default=str(OUTPUT_NOTE))
    return parser.parse_args()


def _load_deadband_selection(validation_csv_path: Path, *, flag_column: str) -> SelectedConfig:
    df = pd.read_csv(validation_csv_path)
    winner = df[df[flag_column] == "yes"].copy()
    if winner.empty:
        raise ValueError(f"No {flag_column}=yes row found in {validation_csv_path}")
    winner = winner.drop_duplicates(subset=["config_key"]).iloc[0]
    params = {
        "delta": float(winner["delta"]),
        "eta_db": float(winner["eta_db"]),
    }

    score = (
        float(winner["mean_positive_cost_delta_sharpe_exec"]),
        int(winner["sum_positive_cost_disagreement_strength"]),
        float(winner["mean_positive_cost_turnover_reduction_pct"]),
    )
    return SelectedConfig(
        family="deadband_partial",
        architecture="arch_deadband_partial",
        config_key=str(winner["config_key"]),
        validation_note_path=DEADBAND_VALIDATION_NOTE,
        validation_csv_path=validation_csv_path,
        family_rank_score=score,
        params=params,
        reference_arm="full_rebalance_baseline",
    )


def _load_selected_pair() -> tuple[SelectedConfig, SelectedConfig]:
    champion = _load_deadband_selection(
        DEADBAND_VALIDATION_CSV,
        flag_column="champion_recommendation",
    )
    runnerup = _load_deadband_selection(
        DEADBAND_VALIDATION_CSV,
        flag_column="runnerup_recommendation",
    )
    return champion, runnerup


def _turnover_reduction_pct(selected_turnover: float | None, reference_turnover: float | None) -> float | None:
    if selected_turnover is None or reference_turnover is None:
        return None
    if not np.isfinite(float(selected_turnover)) or not np.isfinite(float(reference_turnover)) or float(reference_turnover) <= 0.0:
        return None
    return float(((float(reference_turnover) - float(selected_turnover)) / float(reference_turnover)) * 100.0)


def _bool_flag_text(value: bool | None, *, na_text: str = "n/a") -> str:
    if value is None:
        return na_text
    return "yes" if bool(value) else "no"


def _load_period_data_for_family(selection: SelectedConfig):
    cfg = deadband._load_yaml(DEADBAND_CONFIG)
    shared_cfg = deadband._resolve_path(DEADBAND_CONFIG, str(cfg["shared_target_config"]))
    period_data = deadband._prepare_period_data(shared_cfg, period=selection.period, offline=True)
    kappas = [float(kappa) for kappa in ((cfg.get("execution", {}) or {}).get("kappas") or [])]
    return cfg, period_data, kappas


def _evaluate_selection(selection: SelectedConfig, *, role: str) -> pd.DataFrame:
    cfg, period_data, kappas = _load_period_data_for_family(selection)
    near_flat_threshold = TEST_NEAR_FLAT_THRESHOLD
    out_rows: list[dict[str, object]] = []

    for kappa in kappas:
        selected_df, _ = deadband._simulate_execution_rule(
            period_data,
            transaction_cost=float(kappa),
            seed=selection.seed,
            mode="deadband_partial",
            delta=float(selection.params["delta"]),
            eta_db=float(selection.params["eta_db"]),
        )
        reference_df, _ = deadband._simulate_execution_rule(
            period_data,
            transaction_cost=float(kappa),
            seed=selection.seed,
            mode="full_rebalance",
            delta=None,
            eta_db=None,
        )
        selected_arm = deadband._candidate_label(
            delta=float(selection.params["delta"]),
            eta_db=float(selection.params["eta_db"]),
        )

        sharpe_exec_selected = deadband._compute_sharpe(selected_df["net_return_lin"])
        sharpe_exec_reference = deadband._compute_sharpe(reference_df["net_return_lin"])
        sharpe_target_selected = deadband._compute_sharpe(selected_df["net_return_lin_target"])
        sharpe_target_reference = deadband._compute_sharpe(reference_df["net_return_lin_target"])

        audit = classify_pair(
            metric_exec_a=sharpe_exec_selected,
            metric_exec_b=sharpe_exec_reference,
            metric_tgt_a=sharpe_target_selected,
            metric_tgt_b=sharpe_target_reference,
        )
        audit = zero_cost_near_flat_override(
            audit,
            kappa=float(kappa),
            near_flat_threshold=near_flat_threshold,
        )

        turnover_exec = deadband._safe_mean(selected_df["turnover_exec"])
        turnover_target = deadband._safe_mean(selected_df["turnover_target"])
        reference_turnover_exec = deadband._safe_mean(reference_df["turnover_exec"])
        reference_turnover_target = deadband._safe_mean(reference_df["turnover_target"])
        zero_cost_near_flat_flag = None
        positive_cost_direction_flag = None
        if np.isclose(float(kappa), 0.0):
            zero_cost_near_flat_flag = abs(float(audit.delta_exec)) <= near_flat_threshold
        else:
            positive_cost_direction_flag = float(audit.delta_exec) > 0.0

        out_rows.append(
            {
                "role": role,
                "architecture": selection.architecture,
                "family": selection.family,
                "config_key": selection.config_key,
                "selected_arm": selected_arm,
                "reference_arm": selection.reference_arm,
                "period": selection.period,
                "seed": int(selection.seed),
                "kappa": float(kappa),
                "metric_exec_a": float(sharpe_exec_selected),
                "metric_exec_b": float(sharpe_exec_reference),
                "metric_tgt_a": float(sharpe_target_selected),
                "metric_tgt_b": float(sharpe_target_reference),
                "delta_sharpe_exec": float(audit.delta_exec),
                "delta_sharpe_tgt": float(audit.delta_tgt),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": int(audit.disagreement_strength),
                "turnover_exec": turnover_exec,
                "turnover_target": turnover_target,
                "reference_turnover_exec": reference_turnover_exec,
                "reference_turnover_target": reference_turnover_target,
                "turnover_reduction_pct": _turnover_reduction_pct(turnover_exec, reference_turnover_exec),
                "zero_cost_near_flat_flag": _bool_flag_text(zero_cost_near_flat_flag),
                "positive_cost_direction_flag": _bool_flag_text(positive_cost_direction_flag),
                "tracking_error_l2": deadband._safe_mean(selected_df["tracking_error_l2"]),
                "final_path_gap": deadband._safe_last_gap(selected_df["equity_net_lin"], selected_df["equity_net_lin_target"]),
                "cost_exec": deadband._safe_sum(selected_df["cost"]),
                "cost_target": deadband._safe_sum(selected_df["cost_target"]),
                "validation_note_path": str(selection.validation_note_path.resolve()),
                "validation_csv_path": str(selection.validation_csv_path.resolve()),
            }
        )

    out_df = pd.DataFrame(out_rows)
    return out_df.sort_values("kappa", key=lambda s: s.map(kappa_sort_key)).reset_index(drop=True)


def _role_verdict(df: pd.DataFrame) -> str:
    zero_pass = bool((df[df["kappa"] == 0.0]["zero_cost_near_flat_flag"] == "yes").all())
    pos_df = df[df["kappa"] > 0.0].copy()
    pos_pass = bool((pos_df["delta_sharpe_exec"] > 0.0).all())
    disagreement_pass = bool(pos_df["disagreement_type"].isin(["ranking_mismatch", "sign_flip"]).any())
    disagreement_weak = bool(pos_df["disagreement_strength"].fillna(0).max() < 2)

    if zero_pass and pos_pass and disagreement_pass:
        return "Green"
    if pos_pass and (not zero_pass or disagreement_weak or not disagreement_pass):
        return "Yellow"
    return "Red"


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _format_param_summary(selection: SelectedConfig) -> str:
    if selection.family == "deadband_partial":
        return f"`delta={selection.params['delta']:.2f}`, `eta_db={selection.params['eta_db']:.4f}`"
    return (
        f"`alpha={selection.params['alpha']:.3f}`, "
        f"`eta_min={selection.params['eta_min']:.2f}`, "
        f"`lookback={int(selection.params['lookback'])}`"
    )


def _render_kappa_lines(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for row in df.itertuples(index=False):
        lines.append(
            f"- `kappa={kappa_label(float(row.kappa))}`: "
            f"`ΔSharpe_exec={format_float(row.delta_sharpe_exec)}`, "
            f"`ΔSharpe_tgt={format_float(row.delta_sharpe_tgt)}`, "
            f"`turnover_reduction={format_float(row.turnover_reduction_pct, digits=3)}%`, "
            f"`disagreement={row.disagreement_type}`"
        )
    return lines


def _write_note(
    champion_selection: SelectedConfig,
    runnerup_selection: SelectedConfig,
    champion_df: pd.DataFrame,
    runnerup_df: pd.DataFrame,
    output_note: Path,
) -> None:
    output_note.parent.mkdir(parents=True, exist_ok=True)
    champion_verdict = _role_verdict(champion_df)
    runnerup_verdict = _role_verdict(runnerup_df)

    text = "\n".join(
        [
            "# Independent Non-RL Test Note",
            "",
            "This note reports final/test evaluation for both selected independent non-RL comparator configurations.",
            "",
            "Selection policy:",
            "- No post-hoc reselection is performed on test.",
            "- The candidate pair is fixed from the existing validation notes.",
            "- The overall champion and runner-up are both taken from the redesigned deadband validation selection rule.",
            f"- Zero-cost near-flat on test uses the same threshold as validation: `|ΔSharpe_exec(kappa=0)| <= {TEST_NEAR_FLAT_THRESHOLD:.2f}`.",
            "",
            "Champion:",
            f"- Architecture: `{champion_selection.architecture}`",
            f"- Validation source: [{champion_selection.validation_note_path.name}]({champion_selection.validation_note_path}:1)",
            f"- Fixed config: `{champion_selection.config_key}` with {_format_param_summary(champion_selection)}",
            f"- Test verdict: `{champion_verdict}`",
            "Per-kappa summary:",
            *_render_kappa_lines(champion_df),
            "",
            "Runner-up:",
            f"- Architecture: `{runnerup_selection.architecture}`",
            f"- Validation source: [{runnerup_selection.validation_note_path.name}]({runnerup_selection.validation_note_path}:1)",
            f"- Fixed config: `{runnerup_selection.config_key}` with {_format_param_summary(runnerup_selection)}",
            f"- Test verdict: `{runnerup_verdict}`",
            "Per-kappa summary:",
            *_render_kappa_lines(runnerup_df),
            "",
            "Interpretation:",
            "- Both rows are reported because champion status is for emphasis only, not for hiding the runner-up.",
            "- The test readout should be interpreted conservatively and fed into Step 8 only after the broader audit is updated in a later step.",
        ]
    )
    output_note.write_text(text + "\n")


def main() -> int:
    args = parse_args()
    champion_selection, runnerup_selection = _load_selected_pair()
    champion_df = _evaluate_selection(champion_selection, role="champion")
    runnerup_df = _evaluate_selection(runnerup_selection, role="runner_up")

    _write_csv(champion_df, Path(args.output_champion_csv))
    _write_csv(runnerup_df, Path(args.output_runnerup_csv))
    _write_note(champion_selection, runnerup_selection, champion_df, runnerup_df, Path(args.output_note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
