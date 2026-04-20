#!/usr/bin/env python3
"""
Run final/test evaluation for the validation-selected volatility-spike eta champion and runner-up.

Assumptions documented here on purpose:

1. We do not perform post-hoc reselection on test. Champion and runner-up are fixed from the
   validation note and CSV for `arch_vol_spike_eta`.
2. This evaluation remains fully independent from RL replay and uses the same shared deterministic
   target mapping as the validation stage.
3. The zero-cost near-flat screen on test uses the same threshold as validation:
   |ΔSharpe_exec(kappa=0)| <= 0.01.
4. Both selected configurations are reported. Champion status is for emphasis only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_arch_deadband_partial as deadband
import run_arch_vol_spike_eta as vol_spike
from target_exec_audit_utils import classify_pair, format_float, kappa_label, kappa_sort_key, zero_cost_near_flat_override


REPO_ROOT = Path(__file__).resolve().parents[2]
VOL_SPIKE_CONFIG = REPO_ROOT / "configs" / "generalization" / "arch_vol_spike_eta.yaml"
VOL_SPIKE_VALIDATION_CSV = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "arch_vol_spike_eta"
    / "validation_results.csv"
)
VOL_SPIKE_VALIDATION_NOTE = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "notes"
    / "arch_vol_spike_eta_validation_note.md"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "arch_vol_spike_eta"
)
OUTPUT_CHAMPION_CSV = OUTPUT_DIR / "test_results_champion.csv"
OUTPUT_RUNNERUP_CSV = OUTPUT_DIR / "test_results_runnerup.csv"
OUTPUT_NOTE = (
    REPO_ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "notes"
    / "arch_vol_spike_eta_test_note.md"
)

TEST_NEAR_FLAT_THRESHOLD = 0.01


@dataclass(frozen=True)
class SelectedConfig:
    architecture: str
    config_key: str
    validation_note_path: Path
    validation_csv_path: Path
    params: dict[str, float | int]
    reference_arm: str
    period: str = "final"
    seed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final/test evaluation for champion and runner-up volatility-spike eta configs.")
    parser.add_argument("--output-champion-csv", default=str(OUTPUT_CHAMPION_CSV))
    parser.add_argument("--output-runnerup-csv", default=str(OUTPUT_RUNNERUP_CSV))
    parser.add_argument("--output-note", default=str(OUTPUT_NOTE))
    return parser.parse_args()


def _load_selection(validation_csv_path: Path, *, flag_column: str) -> SelectedConfig:
    df = pd.read_csv(validation_csv_path)
    winner = df[df[flag_column] == "yes"].copy()
    if winner.empty:
        raise ValueError(f"No {flag_column}=yes row found in {validation_csv_path}")
    winner = winner.drop_duplicates(subset=["config_key"]).iloc[0]
    params = {
        "trigger": float(winner["trigger"]),
        "eta_low": float(winner["eta_low"]),
        "lookback_sigma": int(winner["lookback_sigma"]),
        "lookback_ref": int(winner["lookback_ref"]),
    }
    return SelectedConfig(
        architecture="arch_vol_spike_eta",
        config_key=str(winner["config_key"]),
        validation_note_path=VOL_SPIKE_VALIDATION_NOTE,
        validation_csv_path=validation_csv_path,
        params=params,
        reference_arm="full_rebalance_baseline",
    )


def _load_selected_pair() -> tuple[SelectedConfig, SelectedConfig]:
    champion = _load_selection(VOL_SPIKE_VALIDATION_CSV, flag_column="champion_recommendation")
    runnerup = _load_selection(VOL_SPIKE_VALIDATION_CSV, flag_column="runnerup_recommendation")
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


def _load_period_data():
    cfg = deadband._load_yaml(VOL_SPIKE_CONFIG)
    shared_cfg = deadband._resolve_path(VOL_SPIKE_CONFIG, str(cfg["shared_target_config"]))
    period_data = deadband._prepare_period_data(shared_cfg, period="final", offline=True)
    kappas = [float(kappa) for kappa in ((cfg.get("execution", {}) or {}).get("kappas") or [])]
    return cfg, period_data, kappas


def _evaluate_selection(selection: SelectedConfig, *, role: str) -> pd.DataFrame:
    cfg, period_data, kappas = _load_period_data()
    execution_cfg = cfg.get("execution", {}) or {}
    eps = float(execution_cfg.get("eps", 1e-8))
    near_flat_threshold = TEST_NEAR_FLAT_THRESHOLD
    out_rows: list[dict[str, object]] = []

    for kappa in kappas:
        selected_df, selected_extras = vol_spike._simulate_execution_rule(
            period_data,
            transaction_cost=float(kappa),
            seed=selection.seed,
            mode="vol_spike_eta",
            trigger=float(selection.params["trigger"]),
            eta_low=float(selection.params["eta_low"]),
            lookback_sigma=int(selection.params["lookback_sigma"]),
            lookback_ref=int(selection.params["lookback_ref"]),
            eps=eps,
        )
        reference_df, _ = vol_spike._simulate_execution_rule(
            period_data,
            transaction_cost=float(kappa),
            seed=selection.seed,
            mode="full_rebalance",
            trigger=None,
            eta_low=None,
            lookback_sigma=int(selection.params["lookback_sigma"]),
            lookback_ref=int(selection.params["lookback_ref"]),
            eps=eps,
        )
        selected_arm = vol_spike._candidate_label(
            trigger=float(selection.params["trigger"]),
            eta_low=float(selection.params["eta_low"]),
            lookback_sigma=int(selection.params["lookback_sigma"]),
            lookback_ref=int(selection.params["lookback_ref"]),
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
                "mean_eta_t": selected_extras.get("mean_eta_t"),
                "mean_spike": selected_extras.get("mean_spike"),
                "activation_rate": selected_extras.get("activation_rate"),
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
    if pos_pass and (disagreement_pass or disagreement_weak or not zero_pass):
        return "Yellow"
    return "Red"


def _write_note(champion_df: pd.DataFrame, runnerup_df: pd.DataFrame, output_note: Path) -> None:
    output_note.parent.mkdir(parents=True, exist_ok=True)
    champion_verdict = _role_verdict(champion_df)
    runnerup_verdict = _role_verdict(runnerup_df)

    def render_rows(df: pd.DataFrame) -> list[str]:
        rows: list[str] = []
        for row in df.itertuples(index=False):
            rows.append(
                f"- `kappa={kappa_label(row.kappa)}`: `ΔSharpe_exec={format_float(row.delta_sharpe_exec)}`, "
                f"`ΔSharpe_tgt={format_float(row.delta_sharpe_tgt)}`, "
                f"`turnover_reduction={format_float(row.turnover_reduction_pct, digits=3)}%`, "
                f"`mean_eta={format_float(row.mean_eta_t, digits=4)}`, "
                f"`activation={format_float(row.activation_rate, digits=3)}`, "
                f"`disagreement={row.disagreement_type}`"
            )
        return rows

    champion_config = champion_df.iloc[0]["config_key"]
    runnerup_config = runnerup_df.iloc[0]["config_key"]

    note = "\n".join(
        [
            "# Volatility-Spike Eta Test Note",
            "",
            "This note reports final/test evaluation for both selected `arch_vol_spike_eta` configurations.",
            "",
            "Selection policy:",
            "- No post-hoc reselection is performed on test.",
            "- The candidate pair is fixed from the redesigned validation-only volatility-spike note and CSV.",
            "- Zero-cost near-flat on test uses the same threshold as validation: `|ΔSharpe_exec(kappa=0)| <= 0.01`.",
            "",
            "Champion:",
            f"- Architecture: `arch_vol_spike_eta`",
            f"- Validation source: [arch_vol_spike_eta_validation_note.md]({VOL_SPIKE_VALIDATION_NOTE.as_posix()}:1)",
            f"- Fixed config: `{champion_config}`",
            f"- Test verdict: `{champion_verdict}`",
            "Per-kappa summary:",
            *render_rows(champion_df),
            "",
            "Runner-up:",
            f"- Architecture: `arch_vol_spike_eta`",
            f"- Validation source: [arch_vol_spike_eta_validation_note.md]({VOL_SPIKE_VALIDATION_NOTE.as_posix()}:1)",
            f"- Fixed config: `{runnerup_config}`",
            f"- Test verdict: `{runnerup_verdict}`",
            "Per-kappa summary:",
            *render_rows(runnerup_df),
            "",
            "Interpretation:",
            "- Both rows are reported because champion status is for emphasis only, not for hiding the runner-up.",
            "- The test readout should be interpreted conservatively and only promoted into Step 8 if the broader architecture audit is updated in a later step.",
        ]
    )
    output_note.write_text(note + "\n")


def main() -> int:
    args = parse_args()
    champion_selection, runnerup_selection = _load_selected_pair()
    champion_df = _evaluate_selection(champion_selection, role="champion")
    runnerup_df = _evaluate_selection(runnerup_selection, role="runnerup")

    Path(args.output_champion_csv).resolve().parent.mkdir(parents=True, exist_ok=True)
    champion_df.to_csv(Path(args.output_champion_csv).resolve(), index=False)
    runnerup_df.to_csv(Path(args.output_runnerup_csv).resolve(), index=False)
    _write_note(champion_df, runnerup_df, Path(args.output_note).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
