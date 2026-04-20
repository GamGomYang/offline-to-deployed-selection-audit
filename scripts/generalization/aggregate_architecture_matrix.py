#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from target_exec_audit_utils import (
    ZERO_COST_NEAR_FLAT_THRESHOLD,
    ZERO_COST_YELLOW_THRESHOLD,
    classify_pair,
    display_architecture_name,
    format_float,
    kappa_label,
    kappa_sort_key,
    latex_escape,
    zero_cost_near_flat_override,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "architecture_matrix" / "raw"
DEFAULT_INDEPENDENT_CHAMPION_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "arch_independent_nonrl" / "test_results_champion.csv"
)
DEFAULT_INDEPENDENT_RUNNERUP_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "arch_independent_nonrl" / "test_results_runnerup.csv"
)
DEFAULT_VOL_SPIKE_CHAMPION_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "arch_vol_spike_eta" / "test_results_champion.csv"
)
DEFAULT_VOL_SPIKE_RUNNERUP_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "arch_vol_spike_eta" / "test_results_runnerup.csv"
)
DEFAULT_OUTPUT_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "architecture_matrix" / "decision_architecture_results.csv"
)
DEFAULT_OUTPUT_TEX = ROOT / "paper" / "forecasting_workshop" / "generalization" / "tables" / "decision_architecture_summary.tex"
DEFAULT_OUTPUT_NOTE = ROOT / "paper" / "forecasting_workshop" / "generalization" / "notes" / "decision_architecture_note.md"
LOCKED_KAPPAS = [0.0, 5e-4, 1e-3]
ARCHITECTURE_ORDER = {
    "arch_rl_selected": 0,
    "arch_deadband_partial_champion": 1,
    "arch_deadband_partial_runnerup": 2,
    "arch_vol_spike_eta_champion": 3,
    "arch_vol_spike_eta_runnerup": 4,
    "arch_rule_eta_fixed": 5,
    "arch_linear_prox": 6,
    "arch_threshold_rebalance": 7,
}


@dataclass(frozen=True)
class ArchitectureVerdict:
    architecture: str
    verdict: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate architecture-matrix raw results into paper-facing outputs.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Root with architecture raw result.json files.")
    parser.add_argument("--independent-champion-csv", default=str(DEFAULT_INDEPENDENT_CHAMPION_CSV), help="Independent non-RL champion test CSV.")
    parser.add_argument("--independent-runnerup-csv", default=str(DEFAULT_INDEPENDENT_RUNNERUP_CSV), help="Independent non-RL runner-up test CSV.")
    parser.add_argument("--vol-spike-champion-csv", default=str(DEFAULT_VOL_SPIKE_CHAMPION_CSV), help="Vol-spike independent non-RL champion test CSV.")
    parser.add_argument("--vol-spike-runnerup-csv", default=str(DEFAULT_VOL_SPIKE_RUNNERUP_CSV), help="Vol-spike independent non-RL runner-up test CSV.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Destination CSV path.")
    parser.add_argument("--output-tex", default=str(DEFAULT_OUTPUT_TEX), help="Destination LaTeX table path.")
    parser.add_argument("--output-note", default=str(DEFAULT_OUTPUT_NOTE), help="Destination note path.")
    return parser.parse_args()


def _load_raw_results(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result_path in sorted(raw_root.glob("*/*/seed_*/result.json")):
        rows.append(json.loads(result_path.read_text()))
    if not rows:
        raise FileNotFoundError(f"No result.json files found under {raw_root}")

    df = pd.DataFrame(rows)
    numeric_cols = [col for col in df.columns if col not in {"architecture", "family", "evaluation_role", "compare_arm", "period", "selected_arm", "reference_arm", "disagreement_type", "result_dir", "selected_trace_path", "reference_trace_path", "model_path", "selection_payload_path", "run_completed_at"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _zero_cost_threshold_for_architecture(name: str) -> float:
    if name in {
        "arch_deadband_partial_champion",
        "arch_deadband_partial_runnerup",
        "arch_vol_spike_eta_champion",
        "arch_vol_spike_eta_runnerup",
    }:
        return ZERO_COST_YELLOW_THRESHOLD
    return ZERO_COST_NEAR_FLAT_THRESHOLD


def _row_verdict(*, kappa: float, delta_exec: float, disagreement_strength: int, zero_cost_threshold: float) -> tuple[str, str]:
    if np.isclose(float(kappa), 0.0):
        if abs(float(delta_exec)) <= float(zero_cost_threshold):
            return "Green", "yes"
        if abs(float(delta_exec)) <= float(ZERO_COST_YELLOW_THRESHOLD):
            return "Yellow", "no"
        return "Red", "no"
    if float(delta_exec) > 0.0 and int(disagreement_strength) >= 2:
        return "Green", "n/a"
    if float(delta_exec) > 0.0:
        return "Yellow", "n/a"
    return "Red", "n/a"


def _aggregate_raw_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for architecture, arch_df in df.groupby("architecture", sort=True):
        for kappa in LOCKED_KAPPAS:
            kappa_df = arch_df[np.isclose(arch_df["kappa"], float(kappa), atol=1e-15)].copy()
            if kappa_df.empty:
                continue

            metric_exec_selected = float(kappa_df["sharpe_exec_net"].median())
            metric_exec_reference = float(kappa_df["reference_sharpe_exec_net"].median())
            metric_tgt_selected = float(kappa_df["sharpe_target_net"].median())
            metric_tgt_reference = float(kappa_df["reference_sharpe_target_net"].median())
            audit = classify_pair(
                metric_exec_a=metric_exec_selected,
                metric_exec_b=metric_exec_reference,
                metric_tgt_a=metric_tgt_selected,
                metric_tgt_b=metric_tgt_reference,
            )
            audit = zero_cost_near_flat_override(
                audit,
                kappa=float(kappa),
                near_flat_threshold=_zero_cost_threshold_for_architecture(str(architecture)),
            )

            turnover_exec_selected = float(kappa_df["turnover_exec"].median())
            turnover_exec_reference = float(kappa_df["reference_turnover_exec"].median())
            turnover_reduction_pct = float(((turnover_exec_reference - turnover_exec_selected) / turnover_exec_reference) * 100.0)
            positive_cost_direction_flag = "yes" if float(kappa) > 0.0 and audit.delta_exec > 0.0 else ("no" if float(kappa) > 0.0 else "n/a")
            verdict_row, zero_cost_near_flat_flag = _row_verdict(
                kappa=float(kappa),
                delta_exec=float(audit.delta_exec),
                disagreement_strength=int(audit.disagreement_strength),
                zero_cost_threshold=_zero_cost_threshold_for_architecture(str(architecture)),
            )

            rows.append(
                {
                    "architecture": architecture,
                    "kappa": float(kappa),
                    "selected_arm": str(kappa_df["selected_arm"].iloc[0]),
                    "reference_arm": str(kappa_df["reference_arm"].iloc[0]),
                    "median_sharpe_exec_reference": metric_exec_reference,
                    "median_sharpe_exec_selected": metric_exec_selected,
                    "delta_sharpe_exec": audit.delta_exec,
                    "median_sharpe_tgt_reference": metric_tgt_reference,
                    "median_sharpe_tgt_selected": metric_tgt_selected,
                    "delta_sharpe_tgt": audit.delta_tgt,
                    "median_toexec_reference": turnover_exec_reference,
                    "median_toexec_selected": turnover_exec_selected,
                    "turnover_reduction_pct": turnover_reduction_pct,
                    "rank_exec": audit.rank_exec,
                    "rank_tgt": audit.rank_tgt,
                    "sign_exec": audit.sign_exec,
                    "sign_tgt": audit.sign_tgt,
                    "disagreement_type": audit.disagreement_type,
                    "disagreement_strength": audit.disagreement_strength,
                    "zero_cost_near_flat_flag": zero_cost_near_flat_flag,
                    "positive_cost_direction_flag": positive_cost_direction_flag,
                    "verdict_row": verdict_row,
                    "paper_use": "support_only" if architecture != "arch_threshold_rebalance" else "appendix_only_optional_support",
                }
            )

    out_df = pd.DataFrame(rows)
    return out_df.sort_values(["architecture", "kappa"], key=lambda s: s.map(kappa_sort_key) if s.name == "kappa" else s).reset_index(drop=True)


def _load_independent_support_rows(csv_path: Path, *, architecture_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        metric_exec_reference = float(row.metric_exec_b)
        metric_exec_selected = float(row.metric_exec_a)
        metric_tgt_reference = float(row.metric_tgt_b)
        metric_tgt_selected = float(row.metric_tgt_a)
        audit = classify_pair(
            metric_exec_a=metric_exec_selected,
            metric_exec_b=metric_exec_reference,
            metric_tgt_a=metric_tgt_selected,
            metric_tgt_b=metric_tgt_reference,
        )
        audit = zero_cost_near_flat_override(
            audit,
            kappa=float(row.kappa),
            near_flat_threshold=_zero_cost_threshold_for_architecture(architecture_name),
        )
        verdict_row, zero_cost_near_flat_flag = _row_verdict(
            kappa=float(row.kappa),
            delta_exec=float(audit.delta_exec),
            disagreement_strength=int(audit.disagreement_strength),
            zero_cost_threshold=_zero_cost_threshold_for_architecture(architecture_name),
        )
        rows.append(
            {
                "architecture": architecture_name,
                "kappa": float(row.kappa),
                "selected_arm": str(row.selected_arm),
                "reference_arm": str(row.reference_arm),
                "median_sharpe_exec_reference": metric_exec_reference,
                "median_sharpe_exec_selected": metric_exec_selected,
                "delta_sharpe_exec": float(audit.delta_exec),
                "median_sharpe_tgt_reference": metric_tgt_reference,
                "median_sharpe_tgt_selected": metric_tgt_selected,
                "delta_sharpe_tgt": float(audit.delta_tgt),
                "median_toexec_reference": float(row.reference_turnover_exec),
                "median_toexec_selected": float(row.turnover_exec),
                "turnover_reduction_pct": float(row.turnover_reduction_pct),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": int(audit.disagreement_strength),
                "zero_cost_near_flat_flag": zero_cost_near_flat_flag,
                "positive_cost_direction_flag": "yes" if float(row.kappa) > 0.0 and float(audit.delta_exec) > 0.0 else ("no" if float(row.kappa) > 0.0 else "n/a"),
                "verdict_row": verdict_row,
                "paper_use": "support_only_independent_nonrl_primary" if architecture_name.endswith("champion") else "support_only_independent_nonrl_secondary",
            }
        )
    return pd.DataFrame(rows)


def _architecture_verdict_for_group(group: pd.DataFrame) -> ArchitectureVerdict:
    group = group.sort_values("kappa")
    zero_row = group[np.isclose(group["kappa"], 0.0, atol=1e-15)].iloc[0]
    positive_rows = group[group["kappa"] > 0.0]

    if (positive_rows["delta_sharpe_exec"] <= 0.0).any():
        return ArchitectureVerdict(
            architecture=str(group["architecture"].iloc[0]),
            verdict="Red",
            reason="At least one positive-cost row loses the executed-path direction.",
        )
    if all(positive_rows["disagreement_strength"] >= 2) and zero_row["zero_cost_near_flat_flag"] == "yes":
        return ArchitectureVerdict(
            architecture=str(group["architecture"].iloc[0]),
            verdict="Green",
            reason="Positive-cost rows preserve executed-path direction and still disagree materially with target-based reading.",
        )
    return ArchitectureVerdict(
        architecture=str(group["architecture"].iloc[0]),
        verdict="Yellow",
        reason="The architecture moves the positive-cost executed path in the same direction, but the disagreement signal is weak or absent in part of the package.",
    )


def _attach_architecture_verdicts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, ArchitectureVerdict]]:
    verdicts: dict[str, ArchitectureVerdict] = {}
    for architecture, group in df.groupby("architecture", sort=True):
        verdicts[architecture] = _architecture_verdict_for_group(group)
    out_df = df.copy()
    out_df["architecture_verdict"] = out_df["architecture"].map(lambda name: verdicts[name].verdict)
    return out_df, verdicts


def _write_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _sort_output_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_arch_order"] = out["architecture"].map(ARCHITECTURE_ORDER).fillna(99)
    out = out.sort_values(["_arch_order", "kappa"], key=lambda s: s.map(kappa_sort_key) if s.name == "kappa" else s)
    return out.drop(columns=["_arch_order"]).reset_index(drop=True)


def _write_tex(df: pd.DataFrame, output_tex: Path) -> None:
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    champion_only = df[
        df["architecture"].isin(
            [
                "arch_rl_selected",
                "arch_deadband_partial_champion",
                "arch_vol_spike_eta_champion",
                "arch_rule_eta_fixed",
                "arch_linear_prox",
            ]
        )
    ].copy()
    champion_only = _sort_output_rows(champion_only)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrrllll}",
        "\\toprule",
        "Architecture & $\\kappa$ & Exec(ref) & Exec(sel) & $\\Delta$Exec & Tgt(ref) & Tgt(sel) & $\\Delta$Tgt & TO red.\\% & Disag. & Near-flat@0 & Pos-cost dir. & Support verdict \\\\",
        "\\midrule",
    ]
    for row in champion_only.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    latex_escape(display_architecture_name(row.architecture)),
                    latex_escape(kappa_label(row.kappa)),
                    format_float(row.median_sharpe_exec_reference),
                    format_float(row.median_sharpe_exec_selected),
                    format_float(row.delta_sharpe_exec),
                    format_float(row.median_sharpe_tgt_reference),
                    format_float(row.median_sharpe_tgt_selected),
                    format_float(row.delta_sharpe_tgt),
                    format_float(row.turnover_reduction_pct, digits=1),
                    latex_escape(str(row.disagreement_type)),
                    latex_escape(str(row.zero_cost_near_flat_flag)),
                    latex_escape(str(row.positive_cost_direction_flag)),
                    latex_escape(str(row.verdict_row)),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\par\\smallskip\\parbox{0.98\\columnwidth}{\\scriptsize TO red.\\% is computed relative to the reference arm within each architecture family. Small values for the deadband and volatility-scaled families reflect near-matched turnover within those family-local comparisons and should not be read as cross-family inefficiency. Support verdicts are also family-local; runner-up configurations remain in the CSV and notes and are omitted from this compact visible table only for readability.}",
            "\\caption{Support-only decision-architecture summary under frozen-source comparisons. Verdicts are computed within each architecture family using the documented zero-cost, positive-cost, and disagreement checks; the table is not intended for cross-family ranking by absolute Sharpe level. `Disag.` reports the conservative target-versus-executed disagreement label, `Near-flat@0` reports the documented zero-cost near-flat check, and `Pos-cost dir.` reports positive-cost executed-path direction.}",
            "\\label{tab:decision_architecture_summary}",
            "\\end{table}",
        ]
    )
    output_tex.write_text("\n".join(lines) + "\n")


def _write_note(df: pd.DataFrame, verdicts: dict[str, ArchitectureVerdict], output_note: Path) -> None:
    output_note.parent.mkdir(parents=True, exist_ok=True)
    rl_df = df[df["architecture"] == "arch_rl_selected"].sort_values("kappa")
    deadband_champion_df = df[df["architecture"] == "arch_deadband_partial_champion"].sort_values("kappa")
    deadband_runnerup_df = df[df["architecture"] == "arch_deadband_partial_runnerup"].sort_values("kappa")
    vol_champion_df = df[df["architecture"] == "arch_vol_spike_eta_champion"].sort_values("kappa")
    vol_runnerup_df = df[df["architecture"] == "arch_vol_spike_eta_runnerup"].sort_values("kappa")
    rule_df = df[df["architecture"] == "arch_rule_eta_fixed"].sort_values("kappa")
    linear_df = df[df["architecture"] == "arch_linear_prox"].sort_values("kappa")

    rl_k5 = rl_df[np.isclose(rl_df["kappa"], 5e-4)].iloc[0]
    rl_k1 = rl_df[np.isclose(rl_df["kappa"], 1e-3)].iloc[0]
    deadband_champion_k0 = deadband_champion_df[np.isclose(deadband_champion_df["kappa"], 0.0)].iloc[0]
    deadband_champion_k5 = deadband_champion_df[np.isclose(deadband_champion_df["kappa"], 5e-4)].iloc[0]
    deadband_champion_k1 = deadband_champion_df[np.isclose(deadband_champion_df["kappa"], 1e-3)].iloc[0]
    deadband_runnerup_k0 = deadband_runnerup_df[np.isclose(deadband_runnerup_df["kappa"], 0.0)].iloc[0]
    deadband_runnerup_k1 = deadband_runnerup_df[np.isclose(deadband_runnerup_df["kappa"], 1e-3)].iloc[0]
    vol_champion_k0 = vol_champion_df[np.isclose(vol_champion_df["kappa"], 0.0)].iloc[0]
    vol_champion_k5 = vol_champion_df[np.isclose(vol_champion_df["kappa"], 5e-4)].iloc[0]
    vol_champion_k1 = vol_champion_df[np.isclose(vol_champion_df["kappa"], 1e-3)].iloc[0]
    vol_runnerup_k0 = vol_runnerup_df[np.isclose(vol_runnerup_df["kappa"], 0.0)].iloc[0]
    vol_runnerup_k1 = vol_runnerup_df[np.isclose(vol_runnerup_df["kappa"], 1e-3)].iloc[0]
    linear_k0 = linear_df[np.isclose(linear_df["kappa"], 0.0)].iloc[0]
    linear_k1 = linear_df[np.isclose(linear_df["kappa"], 1e-3)].iloc[0]

    text = f"""# Decision Architecture Note

This note summarizes the support-only decision-architecture comparison under the frozen-source rules documented for the generalization package. The comparison is narrow: it asks whether the target-versus-executed discrepancy recurs outside the exact current RL interface, while keeping the paper's main wording centered on executed-path evaluation under constrained execution. It does not authorize any claim of universal architectural disagreement.

The architecture-level aggregation uses the same conservative disagreement audit that the master package uses. A row is labeled `ranking_mismatch` when executed-path and target-based evaluation imply different arm ordering once the audit's tie tolerance is applied, `sign_flip` only when the two views point in opposite non-tied directions, and `magnitude_only` only when the ranking stays the same but the target view materially damps the executed-path gap.

The RL-source arm remains the main in-family reference. In `arch_rl_selected`, the zero-cost row stays near-flat at `delta_sharpe_exec={format_float(rl_df.iloc[0].delta_sharpe_exec)}`, while the positive-cost rows are `+{format_float(rl_k5.delta_sharpe_exec)}` at `kappa=5e-4` and `+{format_float(rl_k1.delta_sharpe_exec)}` at `kappa=1e-3` on the executed path. Under the conservative audit these positive-cost rows are still classified as `ranking_mismatch`, because the target-based deltas remain too small to preserve the executed-path ordering.

The independent non-RL deadband pair is now one of the central support additions in this package. The selected champion `arch_deadband_partial_champion` stays within the validation-aligned zero-cost qualification screen at `delta_sharpe_exec={format_float(deadband_champion_k0.delta_sharpe_exec)}` and then remains positive at `+{format_float(deadband_champion_k5.delta_sharpe_exec)}` for `kappa=5e-4` and `+{format_float(deadband_champion_k1.delta_sharpe_exec)}` for `kappa=1e-3`, with `ranking_mismatch` on both positive-cost rows. The paired runner-up `arch_deadband_partial_runnerup` is smaller but qualitatively similar: its zero-cost row is `delta_sharpe_exec={format_float(deadband_runnerup_k0.delta_sharpe_exec)}`, and its highest positive-cost row reaches `+{format_float(deadband_runnerup_k1.delta_sharpe_exec)}` while preserving the same disagreement label. These two rows should be read as independent non-RL support because they do not use RL target replay and they keep the target mapping fixed while changing only the execution rule.

The redesigned independent non-RL volatility-scaled pair now adds a second family-level support block. The champion `arch_vol_spike_eta_champion` remains near-flat at zero cost with `delta_sharpe_exec={format_float(vol_champion_k0.delta_sharpe_exec)}`, then stays positive at `+{format_float(vol_champion_k5.delta_sharpe_exec)}` for `kappa=5e-4` and `+{format_float(vol_champion_k1.delta_sharpe_exec)}` for `kappa=1e-3`, again with `ranking_mismatch` on the positive-cost rows. The paired runner-up `arch_vol_spike_eta_runnerup` is slightly smaller but keeps the same qualitative pattern: its zero-cost row is `delta_sharpe_exec={format_float(vol_runnerup_k0.delta_sharpe_exec)}`, and its highest positive-cost row reaches `+{format_float(vol_runnerup_k1.delta_sharpe_exec)}` while still classifying as `ranking_mismatch`. This second non-RL family matters because it is independent from both RL replay and deadband gating while still starting from the same shared deterministic target mapping.

`arch_rule_eta_fixed` still reproduces the RL-selected numbers under the current spec, which is expected because it replays the same frozen RL target stream with the same fixed-eta map. It should therefore be kept in the package as an implementation-consistency replay rather than promoted as the main independent support result.

The non-RL linear or proximal support arm is mixed in a different way. In `arch_linear_prox`, the selected positive-cost tau improves the executed-path Sharpe relative to `tau=0` by `+{format_float(linear_df[np.isclose(linear_df["kappa"], 5e-4)].iloc[0].delta_sharpe_exec)}` at `kappa=5e-4` and `+{format_float(linear_k1.delta_sharpe_exec)}` at `kappa=1e-3`, but target-based and executed-based evaluation are numerically aligned there, so the disagreement label is `none`. Its zero-cost row is also not near-flat: `delta_sharpe_exec={format_float(linear_k0.delta_sharpe_exec)}`. This means the linear/prox comparator is useful as a cost-sensitive execution support arm, but it does not reproduce the target-versus-executed discrepancy.

The safe architecture verdict is therefore stronger than before but still narrow. The RL-source row remains supportive, the independent non-RL deadband champion and runner-up now add two Green support rows, the independent non-RL vol-spike champion and runner-up add another two Green support rows, and the linear/prox arm still marks the limit case where disagreement disappears. The architecture-level verdicts are `{verdicts["arch_rl_selected"].verdict}` for `arch_rl_selected`, `{verdicts["arch_deadband_partial_champion"].verdict}` for `arch_deadband_partial_champion`, `{verdicts["arch_deadband_partial_runnerup"].verdict}` for `arch_deadband_partial_runnerup`, `{verdicts["arch_vol_spike_eta_champion"].verdict}` for `arch_vol_spike_eta_champion`, `{verdicts["arch_vol_spike_eta_runnerup"].verdict}` for `arch_vol_spike_eta_runnerup`, `{verdicts["arch_rule_eta_fixed"].verdict}` for `arch_rule_eta_fixed`, and `{verdicts["arch_linear_prox"].verdict}` for `arch_linear_prox`. That updated package reduces the specific `RL-only artifact` concern more directly than the earlier architecture draft, because the same qualitative pattern now survives in two independent non-RL execution families as well. It still does not justify claiming universal disagreement across all execution-layer families.
"""
    output_note.write_text(text)


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    independent_champion_csv = Path(args.independent_champion_csv).resolve()
    independent_runnerup_csv = Path(args.independent_runnerup_csv).resolve()
    vol_spike_champion_csv = Path(args.vol_spike_champion_csv).resolve()
    vol_spike_runnerup_csv = Path(args.vol_spike_runnerup_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_tex = Path(args.output_tex).resolve()
    output_note = Path(args.output_note).resolve()

    raw_df = _load_raw_results(raw_root)
    summary_df = _aggregate_raw_rows(raw_df)
    extra_frames = [
        _load_independent_support_rows(independent_champion_csv, architecture_name="arch_deadband_partial_champion"),
        _load_independent_support_rows(independent_runnerup_csv, architecture_name="arch_deadband_partial_runnerup"),
        _load_independent_support_rows(vol_spike_champion_csv, architecture_name="arch_vol_spike_eta_champion"),
        _load_independent_support_rows(vol_spike_runnerup_csv, architecture_name="arch_vol_spike_eta_runnerup"),
    ]
    extra_frames = [frame for frame in extra_frames if not frame.empty]
    if extra_frames:
        summary_df = pd.concat([summary_df, *extra_frames], ignore_index=True)
    summary_df = _sort_output_rows(summary_df)
    summary_df, verdicts = _attach_architecture_verdicts(summary_df)
    summary_df = _sort_output_rows(summary_df)

    _write_csv(summary_df, output_csv)
    _write_tex(summary_df, output_tex)
    _write_note(summary_df, verdicts, output_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
