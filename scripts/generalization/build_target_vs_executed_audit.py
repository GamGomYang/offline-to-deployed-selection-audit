#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from target_exec_audit_utils import (
    classify_pair,
    display_architecture_name,
    display_universe_name,
    format_float,
    kappa_label,
    kappa_sort_key,
    zero_cost_near_flat_override,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_SWEEP_CSV = ROOT / "paper" / "forecasting_workshop" / "generalization" / "cost_sweep_results.csv"
DEFAULT_MULTI_UNIVERSE_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "multi_universe" / "multi_universe_results.csv"
)
DEFAULT_ARCHITECTURE_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "architecture_matrix" / "decision_architecture_results.csv"
)
DEFAULT_SPLIT_REFERENCE_CSV = ROOT / "paper" / "forecasting_workshop" / "results" / "table_target_vs_executed_eval.csv"
DEFAULT_OUTPUT_CSV = ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "target_vs_executed_master.csv"
DEFAULT_OUTPUT_NOTE = ROOT / "paper" / "forecasting_workshop" / "generalization" / "notes" / "target_vs_executed_summary.md"
DEFAULT_OUTPUT_FIG = ROOT / "paper" / "forecasting_workshop" / "generalization" / "figures" / "fig_target_vs_executed_heatmap.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a master target-vs-executed disagreement audit across support settings.")
    parser.add_argument("--cost-sweep-csv", default=str(DEFAULT_COST_SWEEP_CSV), help="Cost-sweep summary CSV.")
    parser.add_argument("--multi-universe-csv", default=str(DEFAULT_MULTI_UNIVERSE_CSV), help="Multi-universe summary CSV.")
    parser.add_argument("--architecture-csv", default=str(DEFAULT_ARCHITECTURE_CSV), help="Decision-architecture summary CSV.")
    parser.add_argument("--split-reference-csv", default=str(DEFAULT_SPLIT_REFERENCE_CSV), help="Optional canonical split reference CSV.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Destination master audit CSV.")
    parser.add_argument("--output-note", default=str(DEFAULT_OUTPUT_NOTE), help="Destination Markdown note.")
    parser.add_argument("--output-fig", default=str(DEFAULT_OUTPUT_FIG), help="Destination heatmap PDF.")
    return parser.parse_args()


def _paper_use_for_architecture(name: str) -> str:
    mapping = {
        "arch_rl_selected": "support_only_reference",
        "arch_rule_eta_fixed": "support_only_nonindependent",
        "arch_linear_prox": "support_only_mixed",
        "arch_threshold_rebalance": "appendix_only_optional_support",
    }
    return mapping.get(name, "support_only")


def _load_cost_sweep_rows(path: Path) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        audit = classify_pair(
            metric_exec_a=float(row.exec_sharpe_eta_0_5),
            metric_exec_b=float(row.exec_sharpe_eta_1_0),
            metric_tgt_a=float(row.target_sharpe_eta_0_5),
            metric_tgt_b=float(row.target_sharpe_eta_1_0),
        )
        audit = zero_cost_near_flat_override(audit, kappa=float(row.kappa))
        rows.append(
            {
                "setting_group": "cost_sweep",
                "setting_name": "locked_eta05_vs_eta1",
                "kappa": float(row.kappa),
                "arm_a": "eta_0.5",
                "arm_b": "eta_1.0",
                "metric_exec_a": float(row.exec_sharpe_eta_0_5),
                "metric_exec_b": float(row.exec_sharpe_eta_1_0),
                "metric_tgt_a": float(row.target_sharpe_eta_0_5),
                "metric_tgt_b": float(row.target_sharpe_eta_1_0),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": audit.disagreement_strength,
                "paper_use": "appendix_support",
                "delta_exec": audit.delta_exec,
                "delta_tgt": audit.delta_tgt,
                "row_label": "Cost Sweep",
            }
        )
    return rows


def _load_multi_universe_rows(path: Path) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        audit = classify_pair(
            metric_exec_a=float(row.median_sharpe_exec_eta05),
            metric_exec_b=float(row.median_sharpe_exec_eta1),
            metric_tgt_a=float(row.median_sharpe_tgt_eta05),
            metric_tgt_b=float(row.median_sharpe_tgt_eta1),
        )
        audit = zero_cost_near_flat_override(audit, kappa=float(row.kappa))
        rows.append(
            {
                "setting_group": "multi_universe",
                "setting_name": str(row.universe),
                "kappa": float(row.kappa),
                "arm_a": "eta_0.5",
                "arm_b": "eta_1.0",
                "metric_exec_a": float(row.median_sharpe_exec_eta05),
                "metric_exec_b": float(row.median_sharpe_exec_eta1),
                "metric_tgt_a": float(row.median_sharpe_tgt_eta05),
                "metric_tgt_b": float(row.median_sharpe_tgt_eta1),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": audit.disagreement_strength,
                "paper_use": "support_only",
                "delta_exec": audit.delta_exec,
                "delta_tgt": audit.delta_tgt,
                "row_label": f"U: {display_universe_name(str(row.universe))}",
            }
        )
    return rows


def _load_architecture_rows(path: Path) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        audit = classify_pair(
            metric_exec_a=float(row.median_sharpe_exec_selected),
            metric_exec_b=float(row.median_sharpe_exec_reference),
            metric_tgt_a=float(row.median_sharpe_tgt_selected),
            metric_tgt_b=float(row.median_sharpe_tgt_reference),
        )
        audit = zero_cost_near_flat_override(audit, kappa=float(row.kappa))
        rows.append(
            {
                "setting_group": "architecture_matrix",
                "setting_name": str(row.architecture),
                "kappa": float(row.kappa),
                "arm_a": str(row.selected_arm),
                "arm_b": str(row.reference_arm),
                "metric_exec_a": float(row.median_sharpe_exec_selected),
                "metric_exec_b": float(row.median_sharpe_exec_reference),
                "metric_tgt_a": float(row.median_sharpe_tgt_selected),
                "metric_tgt_b": float(row.median_sharpe_tgt_reference),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": audit.disagreement_strength,
                "paper_use": _paper_use_for_architecture(str(row.architecture)),
                "delta_exec": audit.delta_exec,
                "delta_tgt": audit.delta_tgt,
                "row_label": f"A: {display_architecture_name(str(row.architecture))}",
            }
        )
    return rows


def _load_split_reference_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        audit = classify_pair(
            metric_exec_a=float(row.exec_sharpe_eta_0_5),
            metric_exec_b=float(row.exec_sharpe_eta_1_0),
            metric_tgt_a=float(row.target_sharpe_eta_0_5),
            metric_tgt_b=float(row.target_sharpe_eta_1_0),
        )
        audit = zero_cost_near_flat_override(audit, kappa=float(row.kappa))
        rows.append(
            {
                "setting_group": "split_reference",
                "setting_name": "split_d_canonical",
                "kappa": float(row.kappa),
                "arm_a": "eta_0.5",
                "arm_b": "eta_1.0",
                "metric_exec_a": float(row.exec_sharpe_eta_0_5),
                "metric_exec_b": float(row.exec_sharpe_eta_1_0),
                "metric_tgt_a": float(row.target_sharpe_eta_0_5),
                "metric_tgt_b": float(row.target_sharpe_eta_1_0),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": audit.disagreement_strength,
                "paper_use": "canonical_reference_optional",
                "delta_exec": audit.delta_exec,
                "delta_tgt": audit.delta_tgt,
                "row_label": "Split D Ref",
            }
        )
    return rows


def _sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    group_order = {
        "cost_sweep": 0,
        "split_reference": 1,
        "multi_universe": 2,
        "architecture_matrix": 3,
    }
    setting_order = {
        "locked_eta05_vs_eta1": 0,
        "split_d_canonical": 0,
        "u27_current": 0,
        "u27_alt_largecap": 1,
        "u27_sector_balanced": 2,
        "u27_random_seed17": 3,
        "arch_rl_selected": 0,
        "arch_rule_eta_fixed": 1,
        "arch_linear_prox": 2,
        "arch_threshold_rebalance": 3,
    }
    out = df.copy()
    out["_group_order"] = out["setting_group"].map(group_order).fillna(99)
    out["_setting_order"] = out["setting_name"].map(setting_order).fillna(99)
    out = out.sort_values(["_group_order", "_setting_order", "kappa"], key=lambda s: s.map(kappa_sort_key) if s.name == "kappa" else s)
    return out.drop(columns=["_group_order", "_setting_order"]).reset_index(drop=True)


def _write_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _write_heatmap(df: pd.DataFrame, output_fig: Path) -> None:
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    row_order = [
        "Cost Sweep",
        "Split D Ref",
        "U: Current",
        "U: Alt-LargeCap",
        "U: Sector-Balanced",
        "A: RL-Selected",
        "A: Rule-EtaFixed",
        "A: Linear-Prox",
    ]
    plot_df = df[df["row_label"].isin(row_order)].copy()
    kappas = sorted(pd.unique(plot_df["kappa"]).tolist(), key=kappa_sort_key)
    matrix = np.full((len(row_order), len(kappas)), np.nan, dtype=np.float64)
    for i, row_label in enumerate(row_order):
        for j, kappa in enumerate(kappas):
            match = plot_df[(plot_df["row_label"] == row_label) & np.isclose(plot_df["kappa"], float(kappa), atol=1e-15)]
            if not match.empty:
                matrix[i, j] = float(match.iloc[0]["disagreement_strength"])

    cmap = ListedColormap(["#e5e7eb", "#fef3c7", "#f59e0b", "#dc2626"])
    cmap.set_bad(color="#ffffff")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(kappas)))
    ax.set_xticklabels([kappa_label(value) for value in kappas], rotation=0)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel(r"$\kappa$")
    ax.set_title("Target-vs-Executed Disagreement Strength")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["0 none", "1 damped", "2 rank", "3 sign"])

    ax.set_xticks(np.arange(-0.5, len(kappas), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="#d1d5db", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(output_fig, bbox_inches="tight")
    plt.close(fig)


def _write_note(df: pd.DataFrame, output_note: Path) -> None:
    output_note.parent.mkdir(parents=True, exist_ok=True)
    counts = df["disagreement_type"].value_counts().to_dict()
    strong_rows = int((df["disagreement_strength"] >= 2).sum())
    cost_df = df[df["setting_group"] == "cost_sweep"].sort_values("kappa")
    universe_df = df[(df["setting_group"] == "multi_universe") & (df["kappa"] > 0.0)]
    arch_df = df[df["setting_group"] == "architecture_matrix"]
    split_df = df[df["setting_group"] == "split_reference"].sort_values("kappa")

    cost_sign_flips = cost_df[cost_df["disagreement_type"] == "sign_flip"]
    cost_rank = cost_df[cost_df["disagreement_type"] == "ranking_mismatch"]
    cost_magnitude = cost_df[cost_df["disagreement_type"] == "magnitude_only"]
    universe_sign_flips = universe_df[universe_df["disagreement_type"] == "sign_flip"]
    universe_rank = universe_df[universe_df["disagreement_type"] == "ranking_mismatch"]
    arch_rank = arch_df[
        (arch_df["setting_name"].isin(["arch_rl_selected", "arch_rule_eta_fixed"]))
        & (arch_df["disagreement_type"] == "ranking_mismatch")
    ]
    linear_none = arch_df[(arch_df["setting_name"] == "arch_linear_prox") & (arch_df["disagreement_type"] == "none")]

    text = f"""# Target-vs-Executed Summary

This note aggregates the available support settings into one master target-versus-executed audit. The wording should stay narrow. The purpose is not to claim that every auxiliary architecture or every support setting produces the same disagreement strength. The purpose is to show, in evidence rather than in wording alone, where executed-path evaluation and target-based evaluation answer different questions under constrained execution.

The master audit currently contains `{len(df)}` rows. The conservative disagreement counts are `{counts.get("none", 0)}` rows with `none`, `{counts.get("magnitude_only", 0)}` with `magnitude_only`, `{counts.get("ranking_mismatch", 0)}` with `ranking_mismatch`, and `{counts.get("sign_flip", 0)}` with `sign_flip`. Using the package's conservative tie rule, `{strong_rows}` rows have disagreement strength at least `2`, meaning either a ranking mismatch or a sign flip.

The cost-sweep rows remain the clearest single block of evidence. Positive-cost disagreement is present across the sweep, but it is not equally strong at every cost level. Most of the lower and middle positive-cost rows are classified as `ranking_mismatch`, meaning the executed-path comparison favors `eta=0.5` while the target-based comparison is too damped to preserve that ordering. The highest-cost sweep row is slightly different again: it is classified as `magnitude_only`, so the target view still points in the same direction there but shrinks the executed-path gap materially. Across the sweep as a whole, the audit records `{len(cost_sign_flips)}` sign-flip row(s), `{len(cost_rank)}` ranking-mismatch row(s), and `{len(cost_magnitude)}` magnitude-only row(s).

The multi-universe package also supports the same narrow reading. Across the positive-cost universe rows, the audit records `{len(universe_rank)}` ranking-mismatch row(s) and `{len(universe_sign_flips)}` sign-flip row(s). This means the evaluation-object discrepancy is not confined to the current U27 basket, even though one zero-cost alternative universe remains mixed and should still be described that way.

The architecture package is mixed by design and should be written that way. The RL-source rows still produce disagreement, with `{len(arch_rank)}` ranking-mismatch row(s) under the conservative audit, while the linear/prox support arm contributes `{len(linear_none)}` row(s) with `none` because target and executed metrics are effectively identical inside that family. That mixed outcome is informative: it supports executed-path primacy for architectures where the realized path diverges from the proposed target, but it does not justify claiming universal target-versus-executed disagreement across all decision-layer designs.

The optional canonical split reference remains aligned with the original main diagnostic. In the available split-D reference rows, the zero-cost row is near-flat, while the positive-cost rows continue to show an executed-path advantage that target-based evaluation does not preserve at the same strength. This reference block is useful as a consistency anchor, not as a new main empirical centerpiece.

The safe paper-facing reading is therefore narrow. Executed-path evaluation remains the primary object in the settings where friction and execution constraints separate realized portfolios from proposed targets. The audit is strongest in the cost sweep, repeats across the fixed-universe package, and stays visible in the RL-source architecture rows. It is weaker or absent in some support architectures, so the paper should explicitly say that the disagreement is setting-dependent rather than universal.
"""
    output_note.write_text(text)


def main() -> int:
    args = parse_args()
    cost_sweep_csv = Path(args.cost_sweep_csv).resolve()
    multi_universe_csv = Path(args.multi_universe_csv).resolve()
    architecture_csv = Path(args.architecture_csv).resolve()
    split_reference_csv = Path(args.split_reference_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_note = Path(args.output_note).resolve()
    output_fig = Path(args.output_fig).resolve()

    rows: list[dict[str, object]] = []
    rows.extend(_load_cost_sweep_rows(cost_sweep_csv))
    rows.extend(_load_multi_universe_rows(multi_universe_csv))
    rows.extend(_load_architecture_rows(architecture_csv))
    rows.extend(_load_split_reference_rows(split_reference_csv))
    if not rows:
        raise RuntimeError("No audit rows were produced.")

    audit_df = _sort_rows(pd.DataFrame(rows))
    _write_csv(audit_df, output_csv)
    _write_note(audit_df, output_note)
    _write_heatmap(audit_df, output_fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
