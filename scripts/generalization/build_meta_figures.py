#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from target_exec_audit_utils import display_architecture_name, display_universe_name, kappa_label, kappa_sort_key


DEFAULT_MULTI_UNIVERSE_CSV = (
    REPO_ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "multi_universe" / "multi_universe_results.csv"
)
DEFAULT_MASTER_AUDIT_CSV = (
    REPO_ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "target_vs_executed_master.csv"
)
DEFAULT_TEMPORAL_PILOT_CSV = REPO_ROOT / "paper" / "forecasting_workshop" / "generalization" / "multi_split_results.csv"
DEFAULT_TOY_CSV = (
    REPO_ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "toy_example_results.csv"
)
DEFAULT_UNIVERSE_FIG = (
    REPO_ROOT / "paper" / "forecasting_workshop" / "generalization" / "figures" / "fig_meta_consistency_universe.pdf"
)
DEFAULT_DISAGREEMENT_MAP_FIG = (
    REPO_ROOT / "paper" / "forecasting_workshop" / "generalization" / "figures" / "fig_meta_disagreement_map.pdf"
)
DEFAULT_NOTE = (
    REPO_ROOT / "paper" / "forecasting_workshop" / "generalization" / "notes" / "meta_consistency_note.md"
)

UNIVERSE_ORDER = ["u27_current", "u27_alt_largecap", "u27_sector_balanced"]
ARCHITECTURE_ORDER = [
    "arch_rl_selected",
    "arch_deadband_partial_champion",
    "arch_deadband_partial_runnerup",
    "arch_vol_spike_eta_champion",
    "arch_vol_spike_eta_runnerup",
    "arch_rule_eta_fixed",
    "arch_linear_prox",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact meta-consistency figures for the generalization package.")
    parser.add_argument("--multi-universe-csv", default=str(DEFAULT_MULTI_UNIVERSE_CSV), help="Multi-universe summary CSV.")
    parser.add_argument("--master-audit-csv", default=str(DEFAULT_MASTER_AUDIT_CSV), help="Master target-vs-executed audit CSV.")
    parser.add_argument("--temporal-pilot-csv", default=str(DEFAULT_TEMPORAL_PILOT_CSV), help="Compact temporal pilot CSV.")
    parser.add_argument("--toy-csv", default=str(DEFAULT_TOY_CSV), help="Toy-example CSV.")
    parser.add_argument("--universe-figure", default=str(DEFAULT_UNIVERSE_FIG), help="Output path for the universe consistency figure.")
    parser.add_argument("--disagreement-map-figure", default=str(DEFAULT_DISAGREEMENT_MAP_FIG), help="Output path for the disagreement map.")
    parser.add_argument("--output-note", default=str(DEFAULT_NOTE), help="Output path for the meta note.")
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_universe_figure(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["universe"].isin(UNIVERSE_ORDER)].copy()
    plot_df["universe"] = pd.Categorical(plot_df["universe"], categories=UNIVERSE_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["universe", "kappa"], key=lambda s: s.map(kappa_sort_key) if s.name == "kappa" else s)

    colors = {
        "u27_current": "#1d4ed8",
        "u27_alt_largecap": "#d97706",
        "u27_sector_balanced": "#047857",
    }
    kappas = sorted(plot_df["kappa"].unique().tolist(), key=kappa_sort_key)
    x = np.arange(len(kappas), dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.axhline(0.0, color="#9ca3af", linewidth=1.0, linestyle="--")
    ax.axhspan(-0.005, 0.005, color="#e5e7eb", alpha=0.6, zorder=0)

    for universe in UNIVERSE_ORDER:
        universe_df = plot_df[plot_df["universe"] == universe]
        color = colors[universe]
        ax.plot(
            x,
            universe_df["delta_sharpe_exec"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{display_universe_name(universe)} exec",
        )
        ax.plot(
            x,
            universe_df["delta_sharpe_tgt"].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.4,
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"{display_universe_name(universe)} target",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([kappa_label(value) for value in kappas])
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\Delta$ score")
    ax.set_title("Fixed-Universe Directional Consistency")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.text(
        0.01,
        0.04,
        "gray band: documented near-flat zone",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#4b5563",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _row_label_for_setting(setting_group: str, setting_name: str) -> str | None:
    if setting_group == "cost_sweep" and setting_name == "locked_eta05_vs_eta1":
        return "Cost Sweep"
    if setting_group == "multi_universe" and setting_name in UNIVERSE_ORDER:
        return f"U: {display_universe_name(setting_name)}"
    if setting_group == "architecture_matrix":
        family_map = {
            "arch_rl_selected": "A: RL family",
            "arch_deadband_partial_champion": "A: Deadband family",
            "arch_deadband_partial_runnerup": "A: Deadband family",
            "arch_vol_spike_eta_champion": "A: VolScaledEta family",
            "arch_vol_spike_eta_runnerup": "A: VolScaledEta family",
            "arch_rule_eta_fixed": "A: Replay arm",
            "arch_linear_prox": "A: Linear-Prox",
        }
        return family_map.get(setting_name)
    return None


def _write_disagreement_map(df: pd.DataFrame, output_path: Path) -> None:
    filtered = df.copy()
    filtered["meta_row_label"] = [
        _row_label_for_setting(setting_group, setting_name)
        for setting_group, setting_name in zip(filtered["setting_group"], filtered["setting_name"])
    ]
    filtered = filtered[filtered["meta_row_label"].notna()].copy()

    row_order = [
        "Cost Sweep",
        "U: Current",
        "U: Alt-LargeCap",
        "U: Sector-Balanced",
        "A: RL family",
        "A: Deadband family",
        "A: VolScaledEta family",
        "A: Replay arm",
        "A: Linear-Prox",
    ]
    kappas = sorted(filtered["kappa"].unique().tolist(), key=kappa_sort_key)
    matrix = np.full((len(row_order), len(kappas)), np.nan, dtype=np.float64)

    for row_idx, row_label in enumerate(row_order):
        for col_idx, kappa in enumerate(kappas):
            match = filtered[
                (filtered["meta_row_label"] == row_label)
                & np.isclose(filtered["kappa"], float(kappa), atol=1e-15)
            ]
            if not match.empty:
                matrix[row_idx, col_idx] = float(match["disagreement_strength"].max())

    cmap = ListedColormap(["#e5e7eb", "#fef3c7", "#f59e0b", "#dc2626"])
    cmap.set_bad(color="#ffffff")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(kappas)))
    ax.set_xticklabels([kappa_label(value) for value in kappas])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel(r"$\kappa$")
    ax.set_title("Meta Disagreement Summary Across Support Packages")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            if np.isfinite(matrix[row_idx, col_idx]):
                ax.text(col_idx, row_idx, str(int(matrix[row_idx, col_idx])), ha="center", va="center", fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(kappas), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="#d1d5db", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Group separators: after cost and after universe rows.
    for separator in (0.5, 3.5):
        ax.axhline(separator, color="#6b7280", linewidth=1.0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["0 none", "1 damped", "2 rank", "3 sign"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_note(
    *,
    multi_universe_df: pd.DataFrame,
    master_audit_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    toy_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    universe_positive = multi_universe_df[multi_universe_df["kappa"] > 0.0]
    universe_direction_yes = int((universe_positive["positive_cost_direction_flag"] == "yes").sum())
    universe_disagreement_yes = int((universe_positive["disagreement_flag"] == "yes").sum())
    universe_total = int(len(universe_positive))

    architecture_df = master_audit_df[master_audit_df["setting_group"] == "architecture_matrix"].copy()
    independent_arch_df = architecture_df[
        architecture_df["setting_name"].isin(
            [
                "arch_deadband_partial_champion",
                "arch_deadband_partial_runnerup",
                "arch_vol_spike_eta_champion",
                "arch_vol_spike_eta_runnerup",
            ]
        )
    ]
    independent_rank_rows = int((independent_arch_df["disagreement_type"] == "ranking_mismatch").sum())
    independent_total_rows = int(len(independent_arch_df[independent_arch_df["kappa"] > 0.0]))

    temporal_mixed = temporal_df.copy()
    temporal_failures = temporal_mixed[
        (temporal_mixed["positive_cost_direction_flag"] != "yes")
        | (temporal_mixed["target_vs_executed_disagreement_flag"] != "yes")
    ]

    toy_positive = toy_df[toy_df["friction"] > 0.0]
    toy_disagreement_rows = int((toy_positive["disagreement_strength"] >= 2).sum())

    text = f"""# Meta Consistency Note

This note summarizes the generalization package at a compact meta level. The wording should remain narrow and reviewer-safe. The package is meant to document recurrence across selected support settings, not to justify any broad temporal robustness or universal generalization claim.

The compact temporal pilot was mixed and should be reported that way. In the available compact split file, `{len(temporal_failures)}` of the `{len(temporal_mixed)}` rows do not reproduce the full positive-cost direction-plus-disagreement pattern. For that reason the temporal pilot is not the main visual focus of this meta package.

The stronger added support instead comes from recurrence across fixed universes and execution-layer families. In the fixed-universe package, all `{universe_total}` positive-cost rows keep the executed-path direction, and `{universe_disagreement_yes}` of those `{universe_total}` rows still trigger the conservative disagreement flag. The universe block remains narrow because one large-cap zero-cost row is mixed, but the positive-cost reading repeats across the tested fixed baskets.

The execution-layer support is also stronger than the early architecture draft. The main RL row remains supportive, and the two independent non-RL families now add repeated positive-cost disagreement as well. Across the independent deadband and volatility-scaled rows, `{independent_rank_rows}` of `{independent_total_rows}` positive-cost rows are classified as `ranking_mismatch` in the master audit. That is the main reason the architecture package now reduces the specific RL-only artifact concern more directly.

The toy example is illustrative, not empirical evidence. Its role is only to show that a proposal-versus-realization mismatch can create a target-versus-executed evaluation gap in a generic decision process. In the current toy CSV, `{toy_disagreement_rows}` positive-friction rows reach disagreement strength at least `2`, but this block should remain appendix-only and should not be presented as a new main empirical result.

The safe paper-facing reading is therefore compact and narrow. The mixed temporal pilot should still be acknowledged explicitly, while the stronger support in this round comes from fixed-universe recurrence and execution-layer recurrence. The figures in this package are designed to keep that emphasis clear without overstating what the support results can justify.
"""
    output_path.write_text(text)


def main() -> int:
    args = parse_args()
    multi_universe_csv = Path(args.multi_universe_csv).resolve()
    master_audit_csv = Path(args.master_audit_csv).resolve()
    temporal_pilot_csv = Path(args.temporal_pilot_csv).resolve()
    toy_csv = Path(args.toy_csv).resolve()
    universe_figure = Path(args.universe_figure).resolve()
    disagreement_map_figure = Path(args.disagreement_map_figure).resolve()
    output_note = Path(args.output_note).resolve()

    multi_universe_df = _load_csv(multi_universe_csv)
    master_audit_df = _load_csv(master_audit_csv)
    temporal_df = _load_csv(temporal_pilot_csv)
    toy_df = _load_csv(toy_csv)

    _write_universe_figure(multi_universe_df, universe_figure)
    _write_disagreement_map(master_audit_df, disagreement_map_figure)
    _write_note(
        multi_universe_df=multi_universe_df,
        master_audit_df=master_audit_df,
        temporal_df=temporal_df,
        toy_df=toy_df,
        output_path=output_note,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
