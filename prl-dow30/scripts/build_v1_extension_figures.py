#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROLLING_KAPPA_STYLE = {
    0.0005: {"color": "#c55a11", "label": r"$\kappa=5\times10^{-4}$"},
    0.001: {"color": "#5b8a3c", "label": r"$\kappa=10^{-3}$"},
}

BENEFIT_STYLE = {
    "global_selected": {"color": "#1f4e79", "marker": "o", "label": r"Global selected $\eta=0.5$"},
    "per_kappa_selected": {"color": "#7a3e9d", "marker": "s", "label": r"Per-$\kappa$ qualifying selector"},
    "best_interior": {"color": "#c55a11", "marker": "D", "label": r"Best interior $\eta<1$"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing extension figures for rolling robustness and dense kappa benefit.")
    parser.add_argument("--rolling-experiment-root", required=True, help="Rolling-origin experiment root.")
    parser.add_argument("--rolling-frontier-csv", required=True, help="Rolling frontier split-kappa summary CSV.")
    parser.add_argument("--kappa-summary-csv", required=True, help="Dense-kappa summary CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to write figure outputs.")
    parser.add_argument("--legacy-dir", default="", help="Optional directory to also copy paper-facing filenames into.")
    return parser.parse_args()


def _kappa_label(kappa: float) -> str:
    if np.isclose(kappa, 0.0):
        return r"$0$"
    if np.isclose(kappa, 2e-4):
        return r"$2\times10^{-4}$"
    if np.isclose(kappa, 5e-4):
        return r"$5\times10^{-4}$"
    if np.isclose(kappa, 1e-3):
        return r"$10^{-3}$"
    if np.isclose(kappa, 2e-3):
        return r"$2\times10^{-3}$"
    return rf"${kappa:g}$"


def _eta_label(eta: float) -> str:
    return f"{eta:g}"


def _rolling_aggregate_root(split_id: str, rolling_root: Path) -> Path:
    if split_id == "split_c":
        split_dir = rolling_root / "splits" / "split_c_reference"
    else:
        split_dir = rolling_root / "splits" / split_id
    return split_dir / "final_eta_full_grid"


def _build_rolling_frontier_figure(rolling_root: Path, frontier_csv: Path, output_path: Path) -> None:
    frontier = pd.read_csv(frontier_csv)
    frontier["kappa"] = pd.to_numeric(frontier["kappa"], errors="coerce")
    frontier["selected_eta"] = pd.to_numeric(frontier["selected_eta"], errors="coerce")
    frontier["best_interior_eta"] = pd.to_numeric(frontier["best_interior_eta"], errors="coerce")

    split_order = ["split_a", "split_b", "split_c"]
    split_titles = {"split_a": "Split A", "split_b": "Split B", "split_c": "Split C"}

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), sharey=False, constrained_layout=True)
    selected_legend = False
    best_legend = False

    for ax, split_id in zip(axes, split_order, strict=False):
        split_rows = frontier[(frontier["split_id"] == split_id) & (frontier["kappa"] > 0)].copy()
        agg_root = _rolling_aggregate_root(split_id, rolling_root)
        aggregate = pd.read_csv(agg_root / "aggregate.csv")
        aggregate["kappa"] = pd.to_numeric(aggregate["kappa"], errors="coerce")
        aggregate["eta"] = pd.to_numeric(aggregate["eta"], errors="coerce")
        aggregate["median_sharpe"] = pd.to_numeric(aggregate["median_sharpe"], errors="coerce")
        aggregate["median_turnover_exec"] = pd.to_numeric(aggregate["median_turnover_exec"], errors="coerce")
        aggregate["iqr_sharpe"] = pd.to_numeric(aggregate["iqr_sharpe"], errors="coerce")

        selected_eta = float(split_rows["selected_eta"].iloc[0])
        for kappa, grp in aggregate[aggregate["kappa"] > 0].groupby("kappa", sort=True):
            style = ROLLING_KAPPA_STYLE.get(float(kappa), {"color": "#444444", "label": rf"$\kappa={kappa:g}$"})
            grp = grp.sort_values("median_turnover_exec", ascending=False)
            ax.plot(
                grp["median_turnover_exec"],
                grp["median_sharpe"],
                color=style["color"],
                marker="o",
                markersize=4.8,
                linewidth=1.5,
                label=style["label"],
                alpha=0.95,
            )

            selected_rows = grp[np.isclose(grp["eta"], selected_eta, atol=1e-12)]
            if not selected_rows.empty:
                ax.scatter(
                    selected_rows["median_turnover_exec"],
                    selected_rows["median_sharpe"],
                    s=120,
                    facecolors=style["color"],
                    edgecolors="#111111",
                    linewidths=1.4,
                    zorder=6,
                    label="Locked selected point" if not selected_legend else None,
                )
                selected_legend = True

            best_eta = float(split_rows.loc[np.isclose(split_rows["kappa"], float(kappa), atol=1e-15), "best_interior_eta"].iloc[0])
            best_rows = grp[np.isclose(grp["eta"], best_eta, atol=1e-12)]
            if not best_rows.empty:
                ax.scatter(
                    best_rows["median_turnover_exec"],
                    best_rows["median_sharpe"],
                    s=110,
                    facecolors="white",
                    edgecolors=style["color"],
                    marker="D",
                    linewidths=1.6,
                    zorder=7,
                    label=r"Best interior $\eta<1$" if not best_legend else None,
                )
                best_legend = True

        ax.set_title(split_titles[split_id])
        ax.set_xlabel("Held-out median executed turnover")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.text(
            0.03,
            0.03,
            rf"selected $\eta={_eta_label(selected_eta)}$",
            transform=ax.transAxes,
            fontsize=9,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    axes[0].set_ylabel("Held-out median net Sharpe")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=10)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_kappa_benefit_curve(summary_csv: Path, output_path: Path) -> None:
    summary = pd.read_csv(summary_csv)
    summary["kappa"] = pd.to_numeric(summary["kappa"], errors="coerce")
    summary = summary[summary["kappa"] > 0].sort_values("kappa").reset_index(drop=True)
    x = np.arange(len(summary))
    xticklabels = [_kappa_label(float(k)) for k in summary["kappa"]]

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True, constrained_layout=True)

    top = axes[0]
    bottom = axes[1]
    top.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    bottom.axhline(0.0, color="#888888", linewidth=1.0, linestyle=":")

    series_map = {
        "global_selected": ("global_selected_final_median_delta_sharpe", "global_selected_final_median_delta_turnover_exec"),
        "per_kappa_selected": ("per_kappa_final_median_delta_sharpe", "per_kappa_final_median_delta_turnover_exec"),
        "best_interior": ("best_interior_final_median_delta_sharpe", "best_interior_final_median_delta_turnover_exec"),
    }

    for key, (sharpe_col, turnover_col) in series_map.items():
        style = BENEFIT_STYLE[key]
        top.plot(
            x,
            summary[sharpe_col],
            color=style["color"],
            marker=style["marker"],
            markersize=6.5,
            linewidth=2.0,
            label=style["label"],
        )
        bottom.plot(
            x,
            summary[turnover_col],
            color=style["color"],
            marker=style["marker"],
            markersize=6.5,
            linewidth=2.0,
            label=style["label"],
        )

    for idx, row in summary.iterrows():
        top.annotate(
            rf"$\eta={_eta_label(float(row['per_kappa_selected_eta']))}$",
            (x[idx], row["per_kappa_final_median_delta_sharpe"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            color=BENEFIT_STYLE["per_kappa_selected"]["color"],
        )

    top.set_ylabel(r"Median $\Delta$ net Sharpe vs $\eta=1.0$")
    top.set_title("Dense canonical friction grid: benefit steepens with kappa")
    top.grid(alpha=0.2, linewidth=0.5)
    top.legend(frameon=False, loc="upper left")

    bottom.set_ylabel(r"Median $\Delta \overline{\mathrm{TO}}_{\mathrm{exec}}$ vs $\eta=1.0$")
    bottom.set_xlabel(r"Transaction cost $\kappa$")
    bottom.grid(alpha=0.2, linewidth=0.5)
    bottom.set_xticks(x, xticklabels)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _copy_legacy(output_path: Path, legacy_dir: Path, legacy_name: str) -> None:
    legacy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, legacy_dir / legacy_name)


def main() -> None:
    args = parse_args()
    rolling_root = Path(args.rolling_experiment_root)
    frontier_csv = Path(args.rolling_frontier_csv)
    kappa_summary_csv = Path(args.kappa_summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = Path(args.legacy_dir) if args.legacy_dir else None

    fig_rolling = output_dir / "fig_rolling_frontier_robustness.png"
    fig_kappa = output_dir / "fig_kappa_benefit_curve.png"

    _build_rolling_frontier_figure(rolling_root, frontier_csv, fig_rolling)
    _build_kappa_benefit_curve(kappa_summary_csv, fig_kappa)

    manifest = {
        "rolling_experiment_root": str(rolling_root.resolve()),
        "rolling_frontier_csv": str(frontier_csv.resolve()),
        "kappa_summary_csv": str(kappa_summary_csv.resolve()),
        "fig_rolling_frontier_robustness": str(fig_rolling.resolve()),
        "fig_kappa_benefit_curve": str(fig_kappa.resolve()),
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if legacy_dir is not None:
        _copy_legacy(fig_rolling, legacy_dir, "fig_rolling_frontier_robustness.png")
        _copy_legacy(fig_kappa, legacy_dir, "fig_kappa_benefit_curve.png")
        _copy_legacy(output_dir / "figure_manifest.json", legacy_dir, "figure_manifest_v1_extensions.json")

    print(f"WROTE_FIGURE={fig_rolling}")
    print(f"WROTE_FIGURE={fig_kappa}")


if __name__ == "__main__":
    main()
