#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
FORECAST_EVAL_DIR = REPO_ROOT / "scripts" / "forecast_eval"
if str(FORECAST_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(FORECAST_EVAL_DIR))

from build_same_interface_rank_summary import build_domain_rank_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build main Q1/Q2 figures for forecasting workshop v2.")
    parser.add_argument(
        "--step2-q1",
        default=str(REPO_ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q1_gap_by_friction.csv"),
        help="Step 2 synthetic Q1 summary CSV.",
    )
    parser.add_argument(
        "--step4-q1",
        default=str(
            REPO_ROOT
            / "outputs"
            / "forecast_eval"
            / "inventory_step4_seed_stability_locked"
            / "inventory_v2_seed_stability_q1_friction_threshold_summary.csv"
        ),
        help="Step 4 inventory Q1 threshold summary CSV.",
    )
    parser.add_argument(
        "--step2-q2-raw",
        default=str(REPO_ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q2_diff_forecasts_same_interface.csv"),
        help="Step 2 synthetic raw Q2 CSV.",
    )
    parser.add_argument(
        "--event-micro-q2-raw",
        default=str(
            REPO_ROOT
            / "outputs"
            / "extensions"
            / "revision_round_20260423"
            / "new_reruns"
            / "event_micro_hardening"
            / "fixed_threshold_tau055_seed100"
            / "q2_diff_forecasts_same_interface.csv"
        ),
        help="Event-micro raw Q2 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "paper" / "forecasting_workshop" / "assets" / "figures"),
        help="Directory for generated figure PDFs.",
    )
    return parser.parse_args()


def _load_rank_corr(raw_path: Path, *, domain: str, expected_interface_id: str) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_path)
    outputs, _meta = build_domain_rank_summary(raw_df, domain=domain, expected_interface_id=expected_interface_id)
    rank_corr = outputs["rank_correlation_by_friction"].copy()
    return rank_corr.sort_values("friction_level", kind="mergesort").reset_index(drop=True)


def _load_selection_summary(raw_path: Path, *, domain: str, expected_interface_id: str) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_path)
    outputs, _meta = build_domain_rank_summary(raw_df, domain=domain, expected_interface_id=expected_interface_id)
    summary = outputs["selection_summary_by_friction"].copy()
    return summary.sort_values("friction_level", kind="mergesort").reset_index(drop=True)


def build_q1_figure(step2_q1: pd.DataFrame, step4_q1: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)

    axes[0].errorbar(
        step2_q1["friction_level"],
        step2_q1["mean_abs_target_executed_gap"],
        yerr=step2_q1["stderr_abs_target_executed_gap"],
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        capsize=3,
    )
    axes[0].set_title("Synthetic Q1")
    axes[0].set_xlabel("Friction")
    axes[0].set_ylabel("Mean abs. target-executed gap")
    axes[0].grid(alpha=0.25, linewidth=0.6)

    axes[1].plot(
        step4_q1["friction_level"],
        step4_q1["mean_executed_delta_tempered_minus_responsive"],
        color="#d62728",
        marker="o",
        linewidth=2.0,
        label="Tempered - responsive",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[1].set_title("Inventory Q1")
    axes[1].set_xlabel("Friction")
    axes[1].set_ylabel("Realized score delta\n(tempered - responsive)")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_q2_figure(
    output_path: Path,
) -> None:
    domains = [
        "Event-micro",
        "Traffic-Hourly\nTop-k",
        "Inventory",
    ]
    frictions = ["0.00", "0.50", "1.00"]
    winners = [
        [
            ("Reactive sharp", "Reactive sharp"),
            ("Reactive sharp", "Calibrated baseline"),
            ("Reactive sharp", "Lagged smoother"),
        ],
        [
            ("Reactive short", "Reactive short"),
            ("Reactive short", "Lagged smoother"),
            ("Reactive short", "Lagged smoother"),
        ],
        [
            ("Small MLP", "Small MLP"),
            ("Small MLP", "Moving average (7)"),
            ("Small MLP", "Moving average (7)"),
        ],
    ]

    same_color = "#d7f0d0"
    mismatch_color = "#f7d8bf"
    edge_color = "#4c4c4c"
    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)

    for row_idx, row in enumerate(winners):
        for col_idx, (forecast_winner, deployed_winner) in enumerate(row):
            same = forecast_winner == deployed_winner
            rect = plt.Rectangle(
                (col_idx, row_idx),
                1.0,
                1.0,
                facecolor=same_color if same else mismatch_color,
                edgecolor=edge_color,
                linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"F: {forecast_winner}\nD: {deployed_winner}",
                ha="center",
                va="center",
                fontsize=8,
                color="#1f1f1f",
            )

    ax.set_xlim(0, len(frictions))
    ax.set_ylim(0, len(domains))
    ax.invert_yaxis()
    ax.set_xticks([idx + 0.5 for idx in range(len(frictions))], frictions)
    ax.set_yticks([idx + 0.5 for idx in range(len(domains))], domains)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_title("Fixed-interface winner inversion", fontsize=11, pad=18)
    ax.set_xlabel("Friction", fontsize=9)
    ax.xaxis.set_label_position("top")

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=same_color, edgecolor=edge_color, linewidth=1.0, label="Forecast winner = deployed winner"),
        plt.Rectangle((0, 0), 1, 1, facecolor=mismatch_color, edgecolor=edge_color, linewidth=1.0, label="Forecast winner != deployed winner"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step2_q1_path = Path(args.step2_q1)
    step4_q1_path = Path(args.step4_q1)
    if step2_q1_path.exists() and step4_q1_path.exists():
        step2_q1 = pd.read_csv(step2_q1_path)
        step4_q1 = pd.read_csv(step4_q1_path)
        build_q1_figure(step2_q1, step4_q1, output_dir / "fig_q1_results_v2.pdf")
    else:
        print("Skipping Q1 figure rebuild because the default Q1 CSV inputs are not available.")

    build_q2_figure(output_dir / "fig_q2_results_v2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
