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
    step2_rank_corr: pd.DataFrame,
    event_rank_corr: pd.DataFrame,
    step2_selection: pd.DataFrame,
    event_selection: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)

    axes[0].plot(
        step2_rank_corr["friction_level"],
        step2_rank_corr["mean_flip_rate"],
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        label="Synthetic",
    )
    axes[0].plot(
        event_rank_corr["friction_level"],
        event_rank_corr["mean_flip_rate"],
        color="#d62728",
        marker="s",
        linewidth=2.0,
        label="Event micro",
    )
    axes[0].set_title("Ranking disagreement")
    axes[0].set_xlabel("Friction")
    axes[0].set_ylabel("Mean flip rate")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[1].plot(
        step2_selection["friction_level"],
        step2_selection["disagreement_rate"],
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        label="Synthetic",
    )
    axes[1].plot(
        event_selection["friction_level"],
        event_selection["disagreement_rate"],
        color="#d62728",
        marker="s",
        linewidth=2.0,
        label="Event micro",
    )
    axes[1].set_title("Deployed-suboptimal share")
    axes[1].set_xlabel("Friction")
    axes[1].set_ylabel("Forecast-selected not deployed-best")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step2_q1 = pd.read_csv(Path(args.step2_q1))
    step4_q1 = pd.read_csv(Path(args.step4_q1))
    step2_rank_corr = _load_rank_corr(Path(args.step2_q2_raw), domain="synthetic", expected_interface_id="tempered")
    event_rank_corr = _load_rank_corr(Path(args.event_micro_q2_raw), domain="event_micro", expected_interface_id="fixed_threshold")
    step2_selection = _load_selection_summary(Path(args.step2_q2_raw), domain="synthetic", expected_interface_id="tempered")
    event_selection = _load_selection_summary(Path(args.event_micro_q2_raw), domain="event_micro", expected_interface_id="fixed_threshold")

    build_q1_figure(step2_q1, step4_q1, output_dir / "fig_q1_results_v2.pdf")
    build_q2_figure(step2_rank_corr, event_rank_corr, step2_selection, event_selection, output_dir / "fig_q2_results_v2.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
