#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = (
    ROOT
    / "repro"
    / "rebuilds"
    / "paper_rebuild_20260324T065755Z"
    / "paper_pack"
    / "diagnostics"
    / "diagnostic_selected_eta_v2.csv"
)
DEFAULT_OUTPUT_FIG = ROOT / "paper" / "forecasting_workshop" / "assets" / "figures" / "fig_accounting_gap.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the main accounting diagnostic figure from locked selected-point diagnostics.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Diagnostic CSV for the selected-point accounting figure.")
    parser.add_argument("--output-fig", default=str(DEFAULT_OUTPUT_FIG), help="Destination PDF path.")
    return parser.parse_args()


def _kappa_label(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    if np.isclose(value, 5e-4):
        return "5e-4"
    if np.isclose(value, 1e-3):
        return "1e-3"
    return f"{value:g}"


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_fig = Path(args.output_fig).resolve()

    df = pd.read_csv(input_csv).sort_values("kappa").reset_index(drop=True)
    kappas = df["kappa"].to_numpy(dtype=np.float64)
    x = np.arange(len(kappas), dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.8), sharex=True)

    exec_turnover = df["median_turnover_exec"].to_numpy(dtype=np.float64)
    target_turnover = df["median_turnover_target"].to_numpy(dtype=np.float64)
    axes[0].plot(x, exec_turnover, marker="o", linewidth=2.1, color="#0f766e", label="Executed")
    axes[0].plot(x, target_turnover, marker="o", linewidth=1.8, linestyle="--", color="#b45309", label="Target")
    axes[0].set_ylabel("Turnover")
    axes[0].set_title("(a) Turnover")
    axes[0].legend(frameon=False, loc="upper right")

    tracking = df["median_tracking_error_l2"].to_numpy(dtype=np.float64)
    axes[1].plot(x, tracking, marker="o", linewidth=2.1, color="#1d4ed8")
    axes[1].set_ylabel(r"$L_2$ gap")
    axes[1].set_title("(b) Tracking discrepancy")

    final_gap = df["median_final_equity_gap"].to_numpy(dtype=np.float64)
    axes[2].plot(x, final_gap, marker="o", linewidth=2.1, color="#7c3aed")
    axes[2].set_ylabel("Path gap")
    axes[2].set_title("(c) Final path gap")
    axes[2].set_xlabel(r"$\kappa$")

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_kappa_label(value) for value in kappas])

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(output_fig, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
