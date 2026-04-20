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
DEFAULT_INPUT_CSV = ROOT / "paper" / "forecasting_workshop" / "generalization" / "cost_sweep_results.csv"
DEFAULT_OUTPUT_FIG = ROOT / "paper" / "forecasting_workshop" / "assets" / "figures" / "fig_cost_sweep_appendix.pdf"
NEAR_FLAT_BAND = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the appendix cost-sweep support figure.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Cost-sweep summary CSV.")
    parser.add_argument("--output-fig", default=str(DEFAULT_OUTPUT_FIG), help="Destination PDF path.")
    return parser.parse_args()


def _kappa_label(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    if np.isclose(value, 1e-4):
        return "1e-4"
    if np.isclose(value, 2e-4):
        return "2e-4"
    if np.isclose(value, 5e-4):
        return "5e-4"
    if np.isclose(value, 1e-3):
        return "1e-3"
    if np.isclose(value, 2e-3):
        return "2e-3"
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

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)

    exec_delta = df["exec_delta_pair_median"].to_numpy(dtype=np.float64)
    tgt_delta = df["target_delta_pair_median"].to_numpy(dtype=np.float64)

    axes[0].axhline(0.0, color="#9ca3af", linewidth=1.0, linestyle="--")
    axes[0].axhspan(-NEAR_FLAT_BAND, NEAR_FLAT_BAND, color="#e5e7eb", alpha=0.7, zorder=0)
    axes[0].plot(x, exec_delta, marker="o", linewidth=2.2, color="#0f766e")
    axes[0].set_ylabel(r"$\Delta$Sharpe")
    axes[0].set_title(r"Executed-path $\Delta$Sharpe")

    axes[1].axhline(0.0, color="#9ca3af", linewidth=1.0, linestyle="--")
    axes[1].axhspan(-NEAR_FLAT_BAND, NEAR_FLAT_BAND, color="#f3f4f6", alpha=0.7, zorder=0)
    axes[1].plot(x, tgt_delta, marker="o", linewidth=2.0, color="#b45309", linestyle="--")
    axes[1].set_ylabel(r"$\Delta$Sharpe")
    axes[1].set_title(r"Target-based $\Delta$Sharpe")
    axes[1].set_xlabel(r"$\kappa$")

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_kappa_label(value) for value in kappas])

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(output_fig, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
