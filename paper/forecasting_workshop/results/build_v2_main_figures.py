#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
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


def build_q2_figure(output_path: Path) -> None:
    latex_source = textwrap.dedent(
        r"""
        \documentclass[varwidth,border=3pt]{standalone}
        \usepackage[T1]{fontenc}
        \usepackage{times}
        \usepackage[table]{xcolor}
        \usepackage{array}
        \renewcommand{\arraystretch}{1.28}
        \setlength{\tabcolsep}{3.5pt}
        \definecolor{agreebg}{HTML}{EDF7E8}
        \definecolor{shiftbg}{HTML}{FBE9DC}
        \definecolor{mixedbg}{HTML}{ECECEC}
        \newcommand{\winnercell}[3]{\cellcolor{#1}{\scriptsize\shortstack[c]{\textbf{#2}\\#3}}}
        \begin{document}
        \sffamily
        \begin{tabular}{@{}>{\raggedright\arraybackslash}m{2.05cm}>{\centering\arraybackslash}m{2.20cm}>{\centering\arraybackslash}m{2.20cm}>{\centering\arraybackslash}m{2.20cm}@{}}
        & \multicolumn{3}{c}{\textbf{Friction}} \\
        \cline{2-4}
        & \textbf{0.00} & \textbf{0.50} & \textbf{1.00} \\
        \shortstack[l]{\textbf{Event-micro}\\{\scriptsize R-sharp}}
          & \winnercell{agreebg}{= R-sharp}{38/100}
          & \winnercell{shiftbg}{$\neq$ Calib.}{69/100}
          & \winnercell{shiftbg}{$\neq$ Smooth}{99/100} \\
        [2pt]
        \shortstack[l]{\textbf{Traffic-Hourly}\\\textbf{Top-k}\\{\scriptsize R-short}}
          & \winnercell{agreebg}{= R-short}{0/100}
          & \winnercell{shiftbg}{$\neq$ Smooth}{100/100}
          & \winnercell{shiftbg}{$\neq$ Smooth}{100/100} \\
        [2pt]
        \shortstack[l]{\textbf{Inventory}\\{\scriptsize S-MLP}}
          & \winnercell{mixedbg}{mixed}{appx.}
          & \winnercell{shiftbg}{$\neq$ MA(7)}{9/10}
          & \winnercell{shiftbg}{$\neq$ MA(7)}{10/10} \\
        \end{tabular}

        \vspace{3pt}

        {\scriptsize $\mathbf{=}$ agree \hspace{1.2em} $\mathbf{\neq}$ mismatch \hspace{1.2em} grey = mixed}
        \end{document}
        """
    ).strip()

    with tempfile.TemporaryDirectory(prefix="q2_winner_matrix_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tex_path = tmp_dir / "fig_q2_winner_inversion_heatmap_v2.tex"
        pdf_path = tmp_dir / "fig_q2_winner_inversion_heatmap_v2.pdf"
        tex_path.write_text(latex_source, encoding="utf-8")

        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(tmp_dir),
            str(tex_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(
                "Failed to build fig_q2_winner_inversion_heatmap_v2.pdf via pdflatex.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf_path, output_path)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step2_q1_path = Path(args.step2_q1)
    step4_q1_path = Path(args.step4_q1)
    q1_output_path = output_dir / "fig_q1_results_v2.pdf"
    if q1_output_path.exists():
        print("Retaining existing Q1 figure asset.")
    elif step2_q1_path.exists() and step4_q1_path.exists():
        step2_q1 = pd.read_csv(step2_q1_path)
        step4_q1 = pd.read_csv(step4_q1_path)
        build_q1_figure(step2_q1, step4_q1, q1_output_path)
    else:
        print("Skipping Q1 figure rebuild because the default Q1 CSV inputs are not available.")

    build_q2_figure(output_dir / "fig_q2_winner_inversion_heatmap_v2.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
