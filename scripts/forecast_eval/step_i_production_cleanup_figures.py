from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper" / "forecasting_workshop"
PAPER_RESULTS_DIR = PAPER_DIR / "results"
PAPER_FIGURES_DIR = PAPER_DIR / "assets" / "figures"
SOURCE_CLEAN = ROOT / "outputs" / "extensions" / "epsilon_tie_audit_20260504" / "source_50b7481_clean"
SOURCE_SCRIPT_DIR = SOURCE_CLEAN / "scripts" / "forecast_eval"
for candidate in (str(SOURCE_SCRIPT_DIR), str(SOURCE_CLEAN)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from build_same_interface_rank_summary import build_domain_rank_summary  # noqa: E402


EVENT_MICRO_RAW = ROOT / "outputs" / "extensions" / "epsilon_tie_audit_20260504" / "event_micro_tau055_seed100" / "q2_diff_forecasts_same_interface.csv"
SYNTHETIC_RAW = ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q2_diff_forecasts_same_interface.csv"
HYSTERESIS_CONFIG = SOURCE_CLEAN / "configs" / "event_micro_revision_round_20260423" / "hardening" / "event_micro_hysteresis_tau055_delta005_seed100.yaml"
RUN_EVENT_MICRO = SOURCE_SCRIPT_DIR / "run_event_micro.py"


EXPECTED_A1 = [
    ("0.00", "Reactive sharp", "Reactive sharp", "0.62", "0.003"),
    ("0.05", "Reactive sharp", "Reactive sharp", "0.60", "0.002"),
    ("0.10", "Reactive sharp", "Reactive sharp", "0.58", "0.002"),
    ("0.25", "Reactive sharp", "Calibrated baseline", "0.51", "0.003"),
    ("0.50", "Reactive sharp", "Calibrated baseline", "0.31", "0.011"),
    ("1.00", "Reactive sharp", "Lagged smoother", "0.01", "0.057"),
]


def apply_paper_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.18,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Step I production-cleanup figures.")
    parser.add_argument("--output-dir", required=True, help="Step I isolated output directory.")
    return parser.parse_args()


def read_event_micro_table() -> pd.DataFrame:
    tex = (PAPER_RESULTS_DIR / "table_q2_selection_drift_event_micro.tex").read_text()
    rows: list[tuple[str, str, str, str, str]] = []
    for line in tex.splitlines():
        stripped = line.strip()
        if "&" not in stripped or "\\\\" not in stripped:
            continue
        if not re.match(r"^(0|1)\.\d{2}\s*&", stripped):
            continue
        cells = [cell.strip().replace("\\", "").strip() for cell in stripped.split("&")]
        if len(cells) == 5:
            rows.append(tuple(cells))  # type: ignore[arg-type]
    if rows != EXPECTED_A1:
        raise RuntimeError(f"A1 source table mismatch: {rows!r}")
    return pd.DataFrame(rows, columns=["friction", "forecast", "deployed", "agreement", "gap"])


def build_a1_compact(table: pd.DataFrame, output_path: Path) -> None:
    friction = table["friction"].astype(float).to_numpy()
    disagreement = 1.0 - table["agreement"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(3.1, 2.25), constrained_layout=True)
    color = "#c44e52"
    ax.plot(
        friction,
        disagreement,
        color=color,
        marker="o",
        markersize=4,
        markeredgewidth=0.6,
        markeredgecolor=color,
        linewidth=1.4,
    )
    ax.set_xlabel("Friction")
    ax.set_ylabel("Disagreement rate")
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def ensure_hysteresis_raw(output_dir: Path) -> Path:
    hysteresis_dir = output_dir / "hysteresis_generated"
    raw_path = hysteresis_dir / "q2_diff_forecasts_same_interface.csv"
    if raw_path.exists():
        return raw_path
    hysteresis_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(RUN_EVENT_MICRO),
            "--config",
            str(HYSTERESIS_CONFIG),
            "--output-dir",
            str(hysteresis_dir),
            "--skip-summary-refresh",
        ],
        cwd=str(SOURCE_CLEAN),
        check=True,
    )
    if not raw_path.exists():
        raise RuntimeError("Hysteresis source regeneration did not produce q2_diff_forecasts_same_interface.csv")
    return raw_path


def domain_outputs(raw_path: Path, *, domain: str, interface_id: str) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(raw_path)
    outputs, _ = build_domain_rank_summary(raw, domain=domain, expected_interface_id=interface_id)
    return outputs


def exact_binom_lows_highs(summary_df: pd.DataFrame) -> tuple[list[float], list[float]]:
    lows: list[float] = []
    highs: list[float] = []
    for row in summary_df.itertuples(index=False):
        ci = binomtest(int(row.deployed_suboptimal_seed_count), int(row.n_seeds)).proportion_ci(
            confidence_level=0.95,
            method="exact",
        )
        lows.append(float(ci.low))
        highs.append(float(ci.high))
    return lows, highs


def check_counts(label: str, summary: pd.DataFrame, expected: dict[float, int]) -> None:
    for friction, count in expected.items():
        got = int(summary.loc[np.isclose(summary["friction_level"], friction), "deployed_suboptimal_seed_count"].iloc[0])
        if got != count:
            raise RuntimeError(f"{label} friction {friction:.2f} count mismatch: expected {count}, got {got}")


def build_a2_interface_figure(fixed_outputs: dict[str, pd.DataFrame], hysteresis_outputs: dict[str, pd.DataFrame], output_path: Path) -> None:
    variants = [
        ("Fixed threshold", "#d62728", fixed_outputs),
        ("Hysteresis threshold", "#2ca02c", hysteresis_outputs),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.2), constrained_layout=True, sharex=True)
    axes[0, 0].set_title("Flip rate")
    axes[0, 1].set_title("Deployed-suboptimal share")
    for row_idx, (row_label, color, outputs) in enumerate(variants):
        rank_df = outputs["rank_correlation_by_friction"].sort_values("friction_level", kind="mergesort")
        summary_df = outputs["selection_summary_by_friction"].sort_values("friction_level", kind="mergesort")

        x = rank_df["friction_level"].to_numpy(dtype=float)
        mean_flip = rank_df["mean_flip_rate"].to_numpy(dtype=float)
        err = 1.96 * rank_df["stderr_flip_rate"].to_numpy(dtype=float)
        axes[row_idx, 0].fill_between(
            x,
            np.clip(mean_flip - err, 0.0, 1.0),
            np.clip(mean_flip + err, 0.0, 1.0),
            color=color,
            alpha=0.12,
            zorder=1,
        )
        axes[row_idx, 0].plot(
            x,
            mean_flip,
            color=color,
            marker="o",
            markersize=4,
            markeredgewidth=0.6,
            markeredgecolor=color,
            linewidth=1.4,
            zorder=2,
        )
        axes[row_idx, 0].set_ylabel("Flip rate")
        axes[row_idx, 0].grid(True)
        axes[row_idx, 0].annotate(row_label, xy=(-0.34, 0.5), xycoords="axes fraction", ha="right", va="center", rotation=90, fontsize=8)

        x2 = summary_df["friction_level"].to_numpy(dtype=float)
        y2 = summary_df["disagreement_rate"].to_numpy(dtype=float)
        lows, highs = exact_binom_lows_highs(summary_df)
        axes[row_idx, 1].fill_between(x2, lows, highs, color=color, alpha=0.12, zorder=1)
        axes[row_idx, 1].plot(
            x2,
            y2,
            color=color,
            marker="o",
            markersize=4,
            markeredgewidth=0.6,
            markeredgecolor=color,
            linewidth=1.4,
            zorder=2,
        )
        axes[row_idx, 1].set_ylabel("")
        axes[row_idx, 1].set_ylim(-0.02, 1.02)
        axes[row_idx, 1].grid(True)
    for ax in axes[1, :]:
        ax.set_xlabel("Friction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_a4_uncertainty_figure(synthetic_outputs: dict[str, pd.DataFrame], event_outputs: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35), constrained_layout=False)
    handles = []
    labels = []
    synthetic_zero_width_band_points = 0
    for label, color, outputs in [
        ("Synthetic", "#1f77b4", synthetic_outputs),
        ("Event-micro", "#d62728", event_outputs),
    ]:
        rank_df = outputs["rank_correlation_by_friction"].sort_values("friction_level", kind="mergesort")
        summary_df = outputs["selection_summary_by_friction"].sort_values("friction_level", kind="mergesort")

        x = rank_df["friction_level"].to_numpy(dtype=float)
        mean_flip = rank_df["mean_flip_rate"].to_numpy(dtype=float)
        err = 1.96 * rank_df["stderr_flip_rate"].to_numpy(dtype=float)
        flip_low = np.clip(mean_flip - err, 0.0, 1.0)
        flip_high = np.clip(mean_flip + err, 0.0, 1.0)
        if label == "Synthetic":
            synthetic_zero_width_band_points += int(np.isclose(flip_low, flip_high).sum())
        axes[0].fill_between(x, flip_low, flip_high, color=color, alpha=0.14, zorder=1)
        line = axes[0].plot(
            x,
            mean_flip,
            color=color,
            marker="o",
            markersize=4,
            markeredgewidth=0.6,
            markeredgecolor=color,
            linewidth=1.4,
            label=label,
            zorder=2,
        )[0]
        handles.append(line)
        labels.append(label)

        x2 = summary_df["friction_level"].to_numpy(dtype=float)
        y2 = summary_df["disagreement_rate"].to_numpy(dtype=float)
        lows, highs = exact_binom_lows_highs(summary_df)
        if label == "Synthetic":
            synthetic_zero_width_band_points += int(np.isclose(lows, highs).sum())
        axes[1].fill_between(x2, lows, highs, color=color, alpha=0.14, zorder=1)
        axes[1].plot(
            x2,
            y2,
            color=color,
            marker="o",
            markersize=4,
            markeredgewidth=0.6,
            markeredgecolor=color,
            linewidth=1.4,
            label=label,
            zorder=2,
        )

    axes[0].set_xlabel("Friction")
    axes[0].set_ylabel("Flip rate")
    axes[0].grid(True)
    axes[1].set_xlabel("Friction")
    axes[1].set_ylabel("Deployed-suboptimal share")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True)
    legend = fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.subplots_adjust(top=0.84, wspace=0.34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=(legend,), pad_inches=0.04)
    plt.close(fig)
    return synthetic_zero_width_band_points


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    source_table = read_event_micro_table()
    build_a1_compact(source_table, PAPER_FIGURES_DIR / "fig_q2_event_micro_support_compact.pdf")

    fixed_outputs = domain_outputs(EVENT_MICRO_RAW, domain="event_micro", interface_id="fixed_threshold")
    hysteresis_raw = ensure_hysteresis_raw(output_dir)
    hysteresis_outputs = domain_outputs(hysteresis_raw, domain="event_micro", interface_id="hysteresis_threshold")
    synthetic_outputs = domain_outputs(SYNTHETIC_RAW, domain="synthetic", interface_id="tempered")

    check_counts("Fixed threshold", fixed_outputs["selection_summary_by_friction"], {0.50: 69, 1.00: 99})
    check_counts("Hysteresis threshold", hysteresis_outputs["selection_summary_by_friction"], {0.50: 55, 1.00: 82})
    check_counts("Event-micro", fixed_outputs["selection_summary_by_friction"], {0.50: 69, 1.00: 99})

    build_a2_interface_figure(
        fixed_outputs,
        hysteresis_outputs,
        PAPER_FIGURES_DIR / "fig_event_micro_interface_uncertainty_appendix.pdf",
    )
    synthetic_zero_width_band_points = build_a4_uncertainty_figure(
        synthetic_outputs,
        fixed_outputs,
        PAPER_FIGURES_DIR / "fig_q2_uncertainty_appendix.pdf",
    )

    figure_report = output_dir / "figure_generation_report.md"
    figure_report.write_text(
        "\n".join(
            [
                "# Step I Figure Generation Report",
                "",
                "- Generated `fig_q2_event_micro_support_compact.pdf`.",
                "- Generated `fig_event_micro_interface_uncertainty_appendix.pdf`.",
                "- Generated `fig_q2_uncertainty_appendix.pdf`.",
                "- Cross-checks passed for canonical event-micro A1 values and A2/A4 deployed-suboptimal counts.",
                "- Applied unified paper rcParams across regenerated figures.",
                "- A4 uses symmetric band code paths for Synthetic and Event-micro, with a shared legend above the panels.",
                f"- Synthetic zero-width band point count in A4: {synthetic_zero_width_band_points}.",
                f"- Hysteresis source generated under `{output_dir / 'hysteresis_generated'}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
