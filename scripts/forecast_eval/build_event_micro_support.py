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
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from build_same_interface_rank_summary import build_domain_rank_summary, validate_q2_source, write_summary_outputs  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "outputs" / "forecast_eval" / "event_micro" / "q2_diff_forecasts_same_interface.csv"
DEFAULT_SEED_METRICS = REPO_ROOT / "outputs" / "forecast_eval" / "event_micro" / "seed_level_metrics.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "event_micro"
DEFAULT_PAPER_RESULTS_DIR = REPO_ROOT / "paper" / "forecasting_workshop" / "results"
DEFAULT_PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "forecasting_workshop" / "assets" / "figures"

PAPER_FORECASTER_LABELS = {
    "calibrated_baseline": "Calibrated baseline",
    "reactive_sharp": "Reactive sharp",
    "lagged_smoother": "Lagged smoother",
    "noisy_heuristic": "Noisy heuristic",
}

SUPPORT_READING = (
    "fit-oriented minimal confirmation of ranking drift under switching friction"
)
SCHEMA_NOTE = (
    "In the shared raw schema, `forecast_metric` is stored as `-brier` only to preserve the higher-is-better "
    "ranking convention used by the summary builder; all paper-facing tables and text should still report Brier "
    "in its standard lower-is-better interpretation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-micro appendix support artifacts.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Raw event-micro Q2 CSV.")
    parser.add_argument("--seed-metrics", default=str(DEFAULT_SEED_METRICS), help="Seed-level diagnostics CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for derived event-micro summaries.")
    parser.add_argument("--paper-results-dir", default=str(DEFAULT_PAPER_RESULTS_DIR), help="Paper results directory.")
    parser.add_argument("--paper-figures-dir", default=str(DEFAULT_PAPER_FIGURES_DIR), help="Paper figures directory.")
    return parser.parse_args()


def _paper_label(model_id: str) -> str:
    return PAPER_FORECASTER_LABELS.get(str(model_id), str(model_id))


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _write_table_tex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Friction & Forecast-side winner & Deployed winner & Agreement rate & Mean deployed gap \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.Friction} & {row.Forecast_side_winner} & {row.Deployed_winner} & "
            f"{row.Agreement_rate} & {row.Mean_deployed_gap} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _build_paper_table(selection_summary: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Friction": selection_summary["friction"].map(lambda value: f"{float(value):.2f}"),
            "Forecast_side_winner": selection_summary["forecast_winner_mode"].map(_paper_label),
            "Deployed_winner": selection_summary["deployed_winner_mode"].map(_paper_label),
            "Agreement_rate": selection_summary["agreement_rate"].map(lambda value: f"{float(value):.2f}"),
            "Mean_deployed_gap": selection_summary["mean_deployed_gap"].map(lambda value: _format_float(value, 3)),
        }
    )
    return table


def _build_friction_summary(
    *,
    rank_outputs: dict[str, pd.DataFrame],
    seed_metrics_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    selection_summary = rank_outputs["selection_summary_by_friction"].copy()
    selection_seed = rank_outputs["seed_level_selection_stats"].copy()
    rank_corr = rank_outputs["rank_correlation_by_friction"].copy()

    forecast_share = (
        selection_seed.groupby(["friction_level", "forecast_selected_representative"], as_index=False)["seed"]
        .count()
        .rename(columns={"forecast_selected_representative": "forecast_winner_mode", "seed": "forecast_winner_count"})
    )
    forecast_share = (
        forecast_share.sort_values(["friction_level", "forecast_winner_count", "forecast_winner_mode"], ascending=[True, False, True])
        .drop_duplicates(subset=["friction_level"], keep="first")
        .reset_index(drop=True)
    )
    forecast_share = forecast_share.merge(
        selection_summary[["friction_level", "n_seeds"]],
        on="friction_level",
        how="left",
    )
    forecast_share["forecast_winner_share"] = forecast_share["forecast_winner_count"] / forecast_share["n_seeds"].clip(lower=1)

    deployed_share = (
        selection_seed.groupby(["friction_level", "deployed_selected_representative"], as_index=False)["seed"]
        .count()
        .rename(columns={"deployed_selected_representative": "deployed_winner_mode", "seed": "deployed_winner_count"})
    )
    deployed_share = (
        deployed_share.sort_values(["friction_level", "deployed_winner_count", "deployed_winner_mode"], ascending=[True, False, True])
        .drop_duplicates(subset=["friction_level"], keep="first")
        .reset_index(drop=True)
    )
    deployed_share = deployed_share.merge(
        selection_summary[["friction_level", "n_seeds"]],
        on="friction_level",
        how="left",
    )
    deployed_share["deployed_winner_share"] = deployed_share["deployed_winner_count"] / deployed_share["n_seeds"].clip(lower=1)

    merged = (
        selection_summary[
            [
                "friction_level",
                "agreement_rate",
                "mean_deployed_gap_of_forecast_selected",
                "median_deployed_gap_of_forecast_selected",
            ]
        ]
        .merge(
            rank_corr[["friction_level", "mean_flip_rate", "mean_spearman_rho"]],
            on="friction_level",
            how="left",
        )
        .merge(
            forecast_share[["friction_level", "forecast_winner_mode", "forecast_winner_share"]],
            on="friction_level",
            how="left",
        )
        .merge(
            deployed_share[["friction_level", "deployed_winner_mode", "deployed_winner_share"]],
            on="friction_level",
            how="left",
        )
        .rename(
            columns={
                "friction_level": "friction",
                "mean_deployed_gap_of_forecast_selected": "mean_deployed_gap",
                "median_deployed_gap_of_forecast_selected": "median_deployed_gap",
            }
        )
        .sort_values("friction")
        .reset_index(drop=True)
    )

    switch_summary = (
        seed_metrics_df.groupby(["friction", "model"], as_index=False)
        .agg(mean_n_switches=("n_switches", "mean"), mean_switch_rate=("switch_rate", "mean"))
        .sort_values(["friction", "model"])
        .reset_index(drop=True)
    )
    switch_summary.to_csv(output_dir / "switch_summary.csv", index=False)
    return merged


def _write_support_note(path: Path, summary: pd.DataFrame, qualitative_pass: bool) -> None:
    zero_row = summary.loc[summary["friction"].eq(0.0)].iloc[0]
    strongest_row = summary.sort_values(["mean_deployed_gap", "friction"], ascending=[False, True]).iloc[0]
    lines = [
        "Minimal event-forecasting micro-benchmark",
        "",
        f"- Verdict: {'appendix_support_ready' if qualitative_pass else 'code_only_not_promoted'}",
        f"- Reading: {SUPPORT_READING}",
        "- Scope: appendix-level support rather than headline evidence",
        "- Canonical forecast ranking: Brier only",
        "- Log loss: robustness-only diagnostic; never used for winner selection",
        f"- Zero-friction agreement rate: {_format_float(float(zero_row['agreement_rate']), 2)}",
        f"- Strongest positive-friction mean deployed gap: {_format_float(float(strongest_row['mean_deployed_gap']), 3)} at friction {_format_float(float(strongest_row['friction']), 2)}",
        f"- Schema note: {SCHEMA_NOTE}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _build_figure(summary: pd.DataFrame, path: Path) -> None:
    plot_df = summary.copy()
    plot_df["disagreement_rate"] = 1.0 - plot_df["agreement_rate"]

    fig, ax = plt.subplots(figsize=(3.3, 2.4), constrained_layout=True)
    ax.plot(
        plot_df["friction"],
        plot_df["disagreement_rate"],
        color="#c44e52",
        marker="o",
        linewidth=2.0,
    )
    ax.set_xlabel("Friction")
    ax.set_ylabel("Disagreement rate")
    ax.set_title("Event micro Q2")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25, linewidth=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _qualitative_pass(summary: pd.DataFrame) -> bool:
    zero_row = summary.loc[summary["friction"].eq(0.0)].iloc[0]
    positive = summary.loc[summary["friction"] > 0.0].copy()
    if positive.empty:
        return False
    strongest = positive.sort_values(["mean_deployed_gap", "friction"], ascending=[False, True]).iloc[0]
    largest_positive_gap = float(positive["mean_deployed_gap"].max())
    gap_ratio = float(zero_row["mean_deployed_gap"]) / largest_positive_gap if largest_positive_gap > 1e-12 else 0.0
    return bool(
        float(zero_row["agreement_rate"]) >= 0.70
        and float(strongest["agreement_rate"]) < float(zero_row["agreement_rate"])
        and gap_ratio <= 0.25
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    seed_metrics_path = Path(args.seed_metrics).resolve()
    output_dir = Path(args.output_dir).resolve()
    paper_results_dir = Path(args.paper_results_dir).resolve()
    paper_figures_dir = Path(args.paper_figures_dir).resolve()

    q2_df = pd.read_csv(input_path)
    seed_metrics_df = pd.read_csv(seed_metrics_path)
    failures = validate_q2_source(q2_df, expected_interface_id="fixed_threshold", min_forecasters_per_seed_friction=4)
    if failures:
        raise SystemExit(f"[event-micro-support] invalid input: {failures}")

    rank_outputs, _meta = build_domain_rank_summary(
        q2_df,
        domain="event_micro",
        expected_interface_id="fixed_threshold",
    )
    write_summary_outputs(rank_outputs, output_dir)

    friction_summary = _build_friction_summary(
        rank_outputs=rank_outputs,
        seed_metrics_df=seed_metrics_df,
        output_dir=output_dir,
    )
    friction_summary_path = output_dir / "friction_summary.csv"
    friction_summary.to_csv(friction_summary_path, index=False)

    paper_table = _build_paper_table(friction_summary)
    paper_results_dir.mkdir(parents=True, exist_ok=True)
    paper_table_csv = paper_results_dir / "table_q2_selection_drift_event_micro.csv"
    paper_table_tex = paper_results_dir / "table_q2_selection_drift_event_micro.tex"
    paper_table.to_csv(paper_table_csv, index=False)
    _write_table_tex(paper_table, paper_table_tex)

    qualitative_pass = _qualitative_pass(friction_summary)
    note_path = paper_results_dir / "event_micro_support_note.md"
    _write_support_note(note_path, friction_summary, qualitative_pass)

    figure_path = paper_figures_dir / "fig_q2_event_micro_support.pdf"
    _build_figure(friction_summary, figure_path)

    print(f"[event-micro-support] wrote {friction_summary_path}")
    print(f"[event-micro-support] wrote {paper_table_csv}")
    print(f"[event-micro-support] wrote {paper_table_tex}")
    print(f"[event-micro-support] wrote {note_path}")
    print(f"[event-micro-support] wrote {figure_path}")
    print(f"[event-micro-support] qualitative_pass={qualitative_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
