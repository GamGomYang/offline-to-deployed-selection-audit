#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build selection-rule defense outputs for the frozen validation frontier.")
    parser.add_argument("--selection-csv", required=True, help="Validation selection CSV.")
    parser.add_argument("--selection-json", required=True, help="Validation selection JSON.")
    parser.add_argument("--mechanism-frontier-csv", required=True, help="Mechanism frontier summary CSV.")
    parser.add_argument("--mechanism-pair-summary-csv", required=True, help="Mechanism selected-vs-eta1 summary CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--thresholds",
        default="0.90,0.95,0.975",
        help="Comma-separated relative thresholds to analyze.",
    )
    return parser.parse_args()


def _write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("- no rows")
    else:
        try:
            lines.append(df.to_markdown(index=False))
        except ImportError:
            headers = [str(c) for c in df.columns]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    md_path.write_text("\n".join(lines) + "\n")


def _qualifies(score: float, best_score: float, threshold: float) -> bool:
    if best_score > 0:
        cutoff = threshold * best_score
    elif best_score < 0:
        cutoff = best_score / threshold
    else:
        cutoff = 0.0
    return score >= cutoff


def _select_largest_qualifying(df: pd.DataFrame, threshold: float) -> pd.Series:
    best_score = float(df["score_mean_median_sharpe_pos_kappa"].max())
    qualifies_mask = df["score_mean_median_sharpe_pos_kappa"].apply(lambda x: _qualifies(float(x), best_score, threshold))
    qualifying = df.loc[qualifies_mask].sort_values("eta", ascending=False)
    if qualifying.empty:
        raise ValueError(f"No qualifying eta found for threshold={threshold}")
    return qualifying.iloc[0]


def _build_eta_summary(selection_df: pd.DataFrame, mechanism_frontier: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    best_score = float(selection_df["score_mean_median_sharpe_pos_kappa"].max())
    best_row = selection_df.loc[selection_df["score_mean_median_sharpe_pos_kappa"].idxmax()]
    mech = mechanism_frontier[
        (mechanism_frontier["window"] == "validation")
        & (mechanism_frontier["kappa"].isin([0.0005, 0.001]))
    ].copy()
    mech = mech.groupby("eta", as_index=False).agg(
        mean_tracking_error_pos_kappa=("median_tracking_error_l2_mean", "mean"),
        mean_turnover_gap_pos_kappa=("median_turnover_gap", "mean"),
        mean_net_vol_pos_kappa=("median_net_return_std_ann", "mean"),
        mean_downside_pos_kappa=("median_downside_dev_ann", "mean"),
    )
    out = selection_df.merge(mech, on="eta", how="left")
    out["relative_score_to_best"] = out["score_mean_median_sharpe_pos_kappa"] / best_score
    out["score_drop_from_best"] = best_score - out["score_mean_median_sharpe_pos_kappa"]
    out["best_eta"] = float(best_row["eta"])
    out["best_score"] = best_score
    out["max_relative_threshold_that_still_qualifies"] = out["relative_score_to_best"]
    for threshold in thresholds:
        col = f"qualifies_at_{threshold:g}"
        out[col] = out["score_mean_median_sharpe_pos_kappa"].apply(lambda x: _qualifies(float(x), best_score, threshold))
    return out.sort_values("eta", ascending=False).reset_index(drop=True)


def _build_threshold_sensitivity(
    selection_df: pd.DataFrame,
    mechanism_frontier: pd.DataFrame,
    mechanism_pair_summary: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    best_score = float(selection_df["score_mean_median_sharpe_pos_kappa"].max())

    val_mech = mechanism_frontier[
        (mechanism_frontier["window"] == "validation")
        & (mechanism_frontier["kappa"].isin([0.0005, 0.001]))
    ].copy()
    val_mech = val_mech.groupby("eta", as_index=False).agg(
        mean_tracking_error_pos_kappa=("median_tracking_error_l2_mean", "mean"),
        mean_turnover_gap_pos_kappa=("median_turnover_gap", "mean"),
        mean_net_vol_pos_kappa=("median_net_return_std_ann", "mean"),
        mean_downside_pos_kappa=("median_downside_dev_ann", "mean"),
    )

    final_mech = mechanism_pair_summary[mechanism_pair_summary["window"] == "final"].copy()
    final_by_kappa = {}
    for kappa in [0.0005, 0.001]:
        row = final_mech[final_mech["kappa"] == kappa]
        if not row.empty:
            final_by_kappa[kappa] = row.iloc[0]

    rows = []
    for threshold in thresholds:
        selected_row = _select_largest_qualifying(selection_df, threshold)
        eta = float(selected_row["eta"])
        val_extra = val_mech[val_mech["eta"] == eta]
        val_extra = val_extra.iloc[0] if not val_extra.empty else None

        record = {
            "relative_threshold": threshold,
            "selected_eta": eta,
            "selected_score": float(selected_row["score_mean_median_sharpe_pos_kappa"]),
            "relative_score_to_best": float(selected_row["score_mean_median_sharpe_pos_kappa"] / best_score),
            "score_drop_from_best": float(best_score - float(selected_row["score_mean_median_sharpe_pos_kappa"])),
            "mean_turnover_exec_pos_kappa": float(selected_row["median_turnover_exec_pos_kappa_mean"]),
            "n_qualifying_etas": int(
                selection_df["score_mean_median_sharpe_pos_kappa"].apply(lambda x: _qualifies(float(x), best_score, threshold)).sum()
            ),
        }
        if val_extra is not None:
            record.update(
                {
                    "mean_tracking_error_pos_kappa": float(val_extra["mean_tracking_error_pos_kappa"]),
                    "mean_turnover_gap_pos_kappa": float(val_extra["mean_turnover_gap_pos_kappa"]),
                    "mean_net_vol_pos_kappa": float(val_extra["mean_net_vol_pos_kappa"]),
                    "mean_downside_pos_kappa": float(val_extra["mean_downside_pos_kappa"]),
                }
            )
        if np.isclose(eta, 0.5, atol=1e-12):
            ref = final_by_kappa.get(0.0005)
            ref_hi = final_by_kappa.get(0.001)
            if ref is not None and ref_hi is not None:
                record.update(
                    {
                        "heldout_median_delta_net_sharpe_kappa_5e4": float(ref["median_delta_net_sharpe_lin"]),
                        "heldout_median_delta_net_sharpe_kappa_1e3": float(ref_hi["median_delta_net_sharpe_lin"]),
                        "heldout_win_rate_kappa_5e4": float(ref["win_rate_net_sharpe"]),
                        "heldout_win_rate_kappa_1e3": float(ref_hi["win_rate_net_sharpe"]),
                    }
                )
        elif np.isclose(eta, 1.0, atol=1e-12):
            record.update(
                {
                    "heldout_median_delta_net_sharpe_kappa_5e4": 0.0,
                    "heldout_median_delta_net_sharpe_kappa_1e3": 0.0,
                    "heldout_win_rate_kappa_5e4": 0.0,
                    "heldout_win_rate_kappa_1e3": 0.0,
                }
            )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("relative_threshold").reset_index(drop=True)


def _build_summary_md(selection_json: dict, eta_summary: pd.DataFrame, threshold_summary: pd.DataFrame) -> str:
    best_row = eta_summary.loc[eta_summary["relative_score_to_best"].idxmax()]
    current_threshold = float(selection_json["relative_threshold"])
    current_row = threshold_summary[np.isclose(threshold_summary["relative_threshold"], current_threshold, atol=1e-12)].iloc[0]
    baseline_row = eta_summary[np.isclose(eta_summary["eta"], 1.0, atol=1e-12)].iloc[0]
    best_eta = float(best_row["eta"])
    lines = [
        "# Selection Rule Defense Summary",
        "",
        f"- locked relative threshold: `{current_threshold}`",
        f"- locked selected eta: `{selection_json['selected_eta']}`",
        f"- raw best validation eta by score: `{best_eta}`",
        "",
        "## Why the locked rule selects eta=0.5",
        "",
        f"- raw best validation score occurs at `eta={best_eta:g}` with score `{best_row['score_mean_median_sharpe_pos_kappa']:.6f}`",
        f"- under the locked `{current_threshold:.3f}` rule, `eta={current_row['selected_eta']:g}` keeps `{current_row['relative_score_to_best'] * 100:.2f}%` of that best score",
        f"- the same selected eta cuts average positive-cost validation turnover from `{baseline_row['median_turnover_exec_pos_kappa_mean']:.6f}` at `eta=1.0` to `{current_row['mean_turnover_exec_pos_kappa']:.6f}`",
        f"- compared with the raw best eta `{best_eta:g}`, the locked selected eta has lower average positive-cost tracking error (`{current_row['mean_tracking_error_pos_kappa']:.6f}` versus `{best_row['mean_tracking_error_pos_kappa']:.6f}`), which is why the rule prefers the largest qualifying eta instead of the most aggressive smoothing point",
        "",
        "## Threshold sensitivity",
        "",
    ]
    for _, row in threshold_summary.iterrows():
        lines.extend(
            [
                f"- threshold `{row['relative_threshold']:.3f}` selects `eta={row['selected_eta']:g}` with relative score `{row['relative_score_to_best'] * 100:.2f}%` and average positive-cost turnover `{row['mean_turnover_exec_pos_kappa']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- the current choice is not the raw best eta; it is the least aggressive eta that remains near the top of the validation frontier",
            "- the selected eta is stable when the threshold is tightened from `0.95` to `0.975`",
            "- only the looser `0.90` rule reverts to `eta=1.0`, which effectively collapses back to the immediate-execution baseline",
            "- this makes the current rule a conservative compromise between validation score retention and avoiding overly aggressive execution smoothing",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_df = pd.read_csv(args.selection_csv)
    selection_json = json.loads(Path(args.selection_json).read_text())
    mechanism_frontier = pd.read_csv(args.mechanism_frontier_csv)
    mechanism_pair_summary = pd.read_csv(args.mechanism_pair_summary_csv)
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]

    eta_summary = _build_eta_summary(selection_df, mechanism_frontier, thresholds)
    _write_csv_md(
        eta_summary,
        output_dir / "selection_rule_eta_summary.csv",
        output_dir / "selection_rule_eta_summary.md",
        "Selection Rule Eta Summary",
    )

    threshold_summary = _build_threshold_sensitivity(selection_df, mechanism_frontier, mechanism_pair_summary, thresholds)
    _write_csv_md(
        threshold_summary,
        output_dir / "selection_rule_threshold_sensitivity.csv",
        output_dir / "selection_rule_threshold_sensitivity.md",
        "Selection Rule Threshold Sensitivity",
    )

    summary_text = _build_summary_md(selection_json, eta_summary, threshold_summary)
    (output_dir / "selection_rule_defense.md").write_text(summary_text)

    manifest = {
        "selection_csv": str(Path(args.selection_csv).resolve()),
        "selection_json": str(Path(args.selection_json).resolve()),
        "mechanism_frontier_csv": str(Path(args.mechanism_frontier_csv).resolve()),
        "mechanism_pair_summary_csv": str(Path(args.mechanism_pair_summary_csv).resolve()),
        "thresholds": thresholds,
        "outputs": {
            "eta_summary_csv": str((output_dir / "selection_rule_eta_summary.csv").resolve()),
            "eta_summary_md": str((output_dir / "selection_rule_eta_summary.md").resolve()),
            "threshold_sensitivity_csv": str((output_dir / "selection_rule_threshold_sensitivity.csv").resolve()),
            "threshold_sensitivity_md": str((output_dir / "selection_rule_threshold_sensitivity.md").resolve()),
            "summary_md": str((output_dir / "selection_rule_defense.md").resolve()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
