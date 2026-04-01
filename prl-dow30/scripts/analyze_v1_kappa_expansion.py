#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen-policy kappa-expansion frontier results.")
    parser.add_argument("--validation-root", required=True, help="Validation full-grid root.")
    parser.add_argument("--final-root", required=True, help="Final full-grid root.")
    parser.add_argument("--selection-json", required=True, help="Global validation selection JSON.")
    parser.add_argument("--selection-csv", required=True, help="Global validation selection CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--baseline-eta", type=float, default=1.0, help="Immediate-execution baseline eta.")
    return parser.parse_args()


def _load_runs(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("kappa_*/*/seed_*/metrics.csv")) + sorted(root.glob("kappa_*/seed_*/metrics.csv"))
    rows: list[dict[str, Any]] = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["metrics_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {root}")

    out = pd.DataFrame(rows)
    for col in [
        "kappa",
        "seed",
        "eta",
        "eta_requested",
        "sharpe_net_lin",
        "cagr",
        "avg_turnover_exec",
        "avg_turnover_target",
        "tracking_error_l2_mean",
        "misalignment_gap_mean",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["seed"] = out["seed"].astype(int)
    out["pair_eta"] = out["eta_requested"].where(~out["eta_requested"].isna(), out["eta"])
    out["turnover_gap"] = out["avg_turnover_target"] - out["avg_turnover_exec"]
    out["realized_cost_proxy"] = out["kappa"] * out["avg_turnover_exec"]
    return out


def _qualifies(score: float, best_score: float, threshold: float) -> bool:
    if best_score > 0:
        cutoff = threshold * best_score
    elif best_score < 0:
        cutoff = best_score / threshold
    else:
        cutoff = 0.0
    return score >= cutoff


def _median_frontier(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        runs.groupby(["kappa", "pair_eta"], as_index=False)
        .agg(
            median_sharpe=("sharpe_net_lin", "median"),
            median_cagr=("cagr", "median"),
            median_turnover_exec=("avg_turnover_exec", "median"),
            median_turnover_target=("avg_turnover_target", "median"),
            median_turnover_gap=("turnover_gap", "median"),
            median_tracking=("tracking_error_l2_mean", "median"),
            median_misalignment_gap=("misalignment_gap_mean", "median"),
            median_realized_cost_proxy=("realized_cost_proxy", "median"),
            n_seeds=("seed", "nunique"),
        )
        .rename(columns={"pair_eta": "eta"})
        .sort_values(["kappa", "eta"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return grouped


def _per_kappa_selector(frontier: pd.DataFrame, threshold: float, baseline_eta: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kappa, grp in frontier.groupby("kappa", sort=True):
        grp = grp.sort_values("eta", ascending=False).reset_index(drop=True)
        best_row = grp.loc[grp["median_sharpe"].idxmax()]
        best_score = float(best_row["median_sharpe"])
        qualifying = grp[grp["median_sharpe"].apply(lambda x: _qualifies(float(x), best_score, threshold))]
        selected = qualifying.sort_values("eta", ascending=False).iloc[0]
        interior = grp[grp["eta"] < baseline_eta - 1e-12].copy()
        if interior.empty:
            best_interior = best_row
        else:
            best_interior = interior.loc[interior["median_sharpe"].idxmax()]
        rows.append(
            {
                "kappa": float(kappa),
                "best_eta": float(best_row["eta"]),
                "best_median_sharpe": float(best_row["median_sharpe"]),
                "per_kappa_selected_eta": float(selected["eta"]),
                "per_kappa_selected_median_sharpe": float(selected["median_sharpe"]),
                "per_kappa_selected_relative_score_to_best": float(selected["median_sharpe"] / best_score) if best_score != 0 else 1.0,
                "best_interior_eta": float(best_interior["eta"]),
                "best_interior_median_sharpe": float(best_interior["median_sharpe"]),
                "best_interior_median_turnover_exec": float(best_interior["median_turnover_exec"]),
                "baseline_eta": float(baseline_eta),
                "baseline_median_sharpe": float(grp.loc[np.isclose(grp["eta"], baseline_eta, atol=1e-12), "median_sharpe"].iloc[0]),
                "baseline_median_turnover_exec": float(grp.loc[np.isclose(grp["eta"], baseline_eta, atol=1e-12), "median_turnover_exec"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("kappa").reset_index(drop=True)


def _paired_summary(runs: pd.DataFrame, *, compare_eta: float, baseline_eta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    compare = runs[np.isclose(runs["pair_eta"], compare_eta, atol=1e-12)].copy()
    baseline = runs[np.isclose(runs["pair_eta"], baseline_eta, atol=1e-12)].copy()
    if compare.empty or baseline.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = compare.merge(
        baseline,
        on=["kappa", "seed"],
        suffixes=("_compare", "_baseline"),
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    for metric in [
        "sharpe_net_lin",
        "cagr",
        "avg_turnover_exec",
        "realized_cost_proxy",
        "tracking_error_l2_mean",
        "turnover_gap",
        "misalignment_gap_mean",
    ]:
        merged[f"delta_{metric}"] = merged[f"{metric}_compare"] - merged[f"{metric}_baseline"]

    seedwise = merged[
        [
            "kappa",
            "seed",
            "pair_eta_compare",
            "pair_eta_baseline",
            "delta_sharpe_net_lin",
            "delta_cagr",
            "delta_avg_turnover_exec",
            "delta_realized_cost_proxy",
            "delta_tracking_error_l2_mean",
            "delta_turnover_gap",
            "delta_misalignment_gap_mean",
        ]
    ].rename(
        columns={
            "pair_eta_compare": "compare_eta",
            "pair_eta_baseline": "baseline_eta",
            "delta_avg_turnover_exec": "delta_turnover_exec",
            "delta_tracking_error_l2_mean": "delta_tracking",
            "delta_misalignment_gap_mean": "delta_misalignment_gap",
        }
    )

    rows: list[dict[str, Any]] = []
    for kappa, grp in seedwise.groupby("kappa", sort=True):
        rows.append(
            {
                "kappa": float(kappa),
                "compare_eta": float(compare_eta),
                "baseline_eta": float(baseline_eta),
                "n_pairs": int(len(grp)),
                "win_rate_sharpe": float((grp["delta_sharpe_net_lin"] > 0.0).mean()),
                "median_delta_sharpe": float(grp["delta_sharpe_net_lin"].median()),
                "mean_delta_sharpe": float(grp["delta_sharpe_net_lin"].mean()),
                "median_delta_cagr": float(grp["delta_cagr"].median()),
                "median_delta_turnover_exec": float(grp["delta_turnover_exec"].median()),
                "median_delta_realized_cost_proxy": float(grp["delta_realized_cost_proxy"].median()),
                "median_delta_tracking": float(grp["delta_tracking"].median()),
                "median_delta_turnover_gap": float(grp["delta_turnover_gap"].median()),
                "median_delta_misalignment_gap": float(grp["delta_misalignment_gap"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("kappa").reset_index(drop=True), seedwise.sort_values(["kappa", "seed"]).reset_index(drop=True)


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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_root = Path(args.validation_root)
    final_root = Path(args.final_root)
    selection_json = json.loads(Path(args.selection_json).read_text())
    selection_df = pd.read_csv(args.selection_csv)
    baseline_eta = float(args.baseline_eta)
    global_selected_eta = float(selection_json["selected_eta"])
    relative_threshold = float(selection_json["relative_threshold"])

    validation_runs = _load_runs(validation_root)
    final_runs = _load_runs(final_root)

    validation_frontier = _median_frontier(validation_runs)
    final_frontier = _median_frontier(final_runs)
    per_kappa = _per_kappa_selector(validation_frontier, relative_threshold, baseline_eta)

    global_summary, global_seedwise = _paired_summary(final_runs, compare_eta=global_selected_eta, baseline_eta=baseline_eta)

    per_kappa_rows: list[dict[str, Any]] = []
    per_kappa_seedwise_rows: list[pd.DataFrame] = []
    best_interior_rows: list[dict[str, Any]] = []
    best_interior_seedwise_rows: list[pd.DataFrame] = []

    for row in per_kappa.itertuples(index=False):
        compare_summary, compare_seedwise = _paired_summary(final_runs, compare_eta=float(row.per_kappa_selected_eta), baseline_eta=baseline_eta)
        best_summary, best_seedwise = _paired_summary(final_runs, compare_eta=float(row.best_interior_eta), baseline_eta=baseline_eta)
        compare_summary = compare_summary[np.isclose(compare_summary["kappa"], float(row.kappa), atol=1e-15)]
        best_summary = best_summary[np.isclose(best_summary["kappa"], float(row.kappa), atol=1e-15)]
        compare_seedwise = compare_seedwise[np.isclose(compare_seedwise["kappa"], float(row.kappa), atol=1e-15)]
        best_seedwise = best_seedwise[np.isclose(best_seedwise["kappa"], float(row.kappa), atol=1e-15)]
        if not compare_summary.empty:
            per_kappa_rows.append(compare_summary.iloc[0].to_dict())
        if not compare_seedwise.empty:
            compare_seedwise = compare_seedwise.copy()
            compare_seedwise["compare_mode"] = "per_kappa_selected"
            per_kappa_seedwise_rows.append(compare_seedwise)
        if not best_summary.empty:
            best_interior_rows.append(best_summary.iloc[0].to_dict())
        if not best_seedwise.empty:
            best_seedwise = best_seedwise.copy()
            best_seedwise["compare_mode"] = "best_interior"
            best_interior_seedwise_rows.append(best_seedwise)

    per_kappa_selected_summary = pd.DataFrame(per_kappa_rows).sort_values("kappa").reset_index(drop=True)
    best_interior_summary = pd.DataFrame(best_interior_rows).sort_values("kappa").reset_index(drop=True)

    merged = per_kappa.merge(
        best_interior_summary.add_prefix("best_interior_final_"),
        left_on="kappa",
        right_on="best_interior_final_kappa",
        how="left",
    )
    merged = merged.merge(
        per_kappa_selected_summary.add_prefix("per_kappa_final_"),
        left_on="kappa",
        right_on="per_kappa_final_kappa",
        how="left",
    )
    if not global_summary.empty:
        merged = merged.merge(
            global_summary.add_prefix("global_selected_final_"),
            left_on="kappa",
            right_on="global_selected_final_kappa",
            how="left",
        )
    merged["global_selected_eta"] = global_selected_eta

    positive = merged[merged["kappa"] > 0].sort_values("kappa").reset_index(drop=True)
    best_sharpe_series = positive["best_interior_final_median_delta_sharpe"] if not positive.empty else pd.Series(dtype=float)
    verdict = {
        "global_selected_eta": global_selected_eta,
        "positive_kappas": [float(x) for x in positive["kappa"].tolist()],
        "positive_cost_frontier_signal_all": bool(
            not positive.empty
            and (positive["best_interior_final_median_delta_sharpe"] > 0.0).all()
            and (positive["best_interior_final_median_delta_turnover_exec"] < 0.0).all()
        ),
        "positive_cost_per_kappa_selected_signal_all": bool(
            not positive.empty
            and (positive["per_kappa_final_median_delta_sharpe"] > 0.0).all()
            and (positive["per_kappa_final_median_delta_turnover_exec"] < 0.0).all()
        ),
        "high_friction_gain_exceeds_low_friction_gain": bool(
            0.002 in set(float(x) for x in positive["kappa"])
            and 0.0002 in set(float(x) for x in positive["kappa"])
            and float(
                positive.loc[np.isclose(positive["kappa"], 0.002, atol=1e-15), "best_interior_final_median_delta_sharpe"].iloc[0]
            )
            > float(
                positive.loc[np.isclose(positive["kappa"], 0.0002, atol=1e-15), "best_interior_final_median_delta_sharpe"].iloc[0]
            )
        ),
        "best_interior_delta_sharpe_non_decreasing": bool(best_sharpe_series.is_monotonic_increasing),
    }

    _write_csv_md(
        validation_frontier,
        output_dir / "validation_kappa_frontier.csv",
        output_dir / "validation_kappa_frontier.md",
        "Validation Kappa Frontier",
    )
    _write_csv_md(
        final_frontier,
        output_dir / "final_kappa_frontier.csv",
        output_dir / "final_kappa_frontier.md",
        "Final Kappa Frontier",
    )
    _write_csv_md(
        per_kappa,
        output_dir / "validation_per_kappa_selection.csv",
        output_dir / "validation_per_kappa_selection.md",
        "Validation Per-Kappa Selection",
    )
    _write_csv_md(
        merged,
        output_dir / "kappa_expansion_summary.csv",
        output_dir / "kappa_expansion_summary.md",
        "Kappa Expansion Summary",
    )

    seedwise_frames = []
    if not global_seedwise.empty:
        global_seedwise = global_seedwise.copy()
        global_seedwise["compare_mode"] = "global_selected"
        seedwise_frames.append(global_seedwise)
    if per_kappa_seedwise_rows:
        seedwise_frames.append(pd.concat(per_kappa_seedwise_rows, ignore_index=True))
    if best_interior_seedwise_rows:
        seedwise_frames.append(pd.concat(best_interior_seedwise_rows, ignore_index=True))
    seedwise_all = pd.concat(seedwise_frames, ignore_index=True) if seedwise_frames else pd.DataFrame()
    if not seedwise_all.empty:
        _write_csv_md(
            seedwise_all.sort_values(["compare_mode", "kappa", "seed"]).reset_index(drop=True),
            output_dir / "kappa_expansion_seedwise.csv",
            output_dir / "kappa_expansion_seedwise.md",
            "Kappa Expansion Seedwise",
        )

    lines = [
        "# Kappa Expansion Summary",
        "",
        f"- global selected eta across expanded positive kappas: `{global_selected_eta}`",
        f"- positive-cost frontier signal on all kappas: `{verdict['positive_cost_frontier_signal_all']}`",
        f"- per-kappa selected signal on all positive kappas: `{verdict['positive_cost_per_kappa_selected_signal_all']}`",
        f"- high-friction gain exceeds low-friction gain: `{verdict['high_friction_gain_exceeds_low_friction_gain']}`",
        f"- best-interior delta Sharpe non-decreasing over positive kappas: `{verdict['best_interior_delta_sharpe_non_decreasing']}`",
        "",
        "## Per-kappa readout",
        "",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"- kappa `{row.kappa:g}`: validation best eta `{row.best_eta:g}`, per-kappa selected eta `{row.per_kappa_selected_eta:g}`, "
            f"best interior final delta Sharpe vs eta=1 `{row.best_interior_final_median_delta_sharpe:+.6f}`, "
            f"best interior final delta TOexec `{row.best_interior_final_median_delta_turnover_exec:+.6f}`"
        )
    (output_dir / "kappa_expansion_summary.txt").write_text("\n".join(lines) + "\n")
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")


if __name__ == "__main__":
    main()
