#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mechanism decomposition tables from frozen-policy traces.")
    parser.add_argument("--validation-root", required=True, help="Validation step6 root.")
    parser.add_argument("--final-root", required=True, help="Final step6 root.")
    parser.add_argument("--selection-json", required=True, help="Validation selection JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory for decomposition artifacts.")
    parser.add_argument("--baseline-eta", type=float, default=1.0, help="Immediate-execution reference eta.")
    return parser.parse_args()


FRONTIER_METRICS = [
    "gross_sharpe_lin",
    "net_sharpe_lin",
    "gross_cagr",
    "net_cagr",
    "cost_mean",
    "cost_total",
    "avg_turnover_exec",
    "avg_turnover_target",
    "turnover_gap",
    "turnover_ratio_target_over_exec",
    "tracking_error_l2_mean",
    "misalignment_gap_mean",
    "mean_abs_return_gap",
    "max_abs_daily_gap",
    "final_equity_gap",
    "gross_net_sharpe_wedge",
    "net_return_std_ann",
    "gross_return_std_ann",
    "downside_dev_ann",
    "collapse_flag_any",
]

PAIR_METRICS = [
    "gross_sharpe_lin",
    "net_sharpe_lin",
    "gross_cagr",
    "net_cagr",
    "cost_mean",
    "cost_total",
    "avg_turnover_exec",
    "avg_turnover_target",
    "turnover_gap",
    "turnover_ratio_target_over_exec",
    "tracking_error_l2_mean",
    "misalignment_gap_mean",
    "mean_abs_return_gap",
    "max_abs_daily_gap",
    "final_equity_gap",
    "gross_net_sharpe_wedge",
    "net_return_std_ann",
    "gross_return_std_ann",
    "downside_dev_ann",
]


def _ann_sharpe(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    if np.isclose(std, 0.0):
        return 0.0
    return float(np.sqrt(252.0) * values.mean() / std)


def _ann_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(np.sqrt(252.0) * values.std(ddof=1))


def _downside_dev(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    downside = np.minimum(values, 0.0)
    return float(np.sqrt(252.0) * np.sqrt(np.mean(np.square(downside))))


def _cagr_from_equity(final_equity: float, periods: int) -> float:
    if periods <= 0 or final_equity <= 0:
        return float("nan")
    return float(final_equity ** (252.0 / periods) - 1.0)


def _parse_component_value(component: str, prefix: str) -> float:
    raw = component.removeprefix(prefix)
    return float(raw)


def _collect_trace_metrics(window: str, root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for trace_path in sorted(root.glob("kappa_*/*/seed_*/trace.parquet")):
        seed_dir = trace_path.parent
        eta_dir = seed_dir.parent
        kappa_dir = eta_dir.parent

        seed = int(seed_dir.name.removeprefix("seed_"))
        eta = _parse_component_value(eta_dir.name, "eta_")
        kappa = _parse_component_value(kappa_dir.name, "kappa_")

        trace = pd.read_parquet(trace_path)
        gross = pd.to_numeric(trace["portfolio_return"], errors="coerce")
        net = pd.to_numeric(trace["net_return_lin"], errors="coerce")
        cost = pd.to_numeric(trace["cost"], errors="coerce")
        turnover_exec = pd.to_numeric(trace["turnover_exec"], errors="coerce")
        turnover_target = pd.to_numeric(trace["turnover_target"], errors="coerce")
        tracking = pd.to_numeric(trace["tracking_error_l2"], errors="coerce")
        misalignment_gap = pd.to_numeric(trace["net_return_lin_target"], errors="coerce") - net
        gross_std = _ann_std(gross)
        net_std = _ann_std(net)
        final_equity_gap = abs(
            float(pd.to_numeric(trace["equity_net_lin_target"], errors="coerce").iloc[-1])
            - float(pd.to_numeric(trace["equity_net_lin"], errors="coerce").iloc[-1])
        )
        periods = len(trace)
        gross_final_equity = float(pd.to_numeric(trace["equity_gross"], errors="coerce").iloc[-1])
        net_final_equity = float(pd.to_numeric(trace["equity_net_lin"], errors="coerce").iloc[-1])

        rows.append(
            {
                "window": window,
                "kappa": kappa,
                "eta": eta,
                "seed": seed,
                "gross_sharpe_lin": _ann_sharpe(gross),
                "net_sharpe_lin": _ann_sharpe(net),
                "gross_cagr": _cagr_from_equity(gross_final_equity, periods),
                "net_cagr": _cagr_from_equity(net_final_equity, periods),
                "cost_mean": float(cost.mean()),
                "cost_total": float(cost.sum()),
                "avg_turnover_exec": float(turnover_exec.mean()),
                "avg_turnover_target": float(turnover_target.mean()),
                "turnover_gap": float((turnover_target - turnover_exec).mean()),
                "turnover_ratio_target_over_exec": float(
                    (turnover_target / turnover_exec.replace(0.0, np.nan)).median()
                ),
                "tracking_error_l2_mean": float(tracking.mean()),
                "misalignment_gap_mean": float(misalignment_gap.mean()),
                "mean_abs_return_gap": float(misalignment_gap.abs().mean()),
                "max_abs_daily_gap": float(misalignment_gap.abs().max()),
                "final_equity_gap": float(final_equity_gap),
                "gross_net_sharpe_wedge": float(_ann_sharpe(gross) - _ann_sharpe(net)),
                "net_return_std_ann": float(net_std),
                "gross_return_std_ann": float(gross_std),
                "downside_dev_ann": float(_downside_dev(net)),
                "collapse_flag_any": bool(pd.Series(trace["collapse_flag"]).fillna(False).astype(bool).any()),
                "trace_path": str(trace_path),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_frontier(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (window, kappa, eta), grp in df.groupby(["window", "kappa", "eta"], sort=True):
        row = {
            "window": window,
            "kappa": float(kappa),
            "eta": float(eta),
            "n_seeds": int(grp["seed"].nunique()),
        }
        for metric in FRONTIER_METRICS:
            if metric == "collapse_flag_any":
                row["collapse_rate"] = float(grp[metric].astype(bool).mean())
                continue
            row[f"median_{metric}"] = float(grp[metric].median())
            row[f"iqr_{metric}"] = float(grp[metric].quantile(0.75) - grp[metric].quantile(0.25))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["window", "kappa", "eta"]).reset_index(drop=True)


def _pair_selected_vs_baseline(df: pd.DataFrame, *, selected_eta: float, baseline_eta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = df[np.isclose(df["eta"], selected_eta, atol=1e-12)].copy()
    baseline = df[np.isclose(df["eta"], baseline_eta, atol=1e-12)].copy()
    merged = selected.merge(
        baseline,
        on=["window", "kappa", "seed"],
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    for metric in PAIR_METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_selected"] - merged[f"{metric}_baseline"]
        merged[f"improvement_{metric}"] = merged[f"{metric}_baseline"] - merged[f"{metric}_selected"]

    merged["selected_eta"] = selected_eta
    merged["baseline_eta"] = baseline_eta
    seedwise_cols = [
        "window",
        "kappa",
        "seed",
        "selected_eta",
        "baseline_eta",
    ]
    for metric in PAIR_METRICS:
        seedwise_cols.extend(
            [
                f"{metric}_selected",
                f"{metric}_baseline",
                f"delta_{metric}",
            ]
        )
    seedwise = merged[seedwise_cols].copy()

    summary_rows: list[dict] = []
    for (window, kappa), grp in merged.groupby(["window", "kappa"], sort=True):
        row = {
            "window": window,
            "kappa": float(kappa),
            "selected_eta": float(selected_eta),
            "baseline_eta": float(baseline_eta),
            "n_pairs": int(len(grp)),
            "n_wins_net_sharpe": int((grp["delta_net_sharpe_lin"] > 0.0).sum()),
            "win_rate_net_sharpe": float((grp["delta_net_sharpe_lin"] > 0.0).mean()),
        }
        for metric in PAIR_METRICS:
            row[f"selected_median_{metric}"] = float(grp[f"{metric}_selected"].median())
            row[f"baseline_median_{metric}"] = float(grp[f"{metric}_baseline"].median())
            row[f"median_delta_{metric}"] = float(grp[f"delta_{metric}"].median())
            row[f"mean_delta_{metric}"] = float(grp[f"delta_{metric}"].mean())
        row["median_cost_reduction"] = float(grp["improvement_cost_mean"].median())
        row["median_turnover_reduction"] = float(grp["improvement_avg_turnover_exec"].median())
        row["median_turnover_gap_reduction"] = float(grp["improvement_turnover_gap"].median())
        row["median_tracking_reduction"] = float(grp["improvement_tracking_error_l2_mean"].median())
        row["median_variance_reduction"] = float(grp["improvement_net_return_std_ann"].median())
        row["median_downside_reduction"] = float(grp["improvement_downside_dev_ann"].median())
        row["median_gross_sharpe_sacrifice"] = float(grp["improvement_gross_sharpe_lin"].median())
        row["median_gross_cagr_sacrifice"] = float(grp["improvement_gross_cagr"].median())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["window", "kappa"]).reset_index(drop=True)
    return seedwise, summary


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


def _build_summary_text(selection_payload: dict, pair_summary: pd.DataFrame) -> str:
    selected_eta = float(selection_payload["selected_eta"])
    lines = [
        "# Mechanism Decomposition Summary",
        "",
        f"- selected eta: `{selected_eta}`",
        f"- baseline eta: `1.0`",
        "",
        "This decomposition compares the validation-selected operating point against the immediate-execution baseline and reports how the net improvement lines up with cost reduction, executed-path stabilization, and any gross-return sacrifice.",
        "",
    ]
    if pair_summary.empty:
        lines.append("- no paired summary rows")
        return "\n".join(lines) + "\n"

    final_rows = pair_summary[pair_summary["window"] == "final"]
    for _, row in final_rows.iterrows():
        kappa = row["kappa"]
        lines.extend(
            [
                f"## Final window, kappa={kappa:g}",
                "",
                f"- median net Sharpe delta: `{row['median_delta_net_sharpe_lin']:+.6f}`",
                f"- median gross Sharpe delta: `{row['median_delta_gross_sharpe_lin']:+.6f}`",
                f"- median realized-cost delta: `{row['median_delta_cost_mean']:+.8f}`",
                f"- median executed-turnover delta: `{row['median_delta_avg_turnover_exec']:+.6f}`",
                f"- median turnover-gap delta: `{row['median_delta_turnover_gap']:+.6f}`",
                f"- median tracking-error delta: `{row['median_delta_tracking_error_l2_mean']:+.6f}`",
                f"- median annualized net-volatility delta: `{row['median_delta_net_return_std_ann']:+.6f}`",
                f"- median downside-deviation delta: `{row['median_delta_downside_dev_ann']:+.6f}`",
                f"- median final-equity-gap delta: `{row['median_delta_final_equity_gap']:+.6f}`",
                f"- net Sharpe wins: `{int(row['n_wins_net_sharpe'])}/{int(row['n_pairs'])}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    validation_root = Path(args.validation_root)
    final_root = Path(args.final_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_payload = json.loads(Path(args.selection_json).read_text())
    selected_eta = float(selection_payload["selected_eta"])

    validation_df = _collect_trace_metrics("validation", validation_root)
    final_df = _collect_trace_metrics("final", final_root)
    combined = pd.concat([validation_df, final_df], ignore_index=True)

    frontier = _aggregate_frontier(combined)
    frontier_csv = output_dir / "mechanism_frontier_summary.csv"
    frontier_md = output_dir / "mechanism_frontier_summary.md"
    _write_csv_md(frontier, frontier_csv, frontier_md, "Mechanism Frontier Summary")

    seedwise, pair_summary = _pair_selected_vs_baseline(
        combined,
        selected_eta=selected_eta,
        baseline_eta=args.baseline_eta,
    )
    seedwise_csv = output_dir / "selected_vs_eta1_mechanism_seedwise.csv"
    seedwise_md = output_dir / "selected_vs_eta1_mechanism_seedwise.md"
    _write_csv_md(seedwise, seedwise_csv, seedwise_md, "Selected Eta Versus Eta=1 Mechanism Seedwise")

    pair_csv = output_dir / "selected_vs_eta1_mechanism_summary.csv"
    pair_md = output_dir / "selected_vs_eta1_mechanism_summary.md"
    _write_csv_md(pair_summary, pair_csv, pair_md, "Selected Eta Versus Eta=1 Mechanism Summary")

    summary_text = _build_summary_text(selection_payload, pair_summary)
    (output_dir / "mechanism_summary.md").write_text(summary_text)

    manifest = {
        "selected_eta": selected_eta,
        "baseline_eta": args.baseline_eta,
        "validation_root": str(validation_root),
        "final_root": str(final_root),
        "outputs": {
            "frontier_summary_csv": str(frontier_csv),
            "frontier_summary_md": str(frontier_md),
            "seedwise_csv": str(seedwise_csv),
            "seedwise_md": str(seedwise_md),
            "pair_summary_csv": str(pair_csv),
            "pair_summary_md": str(pair_md),
            "summary_md": str(output_dir / "mechanism_summary.md"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
