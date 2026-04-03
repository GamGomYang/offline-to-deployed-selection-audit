#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate rolling-origin split results for the v1 frozen-policy study.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_split_payload(split: dict) -> dict:
    if split["status"] == "canonical_reference":
        run_root = Path(split["canonical_run_root"])
    else:
        run_root = Path(split["run_root"])

    selection_json = json.loads((run_root / "validation_eta" / "selection" / "validation_eta_selection.json").read_text())
    selected_eta = float(selection_json["selected_eta"])
    rows = pd.DataFrame(selection_json["rows"])
    best_row = rows.loc[rows["score_mean_median_sharpe_pos_kappa"].idxmax()]
    selected_row = rows.loc[rows["selected"]].iloc[0]

    pair_stats = pd.read_csv(run_root / "paper_pack" / "stats" / "selected_eta_vs_eta1_stats.csv")

    mech_path = run_root / "paper_pack" / "mechanism" / "selected_vs_eta1_mechanism_summary.csv"
    if mech_path.exists():
        mech = pd.read_csv(mech_path)
        mech = mech[mech["window"] == "final"].copy()
    else:
        mech = None

    diagnostics = pd.read_csv(run_root / "paper_pack" / "diagnostics" / "diagnostic_selected_eta_v2.csv")

    split_rows = []
    for _, pair_row in pair_stats.iterrows():
        kappa = float(pair_row["kappa"])
        diag_row = diagnostics[diagnostics["kappa"] == kappa].iloc[0]
        out = {
            "split_id": split["split_id"],
            "label": split["label"],
            "kappa": kappa,
            "selected_eta": selected_eta,
            "raw_best_eta": float(best_row["eta"]),
            "selected_relative_score_to_best": float(
                selected_row["score_mean_median_sharpe_pos_kappa"] / best_row["score_mean_median_sharpe_pos_kappa"]
            ),
            "selected_validation_score": float(selected_row["score_mean_median_sharpe_pos_kappa"]),
            "n_pairs": int(pair_row["n_pairs"]),
            "win_rate_sharpe": float(pair_row["win_rate_sharpe"]),
            "median_delta_sharpe_net_lin": float(pair_row["median_delta_sharpe_net_lin"]),
            "bootstrap_ci_low_median_delta_sharpe_net_lin": float(pair_row["bootstrap_ci_low_median_delta_sharpe_net_lin"]),
            "bootstrap_ci_high_median_delta_sharpe_net_lin": float(pair_row["bootstrap_ci_high_median_delta_sharpe_net_lin"]),
            "selected_median_turnover_exec": float(pair_row["selected_median_turnover_exec"]),
            "baseline_median_turnover_exec": float(pair_row["baseline_median_turnover_exec"]),
            "median_delta_turnover_exec": float(pair_row["median_delta_turnover_exec"]),
            "selected_median_tracking_error_l2": float(diag_row["median_tracking_error_l2"]),
            "selected_median_realized_cost": float(diag_row["median_turnover_exec"] * kappa),
            "selected_median_turnover_target": float(diag_row["median_turnover_target"]),
            "selected_median_turnover_gap": float(diag_row["median_turnover_target"] - diag_row["median_turnover_exec"]),
        }
        if mech is not None:
            mech_row = mech[mech["kappa"] == kappa].iloc[0]
            out["median_delta_cost_mean"] = float(mech_row["median_delta_cost_mean"])
            out["median_delta_tracking_error_l2_mean"] = float(mech_row["median_delta_tracking_error_l2_mean"])
            out["median_delta_turnover_gap"] = float(mech_row["median_delta_turnover_gap"])
        split_rows.append(out)
    return {
        "selection_summary": {
            "split_id": split["split_id"],
            "label": split["label"],
            "selected_eta": selected_eta,
            "raw_best_eta": float(best_row["eta"]),
            "selected_validation_score": float(selected_row["score_mean_median_sharpe_pos_kappa"]),
            "raw_best_validation_score": float(best_row["score_mean_median_sharpe_pos_kappa"]),
            "selected_relative_score_to_best": float(
                selected_row["score_mean_median_sharpe_pos_kappa"] / best_row["score_mean_median_sharpe_pos_kappa"]
            ),
            "run_root": str(run_root),
        },
        "kappa_rows": split_rows,
    }


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((experiment_root / "prepared" / "manifest.json").read_text())
    selection_rows = []
    kappa_rows = []
    missing_splits: list[str] = []
    for split in manifest["splits"].values():
        try:
            payload = _load_split_payload(split)
        except FileNotFoundError:
            missing_splits.append(str(split["split_id"]))
            continue
        selection_rows.append(payload["selection_summary"])
        kappa_rows.extend(payload["kappa_rows"])

    if not selection_rows or not kappa_rows:
        raise SystemExit("No completed split outputs found yet.")

    selection_df = pd.DataFrame(selection_rows).sort_values("split_id").reset_index(drop=True)
    split_kappa_df = pd.DataFrame(kappa_rows).sort_values(["split_id", "kappa"]).reset_index(drop=True)

    positive_df = split_kappa_df[split_kappa_df["kappa"] > 0].copy()
    split_summary = positive_df.groupby(["split_id", "label"], as_index=False).agg(
        selected_eta=("selected_eta", "first"),
        mean_positive_cost_delta_sharpe=("median_delta_sharpe_net_lin", "mean"),
        min_positive_cost_delta_sharpe=("median_delta_sharpe_net_lin", "min"),
        mean_positive_cost_win_rate=("win_rate_sharpe", "mean"),
        mean_selected_turnover_exec=("selected_median_turnover_exec", "mean"),
        mean_selected_realized_cost=("selected_median_realized_cost", "mean"),
        mean_selected_tracking=("selected_median_tracking_error_l2", "mean"),
    )
    split_summary["positive_cost_direction_holds"] = split_summary["min_positive_cost_delta_sharpe"] > 0.0

    split_median_summary = positive_df.groupby("kappa", as_index=False).agg(
        median_of_split_delta_sharpe=("median_delta_sharpe_net_lin", "median"),
        median_of_split_win_rate=("win_rate_sharpe", "median"),
        median_of_split_turnover_exec=("selected_median_turnover_exec", "median"),
        median_of_split_realized_cost=("selected_median_realized_cost", "median"),
        median_of_split_tracking=("selected_median_tracking_error_l2", "median"),
    )

    verdict = {
        "n_splits_total_defined": int(len(manifest["splits"])),
        "n_splits_completed": int(selection_df["split_id"].nunique()),
        "n_splits_positive_cost_direction_holds": int(split_summary["positive_cost_direction_holds"].sum()),
        "passes_two_of_three_rule": bool(split_summary["positive_cost_direction_holds"].sum() >= 2),
        "missing_splits": missing_splits,
    }

    selection_df.to_csv(output_dir / "rolling_selection_summary.csv", index=False)
    split_kappa_df.to_csv(output_dir / "rolling_split_kappa_summary.csv", index=False)
    split_summary.to_csv(output_dir / "rolling_split_summary.csv", index=False)
    split_median_summary.to_csv(output_dir / "rolling_split_median_summary.csv", index=False)
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    lines = [
        "# Rolling-Origin Summary",
        "",
        f"- total splits defined: `{verdict['n_splits_total_defined']}`",
        f"- completed splits: `{verdict['n_splits_completed']}`",
        f"- splits with positive-cost directional hold: `{verdict['n_splits_positive_cost_direction_holds']}`",
        f"- passes two-of-three rule: `{verdict['passes_two_of_three_rule']}`",
        "",
    ]
    for _, row in split_summary.iterrows():
        lines.append(
            f"- {row['label']} ({row['split_id']}): selected eta `{row['selected_eta']}`, "
            f"mean positive-cost delta Sharpe `{row['mean_positive_cost_delta_sharpe']:.6f}`, "
            f"mean win-rate `{row['mean_positive_cost_win_rate']:.3f}`, "
            f"direction holds `{bool(row['positive_cost_direction_holds'])}`"
        )
    (output_dir / "rolling_origin_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
