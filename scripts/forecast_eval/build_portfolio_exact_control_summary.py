#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "forecast_eval" / "portfolio_exact_control" / "portfolio_control_results.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "portfolio_exact_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post-processed exact-control summary tables.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input portfolio exact-control summary CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for summary CSVs.")
    return parser.parse_args()


def _paired_pivot(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index=["universe_id", "seed", "kappa"],
            columns="replay_interface_id",
            values=value_column,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    pivot["delta_eta05_minus_eta1"] = pivot["eta_0_5"] - pivot["eta_1_0"]
    return pivot


def build_target_based_delta_summary(results: pd.DataFrame) -> pd.DataFrame:
    target = _paired_pivot(results, "sharpe_target_net")
    executed = _paired_pivot(results, "sharpe_exec_net").rename(
        columns={
            "eta_0_5": "eta_0_5_executed",
            "eta_1_0": "eta_1_0_executed",
            "delta_eta05_minus_eta1": "executed_delta_eta05_minus_eta1",
        }
    )
    merged = target.merge(
        executed[
            [
                "universe_id",
                "seed",
                "kappa",
                "eta_0_5_executed",
                "eta_1_0_executed",
                "executed_delta_eta05_minus_eta1",
            ]
        ],
        on=["universe_id", "seed", "kappa"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "eta_0_5": "eta_0_5_target",
            "eta_1_0": "eta_1_0_target",
            "delta_eta05_minus_eta1": "target_delta_eta05_minus_eta1",
        }
    )
    merged["evaluation_gap_delta"] = (
        merged["executed_delta_eta05_minus_eta1"] - merged["target_delta_eta05_minus_eta1"]
    )

    summary = (
        merged.groupby(["universe_id", "kappa"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            target_delta_median=("target_delta_eta05_minus_eta1", "median"),
            target_delta_mean=("target_delta_eta05_minus_eta1", "mean"),
            executed_delta_median=("executed_delta_eta05_minus_eta1", "median"),
            executed_delta_mean=("executed_delta_eta05_minus_eta1", "mean"),
            evaluation_gap_delta_median=("evaluation_gap_delta", "median"),
            evaluation_gap_delta_mean=("evaluation_gap_delta", "mean"),
        )
        .sort_values(["universe_id", "kappa"])
        .reset_index(drop=True)
    )
    return summary


def build_effect_size_summary(results: pd.DataFrame) -> pd.DataFrame:
    executed = _paired_pivot(results, "sharpe_exec_net")
    executed["cost_regime"] = executed["kappa"].map(lambda value: "zero_cost" if abs(float(value)) <= 1e-15 else "positive_cost")

    universe_regime = (
        executed.groupby(["universe_id", "cost_regime"], as_index=False)
        .agg(
            seed_groups=("delta_eta05_minus_eta1", "count"),
            paired_delta_median=("delta_eta05_minus_eta1", "median"),
            paired_delta_mean=("delta_eta05_minus_eta1", "mean"),
            paired_delta_abs_median=("delta_eta05_minus_eta1", lambda values: pd.Series(values).abs().median()),
            paired_delta_min=("delta_eta05_minus_eta1", "min"),
            paired_delta_max=("delta_eta05_minus_eta1", "max"),
        )
        .sort_values(["universe_id", "cost_regime"])
        .reset_index(drop=True)
    )

    zero = universe_regime[universe_regime["cost_regime"] == "zero_cost"].rename(
        columns={
            "seed_groups": "zero_seed_groups",
            "paired_delta_median": "zero_paired_delta_median",
            "paired_delta_mean": "zero_paired_delta_mean",
            "paired_delta_abs_median": "zero_paired_delta_abs_median",
            "paired_delta_min": "zero_paired_delta_min",
            "paired_delta_max": "zero_paired_delta_max",
        }
    )
    positive = universe_regime[universe_regime["cost_regime"] == "positive_cost"].rename(
        columns={
            "seed_groups": "positive_seed_groups",
            "paired_delta_median": "positive_paired_delta_median",
            "paired_delta_mean": "positive_paired_delta_mean",
            "paired_delta_abs_median": "positive_paired_delta_abs_median",
            "paired_delta_min": "positive_paired_delta_min",
            "paired_delta_max": "positive_paired_delta_max",
        }
    )
    effect = zero.merge(positive, on="universe_id", how="outer")
    effect["effect_size_median"] = effect["positive_paired_delta_median"] - effect["zero_paired_delta_median"]
    effect["effect_size_mean"] = effect["positive_paired_delta_mean"] - effect["zero_paired_delta_mean"]
    effect["effect_size_abs_median"] = (
        effect["positive_paired_delta_abs_median"] - effect["zero_paired_delta_abs_median"]
    )
    effect = effect.sort_values("universe_id").reset_index(drop=True)
    return effect


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(input_path)
    target_summary = build_target_based_delta_summary(results)
    effect_summary = build_effect_size_summary(results)

    target_path = output_dir / "target_based_delta_summary.csv"
    effect_path = output_dir / "zero_cost_vs_positive_cost_effect_size_summary.csv"
    target_summary.to_csv(target_path, index=False)
    effect_summary.to_csv(effect_path, index=False)

    print(f"[portfolio-exact-control-summary] wrote {target_path}")
    print(f"[portfolio-exact-control-summary] wrote {effect_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
