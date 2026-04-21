#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "inventory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 4 inventory post-processing summaries.")
    parser.add_argument("--q1-csv", default=str(DEFAULT_INVENTORY_DIR / "q1_same_forecast_diff_interface.csv"))
    parser.add_argument("--q2-csv", default=str(DEFAULT_INVENTORY_DIR / "q2_diff_forecasts_same_interface.csv"))
    parser.add_argument("--diagnostics-csv", default=str(DEFAULT_INVENTORY_DIR / "inventory_v2_diagnostics.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_INVENTORY_DIR))
    parser.add_argument("--prefix", default="inventory_v2")
    return parser.parse_args()


def build_q1_threshold_summary(q1_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        q1_df.pivot_table(
            index=["seed", "friction_level"],
            columns="interface_id",
            values=["executed_metric", "target_executed_gap"],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = [
        column[0] if isinstance(column, tuple) and column[1] == "" else f"{column[0]}__{column[1]}"
        for column in pivot.columns
    ]
    pivot["executed_delta_tempered_minus_responsive"] = (
        pivot["executed_metric__tempered"] - pivot["executed_metric__responsive"]
    )
    pivot["tempered_win"] = pivot["executed_delta_tempered_minus_responsive"] > 0.0
    pivot["target_executed_gap_tempered"] = pivot["target_executed_gap__tempered"]
    pivot["target_executed_gap_abs_tempered"] = pivot["target_executed_gap_tempered"].abs()

    summary = (
        pivot.groupby("friction_level", as_index=False)
        .agg(
            seeds=("seed", "count"),
            tempered_win_count=("tempered_win", "sum"),
            tempered_win_rate=("tempered_win", "mean"),
            mean_executed_delta_tempered_minus_responsive=("executed_delta_tempered_minus_responsive", "mean"),
            median_executed_delta_tempered_minus_responsive=("executed_delta_tempered_minus_responsive", "median"),
            mean_target_executed_gap_tempered=("target_executed_gap_tempered", "mean"),
            mean_abs_target_executed_gap_tempered=("target_executed_gap_abs_tempered", "mean"),
            mean_executed_metric_responsive=("executed_metric__responsive", "mean"),
            mean_executed_metric_tempered=("executed_metric__tempered", "mean"),
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )
    return summary


def build_q2_forecast_vs_deployed_summary(q2_df: pd.DataFrame) -> pd.DataFrame:
    disagreement_rows: list[dict[str, object]] = []
    strongest_flip_by_friction: dict[float, tuple[str, float]] = {}
    seed_sets: dict[float, set[int]] = defaultdict(set)
    pair_flip_counts: dict[tuple[float, str, str], int] = defaultdict(int)

    for (seed, friction_level), group in q2_df.groupby(["seed", "friction_level"], sort=True):
        seed_sets[float(friction_level)].add(int(seed))
        items = {
            str(row.forecaster_id): (
                int(row.rank_within_forecast_metric),
                int(row.rank_within_executed_metric),
            )
            for row in group.itertuples(index=False)
        }
        flips = 0
        total = 0
        for left, right in combinations(sorted(items), 2):
            forecast_left, executed_left = items[left]
            forecast_right, executed_right = items[right]
            if forecast_left == forecast_right or executed_left == executed_right:
                continue
            total += 1
            if (forecast_left < forecast_right) != (executed_left < executed_right):
                flips += 1
                pair_flip_counts[(float(friction_level), left, right)] += 1
        disagreement_rows.append(
            {
                "seed": int(seed),
                "friction_level": float(friction_level),
                "disagreement_rate": float(flips / total) if total else 0.0,
            }
        )

    disagreement_df = pd.DataFrame(disagreement_rows)
    zero_mean = float(
        disagreement_df[np.isclose(disagreement_df["friction_level"], 0.0, atol=1e-15)]["disagreement_rate"].mean()
    )

    strongest_rows: list[dict[str, object]] = []
    for friction_level in sorted(seed_sets):
        best_pair = ""
        best_share = 0.0
        seeds_for_friction = max(len(seed_sets[friction_level]), 1)
        for (pair_friction, left, right), count in pair_flip_counts.items():
            if not np.isclose(pair_friction, friction_level, atol=1e-15):
                continue
            share = float(count / seeds_for_friction)
            if share > best_share:
                best_share = share
                best_pair = f"{left}|{right}"
        strongest_rows.append(
            {
                "friction_level": float(friction_level),
                "strongest_flip_pair": best_pair,
                "strongest_flip_share": best_share,
            }
        )

    strongest_df = pd.DataFrame(strongest_rows)
    summary = (
        disagreement_df.groupby("friction_level", as_index=False)
        .agg(
            seeds=("seed", "count"),
            mean_disagreement_rate=("disagreement_rate", "mean"),
            median_disagreement_rate=("disagreement_rate", "median"),
            min_disagreement_rate=("disagreement_rate", "min"),
            max_disagreement_rate=("disagreement_rate", "max"),
        )
        .merge(strongest_df, on="friction_level", how="left")
        .sort_values("friction_level")
        .reset_index(drop=True)
    )
    summary["zero_vs_this_effect_size"] = summary["mean_disagreement_rate"] - zero_mean
    return summary


def build_diagnostics_share_summary(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        diagnostics_df.groupby(["question_id", "scenario_id", "interface_id", "friction_level"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            mean_holding_cost=("mean_holding_cost", "mean"),
            mean_stockout_cost=("mean_stockout_cost", "mean"),
            mean_change_cost=("mean_change_cost", "mean"),
            mean_order_adjustment=("mean_order_adjustment", "mean"),
            mean_inventory=("mean_inventory", "mean"),
            mean_fill_rate=("fill_rate", "mean"),
            mean_stockout_day_rate=("stockout_day_rate", "mean"),
            max_order_cap_hit_rate=("order_cap_hit_rate", "max"),
        )
        .sort_values(["question_id", "interface_id", "friction_level"])
        .reset_index(drop=True)
    )
    total = (
        summary["mean_holding_cost"] + summary["mean_stockout_cost"] + summary["mean_change_cost"]
    ).replace(0.0, np.nan)
    summary["holding_share"] = summary["mean_holding_cost"] / total
    summary["stockout_share"] = summary["mean_stockout_cost"] / total
    summary["change_share"] = summary["mean_change_cost"] / total
    return summary.fillna(0.0)


def main() -> int:
    args = parse_args()
    q1_df = pd.read_csv(args.q1_csv)
    q2_df = pd.read_csv(args.q2_csv)
    diagnostics_df = pd.read_csv(args.diagnostics_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)

    q1_summary = build_q1_threshold_summary(q1_df)
    q2_summary = build_q2_forecast_vs_deployed_summary(q2_df)
    diagnostics_summary = build_diagnostics_share_summary(diagnostics_df)

    q1_path = output_dir / f"{prefix}_q1_friction_threshold_summary.csv"
    q2_path = output_dir / f"{prefix}_q2_forecast_vs_deployed_summary.csv"
    diagnostics_path = output_dir / f"{prefix}_diagnostics_share_by_friction.csv"

    q1_summary.to_csv(q1_path, index=False)
    q2_summary.to_csv(q2_path, index=False)
    diagnostics_summary.to_csv(diagnostics_path, index=False)

    print(f"[inventory-step4-summary] wrote {q1_path}")
    print(f"[inventory-step4-summary] wrote {q2_path}")
    print(f"[inventory-step4-summary] wrote {diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
