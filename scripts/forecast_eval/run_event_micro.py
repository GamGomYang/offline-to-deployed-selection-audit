#!/usr/bin/env python3
"""Run the minimal event-forecasting micro-benchmark.

In the shared raw schema, `forecast_metric` is stored as `-brier` only to
preserve the higher-is-better ranking convention used by the summary builder;
all paper-facing tables and text should still report Brier in its standard
lower-is-better interpretation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from common import build_result_row, prepare_results_frame, save_results  # noqa: E402
from event_micro import EventMicroConfig, brier_score, evaluate_actions, generate_event_stream, generate_forecasts, load_config  # noqa: E402
from event_micro import log_loss_score, threshold_policy  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "event_micro_q2.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "event_micro"
DEFAULT_SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "forecast_eval" / "build_summary.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the event-style Q2 micro-benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the locked event-micro YAML.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for event-micro artifacts.")
    parser.add_argument("--summary-script", default=str(DEFAULT_SUMMARY_SCRIPT), help="Path to build_summary.py.")
    return parser.parse_args()


def _build_rows(config: EventMicroConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    seed_metric_rows: list[dict[str, object]] = []

    for seed in config.seed_list():
        stream = generate_event_stream(config.horizon, seed, config)
        q_true = stream["q_true"]
        y = stream["y"]
        forecasts = generate_forecasts(q_true, seed=seed, config=config)

        metric_cache: dict[tuple[int, float, str], dict[str, float]] = {}
        for friction in config.friction_grid:
            for model_id, probabilities in forecasts.items():
                actions = threshold_policy(probabilities, config.threshold_tau)
                deployed = evaluate_actions(
                    actions,
                    y,
                    friction=float(friction),
                    tp_reward=float(config.tp_reward),
                    fp_penalty=float(config.fp_penalty),
                    fn_penalty=float(config.fn_penalty),
                    initial_action=int(config.initial_action),
                )
                brier = brier_score(probabilities, y)
                logloss = log_loss_score(probabilities, y, eps=float(config.logloss_eps))
                deployed_utility = float(deployed["deployed_utility"])

                raw_rows.append(
                    build_result_row(
                        question_id="Q2",
                        scenario_id=str(config.scenario_id),
                        domain="event_micro",
                        seed=int(seed),
                        forecaster_id=str(model_id),
                        interface_id="fixed_threshold",
                        friction_level=float(friction),
                        forecast_metric=-float(brier),
                        target_metric=deployed_utility,
                        executed_metric=deployed_utility,
                        realized_cost=float(deployed["mean_switch_cost"]),
                        realized_turnover_or_adjustment=float(deployed["switch_rate"]),
                    )
                )
                metric_cache[(int(seed), float(friction), str(model_id))] = {
                    "brier": float(brier),
                    "logloss": float(logloss),
                    "deployed_utility": deployed_utility,
                    "n_switches": int(deployed["n_switches"]),
                    "switch_rate": float(deployed["switch_rate"]),
                }

        raw_df = prepare_results_frame(raw_rows)
        seed_frame = raw_df.loc[raw_df["seed"] == int(seed), ["seed", "friction_level", "forecaster_id", "rank_within_forecast_metric", "rank_within_executed_metric"]].copy()
        for row in seed_frame.itertuples(index=False):
            cache_key = (int(row.seed), float(row.friction_level), str(row.forecaster_id))
            cached = metric_cache[cache_key]
            seed_metric_rows.append(
                {
                    "seed": int(row.seed),
                    "friction": float(row.friction_level),
                    "model": str(row.forecaster_id),
                    "brier": float(cached["brier"]),
                    "logloss": float(cached["logloss"]),
                    "deployed_utility": float(cached["deployed_utility"]),
                    "n_switches": int(cached["n_switches"]),
                    "switch_rate": float(cached["switch_rate"]),
                    "forecast_rank": int(row.rank_within_forecast_metric),
                    "deployed_rank": int(row.rank_within_executed_metric),
                    "is_forecast_winner": bool(int(row.rank_within_forecast_metric) == 1),
                    "is_deployed_winner": bool(int(row.rank_within_executed_metric) == 1),
                }
            )

    final_raw_df = prepare_results_frame(raw_rows)
    seed_metrics_df = pd.DataFrame(seed_metric_rows).sort_values(["friction", "seed", "model"]).reset_index(drop=True)
    return final_raw_df, seed_metrics_df


def _refresh_master_summary(summary_script: Path) -> None:
    subprocess.run([sys.executable, str(summary_script)], cwd=str(REPO_ROOT), check=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, seed_metrics_df = _build_rows(config)
    raw_path = output_dir / "q2_diff_forecasts_same_interface.csv"
    seed_metrics_path = output_dir / "seed_level_metrics.csv"

    save_results(raw_df, raw_path)
    seed_metrics_df.to_csv(seed_metrics_path, index=False)
    _refresh_master_summary(Path(args.summary_script).resolve())

    print(f"[event-micro] wrote {raw_path}")
    print(f"[event-micro] wrote {seed_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
