#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import annualized_sharpe, build_result_row, mse_score, partial_adjustment_path, save_results  # noqa: E402


DEFAULT_RETURNS_PATH = ROOT / "data" / "processed_u27" / "returns.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "forecast_eval" / "portfolio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit portfolio-control results in the shared forecast-eval schema.")
    parser.add_argument("--returns-path", default=str(DEFAULT_RETURNS_PATH), help="Parquet file with asset returns.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for domain-level CSV outputs.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to evaluate.")
    parser.add_argument("--horizon", type=int, default=756, help="Evaluation horizon per seed.")
    parser.add_argument("--warmup", type=int, default=60, help="Warmup history size for rolling forecasts.")
    return parser.parse_args()


def _slice_returns(returns_df: pd.DataFrame, *, seed: int, horizon: int, warmup: int) -> pd.DataFrame:
    total_needed = horizon + warmup
    if len(returns_df) <= total_needed:
        raise ValueError("Not enough return history for the requested horizon and warmup.")
    rng = np.random.default_rng(30_000 + seed)
    start = int(rng.integers(warmup, len(returns_df) - horizon))
    return returns_df.iloc[start - warmup : start + horizon].copy()


def _ewma(history: np.ndarray, halflife: float) -> np.ndarray:
    length = history.shape[0]
    weights = np.exp(-np.linspace(length - 1, 0, length) / float(halflife))
    weights = weights / weights.sum()
    return weights @ history


def _forecast_vectors(history_returns: np.ndarray, forecaster_id: str) -> np.ndarray:
    horizon = history_returns.shape[0]
    n_assets = history_returns.shape[1]
    forecasts = np.zeros((horizon, n_assets), dtype=np.float64)
    for idx in range(1, horizon):
        past = history_returns[:idx]
        if forecaster_id == "last_return":
            forecasts[idx] = past[-1]
        elif forecaster_id == "rolling_mean_5":
            forecasts[idx] = past[-min(5, idx) :].mean(axis=0)
        elif forecaster_id == "rolling_mean_20":
            forecasts[idx] = past[-min(20, idx) :].mean(axis=0)
        elif forecaster_id == "ewma_20":
            forecasts[idx] = _ewma(past[-min(20, idx) :], halflife=6.0)
        else:
            raise ValueError(f"Unknown forecaster: {forecaster_id}")
    return forecasts


def _weights_from_forecast(forecasts: np.ndarray) -> np.ndarray:
    n_assets = forecasts.shape[1]
    equal_weight = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    weights = np.zeros_like(forecasts)
    for idx, forecast in enumerate(forecasts):
        centered = forecast - forecast.mean()
        scale = max(float(centered.std()), 1e-6)
        logits = np.clip(1.8 * centered / scale, -12.0, 12.0)
        tilt = np.exp(logits)
        tilt = tilt / tilt.sum()
        weights[idx] = 0.55 * equal_weight + 0.45 * tilt
    return weights


def _portfolio_metric(weights: np.ndarray, returns: np.ndarray, friction_level: float) -> tuple[float, float, float]:
    n_assets = weights.shape[1]
    previous = np.concatenate([np.full((1, n_assets), 1.0 / n_assets), weights[:-1]], axis=0)
    turnover = 0.5 * np.abs(weights - previous).sum(axis=1)
    gross = (weights * returns).sum(axis=1)
    cost = float(friction_level) * turnover
    net = gross - cost
    return annualized_sharpe(net), float(cost.mean()), float(turnover.mean())


def build_rows(returns_path: Path, seeds: list[int], horizon: int, warmup: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    q1_rows: list[dict[str, object]] = []
    q2_rows: list[dict[str, object]] = []
    returns_df = pd.read_parquet(returns_path)

    interfaces = {
        "responsive": 1.0,
        "tempered": 0.4,
    }
    friction_grid = [0.0, 0.0005, 0.0010]

    for seed in seeds:
        sliced = _slice_returns(returns_df, seed=seed, horizon=horizon, warmup=warmup)
        history_block = sliced.iloc[:-horizon].to_numpy(dtype=np.float64)
        eval_block = sliced.iloc[-horizon:].to_numpy(dtype=np.float64)
        combined = np.concatenate([history_block, eval_block], axis=0)

        def make_eval_forecasts(forecaster_id: str) -> np.ndarray:
            full_forecasts = _forecast_vectors(combined, forecaster_id)
            return full_forecasts[-horizon:]

        q1_forecaster_id = "ewma_20"
        q1_forecasts = make_eval_forecasts(q1_forecaster_id)
        q1_targets = _weights_from_forecast(q1_forecasts)
        q1_forecast_metric = mse_score(q1_forecasts, eval_block)

        for friction_level in friction_grid:
            target_metric, _, _ = _portfolio_metric(q1_targets, eval_block, friction_level)
            for interface_id, interface_strength in interfaces.items():
                executed = partial_adjustment_path(
                    q1_targets,
                    interface_strength=interface_strength,
                    friction_level=friction_level,
                    friction_scale=300.0,
                )
                executed_metric, realized_cost, realized_adjustment = _portfolio_metric(executed, eval_block, friction_level)
                q1_rows.append(
                    build_result_row(
                        question_id="Q1",
                        scenario_id="u27_long_only_v1",
                        domain="portfolio",
                        seed=seed,
                        forecaster_id=q1_forecaster_id,
                        interface_id=interface_id,
                        friction_level=friction_level,
                        forecast_metric=q1_forecast_metric,
                        target_metric=target_metric,
                        executed_metric=executed_metric,
                        realized_cost=realized_cost,
                        realized_turnover_or_adjustment=realized_adjustment,
                    )
                )

        q2_interface_id = "tempered"
        q2_interface_strength = interfaces[q2_interface_id]
        for friction_level in friction_grid:
            for forecaster_id in ["last_return", "rolling_mean_5", "rolling_mean_20", "ewma_20"]:
                forecasts = make_eval_forecasts(forecaster_id)
                targets = _weights_from_forecast(forecasts)
                executed = partial_adjustment_path(
                    targets,
                    interface_strength=q2_interface_strength,
                    friction_level=friction_level,
                    friction_scale=300.0,
                )
                target_metric, _, _ = _portfolio_metric(targets, eval_block, friction_level)
                executed_metric, realized_cost, realized_adjustment = _portfolio_metric(executed, eval_block, friction_level)
                q2_rows.append(
                    build_result_row(
                        question_id="Q2",
                        scenario_id="u27_long_only_v1",
                        domain="portfolio",
                        seed=seed,
                        forecaster_id=forecaster_id,
                        interface_id=q2_interface_id,
                        friction_level=friction_level,
                        forecast_metric=mse_score(forecasts, eval_block),
                        target_metric=target_metric,
                        executed_metric=executed_metric,
                        realized_cost=realized_cost,
                        realized_turnover_or_adjustment=realized_adjustment,
                    )
                )

    return q1_rows, q2_rows


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    q1_rows, q2_rows = build_rows(
        returns_path=Path(args.returns_path),
        seeds=list(args.seeds),
        horizon=int(args.horizon),
        warmup=int(args.warmup),
    )

    q1_path = output_dir / "q1_same_forecast_diff_interface.csv"
    q2_path = output_dir / "q2_diff_forecasts_same_interface.csv"
    q1_df = save_results(q1_rows, q1_path)
    q2_df = save_results(q2_rows, q2_path)

    print(f"[portfolio] wrote {len(q1_df)} Q1 rows to {q1_path}")
    print(f"[portfolio] wrote {len(q2_df)} Q2 rows to {q2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
