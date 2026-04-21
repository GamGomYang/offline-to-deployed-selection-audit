#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_result_row, mae_score, prepare_results_frame, save_results  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "forecast_eval" / "inventory"
DEFAULT_LOCK_DIR = ROOT / "outputs" / "forecast_eval" / "inventory_step4_pre_v2_lock"
DEFAULT_SUMMARY_SCRIPT = ROOT / "scripts" / "forecast_eval" / "build_summary.py"

FRICTION_GRID = (0.0, 0.25, 0.5, 1.0)
SEASONAL_PERIOD = 28
ORDER_CAP = 80.0
HOLDING_W = 0.6
BASE_LEVEL = 18.0
DEFAULT_STOCKOUT_W = 2.0
DEFAULT_SAFETY_STOCK = 4.0
DEFAULT_BURST_AMP = 8.0
DEFAULT_INITIAL_PREV_ORDER = 0.0
TEMPERED_POSITIVE_ETA = 0.6
MLP_EPOCHS = 160
TRAIN_END = 112
DEFAULT_HORIZON = 280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inventory v2 with exact-control Q1 and live Q2.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for canonical inventory outputs.")
    parser.add_argument("--lock-dir", default=str(DEFAULT_LOCK_DIR), help="Directory for locking the pre-v2 inventory outputs.")
    parser.add_argument("--summary-script", default=str(DEFAULT_SUMMARY_SCRIPT), help="Path to build_summary.py.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to evaluate.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Total horizon including train and eval.")
    parser.add_argument("--train-end", type=int, default=TRAIN_END, help="Exclusive end index for the train window.")
    return parser.parse_args()


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _json_hash(values: np.ndarray) -> tuple[str, str]:
    payload = json.dumps(np.asarray(values, dtype=np.float64).round(12).tolist(), separators=(",", ":"), ensure_ascii=True)
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_demand(seed: int, horizon: int, burst_amp: float) -> np.ndarray:
    rng = np.random.default_rng(41_000 + int(seed))
    level = np.zeros(horizon, dtype=np.float64)
    demand = np.zeros(horizon, dtype=np.float64)
    for idx in range(horizon):
        if idx > 0:
            level[idx] = 0.8 * level[idx - 1] + rng.normal(0.0, 1.2)
        weekly = 3.5 * np.sin(2.0 * np.pi * idx / 7.0) + 1.5 * np.cos(2.0 * np.pi * idx / 14.0)
        cycle_day = idx % SEASONAL_PERIOD
        burst = 0.0
        if cycle_day in {18, 19, 20}:
            burst = float(burst_amp)
        elif cycle_day in {21, 22}:
            burst = -0.5 * float(burst_amp)
        noise = rng.normal(0.0, 1.0)
        demand[idx] = max(0.0, BASE_LEVEL + weekly + level[idx] + burst + noise)
    return demand


def _feature_row(demand: np.ndarray, idx: int) -> np.ndarray:
    return np.array(
        [
            demand[idx - 1],
            demand[idx - 2],
            demand[idx - 7],
            demand[idx - 7 : idx].mean(),
            np.sin(2.0 * np.pi * idx / 7.0),
            np.cos(2.0 * np.pi * idx / 7.0),
        ],
        dtype=np.float64,
    )


def _build_supervised_arrays(demand: np.ndarray, train_end: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_indices = np.arange(7, train_end, dtype=np.int64)
    eval_indices = np.arange(train_end, horizon, dtype=np.int64)
    x_train = np.vstack([_feature_row(demand, idx) for idx in train_indices]).astype(np.float64)
    y_train = demand[train_indices].astype(np.float64)
    x_eval = np.vstack([_feature_row(demand, idx) for idx in eval_indices]).astype(np.float64)
    return x_train, y_train, x_eval


def _fit_linear_ar_ridge(x_train: np.ndarray, y_train: np.ndarray, ridge_penalty: float = 1e-3) -> dict[str, np.ndarray]:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_scaled = (x_train - x_mean) / x_std
    x_aug = np.concatenate([np.ones((x_scaled.shape[0], 1), dtype=np.float64), x_scaled], axis=1)
    ridge = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge_penalty)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + ridge, x_aug.T @ y_train)
    return {"x_mean": x_mean, "x_std": x_std, "beta": beta}


def _predict_linear_ar(model: dict[str, np.ndarray], x_values: np.ndarray) -> np.ndarray:
    x_scaled = (x_values - model["x_mean"]) / model["x_std"]
    x_aug = np.concatenate([np.ones((x_scaled.shape[0], 1), dtype=np.float64), x_scaled], axis=1)
    return np.clip(x_aug @ model["beta"], 0.0, None)


def _fit_and_predict_mlp(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, *, seed: int) -> np.ndarray:
    torch.manual_seed(52_000 + int(seed))
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_std = 1.0 if y_std < 1e-8 else y_std

    x_train_t = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32)
    y_train_t = torch.tensor(((y_train - y_mean) / y_std)[:, None], dtype=torch.float32)
    x_eval_t = torch.tensor((x_eval - x_mean) / x_std, dtype=torch.float32)

    model = SmallMLP(x_train.shape[1]).cpu()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    model.train()
    for _epoch in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_eval_scaled = model(x_eval_t).squeeze(-1).cpu().numpy()
    return np.clip(y_eval_scaled * y_std + y_mean, 0.0, None)


def _build_forecast_cache(seed: int, burst_amp: float, horizon: int, train_end: int) -> dict[str, Any]:
    demand = _generate_demand(seed=seed, horizon=horizon, burst_amp=burst_amp)
    x_train, y_train, x_eval = _build_supervised_arrays(demand, train_end=train_end, horizon=horizon)
    eval_indices = np.arange(train_end, horizon, dtype=np.int64)
    eval_demand = demand[eval_indices]

    linear_model = _fit_linear_ar_ridge(x_train, y_train)
    forecasts = {
        "naive_last": np.clip(demand[eval_indices - 1], 0.0, None),
        "moving_average_7": np.array([demand[idx - 7 : idx].mean() for idx in eval_indices], dtype=np.float64),
        "linear_ar_ridge": _predict_linear_ar(linear_model, x_eval),
        "mlp_small": _fit_and_predict_mlp(x_train, y_train, x_eval, seed=seed),
    }
    forecast_metrics = {name: mae_score(values, eval_demand) for name, values in forecasts.items()}
    return {
        "demand": demand,
        "train_mean": float(demand[:train_end].mean()),
        "eval_indices": eval_indices,
        "eval_demand": eval_demand,
        "forecast_map": forecasts,
        "forecast_metrics": forecast_metrics,
    }


def _simulate_order_path(
    *,
    demand_eval: np.ndarray,
    orders: np.ndarray,
    friction_level: float,
    stockout_w: float,
    initial_inventory: float,
    initial_prev_order: float,
) -> dict[str, Any]:
    inventory = float(initial_inventory)
    prev_order = float(initial_prev_order)

    holding_costs: list[float] = []
    stockout_costs: list[float] = []
    change_costs: list[float] = []
    adjustments: list[float] = []
    inventories: list[float] = []
    stockout_units: list[float] = []
    executed_orders: list[float] = []
    cap_hits: list[bool] = []

    for demand_t, order_t in zip(demand_eval, orders, strict=True):
        q_exec = float(np.clip(order_t, 0.0, ORDER_CAP))
        adjustment = abs(q_exec - prev_order)
        available = inventory + q_exec
        stockout_units_t = max(float(demand_t) - available, 0.0)
        ending_inventory = max(available - float(demand_t), 0.0)
        holding_cost = HOLDING_W * ending_inventory
        stockout_cost = float(stockout_w) * stockout_units_t
        change_cost = float(friction_level) * adjustment

        holding_costs.append(holding_cost)
        stockout_costs.append(stockout_cost)
        change_costs.append(change_cost)
        adjustments.append(adjustment)
        inventories.append(ending_inventory)
        stockout_units.append(stockout_units_t)
        executed_orders.append(q_exec)
        cap_hits.append(bool(q_exec >= ORDER_CAP - 1e-8))

        inventory = ending_inventory
        prev_order = q_exec

    holding_arr = np.asarray(holding_costs, dtype=np.float64)
    stockout_arr = np.asarray(stockout_costs, dtype=np.float64)
    change_arr = np.asarray(change_costs, dtype=np.float64)
    adjustments_arr = np.asarray(adjustments, dtype=np.float64)
    inventories_arr = np.asarray(inventories, dtype=np.float64)
    stockout_units_arr = np.asarray(stockout_units, dtype=np.float64)
    executed_orders_arr = np.asarray(executed_orders, dtype=np.float64)
    total_cost_arr = holding_arr + stockout_arr + change_arr
    demand_total = float(np.asarray(demand_eval, dtype=np.float64).sum())
    stockout_total = float(stockout_units_arr.sum())

    return {
        "score": -float(total_cost_arr.mean()),
        "mean_holding_cost": float(holding_arr.mean()),
        "mean_stockout_cost": float(stockout_arr.mean()),
        "mean_change_cost": float(change_arr.mean()),
        "mean_order_adjustment": float(adjustments_arr.mean()),
        "order_cap_hit_rate": float(np.mean(cap_hits)),
        "mean_inventory": float(inventories_arr.mean()),
        "fill_rate": float(1.0 - stockout_total / max(demand_total, 1e-12)),
        "stockout_day_rate": float(np.mean(stockout_units_arr > 1e-12)),
        "final_inventory": float(inventory),
        "final_prev_order": float(prev_order),
        "executed_orders": executed_orders_arr,
        "change_costs": change_arr,
    }


def _run_live_inventory(
    *,
    demand_eval: np.ndarray,
    forecasts_eval: np.ndarray,
    safety_stock: float,
    stockout_w: float,
    friction_level: float,
    interface_id: str,
    initial_inventory: float,
    initial_prev_order: float,
) -> dict[str, Any]:
    inventory = float(initial_inventory)
    prev_order = float(initial_prev_order)

    q_targets: list[float] = []
    q_execs: list[float] = []
    inventories: list[float] = []
    holding_costs: list[float] = []
    stockout_costs: list[float] = []
    change_costs: list[float] = []
    adjustments: list[float] = []
    stockout_units: list[float] = []
    cap_hits: list[bool] = []

    for forecast_t, demand_t in zip(forecasts_eval, demand_eval, strict=True):
        q_target = float(np.clip(float(forecast_t) + float(safety_stock) - inventory, 0.0, ORDER_CAP))
        q_exec = q_target
        if interface_id == "tempered":
            eta = 1.0 if np.isclose(float(friction_level), 0.0, atol=1e-15) else TEMPERED_POSITIVE_ETA
            q_exec = float(np.clip(prev_order + eta * (q_target - prev_order), 0.0, ORDER_CAP))

        adjustment = abs(q_exec - prev_order)
        available = inventory + q_exec
        stockout_units_t = max(float(demand_t) - available, 0.0)
        ending_inventory = max(available - float(demand_t), 0.0)
        holding_cost = HOLDING_W * ending_inventory
        stockout_cost = float(stockout_w) * stockout_units_t
        change_cost = float(friction_level) * adjustment

        q_targets.append(q_target)
        q_execs.append(q_exec)
        inventories.append(ending_inventory)
        holding_costs.append(holding_cost)
        stockout_costs.append(stockout_cost)
        change_costs.append(change_cost)
        adjustments.append(adjustment)
        stockout_units.append(stockout_units_t)
        cap_hits.append(bool(q_exec >= ORDER_CAP - 1e-8))

        inventory = ending_inventory
        prev_order = q_exec

    q_targets_arr = np.asarray(q_targets, dtype=np.float64)
    q_execs_arr = np.asarray(q_execs, dtype=np.float64)
    holding_arr = np.asarray(holding_costs, dtype=np.float64)
    stockout_arr = np.asarray(stockout_costs, dtype=np.float64)
    change_arr = np.asarray(change_costs, dtype=np.float64)
    adjustments_arr = np.asarray(adjustments, dtype=np.float64)
    inventories_arr = np.asarray(inventories, dtype=np.float64)
    stockout_units_arr = np.asarray(stockout_units, dtype=np.float64)
    total_cost_arr = holding_arr + stockout_arr + change_arr
    demand_total = float(np.asarray(demand_eval, dtype=np.float64).sum())
    stockout_total = float(stockout_units_arr.sum())

    return {
        "score": -float(total_cost_arr.mean()),
        "q_targets": q_targets_arr,
        "q_execs": q_execs_arr,
        "final_inventory": float(inventory),
        "final_prev_order": float(prev_order),
        "mean_holding_cost": float(holding_arr.mean()),
        "mean_stockout_cost": float(stockout_arr.mean()),
        "mean_change_cost": float(change_arr.mean()),
        "mean_order_adjustment": float(adjustments_arr.mean()),
        "order_cap_hit_rate": float(np.mean(cap_hits)),
        "mean_inventory": float(inventories_arr.mean()),
        "fill_rate": float(1.0 - stockout_total / max(demand_total, 1e-12)),
        "stockout_day_rate": float(np.mean(stockout_units_arr > 1e-12)),
    }


def _run_q1_source_and_replay(
    *,
    seed: int,
    cache_entry: dict[str, Any],
    safety_stock: float,
    stockout_w: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    eval_demand = cache_entry["eval_demand"]
    forecasts_eval = np.asarray(cache_entry["forecast_map"]["linear_ar_ridge"], dtype=np.float64)
    forecast_metric = float(cache_entry["forecast_metrics"]["linear_ar_ridge"])
    initial_inventory_source = float(cache_entry["train_mean"] + float(safety_stock))
    initial_prev_order_source = float(DEFAULT_INITIAL_PREV_ORDER)

    source_result = _run_live_inventory(
        demand_eval=eval_demand,
        forecasts_eval=forecasts_eval,
        safety_stock=safety_stock,
        stockout_w=stockout_w,
        friction_level=0.0,
        interface_id="responsive",
        initial_inventory=initial_inventory_source,
        initial_prev_order=initial_prev_order_source,
    )
    q_target = np.asarray(source_result["q_targets"], dtype=np.float64)
    forecast_json, forecast_hash = _json_hash(forecasts_eval)
    proposal_json, proposal_hash = _json_hash(q_target)

    q1_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    pairing_failure_count = 0

    for friction_level in FRICTION_GRID:
        target_path_result = _simulate_order_path(
            demand_eval=eval_demand,
            orders=q_target,
            friction_level=friction_level,
            stockout_w=stockout_w,
            initial_inventory=initial_inventory_source,
            initial_prev_order=initial_prev_order_source,
        )
        for interface_id in ("responsive", "tempered"):
            if interface_id == "responsive":
                executed_orders = q_target.copy()
            else:
                eta = 1.0 if np.isclose(float(friction_level), 0.0, atol=1e-15) else TEMPERED_POSITIVE_ETA
                replay_orders = np.zeros_like(q_target, dtype=np.float64)
                prev_order = float(initial_prev_order_source)
                for idx, q_target_t in enumerate(q_target):
                    replay_orders[idx] = float(np.clip(prev_order + eta * (q_target_t - prev_order), 0.0, ORDER_CAP))
                    prev_order = replay_orders[idx]
                executed_orders = replay_orders

            executed_result = _simulate_order_path(
                demand_eval=eval_demand,
                orders=executed_orders,
                friction_level=friction_level,
                stockout_w=stockout_w,
                initial_inventory=initial_inventory_source,
                initial_prev_order=initial_prev_order_source,
            )
            q1_rows.append(
                build_result_row(
                    question_id="Q1",
                    scenario_id="inventory_control_v2_q1",
                    domain="inventory",
                    seed=seed,
                    forecaster_id="linear_ar_ridge",
                    interface_id=interface_id,
                    friction_level=float(friction_level),
                    forecast_metric=forecast_metric,
                    target_metric=float(target_path_result["score"]),
                    executed_metric=float(executed_result["score"]),
                    realized_cost=float(executed_result["mean_change_cost"]),
                    realized_turnover_or_adjustment=float(executed_result["mean_order_adjustment"]),
                )
            )
            diagnostics_rows.append(
                {
                    "question_id": "Q1",
                    "scenario_id": "inventory_control_v2_q1",
                    "domain": "inventory",
                    "seed": int(seed),
                    "forecaster_id": "linear_ar_ridge",
                    "interface_id": interface_id,
                    "friction_level": float(friction_level),
                    "mean_holding_cost": float(executed_result["mean_holding_cost"]),
                    "mean_stockout_cost": float(executed_result["mean_stockout_cost"]),
                    "mean_change_cost": float(executed_result["mean_change_cost"]),
                    "mean_order_adjustment": float(executed_result["mean_order_adjustment"]),
                    "order_cap_hit_rate": float(executed_result["order_cap_hit_rate"]),
                    "mean_inventory": float(executed_result["mean_inventory"]),
                    "fill_rate": float(executed_result["fill_rate"]),
                    "stockout_day_rate": float(executed_result["stockout_day_rate"]),
                }
            )

        if 2 != len([row for row in q1_rows if row["seed"] == seed and np.isclose(row["friction_level"], friction_level)]):
            pairing_failure_count += 1

    freeze_check = {
        "seed": int(seed),
        "forecast_hash": forecast_hash,
        "proposal_hash": proposal_hash,
        "forecast_hash_identical_flag": True,
        "proposal_hash_identical_flag": True,
        "initial_inventory_source": float(initial_inventory_source),
        "initial_prev_order_source": float(initial_prev_order_source),
        "initial_inventory_match_flag": True,
        "initial_prev_order_match_flag": True,
        "pairing_failure_count": int(pairing_failure_count),
        "forecast_sequence_json": forecast_json,
        "proposal_path_json": proposal_json,
    }
    return q1_rows, diagnostics_rows, freeze_check


def _run_q2_live(
    *,
    seed: int,
    cache_entry: dict[str, Any],
    safety_stock: float,
    stockout_w: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    eval_demand = cache_entry["eval_demand"]
    initial_inventory = float(cache_entry["train_mean"] + float(safety_stock))
    initial_prev_order = float(DEFAULT_INITIAL_PREV_ORDER)

    q2_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    for friction_level in FRICTION_GRID:
        for forecaster_id in ("naive_last", "moving_average_7", "linear_ar_ridge", "mlp_small"):
            live_result = _run_live_inventory(
                demand_eval=eval_demand,
                forecasts_eval=np.asarray(cache_entry["forecast_map"][forecaster_id], dtype=np.float64),
                safety_stock=safety_stock,
                stockout_w=stockout_w,
                friction_level=friction_level,
                interface_id="responsive",
                initial_inventory=initial_inventory,
                initial_prev_order=initial_prev_order,
            )
            q2_rows.append(
                build_result_row(
                    question_id="Q2",
                    scenario_id="inventory_control_v2_q2",
                    domain="inventory",
                    seed=seed,
                    forecaster_id=forecaster_id,
                    interface_id="responsive",
                    friction_level=float(friction_level),
                    forecast_metric=float(cache_entry["forecast_metrics"][forecaster_id]),
                    target_metric=float(live_result["score"]),
                    executed_metric=float(live_result["score"]),
                    realized_cost=float(live_result["mean_change_cost"]),
                    realized_turnover_or_adjustment=float(live_result["mean_order_adjustment"]),
                )
            )
            diagnostics_rows.append(
                {
                    "question_id": "Q2",
                    "scenario_id": "inventory_control_v2_q2",
                    "domain": "inventory",
                    "seed": int(seed),
                    "forecaster_id": forecaster_id,
                    "interface_id": "responsive",
                    "friction_level": float(friction_level),
                    "mean_holding_cost": float(live_result["mean_holding_cost"]),
                    "mean_stockout_cost": float(live_result["mean_stockout_cost"]),
                    "mean_change_cost": float(live_result["mean_change_cost"]),
                    "mean_order_adjustment": float(live_result["mean_order_adjustment"]),
                    "order_cap_hit_rate": float(live_result["order_cap_hit_rate"]),
                    "mean_inventory": float(live_result["mean_inventory"]),
                    "fill_rate": float(live_result["fill_rate"]),
                    "stockout_day_rate": float(live_result["stockout_day_rate"]),
                }
            )

    return q2_rows, diagnostics_rows


def _pairwise_disagreement_stats(q2_df: pd.DataFrame) -> dict[str, Any]:
    rates_by_friction: dict[float, list[float]] = {float(level): [] for level in FRICTION_GRID}
    flip_counts: dict[tuple[float, str, str], int] = {}
    seeds_by_friction: dict[float, set[int]] = {float(level): set() for level in FRICTION_GRID}

    for (seed, friction_level), group in q2_df.groupby(["seed", "friction_level"], sort=True):
        seeds_by_friction[float(friction_level)].add(int(seed))
        group = group.sort_values("forecaster_id").reset_index(drop=True)
        flips = 0
        total = 0
        for left, right in itertools.combinations(group.itertuples(index=False), 2):
            forecast_order = np.sign(int(left.rank_within_forecast_metric) - int(right.rank_within_forecast_metric))
            executed_order = np.sign(int(left.rank_within_executed_metric) - int(right.rank_within_executed_metric))
            if forecast_order == 0 or executed_order == 0:
                continue
            total += 1
            if forecast_order != executed_order:
                flips += 1
                pair_key = tuple(sorted((str(left.forecaster_id), str(right.forecaster_id))))
                flip_counts[(float(friction_level), pair_key[0], pair_key[1])] = flip_counts.get(
                    (float(friction_level), pair_key[0], pair_key[1]),
                    0,
                ) + 1
        rates_by_friction[float(friction_level)].append(float(flips / total) if total else 0.0)

    mean_rates = {level: (float(np.mean(values)) if values else 0.0) for level, values in rates_by_friction.items()}
    max_pair_flip_share = 0.0
    for friction_level in (0.5, 1.0):
        seed_count = max(len(seeds_by_friction[float(friction_level)]), 1)
        for key, flips in flip_counts.items():
            if np.isclose(key[0], friction_level, atol=1e-15):
                max_pair_flip_share = max(max_pair_flip_share, float(flips / seed_count))
    return {
        "mean_disagreement_by_friction": mean_rates,
        "max_pair_flip_share_positive": max_pair_flip_share,
    }


def _q1_acceptance(q1_df: pd.DataFrame, freeze_df: pd.DataFrame) -> tuple[bool, dict[str, Any], list[str]]:
    fail_reasons: list[str] = []
    freeze_ok = bool(
        freeze_df["forecast_hash_identical_flag"].all()
        and freeze_df["proposal_hash_identical_flag"].all()
        and freeze_df["initial_inventory_match_flag"].all()
        and freeze_df["initial_prev_order_match_flag"].all()
        and (freeze_df["pairing_failure_count"] == 0).all()
    )
    if not freeze_ok:
        fail_reasons.append("freeze_check_failed")

    zero_gap = float(
        q1_df[np.isclose(q1_df["friction_level"], 0.0, atol=1e-15)]["target_executed_gap"].abs().mean()
    )
    if zero_gap > 1e-10:
        fail_reasons.append("q1_zero_gap_not_zero")

    positive_gap_ok = True
    positive_gap_map: dict[float, float] = {}
    for friction_level in (0.25, 0.5, 1.0):
        mean_abs_gap = float(
            q1_df[np.isclose(q1_df["friction_level"], friction_level, atol=1e-15)]["target_executed_gap"].abs().mean()
        )
        positive_gap_map[float(friction_level)] = mean_abs_gap
        if mean_abs_gap <= 1e-10:
            positive_gap_ok = False
    if not positive_gap_ok:
        fail_reasons.append("q1_positive_gap_missing")

    pivot = (
        q1_df.pivot_table(
            index=["seed", "friction_level"],
            columns="interface_id",
            values="executed_metric",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    wins_05 = int(
        (
            pivot[np.isclose(pivot["friction_level"], 0.5, atol=1e-15)]["tempered"]
            > pivot[np.isclose(pivot["friction_level"], 0.5, atol=1e-15)]["responsive"]
        ).sum()
    )
    wins_10 = int(
        (
            pivot[np.isclose(pivot["friction_level"], 1.0, atol=1e-15)]["tempered"]
            > pivot[np.isclose(pivot["friction_level"], 1.0, atol=1e-15)]["responsive"]
        ).sum()
    )
    if wins_05 < 2 or wins_10 < 2:
        fail_reasons.append("q1_tempered_not_winning_enough")

    return (
        len(fail_reasons) == 0,
        {
            "q1_zero_gap_abs_mean": zero_gap,
            "q1_positive_gap_abs_mean_025": positive_gap_map[0.25],
            "q1_positive_gap_abs_mean_05": positive_gap_map[0.5],
            "q1_positive_gap_abs_mean_10": positive_gap_map[1.0],
            "q1_tempered_wins_at_05": wins_05,
            "q1_tempered_wins_at_10": wins_10,
        },
        fail_reasons,
    )


def _q2_acceptance(q2_df: pd.DataFrame) -> tuple[bool, dict[str, Any], list[str]]:
    fail_reasons: list[str] = []
    disagreement = _pairwise_disagreement_stats(q2_df)
    zero_mean = float(disagreement["mean_disagreement_by_friction"][0.0])
    mean_05 = float(disagreement["mean_disagreement_by_friction"][0.5])
    mean_10 = float(disagreement["mean_disagreement_by_friction"][1.0])
    max_flip_share = float(disagreement["max_pair_flip_share_positive"])

    if zero_mean > 0.10:
        fail_reasons.append("q2_zero_disagreement_too_high")
    if mean_05 <= 0.0 or mean_10 <= 0.0:
        fail_reasons.append("q2_positive_disagreement_missing")
    if max_flip_share < (2.0 / 3.0):
        fail_reasons.append("q2_pair_flip_share_too_low")

    return (
        len(fail_reasons) == 0,
        {
            "q2_zero_disagreement_mean": zero_mean,
            "q2_disagreement_mean_025": float(disagreement["mean_disagreement_by_friction"][0.25]),
            "q2_disagreement_mean_05": mean_05,
            "q2_disagreement_mean_10": mean_10,
            "q2_max_pair_flip_share_positive": max_flip_share,
        },
        fail_reasons,
    )


def _diagnostics_acceptance(diagnostics_df: pd.DataFrame) -> tuple[bool, dict[str, Any], list[str]]:
    fail_reasons: list[str] = []
    max_cap_hit_rate = float(diagnostics_df["order_cap_hit_rate"].max())
    if max_cap_hit_rate >= 0.01:
        fail_reasons.append("diagnostics_cap_hit_too_high")

    q2_diag = diagnostics_df[diagnostics_df["question_id"] == "Q2"].copy()
    holding_total = float(q2_diag["mean_holding_cost"].sum())
    stockout_total = float(q2_diag["mean_stockout_cost"].sum())
    non_change_total = holding_total + stockout_total
    holding_share = holding_total / max(non_change_total, 1e-12)
    stockout_share = stockout_total / max(non_change_total, 1e-12)
    if holding_share < 0.15 or stockout_share < 0.15:
        fail_reasons.append("diagnostics_cost_balance_bad")

    return (
        len(fail_reasons) == 0,
        {
            "diagnostics_max_cap_hit_rate": max_cap_hit_rate,
            "diagnostics_holding_share": holding_share,
            "diagnostics_stockout_share": stockout_share,
        },
        fail_reasons,
    )


def _search_order() -> list[tuple[float, float, float]]:
    burst_grid = [6.0, 8.0, 10.0]
    safety_grid = [3.0, 4.0, 5.0]
    stockout_grid = [1.8, 2.0, 2.2]
    center = (8.0, 4.0, 2.0)
    one_hop = [
        (6.0, 4.0, 2.0),
        (10.0, 4.0, 2.0),
        (8.0, 3.0, 2.0),
        (8.0, 5.0, 2.0),
        (8.0, 4.0, 1.8),
        (8.0, 4.0, 2.2),
    ]
    remaining = sorted(set(itertools.product(burst_grid, safety_grid, stockout_grid)) - {center} - set(one_hop))
    return [center] + one_hop + remaining


def _lock_old_inventory_outputs(lock_dir: Path, output_dir: Path) -> None:
    lock_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("q1_same_forecast_diff_interface.csv", "q2_diff_forecasts_same_interface.csv"):
        src = output_dir / filename
        if src.exists():
            shutil.copy2(src, lock_dir / filename)


def _refresh_master_summary(summary_script: Path) -> None:
    subprocess.run([sys.executable, str(summary_script)], cwd=str(ROOT), check=True)


def _evaluate_candidate(
    *,
    seeds: list[int],
    data_cache: dict[tuple[int, float], dict[str, Any]],
    burst_amp: float,
    safety_stock: float,
    stockout_w: float,
) -> dict[str, Any]:
    q1_rows: list[dict[str, object]] = []
    q2_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    freeze_rows: list[dict[str, object]] = []

    for seed in seeds:
        cache_entry = data_cache[(int(seed), float(burst_amp))]
        seed_q1_rows, seed_q1_diag, freeze_check = _run_q1_source_and_replay(
            seed=int(seed),
            cache_entry=cache_entry,
            safety_stock=safety_stock,
            stockout_w=stockout_w,
        )
        seed_q2_rows, seed_q2_diag = _run_q2_live(
            seed=int(seed),
            cache_entry=cache_entry,
            safety_stock=safety_stock,
            stockout_w=stockout_w,
        )
        q1_rows.extend(seed_q1_rows)
        q2_rows.extend(seed_q2_rows)
        diagnostics_rows.extend(seed_q1_diag)
        diagnostics_rows.extend(seed_q2_diag)
        freeze_rows.append(freeze_check)

    q1_df = prepare_results_frame(q1_rows)
    q2_df = prepare_results_frame(q2_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values(
        ["question_id", "seed", "friction_level", "forecaster_id", "interface_id"]
    ).reset_index(drop=True)
    freeze_df = pd.DataFrame(freeze_rows).sort_values("seed").reset_index(drop=True)

    q1_pass, q1_metrics, q1_fail = _q1_acceptance(q1_df, freeze_df)
    q2_pass, q2_metrics, q2_fail = _q2_acceptance(q2_df)
    diagnostics_pass, diag_metrics, diag_fail = _diagnostics_acceptance(diagnostics_df)
    fail_reasons = q1_fail + q2_fail + diag_fail

    return {
        "q1_df": q1_df,
        "q2_df": q2_df,
        "diagnostics_df": diagnostics_df,
        "freeze_df": freeze_df,
        "q1_pass": q1_pass,
        "q2_pass": q2_pass,
        "diagnostics_pass": diagnostics_pass,
        "pass_all": q1_pass and q2_pass and diagnostics_pass,
        "metrics": {**q1_metrics, **q2_metrics, **diag_metrics},
        "fail_reason": ";".join(fail_reasons) if fail_reasons else "",
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    lock_dir = Path(args.lock_dir)
    summary_script = Path(args.summary_script)
    horizon = int(args.horizon)
    train_end = int(args.train_end)
    seeds = [int(seed) for seed in args.seeds]

    if horizon <= train_end or train_end < 8:
        raise ValueError("Expected horizon > train_end and train_end >= 8.")

    output_dir.mkdir(parents=True, exist_ok=True)

    burst_values = {config[0] for config in _search_order()}
    data_cache: dict[tuple[int, float], dict[str, Any]] = {}
    for burst_amp in sorted(burst_values):
        for seed in seeds:
            data_cache[(int(seed), float(burst_amp))] = _build_forecast_cache(
                seed=int(seed),
                burst_amp=float(burst_amp),
                horizon=horizon,
                train_end=train_end,
            )

    calibration_rows: list[dict[str, object]] = []
    selected_candidate: dict[str, Any] | None = None
    selected_config: tuple[float, float, float] | None = None

    for burst_amp, safety_stock, stockout_w in _search_order():
        candidate = _evaluate_candidate(
            seeds=seeds,
            data_cache=data_cache,
            burst_amp=float(burst_amp),
            safety_stock=float(safety_stock),
            stockout_w=float(stockout_w),
        )
        calibration_row = {
            "burst_amp": float(burst_amp),
            "safety_stock": float(safety_stock),
            "stockout_w": float(stockout_w),
            "q1_pass": bool(candidate["q1_pass"]),
            "q2_pass": bool(candidate["q2_pass"]),
            "diagnostics_pass": bool(candidate["diagnostics_pass"]),
            "selected_flag": False,
            "fail_reason": str(candidate["fail_reason"]),
            **candidate["metrics"],
        }
        calibration_rows.append(calibration_row)
        if candidate["pass_all"]:
            selected_candidate = candidate
            selected_config = (float(burst_amp), float(safety_stock), float(stockout_w))
            calibration_row["selected_flag"] = True
            break

    calibration_log_df = pd.DataFrame(calibration_rows)
    calibration_log_df.to_csv(output_dir / "inventory_v2_calibration_log.csv", index=False)

    if selected_candidate is None or selected_config is None:
        raise RuntimeError(
            "No inventory v2 config satisfied the acceptance criteria. "
            f"See {output_dir / 'inventory_v2_calibration_log.csv'} for details."
        )

    _lock_old_inventory_outputs(lock_dir=lock_dir, output_dir=output_dir)

    save_results(selected_candidate["q1_df"], output_dir / "q1_same_forecast_diff_interface.csv")
    save_results(selected_candidate["q2_df"], output_dir / "q2_diff_forecasts_same_interface.csv")
    selected_candidate["diagnostics_df"].to_csv(output_dir / "inventory_v2_diagnostics.csv", index=False)
    selected_candidate["freeze_df"].to_csv(output_dir / "inventory_v2_q1_freeze_check.csv", index=False)

    _refresh_master_summary(summary_script=summary_script)

    print(
        "[inventory-v2] selected config "
        f"burst_amp={selected_config[0]} safety_stock={selected_config[1]} stockout_w={selected_config[2]}"
    )
    print(f"[inventory-v2] wrote Q1 rows={len(selected_candidate['q1_df'])} to {output_dir / 'q1_same_forecast_diff_interface.csv'}")
    print(f"[inventory-v2] wrote Q2 rows={len(selected_candidate['q2_df'])} to {output_dir / 'q2_diff_forecasts_same_interface.csv'}")
    print(f"[inventory-v2] wrote diagnostics to {output_dir / 'inventory_v2_diagnostics.csv'}")
    print(f"[inventory-v2] wrote freeze check to {output_dir / 'inventory_v2_q1_freeze_check.csv'}")
    print(f"[inventory-v2] wrote calibration log to {output_dir / 'inventory_v2_calibration_log.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
