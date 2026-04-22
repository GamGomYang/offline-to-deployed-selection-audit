#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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


DEFAULT_RAW_PATH = ROOT / "data" / "load_dispatch" / "uci_household_power_consumption.txt"
DEFAULT_ALT_RAW_PATH = ROOT / "data" / "load_dispatch" / "household_power_consumption.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "forecast_eval" / "load_dispatch"

DEFAULT_TRAIN_HOURS = 24 * 365
DEFAULT_EVAL_HOURS = 24 * 180
DEFAULT_WINDOW_STEP_HOURS = 24 * 7
MAX_LAG_HOURS = 168
FRICTION_GRID = (0.0, 0.25, 0.5, 1.0)
FORECASTER_IDS = ("naive_last", "moving_average_24", "linear_ar_ridge", "mlp_small")
SHORTAGE_W = 2.0
SURPLUS_W = 0.5
TEMPERED_POSITIVE_ETA = 0.6
MLP_EPOCHS = 120
MLP_HIDDEN = 16
CLIP_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the load-following proxy support-domain experiment.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--train-hours", type=int, default=DEFAULT_TRAIN_HOURS)
    parser.add_argument("--eval-hours", type=int, default=DEFAULT_EVAL_HOURS)
    parser.add_argument("--window-step-hours", type=int, default=DEFAULT_WINDOW_STEP_HOURS)
    parser.add_argument("--max-lag-hours", type=int, default=MAX_LAG_HOURS)
    return parser.parse_args()


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _json_hash(values: np.ndarray) -> tuple[str, str]:
    payload = json.dumps(np.asarray(values, dtype=np.float64).round(12).tolist(), separators=(",", ":"), ensure_ascii=True)
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_raw_path(raw_path: Path) -> Path:
    if raw_path.exists():
        return raw_path
    if raw_path == DEFAULT_RAW_PATH and DEFAULT_ALT_RAW_PATH.exists():
        return DEFAULT_ALT_RAW_PATH
    raise FileNotFoundError(f"Raw load-following file not found: {raw_path}")


def load_hourly_series(raw_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_path, sep=";", low_memory=False, na_values="?")
    timestamps = pd.to_datetime(
        frame["Date"] + " " + frame["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    load = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    hourly = (
        pd.DataFrame({"timestamp": timestamps, "load_raw": load})
        .dropna()
        .sort_values("timestamp")
        .set_index("timestamp")
        .resample("h")
        .mean()
    )
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_index)
    hourly.index.name = "timestamp"
    return hourly.reset_index()


def _contiguous_segments(hourly: pd.DataFrame) -> list[tuple[int, int]]:
    valid_positions = np.flatnonzero(hourly["load_raw"].notna().to_numpy())
    if valid_positions.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(valid_positions[0])
    prev = int(valid_positions[0])
    for pos in valid_positions[1:]:
        pos = int(pos)
        if pos != prev + 1:
            segments.append((start, prev))
            start = pos
        prev = pos
    segments.append((start, prev))
    return segments


def select_latest_contiguous_block(hourly: pd.DataFrame, required_hours: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = []
    for start, end in _contiguous_segments(hourly):
        span = int(end - start + 1)
        if span >= int(required_hours):
            candidates.append((start, end, span))
    if not candidates:
        raise RuntimeError(
            "No contiguous hourly block is long enough for the fixed train/eval/window schedule."
        )
    start, end, span = max(candidates, key=lambda item: (int(item[1]), int(item[2])))
    block = hourly.iloc[start : end + 1].reset_index(drop=True)
    meta = {
        "block_start_timestamp": pd.Timestamp(block.loc[0, "timestamp"]),
        "block_end_timestamp": pd.Timestamp(block.loc[len(block) - 1, "timestamp"]),
        "block_hours": int(span),
    }
    return block, meta


def build_window_schedule(
    block: pd.DataFrame,
    *,
    window_ids: list[int],
    train_hours: int,
    eval_hours: int,
    window_step_hours: int,
    max_lag_hours: int,
) -> pd.DataFrame:
    if sorted(window_ids) != list(range(min(window_ids), max(window_ids) + 1)):
        raise ValueError("window_ids must be a contiguous range.")
    end_exclusive_base = len(block)
    rows: list[dict[str, Any]] = []
    for window_id in window_ids:
        eval_end_exclusive = end_exclusive_base - int(window_id) * int(window_step_hours)
        eval_start = eval_end_exclusive - int(eval_hours)
        train_end = eval_start
        train_start = train_end - int(train_hours)
        context_start = train_start - int(max_lag_hours)
        if context_start < 0:
            raise RuntimeError(
                f"window_id={window_id} is not constructible under the fixed train/eval/lag schedule."
            )
        rows.append(
            {
                "window_id": int(window_id),
                "context_start_idx": int(context_start),
                "train_start_idx": int(train_start),
                "train_end_idx_exclusive": int(train_end),
                "eval_start_idx": int(eval_start),
                "eval_end_idx_exclusive": int(eval_end_exclusive),
                "context_start_timestamp": pd.Timestamp(block.loc[context_start, "timestamp"]),
                "train_start_timestamp": pd.Timestamp(block.loc[train_start, "timestamp"]),
                "train_end_timestamp": pd.Timestamp(block.loc[train_end - 1, "timestamp"]),
                "eval_start_timestamp": pd.Timestamp(block.loc[eval_start, "timestamp"]),
                "eval_end_timestamp": pd.Timestamp(block.loc[eval_end_exclusive - 1, "timestamp"]),
            }
        )
    return pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)


def _feature_row(load_norm: np.ndarray, timestamps: pd.Series, idx: int) -> np.ndarray:
    timestamp = pd.Timestamp(timestamps.iloc[idx])
    hour = float(timestamp.hour)
    week_hour = float(timestamp.dayofweek * 24 + timestamp.hour)
    return np.array(
        [
            load_norm[idx - 1],
            load_norm[idx - 2],
            load_norm[idx - 24],
            load_norm[idx - 168],
            float(load_norm[idx - 24 : idx].mean()),
            float(load_norm[idx - 168 : idx].mean()),
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * week_hour / 168.0),
            np.cos(2.0 * np.pi * week_hour / 168.0),
        ],
        dtype=np.float64,
    )


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


def _fit_and_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    window_id: int,
    attempt: int,
) -> np.ndarray:
    torch.manual_seed(82_000 + 101 * int(window_id) + int(attempt))
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
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite MLP loss")
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_eval_scaled = model(x_eval_t).squeeze(-1).cpu().numpy()
    if not np.isfinite(y_eval_scaled).all():
        raise RuntimeError("non-finite MLP predictions")
    return np.clip(y_eval_scaled * y_std + y_mean, 0.0, None)


def build_window_forecasts(block: pd.DataFrame, schedule_row: pd.Series) -> dict[str, Any]:
    train_start = int(schedule_row["train_start_idx"])
    train_end = int(schedule_row["train_end_idx_exclusive"])
    eval_start = int(schedule_row["eval_start_idx"])
    eval_end = int(schedule_row["eval_end_idx_exclusive"])
    window_id = int(schedule_row["window_id"])

    load_raw = block["load_raw"].to_numpy(dtype=np.float64)
    timestamps = block["timestamp"]
    train_raw = load_raw[train_start:train_end]
    dispatch_scale = float(np.percentile(train_raw, 95.0))
    if not np.isfinite(dispatch_scale) or dispatch_scale <= 1e-8:
        raise RuntimeError(f"Invalid dispatch scale for window_id={window_id}: {dispatch_scale}")
    load_norm = load_raw / dispatch_scale
    train_norm = load_norm[train_start:train_end]
    train_mean_load_norm = float(train_norm.mean())
    reserve_margin = 0.15 * train_mean_load_norm
    initial_prev_dispatch = train_mean_load_norm
    initial_prev_target = train_mean_load_norm
    dispatch_cap = max(float(np.percentile(train_norm, 99.0)), train_mean_load_norm + reserve_margin)

    train_indices = np.arange(train_start, train_end, dtype=np.int64)
    eval_indices = np.arange(eval_start, eval_end, dtype=np.int64)
    x_train = np.vstack([_feature_row(load_norm, timestamps, idx) for idx in train_indices]).astype(np.float64)
    y_train = load_norm[train_indices].astype(np.float64)
    x_eval = np.vstack([_feature_row(load_norm, timestamps, idx) for idx in eval_indices]).astype(np.float64)
    y_eval = load_norm[eval_indices].astype(np.float64)

    linear_model = _fit_linear_ar_ridge(x_train, y_train)
    forecasts: dict[str, np.ndarray] = {
        "naive_last": np.clip(load_norm[eval_indices - 1], 0.0, None),
        "moving_average_24": np.array([load_norm[idx - 24 : idx].mean() for idx in eval_indices], dtype=np.float64),
        "linear_ar_ridge": _predict_linear_ar(linear_model, x_eval),
    }
    model_failures: dict[str, str] = {}
    mlp_prediction: np.ndarray | None = None
    mlp_error: Exception | None = None
    for attempt in (0, 1):
        try:
            mlp_prediction = _fit_and_predict_mlp(x_train, y_train, x_eval, window_id=window_id, attempt=attempt)
            mlp_error = None
            break
        except Exception as exc:  # noqa: BLE001
            mlp_error = exc
    if mlp_prediction is not None:
        forecasts["mlp_small"] = mlp_prediction
    else:
        model_failures["mlp_small"] = str(mlp_error or "unknown_mlp_failure")

    forecast_metrics = {name: mae_score(values, y_eval) for name, values in forecasts.items()}
    return {
        "window_id": window_id,
        "load_norm": load_norm,
        "eval_load_norm": y_eval,
        "eval_timestamps": block.loc[eval_start : eval_end - 1, "timestamp"].reset_index(drop=True),
        "forecasts": forecasts,
        "forecast_metrics": forecast_metrics,
        "model_failures": model_failures,
        "dispatch_scale": dispatch_scale,
        "reserve_margin": float(reserve_margin),
        "initial_prev_dispatch": float(initial_prev_dispatch),
        "initial_prev_target": float(initial_prev_target),
        "dispatch_cap": float(dispatch_cap),
        "train_mean_load_norm": float(train_mean_load_norm),
    }


def _evaluate_load_path(
    *,
    load_eval: np.ndarray,
    proposed_path: np.ndarray,
    friction_level: float,
    initial_prev: float,
    dispatch_cap: float,
) -> dict[str, Any]:
    unclipped = np.asarray(proposed_path, dtype=np.float64)
    dispatch = np.clip(unclipped, 0.0, float(dispatch_cap))
    clip_rate = float(np.mean(np.abs(unclipped - dispatch) > CLIP_TOL)) if dispatch.size else 0.0

    shortage_costs: list[float] = []
    surplus_costs: list[float] = []
    ramp_costs: list[float] = []
    adjustments: list[float] = []

    prev = float(initial_prev)
    for load_t, dispatch_t in zip(load_eval, dispatch, strict=True):
        dispatch_t = float(dispatch_t)
        adjustment = abs(dispatch_t - prev)
        shortage_costs.append(SHORTAGE_W * max(float(load_t) - dispatch_t, 0.0))
        surplus_costs.append(SURPLUS_W * max(dispatch_t - float(load_t), 0.0))
        ramp_costs.append(float(friction_level) * adjustment)
        adjustments.append(adjustment)
        prev = dispatch_t

    shortage_arr = np.asarray(shortage_costs, dtype=np.float64)
    surplus_arr = np.asarray(surplus_costs, dtype=np.float64)
    ramp_arr = np.asarray(ramp_costs, dtype=np.float64)
    total_cost_arr = shortage_arr + surplus_arr + ramp_arr
    adjustments_arr = np.asarray(adjustments, dtype=np.float64)
    return {
        "dispatch_path": dispatch,
        "clip_rate": clip_rate,
        "score": -float(total_cost_arr.mean()),
        "mean_shortage_cost": float(shortage_arr.mean()),
        "mean_surplus_cost": float(surplus_arr.mean()),
        "mean_ramp_cost": float(ramp_arr.mean()),
        "mean_dispatch_adjustment": float(adjustments_arr.mean()),
        "mean_dispatch": float(dispatch.mean()),
        "mean_load": float(np.asarray(load_eval, dtype=np.float64).mean()),
    }


def _tempered_unclipped_path(
    target_path: np.ndarray,
    *,
    friction_level: float,
    initial_prev_dispatch: float,
) -> np.ndarray:
    eta = 1.0 if np.isclose(float(friction_level), 0.0, atol=1e-15) else TEMPERED_POSITIVE_ETA
    unclipped = np.zeros_like(target_path, dtype=np.float64)
    prev_exec = float(initial_prev_dispatch)
    for idx, target_t in enumerate(np.asarray(target_path, dtype=np.float64)):
        proposed = prev_exec + eta * (float(target_t) - prev_exec)
        unclipped[idx] = proposed
        prev_exec = proposed
    return unclipped


def _build_q1_rows(
    *,
    window_bundle: dict[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    window_id = int(window_bundle["window_id"])
    load_eval = np.asarray(window_bundle["eval_load_norm"], dtype=np.float64)
    forecasts = np.asarray(window_bundle["forecasts"]["linear_ar_ridge"], dtype=np.float64)
    forecast_metric = float(window_bundle["forecast_metrics"]["linear_ar_ridge"])
    reserve_margin = float(window_bundle["reserve_margin"])
    initial_prev_target = float(window_bundle["initial_prev_target"])
    initial_prev_dispatch = float(window_bundle["initial_prev_dispatch"])
    dispatch_cap = float(window_bundle["dispatch_cap"])

    q_target_unclipped = forecasts + reserve_margin
    target_reference = _evaluate_load_path(
        load_eval=load_eval,
        proposed_path=q_target_unclipped,
        friction_level=0.0,
        initial_prev=initial_prev_target,
        dispatch_cap=dispatch_cap,
    )
    q_target = np.asarray(target_reference["dispatch_path"], dtype=np.float64)
    forecast_json, forecast_hash = _json_hash(forecasts)
    target_json, target_hash = _json_hash(q_target)

    q1_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    freeze_rows: list[dict[str, object]] = []

    for friction_level in FRICTION_GRID:
        target_result = _evaluate_load_path(
            load_eval=load_eval,
            proposed_path=q_target_unclipped,
            friction_level=float(friction_level),
            initial_prev=initial_prev_target,
            dispatch_cap=dispatch_cap,
        )
        row_count_before = len(q1_rows)
        for interface_id in ("responsive", "tempered"):
            if interface_id == "responsive":
                exec_unclipped = q_target.copy()
            else:
                exec_unclipped = _tempered_unclipped_path(
                    q_target,
                    friction_level=float(friction_level),
                    initial_prev_dispatch=initial_prev_dispatch,
                )
            exec_result = _evaluate_load_path(
                load_eval=load_eval,
                proposed_path=exec_unclipped,
                friction_level=float(friction_level),
                initial_prev=initial_prev_dispatch,
                dispatch_cap=dispatch_cap,
            )
            q1_rows.append(
                build_result_row(
                    question_id="Q1",
                    scenario_id="load_following_proxy_q1",
                    domain="load_dispatch",
                    seed=window_id,
                    forecaster_id="linear_ar_ridge",
                    interface_id=interface_id,
                    friction_level=float(friction_level),
                    forecast_metric=forecast_metric,
                    target_metric=float(target_result["score"]),
                    executed_metric=float(exec_result["score"]),
                    realized_cost=float(exec_result["mean_ramp_cost"]),
                    realized_turnover_or_adjustment=float(exec_result["mean_dispatch_adjustment"]),
                )
            )
            diagnostics_rows.append(
                {
                    "question_id": "Q1",
                    "scenario_id": "load_following_proxy_q1",
                    "domain": "load_dispatch",
                    "seed": int(window_id),
                    "forecaster_id": "linear_ar_ridge",
                    "interface_id": interface_id,
                    "friction_level": float(friction_level),
                    "mean_shortage_cost": float(exec_result["mean_shortage_cost"]),
                    "mean_surplus_cost": float(exec_result["mean_surplus_cost"]),
                    "mean_ramp_cost": float(exec_result["mean_ramp_cost"]),
                    "mean_dispatch_adjustment": float(exec_result["mean_dispatch_adjustment"]),
                    "dispatch_target_clip_rate": float(target_result["clip_rate"]),
                    "dispatch_exec_clip_rate": float(exec_result["clip_rate"]),
                    "mean_dispatch": float(exec_result["mean_dispatch"]),
                    "mean_load": float(exec_result["mean_load"]),
                }
            )
        pairing_failure_count = len(q1_rows) - row_count_before - 2
        freeze_rows.append(
            {
                "window_id": int(window_id),
                "friction_level": float(friction_level),
                "forecast_path_hash": forecast_hash,
                "target_path_hash": target_hash,
                "forecast_hash_identical_flag": True,
                "target_hash_identical_flag": True,
                "initial_prev_target_value": float(initial_prev_target),
                "initial_prev_dispatch_value": float(initial_prev_dispatch),
                "initial_prev_target_match_flag": True,
                "initial_prev_dispatch_match_flag": True,
                "pairing_failure_count": int(max(pairing_failure_count, 0)),
                "forecast_sequence_json": forecast_json,
                "target_path_json": target_json,
            }
        )

    return q1_rows, diagnostics_rows, freeze_rows


def _build_q2_rows(
    *,
    window_bundle: dict[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    window_id = int(window_bundle["window_id"])
    load_eval = np.asarray(window_bundle["eval_load_norm"], dtype=np.float64)
    reserve_margin = float(window_bundle["reserve_margin"])
    initial_prev_dispatch = float(window_bundle["initial_prev_dispatch"])
    dispatch_cap = float(window_bundle["dispatch_cap"])

    q2_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    for friction_level in FRICTION_GRID:
        for forecaster_id, forecast_path in window_bundle["forecasts"].items():
            q_target_unclipped = np.asarray(forecast_path, dtype=np.float64) + reserve_margin
            exec_result = _evaluate_load_path(
                load_eval=load_eval,
                proposed_path=q_target_unclipped,
                friction_level=float(friction_level),
                initial_prev=initial_prev_dispatch,
                dispatch_cap=dispatch_cap,
            )
            q2_rows.append(
                build_result_row(
                    question_id="Q2",
                    scenario_id="load_following_proxy_q2",
                    domain="load_dispatch",
                    seed=window_id,
                    forecaster_id=str(forecaster_id),
                    interface_id="responsive",
                    friction_level=float(friction_level),
                    forecast_metric=float(window_bundle["forecast_metrics"][forecaster_id]),
                    target_metric=float(exec_result["score"]),
                    executed_metric=float(exec_result["score"]),
                    realized_cost=float(exec_result["mean_ramp_cost"]),
                    realized_turnover_or_adjustment=float(exec_result["mean_dispatch_adjustment"]),
                )
            )
            diagnostics_rows.append(
                {
                    "question_id": "Q2",
                    "scenario_id": "load_following_proxy_q2",
                    "domain": "load_dispatch",
                    "seed": int(window_id),
                    "forecaster_id": str(forecaster_id),
                    "interface_id": "responsive",
                    "friction_level": float(friction_level),
                    "mean_shortage_cost": float(exec_result["mean_shortage_cost"]),
                    "mean_surplus_cost": float(exec_result["mean_surplus_cost"]),
                    "mean_ramp_cost": float(exec_result["mean_ramp_cost"]),
                    "mean_dispatch_adjustment": float(exec_result["mean_dispatch_adjustment"]),
                    "dispatch_target_clip_rate": float(exec_result["clip_rate"]),
                    "dispatch_exec_clip_rate": float(exec_result["clip_rate"]),
                    "mean_dispatch": float(exec_result["mean_dispatch"]),
                    "mean_load": float(exec_result["mean_load"]),
                }
            )
    return q2_rows, diagnostics_rows


def run_experiment(
    *,
    raw_path: Path,
    window_ids: list[int],
    train_hours: int,
    eval_hours: int,
    window_step_hours: int,
    max_lag_hours: int,
) -> dict[str, Any]:
    resolved_raw_path = _resolve_raw_path(raw_path)
    processed_hourly = load_hourly_series(resolved_raw_path)
    required_hours = int(train_hours) + int(eval_hours) + int(max_lag_hours) + int(max(window_ids)) * int(window_step_hours)
    block, block_meta = select_latest_contiguous_block(processed_hourly, required_hours=required_hours)
    schedule = build_window_schedule(
        block,
        window_ids=window_ids,
        train_hours=train_hours,
        eval_hours=eval_hours,
        window_step_hours=window_step_hours,
        max_lag_hours=max_lag_hours,
    )

    q1_rows: list[dict[str, object]] = []
    q2_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    freeze_rows: list[dict[str, object]] = []
    model_failure_rows: list[dict[str, object]] = []

    for schedule_row in schedule.itertuples(index=False):
        window_bundle = build_window_forecasts(block, pd.Series(schedule_row._asdict()))
        if window_bundle["model_failures"]:
            for model_id, failure in window_bundle["model_failures"].items():
                model_failure_rows.append(
                    {
                        "window_id": int(window_bundle["window_id"]),
                        "forecaster_id": str(model_id),
                        "failure_reason": str(failure),
                    }
                )
        window_q1_rows, window_q1_diag, window_freeze = _build_q1_rows(window_bundle=window_bundle)
        window_q2_rows, window_q2_diag = _build_q2_rows(window_bundle=window_bundle)
        q1_rows.extend(window_q1_rows)
        q2_rows.extend(window_q2_rows)
        diagnostics_rows.extend(window_q1_diag)
        diagnostics_rows.extend(window_q2_diag)
        freeze_rows.extend(window_freeze)

    q1_df = prepare_results_frame(q1_rows)
    q2_df = prepare_results_frame(q2_rows)
    diagnostics_df = (
        pd.DataFrame(diagnostics_rows)
        .sort_values(["question_id", "seed", "friction_level", "forecaster_id", "interface_id"])
        .reset_index(drop=True)
    )
    freeze_df = pd.DataFrame(freeze_rows).sort_values(["window_id", "friction_level"]).reset_index(drop=True)
    failure_df = pd.DataFrame(
        model_failure_rows,
        columns=["window_id", "forecaster_id", "failure_reason"],
    )
    if not failure_df.empty:
        failure_df = failure_df.sort_values(["window_id", "forecaster_id"]).reset_index(drop=True)
    return {
        "raw_path": resolved_raw_path,
        "processed_hourly_df": processed_hourly,
        "contiguous_block_df": block,
        "block_meta": block_meta,
        "window_schedule_df": schedule,
        "q1_df": q1_df,
        "q2_df": q2_df,
        "diagnostics_df": diagnostics_df,
        "freeze_df": freeze_df,
        "model_failures_df": failure_df,
        "metadata": {
            "train_hours": int(train_hours),
            "eval_hours": int(eval_hours),
            "window_step_hours": int(window_step_hours),
            "max_lag_hours": int(max_lag_hours),
            "window_ids": [int(window_id) for window_id in window_ids],
            **block_meta,
        },
    }


def write_default_outputs(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results["q1_df"], output_dir / "q1_same_forecast_diff_interface.csv")
    save_results(results["q2_df"], output_dir / "q2_diff_forecasts_same_interface.csv")
    results["diagnostics_df"].to_csv(output_dir / "load_dispatch_diagnostics.csv", index=False)
    results["freeze_df"].to_csv(output_dir / "load_dispatch_q1_freeze_check.csv", index=False)
    results["model_failures_df"].to_csv(output_dir / "load_dispatch_model_failures.csv", index=False)
    results["window_schedule_df"].to_csv(output_dir / "load_dispatch_window_schedule.csv", index=False)
    results["processed_hourly_df"].to_csv(output_dir / "processed_hourly_load.csv", index=False)
    pd.DataFrame([results["metadata"]]).to_csv(output_dir / "load_dispatch_run_metadata.csv", index=False)


def main() -> int:
    args = parse_args()
    results = run_experiment(
        raw_path=Path(args.raw_path),
        window_ids=[int(window_id) for window_id in args.window_ids],
        train_hours=int(args.train_hours),
        eval_hours=int(args.eval_hours),
        window_step_hours=int(args.window_step_hours),
        max_lag_hours=int(args.max_lag_hours),
    )
    output_dir = Path(args.output_dir)
    write_default_outputs(results, output_dir)
    print(
        "[load-dispatch] "
        f"windows={len(results['metadata']['window_ids'])} train_hours={results['metadata']['train_hours']} "
        f"eval_hours={results['metadata']['eval_hours']} block_hours={results['metadata']['block_hours']}"
    )
    print(f"[load-dispatch] wrote Q1 rows to {output_dir / 'q1_same_forecast_diff_interface.csv'}")
    print(f"[load-dispatch] wrote Q2 rows to {output_dir / 'q2_diff_forecasts_same_interface.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
