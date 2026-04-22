#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_same_interface_rank_summary import build_domain_rank_summary  # noqa: E402
from common import build_result_row, mae_score, prepare_results_frame  # noqa: E402


DOMAIN_ID = "load_following_elecdiag"
Q1_SCENARIO_ID = "load_following_elecdiag_q1"
Q2_SCENARIO_ID = "load_following_elecdiag_q2"
GROUPING_STRATEGY_BASELINE = "baseline"
GROUPING_STRATEGY_BALANCE_REPAIR_V1 = "balance_repair_v1"

DEFAULT_RAW_PATH = REPO_ROOT / "data" / "load_dispatch" / "ElectricityLoadDiagrams20112014.csv"
DEFAULT_ALT_RAW_PATH = REPO_ROOT / "data" / "load_dispatch" / "LD2011_2014.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag"

TRAIN_DAYS = 365 * 2
EVAL_DAYS = 180
RAW_FREQ_MINUTES = 15
MAX_LAG_HOURS = 168
RESOLUTION_CHOICES = (30, 60)
FRICTION_GRID = (0.0, 0.25, 0.5, 1.0)
FORECASTER_IDS = ("naive_last", "moving_average_24h", "linear_ar_ridge", "mlp_small")
CALIBRATION_GROUP_IDS = (0, 1)
EVALUATION_GROUP_IDS = tuple(range(2, 10))
MIN_CLIENTS_PER_GROUP = 20
NONZERO_FRACTION_MIN = 0.10
VARIANCE_MIN = 1e-8
SHORTAGE_W = 2.0
SURPLUS_W = 0.5
TEMPERED_POSITIVE_ETA = 0.6
MLP_HIDDEN = 16
MLP_EPOCHS = 120
CLIP_TOL = 1e-12
BALANCE_MEAN_RATIO_MAX = 1.5
BALANCE_STD_RATIO_MAX = 1.5
BALANCE_COUNT_RATIO_MAX = 1.25
CALIBRATION_END_STEP_RAW = 96  # one day at 15-minute resolution


@dataclass(frozen=True)
class SharedBlock:
    start_idx: int
    end_idx_exclusive: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    n_raw_steps: int
    eligible_client_ids: tuple[str, ...]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fixed-config ElectricityLoadDiagrams20112014 group experiment.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resolution-minutes", type=int, choices=list(RESOLUTION_CHOICES), default=60)
    parser.add_argument("--reserve-margin-multiplier", type=float, choices=[0.05, 0.10, 0.15], default=0.10)
    parser.add_argument("--group-ids", nargs="+", type=int, default=list(range(10)))
    return parser.parse_args()


def _json_hash(values: np.ndarray) -> tuple[str, str]:
    payload = json.dumps(np.asarray(values, dtype=np.float64).round(12).tolist(), separators=(",", ":"), ensure_ascii=True)
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stderr(values: pd.Series) -> float:
    array = values.to_numpy(dtype=np.float64)
    if array.size <= 1:
        return 0.0
    return float(array.std(ddof=1) / math.sqrt(array.size))


def _resolve_raw_path(raw_path: Path) -> Path:
    if raw_path.exists():
        return raw_path
    if raw_path == DEFAULT_RAW_PATH and DEFAULT_ALT_RAW_PATH.exists():
        return DEFAULT_ALT_RAW_PATH
    raise FileNotFoundError(f"Electricity load file not found: {raw_path}")


def load_raw_electricity_panel(raw_path: Path) -> tuple[pd.Series, pd.DataFrame]:
    resolved = _resolve_raw_path(raw_path)
    frame = pd.read_csv(resolved, sep=";", decimal=",", low_memory=False)
    timestamp_col = frame.columns[0]
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    values = frame.drop(columns=[timestamp_col]).apply(pd.to_numeric, errors="coerce")
    values = values.astype(np.float32)
    valid_rows = timestamps.notna() & values.notna().all(axis=1)
    timestamps = timestamps.loc[valid_rows].reset_index(drop=True)
    values = values.loc[valid_rows].reset_index(drop=True)
    return timestamps, values


def _find_contiguous_segments(timestamps: pd.Series) -> list[tuple[int, int]]:
    diffs = timestamps.diff().fillna(pd.Timedelta(minutes=RAW_FREQ_MINUTES))
    expected = pd.Timedelta(minutes=RAW_FREQ_MINUTES)
    segments: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(timestamps)):
        if diffs.iloc[idx] != expected:
            segments.append((start, idx))
            start = idx
    segments.append((start, len(timestamps)))
    return segments


def _required_raw_steps() -> int:
    train_steps = TRAIN_DAYS * 24 * (60 // RAW_FREQ_MINUTES)
    eval_steps = EVAL_DAYS * 24 * (60 // RAW_FREQ_MINUTES)
    lag_steps = MAX_LAG_HOURS * (60 // RAW_FREQ_MINUTES)
    return int(train_steps + eval_steps + lag_steps)


def _raw_train_segment_bounds() -> tuple[int, int, int]:
    lag_steps = MAX_LAG_HOURS * (60 // RAW_FREQ_MINUTES)
    train_steps = TRAIN_DAYS * 24 * (60 // RAW_FREQ_MINUTES)
    train_start = lag_steps
    train_end = train_start + train_steps
    return lag_steps, train_start, train_end


def _eligible_clients_for_block(block_values: np.ndarray, client_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coverage_ok = np.isfinite(block_values).all(axis=0)
    nonzero_fraction = (np.abs(block_values) > 1e-12).mean(axis=0)
    variance = block_values.var(axis=0)
    eligible = coverage_ok & (nonzero_fraction >= NONZERO_FRACTION_MIN) & (variance > VARIANCE_MIN)
    return client_ids[eligible], eligible


def compute_eligible_client_stats(block_values: np.ndarray, client_ids: np.ndarray) -> pd.DataFrame:
    eligible_client_ids, eligible_mask = _eligible_clients_for_block(block_values, client_ids)
    _, train_start, train_end = _raw_train_segment_bounds()
    train_values = block_values[train_start:train_end, eligible_mask]
    return pd.DataFrame(
        {
            "client_id": eligible_client_ids,
            "train_mean_load": train_values.mean(axis=0).astype(np.float64),
            "train_std_load": train_values.std(axis=0, ddof=0).astype(np.float64),
        }
    ).reset_index(drop=True)


def _build_snake_group_sequence(n_groups: int, n_items: int) -> list[int]:
    if n_groups <= 0:
        raise ValueError("n_groups must be positive.")
    if n_groups == 1:
        return [0] * n_items
    seq: list[int] = []
    current = 0
    direction = 1
    for _ in range(n_items):
        seq.append(current)
        if current == n_groups - 1:
            direction = -1
        elif current == 0:
            direction = 1
        current += direction
    return seq


def resolve_shared_block_from_metadata(timestamps: pd.Series, metadata: pd.Series, *, eligible_client_ids: tuple[str, ...]) -> SharedBlock:
    start_timestamp = pd.Timestamp(metadata["block_start_timestamp"])
    end_timestamp = pd.Timestamp(metadata["block_end_timestamp"])
    n_raw_steps = int(metadata["n_raw_steps"])
    start_matches = timestamps.index[timestamps == start_timestamp].to_list()
    end_matches = timestamps.index[timestamps == end_timestamp].to_list()
    for start_idx in start_matches:
        for end_idx in end_matches:
            if end_idx >= start_idx and (end_idx - start_idx + 1) == n_raw_steps:
                return SharedBlock(
                    start_idx=int(start_idx),
                    end_idx_exclusive=int(end_idx + 1),
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    n_raw_steps=n_raw_steps,
                    eligible_client_ids=tuple(eligible_client_ids),
                )
    raise RuntimeError(
        "Unable to reconstruct the frozen shared block from run_metadata.csv "
        f"(start={start_timestamp}, end={end_timestamp}, n_raw_steps={n_raw_steps})."
    )


def build_balance_repair_assignments(eligible_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if eligible_stats_df.empty:
        raise RuntimeError("Eligible-client stats are empty.")
    fully_sorted = (
        eligible_stats_df.sort_values(
            ["train_mean_load", "train_std_load", "client_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    m = int(len(fully_sorted) // 10)
    if m < MIN_CLIENTS_PER_GROUP:
        raise RuntimeError(
            f"balance_repair_v1 requires floor(n_eligible/10) >= {MIN_CLIENTS_PER_GROUP}, "
            f"got m={m} from n_eligible={len(fully_sorted)}."
        )
    retained_count = int(10 * m)
    retained = fully_sorted.iloc[:retained_count].copy().reset_index(drop=True)
    dropped = fully_sorted.iloc[retained_count:].copy().reset_index(drop=True)

    assignment_frames: list[pd.DataFrame] = []
    for stratum_id in range(m):
        start = stratum_id * 10
        stop = start + 10
        stratum = (
            retained.iloc[start:stop]
            .copy()
            .sort_values(["train_std_load", "client_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        group_order = list(range(10)) if (stratum_id % 2 == 0) else list(range(9, -1, -1))
        stratum["stratum_id"] = int(stratum_id)
        stratum["group_id"] = group_order
        stratum["stratum_position"] = np.arange(10, dtype=np.int64)
        assignment_frames.append(stratum)
    assignments = (
        pd.concat(assignment_frames, ignore_index=True)
        .sort_values(["group_id", "stratum_id", "client_id"])
        .reset_index(drop=True)
    )
    retention_diagnostics_df = pd.DataFrame(
        [
            {
                "grouping_strategy": GROUPING_STRATEGY_BALANCE_REPAIR_V1,
                "eligible_client_count": int(len(fully_sorted)),
                "retained_client_count": int(retained_count),
                "dropped_client_count": int(len(dropped)),
                "retained_client_fraction": float(retained_count / max(len(fully_sorted), 1)),
            }
        ]
    )
    return assignments, dropped, retention_diagnostics_df


def choose_shared_block_and_assign_groups(
    timestamps: pd.Series,
    values: pd.DataFrame,
) -> tuple[SharedBlock, pd.DataFrame]:
    client_ids = values.columns.to_numpy(dtype=str)
    matrix = values.to_numpy(dtype=np.float32)
    required_steps = _required_raw_steps()
    segments = _find_contiguous_segments(timestamps)
    for seg_start, seg_end_exclusive in sorted(segments, key=lambda item: item[1], reverse=True):
        if seg_end_exclusive - seg_start < required_steps:
            continue
        latest_end_exclusive = seg_end_exclusive
        earliest_end_exclusive = seg_start + required_steps
        for end_exclusive in range(latest_end_exclusive, earliest_end_exclusive - 1, -CALIBRATION_END_STEP_RAW):
            start_idx = end_exclusive - required_steps
            block_values = matrix[start_idx:end_exclusive]
            eligible_client_ids, eligible_mask = _eligible_clients_for_block(block_values, client_ids)
            if len(eligible_client_ids) < MIN_CLIENTS_PER_GROUP * 10:
                continue

            lag_steps = MAX_LAG_HOURS * (60 // RAW_FREQ_MINUTES)
            train_steps = TRAIN_DAYS * 24 * (60 // RAW_FREQ_MINUTES)
            train_start = lag_steps
            train_end = train_start + train_steps
            train_means = block_values[train_start:train_end, eligible_mask].mean(axis=0)
            order = np.argsort(-train_means, kind="stable")
            sorted_client_ids = eligible_client_ids[order]
            sorted_train_means = train_means[order]
            snake_groups = _build_snake_group_sequence(10, len(sorted_client_ids))
            assignments = pd.DataFrame(
                {
                    "client_id": sorted_client_ids,
                    "train_mean_load": sorted_train_means.astype(np.float64),
                    "group_id": snake_groups,
                }
            )
            counts = assignments.groupby("group_id")["client_id"].count()
            if counts.min() < MIN_CLIENTS_PER_GROUP or len(counts) != 10:
                continue
            shared_block = SharedBlock(
                start_idx=int(start_idx),
                end_idx_exclusive=int(end_exclusive),
                start_timestamp=pd.Timestamp(timestamps.iloc[start_idx]),
                end_timestamp=pd.Timestamp(timestamps.iloc[end_exclusive - 1]),
                n_raw_steps=int(required_steps),
                eligible_client_ids=tuple(assignments["client_id"].tolist()),
            )
            assignments = assignments.sort_values(["group_id", "client_id"]).reset_index(drop=True)
            return shared_block, assignments
    raise RuntimeError("No contiguous candidate block can produce 10 viable disjoint client groups.")


def run_config_from_assignments(
    *,
    values: pd.DataFrame,
    timestamps: pd.Series,
    shared_block: SharedBlock,
    assignments: pd.DataFrame,
    resolution_minutes: int,
    reserve_margin_multiplier: float,
    dispatch_cap_quantile: float = 0.99,
    group_ids: list[int],
) -> dict[str, Any]:
    block_timestamps, group_series_15m = build_group_aggregates(values, timestamps, shared_block, assignments)
    bundles, group_summary_df, balance_audit_df = prepare_resolution_bundles(
        block_timestamps,
        group_series_15m,
        assignments,
        resolution_minutes=resolution_minutes,
    )
    results = run_config(
        bundles,
        reserve_margin_multiplier=reserve_margin_multiplier,
        dispatch_cap_quantile=dispatch_cap_quantile,
        group_ids=group_ids,
    )
    results.update(
        {
            "shared_block": shared_block,
            "assignments_df": assignments,
            "group_summary_df": group_summary_df,
            "group_balance_audit_df": balance_audit_df,
            "resolution_minutes": int(resolution_minutes),
            "reserve_margin_multiplier": float(reserve_margin_multiplier),
            "dispatch_cap_quantile": float(dispatch_cap_quantile),
        }
    )
    return results


def build_group_aggregates(
    values: pd.DataFrame,
    timestamps: pd.Series,
    shared_block: SharedBlock,
    assignments: pd.DataFrame,
) -> tuple[pd.Series, dict[int, np.ndarray]]:
    block_values = values.iloc[shared_block.start_idx : shared_block.end_idx_exclusive].to_numpy(dtype=np.float64)
    block_timestamps = timestamps.iloc[shared_block.start_idx : shared_block.end_idx_exclusive].reset_index(drop=True)
    client_to_idx = {client_id: idx for idx, client_id in enumerate(values.columns.to_numpy(dtype=str))}
    group_series: dict[int, np.ndarray] = {}
    for group_id, group_df in assignments.groupby("group_id", sort=True):
        indices = [client_to_idx[client_id] for client_id in group_df["client_id"]]
        group_series[int(group_id)] = block_values[:, indices].sum(axis=1, dtype=np.float64)
    return block_timestamps, group_series


def aggregate_resolution(
    raw_series: np.ndarray,
    raw_timestamps: pd.Series,
    *,
    resolution_minutes: int,
) -> tuple[np.ndarray, pd.Series]:
    factor = int(resolution_minutes // RAW_FREQ_MINUTES)
    if raw_series.size % factor != 0:
        raise RuntimeError(f"Raw block length {raw_series.size} is not divisible by factor={factor}.")
    reshaped = raw_series.reshape(-1, factor)
    aggregated = reshaped.mean(axis=1)
    agg_timestamps = raw_timestamps.iloc[factor - 1 :: factor].reset_index(drop=True)
    return aggregated.astype(np.float64), agg_timestamps


def _resolution_steps(hours: int, resolution_minutes: int) -> int:
    return int(hours * 60 // resolution_minutes)


def _resolution_train_eval_steps(resolution_minutes: int) -> tuple[int, int, int]:
    lag_steps = _resolution_steps(MAX_LAG_HOURS, resolution_minutes)
    train_steps = int(TRAIN_DAYS * 24 * 60 // resolution_minutes)
    eval_steps = int(EVAL_DAYS * 24 * 60 // resolution_minutes)
    return lag_steps, train_steps, eval_steps


def _feature_row(load_norm: np.ndarray, timestamps: pd.Series, idx: int, resolution_minutes: int) -> np.ndarray:
    timestamp = pd.Timestamp(timestamps.iloc[idx])
    steps_per_day = int(24 * 60 // resolution_minutes)
    steps_per_week = int(7 * 24 * 60 // resolution_minutes)
    hour_float = timestamp.hour + timestamp.minute / 60.0
    week_position = float(timestamp.dayofweek * steps_per_day + idx % steps_per_day)
    return np.array(
        [
            load_norm[idx - 1],
            load_norm[idx - 2],
            load_norm[idx - steps_per_day],
            load_norm[idx - steps_per_week],
            float(load_norm[idx - steps_per_day : idx].mean()),
            float(load_norm[idx - steps_per_week : idx].mean()),
            np.sin(2.0 * np.pi * hour_float / 24.0),
            np.cos(2.0 * np.pi * hour_float / 24.0),
            np.sin(2.0 * np.pi * week_position / steps_per_week),
            np.cos(2.0 * np.pi * week_position / steps_per_week),
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
    group_id: int,
    resolution_minutes: int,
    attempt: int,
) -> np.ndarray:
    torch.manual_seed(91_000 + 101 * int(group_id) + 7 * int(resolution_minutes) + int(attempt))
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


def prepare_resolution_bundles(
    raw_timestamps: pd.Series,
    group_series_15m: dict[int, np.ndarray],
    assignments: pd.DataFrame,
    *,
    resolution_minutes: int,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    lag_steps, train_steps, eval_steps = _resolution_train_eval_steps(resolution_minutes)
    bundles: dict[int, dict[str, Any]] = {}
    group_rows: list[dict[str, Any]] = []
    for group_id, raw_series in group_series_15m.items():
        group_df = assignments[assignments["group_id"] == group_id]
        load_raw, agg_timestamps = aggregate_resolution(raw_series, raw_timestamps, resolution_minutes=resolution_minutes)
        if len(load_raw) != lag_steps + train_steps + eval_steps:
            raise RuntimeError(
                f"group_id={group_id} produced {len(load_raw)} steps at {resolution_minutes}min, "
                f"expected {lag_steps + train_steps + eval_steps}."
            )

        train_start = lag_steps
        train_end = train_start + train_steps
        eval_start = train_end
        eval_end = eval_start + eval_steps
        train_raw = load_raw[train_start:train_end]
        dispatch_scale = float(np.percentile(train_raw, 95.0))
        if not np.isfinite(dispatch_scale) or dispatch_scale <= 1e-8:
            raise RuntimeError(f"Invalid dispatch scale for group_id={group_id}: {dispatch_scale}")

        load_norm = load_raw / dispatch_scale
        train_norm = load_norm[train_start:train_end]
        train_mean_load_norm = float(train_norm.mean())
        train_p99_norm = float(np.percentile(train_norm, 99.0))
        train_p995_norm = float(np.percentile(train_norm, 99.5))
        train_variance = float(train_raw.var())
        zero_fraction = float(np.mean(np.abs(train_raw) <= 1e-12))

        train_indices = np.arange(train_start, train_end, dtype=np.int64)
        eval_indices = np.arange(eval_start, eval_end, dtype=np.int64)
        x_train = np.vstack([_feature_row(load_norm, agg_timestamps, idx, resolution_minutes) for idx in train_indices]).astype(np.float64)
        y_train = load_norm[train_indices].astype(np.float64)
        x_eval = np.vstack([_feature_row(load_norm, agg_timestamps, idx, resolution_minutes) for idx in eval_indices]).astype(np.float64)
        y_eval = load_norm[eval_indices].astype(np.float64)

        linear_model = _fit_linear_ar_ridge(x_train, y_train)
        forecasts: dict[str, np.ndarray] = {
            "naive_last": np.clip(load_norm[eval_indices - 1], 0.0, None),
            "moving_average_24h": np.array([load_norm[idx - int(24 * 60 // resolution_minutes) : idx].mean() for idx in eval_indices], dtype=np.float64),
            "linear_ar_ridge": _predict_linear_ar(linear_model, x_eval),
        }
        model_failures: dict[str, str] = {}
        mlp_prediction: np.ndarray | None = None
        mlp_error: Exception | None = None
        for attempt in (0, 1):
            try:
                mlp_prediction = _fit_and_predict_mlp(
                    x_train,
                    y_train,
                    x_eval,
                    group_id=int(group_id),
                    resolution_minutes=int(resolution_minutes),
                    attempt=int(attempt),
                )
                mlp_error = None
                break
            except Exception as exc:  # noqa: BLE001
                mlp_error = exc
        if mlp_prediction is not None:
            forecasts["mlp_small"] = mlp_prediction
        else:
            model_failures["mlp_small"] = str(mlp_error or "unknown_mlp_failure")

        forecast_metrics = {name: mae_score(values, y_eval) for name, values in forecasts.items()}
        bundles[int(group_id)] = {
            "group_id": int(group_id),
            "resolution_minutes": int(resolution_minutes),
            "load_raw": load_raw,
            "load_norm": load_norm,
            "eval_load_norm": y_eval,
            "eval_timestamps": agg_timestamps.iloc[eval_start:eval_end].reset_index(drop=True),
            "forecasts": forecasts,
            "forecast_metrics": forecast_metrics,
            "model_failures": model_failures,
            "dispatch_scale": float(dispatch_scale),
            "train_mean_load_norm": float(train_mean_load_norm),
            "train_p99_norm": float(train_p99_norm),
            "train_p995_norm": float(train_p995_norm),
            "initial_prev_dispatch": float(train_mean_load_norm),
            "initial_prev_target": float(train_mean_load_norm),
            "group_client_count": int(len(group_df)),
            "aggregate_load_variance": float(train_variance),
            "zero_fraction": float(zero_fraction),
            "train_mean_load": float(train_raw.mean()),
            "train_std_load": float(train_raw.std(ddof=0)),
        }
        group_rows.append(
            {
                "group_id": int(group_id),
                "resolution_minutes": int(resolution_minutes),
                "client_count": int(len(group_df)),
                "train_mean_load": float(train_raw.mean()),
                "train_std_load": float(train_raw.std(ddof=0)),
                "zero_fraction": float(zero_fraction),
            }
        )

    group_summary_df = pd.DataFrame(group_rows).sort_values("group_id").reset_index(drop=True)
    balance_audit_df = build_group_balance_audit(group_summary_df)
    return bundles, group_summary_df, balance_audit_df


def build_group_balance_audit(group_summary_df: pd.DataFrame) -> pd.DataFrame:
    mean_ratio = float(group_summary_df["train_mean_load"].max() / max(group_summary_df["train_mean_load"].min(), 1e-12))
    std_ratio = float(group_summary_df["train_std_load"].max() / max(group_summary_df["train_std_load"].min(), 1e-12))
    count_ratio = float(group_summary_df["client_count"].max() / max(group_summary_df["client_count"].min(), 1))
    balance_ok = (
        mean_ratio <= BALANCE_MEAN_RATIO_MAX
        and std_ratio <= BALANCE_STD_RATIO_MAX
        and count_ratio <= BALANCE_COUNT_RATIO_MAX
    )
    return pd.DataFrame(
        [
            {
                "max_to_min_train_mean_load_ratio": mean_ratio,
                "max_to_min_train_std_ratio": std_ratio,
                "max_to_min_client_count_ratio": count_ratio,
                "balance_status": "balance_ok" if balance_ok else "balance_warn",
            }
        ]
    )


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


def evaluate_group_bundle(
    bundle: dict[str, Any],
    *,
    reserve_margin_multiplier: float,
    dispatch_cap_quantile: float = 0.99,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    group_id = int(bundle["group_id"])
    load_eval = np.asarray(bundle["eval_load_norm"], dtype=np.float64)
    reserve_margin = float(reserve_margin_multiplier) * float(bundle["train_mean_load_norm"])
    initial_prev_target = float(bundle["initial_prev_target"])
    initial_prev_dispatch = float(bundle["initial_prev_dispatch"])
    if np.isclose(float(dispatch_cap_quantile), 0.99, atol=1e-12):
        train_cap_quantile_norm = float(bundle["train_p99_norm"])
    elif np.isclose(float(dispatch_cap_quantile), 0.995, atol=1e-12):
        train_cap_quantile_norm = float(bundle["train_p995_norm"])
    else:
        raise ValueError(f"Unsupported dispatch_cap_quantile={dispatch_cap_quantile}.")
    dispatch_cap = max(train_cap_quantile_norm, float(bundle["train_mean_load_norm"]) + reserve_margin)

    q1_rows: list[dict[str, object]] = []
    q2_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    freeze_rows: list[dict[str, object]] = []

    forecasts_linear = np.asarray(bundle["forecasts"]["linear_ar_ridge"], dtype=np.float64)
    forecast_metric_linear = float(bundle["forecast_metrics"]["linear_ar_ridge"])
    q_target_unclipped = forecasts_linear + reserve_margin
    target_reference_zero = _evaluate_load_path(
        load_eval=load_eval,
        proposed_path=q_target_unclipped,
        friction_level=0.0,
        initial_prev=initial_prev_target,
        dispatch_cap=dispatch_cap,
    )
    q_target = np.asarray(target_reference_zero["dispatch_path"], dtype=np.float64)
    forecast_json, forecast_hash = _json_hash(forecasts_linear)
    target_json, target_hash = _json_hash(q_target)

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
                    scenario_id=Q1_SCENARIO_ID,
                    domain=DOMAIN_ID,
                    seed=group_id,
                    forecaster_id="linear_ar_ridge",
                    interface_id=interface_id,
                    friction_level=float(friction_level),
                    forecast_metric=forecast_metric_linear,
                    target_metric=float(target_result["score"]),
                    executed_metric=float(exec_result["score"]),
                    realized_cost=float(exec_result["mean_ramp_cost"]),
                    realized_turnover_or_adjustment=float(exec_result["mean_dispatch_adjustment"]),
                )
            )
            diagnostics_rows.append(
                {
                    "question_id": "Q1",
                    "scenario_id": Q1_SCENARIO_ID,
                    "domain": DOMAIN_ID,
                    "seed": int(group_id),
                    "group_id": int(group_id),
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
                    "group_client_count": int(bundle["group_client_count"]),
                    "aggregate_load_variance": float(bundle["aggregate_load_variance"]),
                    "forecast_residual_mae": float(-forecast_metric_linear),
                    "resolution_minutes": int(bundle["resolution_minutes"]),
                    "dispatch_cap_quantile": float(dispatch_cap_quantile),
                    "train_cap_quantile_norm": float(train_cap_quantile_norm),
                }
            )
        freeze_rows.append(
            {
                "group_id": int(group_id),
                "friction_level": float(friction_level),
                "forecast_path_hash": forecast_hash,
                "target_path_hash": target_hash,
                "forecast_hash_identical_flag": True,
                "target_hash_identical_flag": True,
                "initial_prev_target_value": float(initial_prev_target),
                "initial_prev_dispatch_value": float(initial_prev_dispatch),
                "initial_prev_target_match_flag": True,
                "initial_prev_dispatch_match_flag": True,
                "pairing_failure_count": int(max(len(q1_rows) - row_count_before - 2, 0)),
                "forecast_sequence_json": forecast_json,
                "target_path_json": target_json,
            }
        )

    for friction_level in FRICTION_GRID:
        for forecaster_id, forecast_path in bundle["forecasts"].items():
            q_target_unclipped_forecaster = np.asarray(forecast_path, dtype=np.float64) + reserve_margin
            exec_result = _evaluate_load_path(
                load_eval=load_eval,
                proposed_path=q_target_unclipped_forecaster,
                friction_level=float(friction_level),
                initial_prev=initial_prev_dispatch,
                dispatch_cap=dispatch_cap,
            )
            q2_rows.append(
                build_result_row(
                    question_id="Q2",
                    scenario_id=Q2_SCENARIO_ID,
                    domain=DOMAIN_ID,
                    seed=group_id,
                    forecaster_id=forecaster_id,
                    interface_id="responsive",
                    friction_level=float(friction_level),
                    forecast_metric=float(bundle["forecast_metrics"][forecaster_id]),
                    target_metric=float(exec_result["score"]),
                    executed_metric=float(exec_result["score"]),
                    realized_cost=float(exec_result["mean_ramp_cost"]),
                    realized_turnover_or_adjustment=float(exec_result["mean_dispatch_adjustment"]),
                )
            )
            diagnostics_rows.append(
                {
                    "question_id": "Q2",
                    "scenario_id": Q2_SCENARIO_ID,
                    "domain": DOMAIN_ID,
                    "seed": int(group_id),
                    "group_id": int(group_id),
                    "forecaster_id": forecaster_id,
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
                    "group_client_count": int(bundle["group_client_count"]),
                    "aggregate_load_variance": float(bundle["aggregate_load_variance"]),
                    "forecast_residual_mae": float(-bundle["forecast_metrics"][forecaster_id]),
                    "resolution_minutes": int(bundle["resolution_minutes"]),
                    "dispatch_cap_quantile": float(dispatch_cap_quantile),
                    "train_cap_quantile_norm": float(train_cap_quantile_norm),
                }
            )
    return q1_rows, q2_rows, diagnostics_rows, freeze_rows


def run_config(
    bundles: dict[int, dict[str, Any]],
    *,
    reserve_margin_multiplier: float,
    dispatch_cap_quantile: float = 0.99,
    group_ids: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    q1_rows: list[dict[str, object]] = []
    q2_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    freeze_rows: list[dict[str, object]] = []
    model_failures: list[dict[str, object]] = []
    for group_id in group_ids:
        bundle = bundles[int(group_id)]
        for forecaster_id, error_message in bundle["model_failures"].items():
            model_failures.append(
                {
                    "group_id": int(group_id),
                    "resolution_minutes": int(bundle["resolution_minutes"]),
                    "forecaster_id": forecaster_id,
                    "error_message": error_message,
                }
            )
        group_q1, group_q2, group_diag, group_freeze = evaluate_group_bundle(
            bundle,
            reserve_margin_multiplier=reserve_margin_multiplier,
            dispatch_cap_quantile=dispatch_cap_quantile,
        )
        q1_rows.extend(group_q1)
        q2_rows.extend(group_q2)
        diagnostics_rows.extend(group_diag)
        freeze_rows.extend(group_freeze)

    q1_df = prepare_results_frame(q1_rows)
    q2_df = prepare_results_frame(q2_rows)
    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values(
        ["question_id", "seed", "friction_level", "forecaster_id", "interface_id"]
    ).reset_index(drop=True)
    freeze_df = pd.DataFrame(freeze_rows).sort_values(["group_id", "friction_level"]).reset_index(drop=True)
    if model_failures:
        model_failures_df = pd.DataFrame(model_failures).sort_values(["group_id", "forecaster_id"]).reset_index(drop=True)
    else:
        model_failures_df = pd.DataFrame(
            columns=["group_id", "resolution_minutes", "forecaster_id", "error_message"]
        )
    return {
        "q1_df": q1_df,
        "q2_df": q2_df,
        "diagnostics_df": diagnostics_df,
        "freeze_df": freeze_df,
        "model_failures_df": model_failures_df,
    }


def build_q1_threshold_summary(q1_df: pd.DataFrame, *, eval_group_ids: tuple[int, ...] = EVALUATION_GROUP_IDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = q1_df[q1_df["seed"].isin(eval_group_ids)].copy()
    pivot = (
        filtered.pivot_table(
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
    pivot["group_abs_gap_mean"] = pivot[["target_executed_gap__responsive", "target_executed_gap__tempered"]].abs().mean(axis=1)
    pivot["executed_delta_tempered_minus_responsive"] = (
        pivot["executed_metric__tempered"] - pivot["executed_metric__responsive"]
    )
    pivot["tempered_win"] = pivot["executed_delta_tempered_minus_responsive"] > 0.0
    summary = (
        pivot.groupby("friction_level", as_index=False)
        .agg(
            n_groups=("seed", "count"),
            tempered_win_count=("tempered_win", "sum"),
            tempered_win_rate=("tempered_win", "mean"),
            mean_executed_delta_tempered_minus_responsive=("executed_delta_tempered_minus_responsive", "mean"),
            median_executed_delta_tempered_minus_responsive=("executed_delta_tempered_minus_responsive", "median"),
            mean_group_abs_gap=("group_abs_gap_mean", "mean"),
            median_group_abs_gap=("group_abs_gap_mean", "median"),
            mean_abs_target_executed_gap_tempered=("target_executed_gap__tempered", lambda s: float(np.mean(np.abs(s)))),
            mean_executed_metric_responsive=("executed_metric__responsive", "mean"),
            mean_executed_metric_tempered=("executed_metric__tempered", "mean"),
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )
    return summary, pivot


def build_diagnostics_share_summary(diagnostics_df: pd.DataFrame, *, eval_group_ids: tuple[int, ...] = EVALUATION_GROUP_IDS) -> pd.DataFrame:
    filtered = diagnostics_df[diagnostics_df["seed"].isin(eval_group_ids)].copy()
    summary = (
        filtered.groupby(["question_id", "scenario_id", "interface_id", "friction_level"], as_index=False)
        .agg(
            n_groups=("seed", "count"),
            mean_shortage_cost=("mean_shortage_cost", "mean"),
            mean_surplus_cost=("mean_surplus_cost", "mean"),
            mean_ramp_cost=("mean_ramp_cost", "mean"),
            mean_dispatch_adjustment=("mean_dispatch_adjustment", "mean"),
            mean_dispatch_target_clip_rate=("dispatch_target_clip_rate", "mean"),
            mean_dispatch_exec_clip_rate=("dispatch_exec_clip_rate", "mean"),
            mean_dispatch=("mean_dispatch", "mean"),
            mean_load=("mean_load", "mean"),
        )
        .sort_values(["question_id", "interface_id", "friction_level"])
        .reset_index(drop=True)
    )
    total = (summary["mean_shortage_cost"] + summary["mean_surplus_cost"] + summary["mean_ramp_cost"]).replace(0.0, np.nan)
    summary["shortage_cost_share"] = summary["mean_shortage_cost"] / total
    summary["surplus_cost_share"] = summary["mean_surplus_cost"] / total
    summary["ramp_cost_share"] = summary["mean_ramp_cost"] / total
    return summary.fillna(0.0)


def build_q2_forecast_vs_deployed_summary(rank_corr: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    strongest = (
        pairwise.sort_values(
            ["friction_level", "flip_seed_share", "model_a", "model_b"],
            ascending=[True, False, True, True],
        )
        .groupby("friction_level", as_index=False)
        .first()
    )
    strongest["strongest_flip_pair"] = strongest["model_a"].fillna("") + "|" + strongest["model_b"].fillna("")
    strongest.loc[strongest["model_a"].isna() | strongest["model_b"].isna(), "strongest_flip_pair"] = ""
    summary = rank_corr.merge(
        strongest[["friction_level", "strongest_flip_pair", "flip_seed_share"]],
        on="friction_level",
        how="left",
    ).rename(columns={"flip_seed_share": "strongest_flip_share"})
    return summary


def _bool_all(frame: pd.DataFrame, column: str) -> bool:
    return bool(frame[column].astype(bool).all()) if not frame.empty else False


def _first_drift_friction(rank_corr: pd.DataFrame) -> float | None:
    zero = rank_corr[np.isclose(rank_corr["friction_level"], 0.0, atol=1e-15)].iloc[0]
    positive = rank_corr[rank_corr["friction_level"] > 0.0].copy()
    drift_levels = positive[
        (positive["mean_flip_rate"] > float(zero["mean_flip_rate"]))
        & (positive["mean_spearman_rho"] < float(zero["mean_spearman_rho"]))
    ]["friction_level"].tolist()
    if not drift_levels:
        return None
    return float(min(drift_levels))


def assess_q1(
    q1_df: pd.DataFrame,
    freeze_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    *,
    eval_group_ids: tuple[int, ...] = EVALUATION_GROUP_IDS,
) -> dict[str, Any]:
    summary, group_level = build_q1_threshold_summary(q1_df, eval_group_ids=eval_group_ids)
    zero_row = summary[np.isclose(summary["friction_level"], 0.0, atol=1e-15)].iloc[0]
    positive = summary[summary["friction_level"] > 0.0].copy()
    row_05 = summary[np.isclose(summary["friction_level"], 0.5, atol=1e-15)]
    row_10 = summary[np.isclose(summary["friction_level"], 1.0, atol=1e-15)]
    high_win_rate_05 = float(row_05.iloc[0]["tempered_win_rate"]) if not row_05.empty else 0.0
    high_win_rate_10 = float(row_10.iloc[0]["tempered_win_rate"]) if not row_10.empty else 0.0

    q1_diag = diagnostics_df[
        (diagnostics_df["question_id"] == "Q1") & diagnostics_df["seed"].isin(eval_group_ids)
    ]
    clip_summary = (
        q1_diag.groupby("friction_level", as_index=False)
        .agg(
            mean_target_clip_rate=("dispatch_target_clip_rate", "mean"),
            mean_exec_clip_rate=("dispatch_exec_clip_rate", "mean"),
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )
    clip_ok = bool(
        (clip_summary["mean_target_clip_rate"] <= 0.02).all()
        and (clip_summary["mean_exec_clip_rate"] <= 0.02).all()
    )

    filtered_freeze = freeze_df[freeze_df["group_id"].isin(eval_group_ids)]
    freeze_ok = bool(
        _bool_all(filtered_freeze, "forecast_hash_identical_flag")
        and _bool_all(filtered_freeze, "target_hash_identical_flag")
        and _bool_all(filtered_freeze, "initial_prev_target_match_flag")
        and _bool_all(filtered_freeze, "initial_prev_dispatch_match_flag")
        and int(filtered_freeze["pairing_failure_count"].sum()) == 0
    )
    zero_gap_ok = float(zero_row["mean_group_abs_gap"]) <= 1e-12
    positive_gap_exists = bool((positive["mean_group_abs_gap"] > 1e-12).any())
    gate_05 = high_win_rate_05 >= 0.7
    gate_10 = high_win_rate_10 >= 0.7
    promotion_gate_pass = bool(freeze_ok and zero_gap_ok and positive_gap_exists and clip_ok and (gate_05 or gate_10))
    minimum_support = bool(freeze_ok and zero_gap_ok and positive_gap_exists)
    return {
        "promotion_gate_pass": promotion_gate_pass,
        "minimum_support": minimum_support,
        "freeze_ok": freeze_ok,
        "zero_gap_ok": zero_gap_ok,
        "positive_gap_exists": positive_gap_exists,
        "clip_ok": clip_ok,
        "clip_summary_df": clip_summary,
        "high_friction_tempered_win_rate_05": high_win_rate_05,
        "high_friction_tempered_win_rate_10": high_win_rate_10,
        "pass_friction_summary": "0.5_and_1.0" if gate_05 and gate_10 else ("0.5_only" if gate_05 else ("1.0_only" if gate_10 else "none")),
        "zero_friction_mean_group_abs_gap": float(zero_row["mean_group_abs_gap"]),
        "summary_df": summary,
        "group_level_df": group_level,
    }


def assess_q2(
    q2_df: pd.DataFrame,
    *,
    eval_group_ids: tuple[int, ...] = EVALUATION_GROUP_IDS,
) -> dict[str, Any]:
    filtered = q2_df[q2_df["seed"].isin(eval_group_ids)].copy()
    outputs, meta = build_domain_rank_summary(
        filtered,
        domain=DOMAIN_ID,
        expected_interface_id="responsive",
    )
    rank_corr = outputs["rank_correlation_by_friction"].copy()
    pairwise = outputs["pairwise_flips_by_friction"].copy()
    zero_row = rank_corr[np.isclose(rank_corr["friction_level"], 0.0, atol=1e-15)].iloc[0]
    positive = rank_corr[rank_corr["friction_level"] > 0.0].copy()
    drift_positive = positive[
        (positive["mean_flip_rate"] > float(zero_row["mean_flip_rate"]))
        & (positive["mean_spearman_rho"] < float(zero_row["mean_spearman_rho"]))
    ].copy()
    first_drift_friction = _first_drift_friction(rank_corr)
    pair_share_ok = bool((pairwise["flip_seed_share"] >= 0.50).any()) if not pairwise.empty else False
    strongest_pair = ""
    strongest_share = 0.0
    if not pairwise.empty:
        strongest = pairwise.sort_values(["flip_seed_share", "friction_level", "model_a", "model_b"], ascending=[False, True, True, True]).iloc[0]
        strongest_pair = f"{strongest['model_a']}|{strongest['model_b']}"
        strongest_share = float(strongest["flip_seed_share"])
    min_forecasters_ok = int(meta.min_n_forecasters_per_seed_friction) >= 4
    promotion_gate_pass = bool(
        float(zero_row["mean_flip_rate"]) <= 0.10
        and len(drift_positive) >= 2
        and pair_share_ok
        and min_forecasters_ok
    )
    qualitative_minimum = bool(
        float(zero_row["mean_flip_rate"]) <= 0.10
        and pair_share_ok
        and any(np.isclose(float(value), 1.0, atol=1e-12) for value in drift_positive["friction_level"].tolist())
    )
    return {
        "promotion_gate_pass": promotion_gate_pass,
        "qualitative_minimum": qualitative_minimum,
        "paper_facing_valid": bool(min_forecasters_ok),
        "min_forecasters_per_seed_friction": int(meta.min_n_forecasters_per_seed_friction),
        "zero_friction_mean_flip_rate": float(zero_row["mean_flip_rate"]),
        "zero_friction_mean_spearman_rho": float(zero_row["mean_spearman_rho"]),
        "drift_positive_frictions": tuple(float(value) for value in drift_positive["friction_level"].tolist()),
        "first_drift_friction": first_drift_friction,
        "strongest_flip_pair": strongest_pair,
        "strongest_flip_share": strongest_share,
        "pair_share_ok": pair_share_ok,
        "outputs": outputs,
        "meta": meta,
    }


def run_group_config(
    *,
    raw_path: Path,
    resolution_minutes: int,
    reserve_margin_multiplier: float,
    dispatch_cap_quantile: float = 0.99,
    group_ids: list[int],
) -> dict[str, Any]:
    timestamps, values = load_raw_electricity_panel(raw_path)
    shared_block, assignments = choose_shared_block_and_assign_groups(timestamps, values)
    results = run_config_from_assignments(
        values=values,
        timestamps=timestamps,
        shared_block=shared_block,
        assignments=assignments,
        resolution_minutes=resolution_minutes,
        reserve_margin_multiplier=reserve_margin_multiplier,
        dispatch_cap_quantile=dispatch_cap_quantile,
        group_ids=group_ids,
    )
    return results


def write_group_config_outputs(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results["q1_df"].to_csv(output_dir / "q1_same_forecast_diff_interface.csv", index=False)
    results["q2_df"].to_csv(output_dir / "q2_diff_forecasts_same_interface.csv", index=False)
    results["diagnostics_df"].to_csv(output_dir / "load_following_elecdiag_diagnostics.csv", index=False)
    results["freeze_df"].to_csv(output_dir / "load_following_elecdiag_q1_freeze_check.csv", index=False)
    results["model_failures_df"].to_csv(output_dir / "load_following_elecdiag_model_failures.csv", index=False)
    results["assignments_df"].to_csv(output_dir / "group_assignments.csv", index=False)
    results["group_summary_df"].to_csv(output_dir / "group_summary.csv", index=False)
    results["group_balance_audit_df"].to_csv(output_dir / "group_balance_audit.csv", index=False)
    if "dropped_clients_df" in results:
        results["dropped_clients_df"].to_csv(output_dir / "dropped_client_ids.csv", index=False)
    if "retention_diagnostics_df" in results:
        results["retention_diagnostics_df"].to_csv(output_dir / "retention_diagnostics.csv", index=False)
    shared_block: SharedBlock = results["shared_block"]
    metadata_row: dict[str, Any] = {
        "domain": DOMAIN_ID,
        "resolution_minutes": int(results["resolution_minutes"]),
        "reserve_margin_multiplier": float(results["reserve_margin_multiplier"]),
        "dispatch_cap_quantile": float(results.get("dispatch_cap_quantile", 0.99)),
        "block_start_timestamp": shared_block.start_timestamp,
        "block_end_timestamp": shared_block.end_timestamp,
        "n_raw_steps": int(shared_block.n_raw_steps),
        "calibration_group_ids": "|".join(str(v) for v in CALIBRATION_GROUP_IDS),
        "evaluation_group_ids": "|".join(str(v) for v in EVALUATION_GROUP_IDS),
    }
    for key in [
        "grouping_strategy",
        "baseline_work_dir",
        "eligible_client_set_matches_baseline_flag",
        "baseline_eligible_client_count",
        "rerun_eligible_client_count",
        "retained_client_count",
        "dropped_client_count",
        "retained_client_fraction",
    ]:
        if key in results:
            metadata_row[key] = results[key]
    pd.DataFrame(
        [
            metadata_row
        ]
    ).to_csv(output_dir / "run_metadata.csv", index=False)


def main() -> int:
    args = parse_args()
    results = run_group_config(
        raw_path=Path(args.raw_path),
        resolution_minutes=int(args.resolution_minutes),
        reserve_margin_multiplier=float(args.reserve_margin_multiplier),
        dispatch_cap_quantile=0.99,
        group_ids=[int(value) for value in args.group_ids],
    )
    write_group_config_outputs(results, Path(args.output_dir))
    print(f"[load-following-elecdiag-groups] wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
