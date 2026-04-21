from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "domain",
    "seed",
    "forecaster_id",
    "interface_id",
    "friction_level",
    "forecast_metric",
    "target_metric",
    "executed_metric",
    "target_executed_gap",
    "realized_cost",
    "realized_turnover_or_adjustment",
    "rank_within_forecast_metric",
    "rank_within_executed_metric",
]
META_COLUMNS = [
    "question_id",
    "scenario_id",
]
RESULT_COLUMNS = META_COLUMNS + REQUIRED_COLUMNS
RESULT_BASENAMES = {
    "q1_same_forecast_diff_interface.csv",
    "q2_diff_forecasts_same_interface.csv",
}


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    series = np.asarray(returns, dtype=np.float64)
    if series.size == 0:
        return 0.0
    std = float(series.std(ddof=1)) if series.size > 1 else 0.0
    if std <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * series.mean() / std)


def mse_score(forecast: np.ndarray, truth: np.ndarray) -> float:
    return -float(np.mean((np.asarray(forecast, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) ** 2))


def mae_score(forecast: np.ndarray, truth: np.ndarray) -> float:
    return -float(np.mean(np.abs(np.asarray(forecast, dtype=np.float64) - np.asarray(truth, dtype=np.float64))))


def l1_adjustments(path: np.ndarray, *, initial_action: np.ndarray | float | None = None) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    if initial_action is None:
        if values.shape[0] == 1:
            return np.zeros(1, dtype=np.float64)
        deltas = values[1:] - values[:-1]
    else:
        initial = np.asarray(initial_action, dtype=np.float64)
        if initial.ndim == 0:
            initial = np.array([float(initial)], dtype=np.float64)
        initial = initial.reshape(1, -1)
        deltas = np.concatenate([values[:1] - initial, values[1:] - values[:-1]], axis=0)
    return np.abs(deltas).sum(axis=1)


def mean_adjustment(path: np.ndarray, *, initial_action: np.ndarray | float | None = None) -> float:
    adjustments = l1_adjustments(path, initial_action=initial_action)
    if adjustments.size == 0:
        return 0.0
    return float(adjustments.mean())


def partial_adjustment_path(
    targets: np.ndarray,
    *,
    interface_strength: float,
    friction_level: float,
    friction_scale: float,
) -> np.ndarray:
    target_array = np.asarray(targets, dtype=np.float64)
    executed = np.array(target_array, copy=True)
    if target_array.shape[0] == 0:
        return executed
    if np.isclose(float(friction_level), 0.0):
        return executed

    effective_rate = float(interface_strength) / (1.0 + float(friction_scale) * float(friction_level))
    effective_rate = float(np.clip(effective_rate, 0.0, 1.0))
    executed[0] = target_array[0]
    for idx in range(1, target_array.shape[0]):
        executed[idx] = executed[idx - 1] + effective_rate * (target_array[idx] - executed[idx - 1])
    return executed


def build_result_row(
    *,
    question_id: str,
    scenario_id: str,
    domain: str,
    seed: int,
    forecaster_id: str,
    interface_id: str,
    friction_level: float,
    forecast_metric: float,
    target_metric: float,
    executed_metric: float,
    realized_cost: float,
    realized_turnover_or_adjustment: float,
) -> dict[str, object]:
    return {
        "question_id": str(question_id),
        "scenario_id": str(scenario_id),
        "domain": str(domain),
        "seed": int(seed),
        "forecaster_id": str(forecaster_id),
        "interface_id": str(interface_id),
        "friction_level": float(friction_level),
        "forecast_metric": float(forecast_metric),
        "target_metric": float(target_metric),
        "executed_metric": float(executed_metric),
        "target_executed_gap": float(target_metric) - float(executed_metric),
        "realized_cost": float(realized_cost),
        "realized_turnover_or_adjustment": float(realized_turnover_or_adjustment),
    }


def _ranking_group_columns(question_id: str) -> list[str]:
    base = ["question_id", "scenario_id", "domain", "seed", "friction_level"]
    if str(question_id).upper() == "Q1":
        return base + ["forecaster_id"]
    if str(question_id).upper() == "Q2":
        return base + ["interface_id"]
    return base + ["forecaster_id", "interface_id"]


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_within_forecast_metric"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["rank_within_executed_metric"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    for question_id, idx in out.groupby("question_id").groups.items():
        group_cols = _ranking_group_columns(str(question_id))
        question_df = out.loc[idx]
        forecast_ranks = question_df.groupby(group_cols)["forecast_metric"].rank(method="dense", ascending=False)
        executed_ranks = question_df.groupby(group_cols)["executed_metric"].rank(method="dense", ascending=False)
        out.loc[idx, "rank_within_forecast_metric"] = forecast_ranks.astype("Int64")
        out.loc[idx, "rank_within_executed_metric"] = executed_ranks.astype("Int64")

    return out


def validate_results_schema(df: pd.DataFrame) -> None:
    missing = [column for column in RESULT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required result columns: {missing}")

    for column in META_COLUMNS:
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' must not contain null values.")

    numeric_columns = [
        "seed",
        "friction_level",
        "forecast_metric",
        "target_metric",
        "executed_metric",
        "target_executed_gap",
        "realized_cost",
        "realized_turnover_or_adjustment",
        "rank_within_forecast_metric",
        "rank_within_executed_metric",
    ]
    for column in numeric_columns:
        if pd.to_numeric(df[column], errors="coerce").isna().any():
            raise ValueError(f"Column '{column}' must be fully numeric.")


def prepare_results_frame(rows: Iterable[dict[str, object]] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy() if not isinstance(rows, pd.DataFrame) else rows.copy()
    if "scenario_id" not in df.columns:
        df["scenario_id"] = "default"
    if "question_id" not in df.columns:
        raise ValueError("Each result row must include a question_id so ranks can be computed.")
    if "target_executed_gap" not in df.columns and {"target_metric", "executed_metric"}.issubset(df.columns):
        df["target_executed_gap"] = df["target_metric"] - df["executed_metric"]
    df = add_rank_columns(df)
    df = df.loc[:, RESULT_COLUMNS]
    validate_results_schema(df)
    return df.sort_values(["question_id", "scenario_id", "domain", "seed", "friction_level", "forecaster_id", "interface_id"]).reset_index(drop=True)


def save_results(rows: Iterable[dict[str, object]] | pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    output = Path(output_path)
    df = prepare_results_frame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def discover_result_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    files = sorted(
        path
        for path in root_path.rglob("*.csv")
        if "summary" not in path.parts
        and not any("baseline" in part for part in path.parts)
        and not any("lock" in part for part in path.parts)
        and not any("candidate_lock" in part for part in path.parts)
        and path.name in RESULT_BASENAMES
    )
    return files


def merge_result_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    csv_paths = [Path(path) for path in paths]
    if not csv_paths:
        raise ValueError("No result CSVs were found to merge.")

    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for path in csv_paths:
        frame = pd.read_csv(path)
        validate_results_schema(frame)
        columns = frame.columns.tolist()
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"Schema mismatch for {path}: expected {expected_columns}, got {columns}")
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged = prepare_results_frame(merged)
    return merged


def build_pairing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for question_id, question_df in df.groupby("question_id"):
        group_cols = _ranking_group_columns(str(question_id))
        grouped = question_df.groupby(group_cols, dropna=False).size().reset_index(name="rows_in_group")
        grouped["question_id"] = str(question_id)
        grouped["paired_ok"] = grouped["rows_in_group"] >= 2
        rows.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(rows)
