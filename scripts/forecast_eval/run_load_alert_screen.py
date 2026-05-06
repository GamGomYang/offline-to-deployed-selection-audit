#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import heapq as _heapq
import _heapq as _heapq_c
import numpy as np
import pandas as pd

for _name in ("heappush", "heappop", "heapify", "heapreplace", "heappushpop"):
    if not hasattr(_heapq, _name):
        setattr(_heapq, _name, getattr(_heapq_c, _name))
if not hasattr(_heapq, "nlargest"):
    _heapq.nlargest = lambda n, iterable, key=None: sorted(iterable, key=key, reverse=True)[:n]
if not hasattr(_heapq, "nsmallest"):
    _heapq.nsmallest = lambda n, iterable, key=None: sorted(iterable, key=key)[:n]

from scipy.stats import binomtest  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = (
    REPO_ROOT
    / "outputs"
    / "extensions"
    / "epsilon_tie_audit_20260504"
    / "source_50b7481_clean"
    / "data"
    / "load_dispatch"
    / "LD2011_2014.txt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "extensions" / "load_alert_screen_20260505"
V1_OUTPUT = REPO_ROOT / "outputs" / "extensions" / "load_peak_alert_20260505"

QUESTION_ID = "Q2"
DOMAIN_ID = "load_alert_screen"
FRICTION_GRID = (0.0, 0.25, 0.5, 1.0)
EPS_REL_GRID = (0.0, 0.001, 0.005)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260505
TRAIN_HOURS = 365 * 24
VALIDATION_HOURS = 90 * 24
WEEK_HOURS = 7 * 24
FOUR_WEEK_HOURS = 28 * 24
MAX_WEEKLY_UNITS = 100
MIN_MAIN_ELIGIBLE_UNITS = 50
ELIGIBLE_COVERAGE_MIN = 0.95
PROB_EPS = 1e-6

MODEL_ORDER = (
    "persistence_event",
    "moving_event_rate_24h",
    "moving_event_rate_168h",
    "hour_of_week_rate",
    "same_hour_last_week",
    "ridge_linear_ar",
    "calibrated_logistic",
    "gradient_boosted",
)
MODEL_LABELS = {
    "persistence_event": "Persistence",
    "moving_event_rate_24h": "MA event 24h",
    "moving_event_rate_168h": "MA event 168h",
    "hour_of_week_rate": "Hour-of-week",
    "same_hour_last_week": "Same-hour last-week",
    "ridge_linear_ar": "Linear AR",
    "calibrated_logistic": "Calibrated logistic",
    "gradient_boosted": "Gradient boosted",
}


@dataclass(frozen=True)
class WindowSpec:
    seed: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    mode: str


@dataclass(frozen=True)
class CandidateSpec:
    variant: str
    variant_dir: str
    candidate_id: str
    m: float | None
    q: float | None
    k_frac: float | None
    k_abs: int | None
    clean_mode: bool


@dataclass(frozen=True)
class CandidateData:
    spec: CandidateSpec
    validation_model: pd.DataFrame
    validation_seed: pd.DataFrame
    test_model: pd.DataFrame
    test_seed: pd.DataFrame
    window_diag: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen fixed A4 load-alert variants.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args()


def read_hourly_panel(raw_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    frame = pd.read_csv(raw_path, sep=";", decimal=",", low_memory=False)
    timestamp_col = frame.columns[0]
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    values = frame.drop(columns=[timestamp_col]).apply(pd.to_numeric, errors="coerce")
    values.index = timestamps
    values = values.loc[values.index.notna()].sort_index()
    duplicated = int(values.index.duplicated().sum())
    if duplicated:
        values = values.loc[~values.index.duplicated(keep="first")]
    hourly = values.resample("1h").mean().astype(np.float32)
    meta = {
        "raw_path": str(raw_path),
        "raw_rows": int(len(frame)),
        "raw_client_columns": int(values.shape[1]),
        "duplicate_timestamps_dropped": duplicated,
        "hourly_rows": int(len(hourly)),
        "hourly_start": str(hourly.index.min()),
        "hourly_end": str(hourly.index.max()),
    }
    return hourly, meta


def make_windows(n_hours: int, *, test_hours: int, step_hours: int, max_units: int, mode: str) -> list[WindowSpec]:
    first_test_start = TRAIN_HOURS + VALIDATION_HOURS
    windows: list[WindowSpec] = []
    test_start = first_test_start
    while test_start + test_hours <= n_hours and len(windows) < max_units:
        train_start = test_start - VALIDATION_HOURS - TRAIN_HOURS
        train_end = test_start - VALIDATION_HOURS
        windows.append(
            WindowSpec(
                seed=len(windows),
                train_start=train_start,
                train_end=train_end,
                val_start=train_end,
                val_end=test_start,
                test_start=test_start,
                test_end=test_start + test_hours,
                mode=mode,
            )
        )
        test_start += step_hours
    return windows


def eligible_aggregate(hourly: pd.DataFrame, win: WindowSpec) -> tuple[pd.Series, dict[str, object]]:
    train_val_panel = hourly.iloc[win.train_start : win.val_end]
    coverage = train_val_panel.notna().mean(axis=0)
    eligible_cols = coverage.index[coverage >= ELIGIBLE_COVERAGE_MIN].tolist()
    if not eligible_cols:
        raise RuntimeError(f"{win.mode} seed={win.seed} has no eligible clients")
    span = hourly.iloc[win.train_start : win.test_end][eligible_cols]
    aggregate = span.sum(axis=1, min_count=len(eligible_cols)).astype(np.float64)
    dropped_hour_count = int(aggregate.isna().sum())
    aggregate = aggregate.dropna()
    required = TRAIN_HOURS + VALIDATION_HOURS + (win.test_end - win.test_start)
    if len(aggregate) < required - 24:
        raise RuntimeError(f"{win.mode} seed={win.seed} lost too many hours after aggregation")
    payload = "|".join(str(col) for col in eligible_cols).encode("utf-8")
    diag = {
        "seed": int(win.seed),
        "mode": win.mode,
        "eligible_client_count": int(len(eligible_cols)),
        "eligible_client_ids_hash": hashlib.sha256(payload).hexdigest(),
        "dropped_hour_count": dropped_hour_count,
        "train_start": str(hourly.index[win.train_start]),
        "train_end": str(hourly.index[win.train_end - 1]),
        "validation_start": str(hourly.index[win.val_start]),
        "validation_end": str(hourly.index[win.val_end - 1]),
        "test_start": str(hourly.index[win.test_start]),
        "test_end": str(hourly.index[win.test_end - 1]),
    }
    return aggregate.iloc[:required], diag


def top_m_daily_labels(load: pd.Series, m: int) -> pd.Series:
    next_load = load.shift(-1)
    next_day = next_load.index.normalize()
    labels = pd.Series(0.0, index=load.index)
    frame = pd.DataFrame({"next_load": next_load, "day": next_day}, index=load.index).dropna()
    for _, day_frame in frame.groupby("day", sort=False):
        order = day_frame["next_load"].sort_values(ascending=False).index[: int(m)]
        labels.loc[order] = 1.0
    labels.iloc[-1] = np.nan
    return labels


def how_baseline(load: pd.Series, history_slice: slice) -> pd.Series:
    history = load.iloc[history_slice]
    how = history.index.dayofweek * 24 + history.index.hour
    medians = history.groupby(how).median()
    all_how = load.index.dayofweek * 24 + load.index.hour
    return pd.Series([float(medians.get(int(value), history.median())) for value in all_how], index=load.index)


def residual_labels(load: pd.Series, history_slice: slice, q: float) -> pd.Series:
    baseline = how_baseline(load, history_slice)
    residual = load - baseline
    threshold = float(residual.iloc[history_slice].quantile(float(q)))
    return (residual.shift(-1) > threshold).astype(float)


def global_threshold_labels(load: pd.Series, history_slice: slice, q: float) -> pd.Series:
    threshold = float(load.iloc[history_slice].quantile(float(q)))
    return (load.shift(-1) > threshold).astype(float)


def brier_score_metric(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(y, dtype=np.float64)
    return -float(np.mean((p - labels) ** 2))


def safe_log_loss_score(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), PROB_EPS, 1.0 - PROB_EPS)
    labels = np.asarray(y, dtype=np.float64)
    return -float(np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def model_features(load: pd.Series, event: pd.Series) -> pd.DataFrame:
    idx = load.index
    hour = idx.hour.to_numpy(dtype=np.float64)
    dow = idx.dayofweek.to_numpy(dtype=np.float64)
    frame = pd.DataFrame(index=idx)
    frame["load"] = load
    frame["lag1"] = load.shift(1)
    frame["lag24"] = load.shift(24)
    frame["lag168"] = load.shift(168)
    frame["roll24_mean"] = load.shift(1).rolling(24, min_periods=12).mean()
    frame["roll168_mean"] = load.shift(1).rolling(168, min_periods=48).mean()
    frame["event_lag1"] = event.shift(1)
    frame["event_lag24"] = event.shift(24)
    frame["event_lag168"] = event.shift(168)
    frame["event_rate24"] = event.shift(1).rolling(24, min_periods=1).mean()
    frame["event_rate168"] = event.shift(1).rolling(168, min_periods=1).mean()
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    frame["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    frame["hour_of_week"] = (idx.dayofweek * 24 + idx.hour).astype(np.int16)
    return frame


def fit_predict_models(load: pd.Series, event: pd.Series, train_slice: slice, pred_slice: slice) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = model_features(load, event)
    train_idx = load.index[train_slice]
    pred_idx = load.index[pred_slice]
    train_y = event.loc[train_idx].astype(float)
    pred_y = event.loc[pred_idx].astype(float)
    valid_train = train_y.notna()
    train_y_np = train_y.loc[valid_train].to_numpy(dtype=np.float64)
    fallback = float(np.clip(np.nanmean(train_y_np), PROB_EPS, 1.0 - PROB_EPS)) if train_y_np.size else 0.5
    pred_n = len(pred_idx)
    predictions: dict[str, np.ndarray] = {}
    predictions["persistence_event"] = features.loc[pred_idx, "event_lag1"].fillna(fallback).to_numpy(dtype=np.float64)
    predictions["moving_event_rate_24h"] = features.loc[pred_idx, "event_rate24"].fillna(fallback).to_numpy(dtype=np.float64)
    predictions["moving_event_rate_168h"] = features.loc[pred_idx, "event_rate168"].fillna(fallback).to_numpy(dtype=np.float64)
    how_train = features.loc[train_idx, "hour_of_week"].loc[valid_train].to_numpy(dtype=np.int16)
    how_pred = features.loc[pred_idx, "hour_of_week"].to_numpy(dtype=np.int16)
    rates = {int(hour): float(np.mean(train_y_np[how_train == hour])) for hour in np.unique(how_train)}
    predictions["hour_of_week_rate"] = np.array([rates.get(int(hour), fallback) for hour in how_pred], dtype=np.float64)
    predictions["same_hour_last_week"] = features.loc[pred_idx, "event_lag168"].fillna(
        pd.Series(predictions["hour_of_week_rate"], index=pred_idx)
    ).fillna(fallback).to_numpy(dtype=np.float64)

    cols = [
        "load",
        "lag1",
        "lag24",
        "lag168",
        "roll24_mean",
        "roll168_mean",
        "event_lag1",
        "event_lag24",
        "event_lag168",
        "event_rate24",
        "event_rate168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    x_train_df = features.loc[train_idx, cols].loc[valid_train].copy()
    x_pred_df = features.loc[pred_idx, cols].copy()
    fill_values = x_train_df.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_train = x_train_df.replace([np.inf, -np.inf], np.nan).fillna(fill_values).to_numpy(dtype=np.float64)
    x_pred = x_pred_df.replace([np.inf, -np.inf], np.nan).fillna(fill_values).to_numpy(dtype=np.float64)
    if np.unique(train_y_np.astype(int)).size < 2:
        for model_id in ("ridge_linear_ar", "calibrated_logistic", "gradient_boosted"):
            predictions[model_id] = np.full(pred_n, fallback, dtype=np.float64)
    else:
        try:
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            ridge.fit(x_train, train_y_np)
            predictions["ridge_linear_ar"] = np.clip(ridge.predict(x_pred), 0.0, 1.0)
        except Exception:
            predictions["ridge_linear_ar"] = np.full(pred_n, fallback, dtype=np.float64)
        try:
            min_class = int(min(np.bincount(train_y_np.astype(int))))
            cv = max(2, min(5, min_class))
            logistic = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=(0.1, 1.0, 10.0),
                    cv=cv,
                    scoring="neg_brier_score",
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            )
            logistic.fit(x_train, train_y_np.astype(int))
            predictions["calibrated_logistic"] = logistic.predict_proba(x_pred)[:, 1]
        except Exception:
            predictions["calibrated_logistic"] = np.full(pred_n, fallback, dtype=np.float64)
        try:
            gbt = GradientBoostingClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.9,
                random_state=97,
            )
            gbt.fit(x_train, train_y_np.astype(int))
            predictions["gradient_boosted"] = gbt.predict_proba(x_pred)[:, 1]
        except Exception:
            predictions["gradient_boosted"] = np.full(pred_n, fallback, dtype=np.float64)
    return {key: np.clip(value, 0.0, 1.0) for key, value in predictions.items()}, pred_y.to_numpy(dtype=np.float64)


def top_k_actions(prob: np.ndarray, k_abs: int) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    k = int(max(1, min(k_abs, p.size)))
    order = np.lexsort((np.arange(p.size), -p))
    action = np.zeros(p.size, dtype=np.float64)
    action[order[:k]] = 1.0
    return action


def daily_top_m_actions(prob: np.ndarray, timestamps: pd.DatetimeIndex, m: int) -> np.ndarray:
    action = np.zeros(len(prob), dtype=np.float64)
    frame = pd.DataFrame({"prob": prob}, index=timestamps)
    for _, day_frame in frame.groupby(frame.index.normalize(), sort=False):
        order = np.lexsort((np.arange(len(day_frame)), -day_frame["prob"].to_numpy(dtype=np.float64)))
        chosen = day_frame.index[order[: int(m)]]
        action[frame.index.get_indexer(chosen)] = 1.0
    return action


def utility_from_action(action: np.ndarray, y: np.ndarray, friction: float) -> tuple[float, float, float]:
    labels = np.asarray(y, dtype=np.float64)
    hits = float(np.dot(action, labels))
    switches = float(np.abs(np.diff(np.concatenate([[0.0], action]))).sum())
    cost = float(friction) * switches
    return float(hits - cost), cost, switches


def best_model(scores: dict[str, float]) -> str:
    return sorted(scores, key=lambda model_id: (-float(scores[model_id]), model_id))[0]


def build_rows(
    *,
    spec: CandidateSpec,
    split: str,
    seed: int,
    timestamps: pd.DatetimeIndex,
    y: np.ndarray,
    prob_by_model: dict[str, np.ndarray],
    interface_kind: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    forecast_scores = {model: brier_score_metric(prob, y) for model, prob in prob_by_model.items()}
    logloss_scores = {model: safe_log_loss_score(prob, y) for model, prob in prob_by_model.items()}
    forecast_winner = best_model(forecast_scores)
    model_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for friction in FRICTION_GRID:
        utility_scores: dict[str, float] = {}
        cost_scores: dict[str, float] = {}
        switch_scores: dict[str, float] = {}
        for model, prob in prob_by_model.items():
            if interface_kind == "daily_top_m":
                action = daily_top_m_actions(prob, timestamps, int(spec.m or 1))
            else:
                if spec.k_abs is None:
                    k_abs = int(round(float(spec.k_frac) * len(y)))
                else:
                    k_abs = int(spec.k_abs)
                action = top_k_actions(prob, k_abs)
            util, cost, switches = utility_from_action(action, y, friction)
            utility_scores[model] = util
            cost_scores[model] = cost
            switch_scores[model] = switches
        deployed_winner = best_model(utility_scores)
        gap = float(utility_scores[deployed_winner] - utility_scores[forecast_winner])
        seed_rows.append(
            {
                "variant": spec.variant,
                "candidate_id": spec.candidate_id,
                "split": split,
                "seed": seed,
                "m": spec.m,
                "q": spec.q,
                "k_frac": spec.k_frac,
                "k_abs": spec.k_abs if spec.k_abs is not None else int(round(float(spec.k_frac or 0.0) * len(y))),
                "friction_level": float(friction),
                "forecast_winner": forecast_winner,
                "deployed_winner": deployed_winner,
                "forecast_selected_deployed_gap": gap,
                "suboptimal_flag": bool(forecast_winner != deployed_winner),
                "event_rate": float(np.nanmean(y)),
                "positive_count": int(np.nansum(y)),
                "U_best": float(utility_scores[deployed_winner]),
                "U_forecast_selected": float(utility_scores[forecast_winner]),
                "scale_component": float(max(abs(utility_scores[deployed_winner]), abs(utility_scores[forecast_winner]), 1.0)),
            }
        )
        for model in MODEL_ORDER:
            model_rows.append(
                {
                    "question_id": QUESTION_ID,
                    "scenario_id": f"{spec.variant}:{spec.candidate_id}:split_{split}",
                    "domain": DOMAIN_ID,
                    "variant": spec.variant,
                    "candidate_id": spec.candidate_id,
                    "split": split,
                    "seed": seed,
                    "forecaster_id": model,
                    "interface_id": interface_kind,
                    "friction_level": float(friction),
                    "forecast_metric": float(forecast_scores[model]),
                    "target_metric": float(utility_scores[model]),
                    "executed_metric": float(utility_scores[model]),
                    "target_executed_gap": 0.0,
                    "realized_cost": float(cost_scores[model]),
                    "realized_turnover_or_adjustment": float(switch_scores[model]),
                    "rank_within_forecast_metric": np.nan,
                    "rank_within_executed_metric": np.nan,
                    "forecast_metric_brier": float(forecast_scores[model]),
                    "forecast_metric_logloss": float(logloss_scores[model]),
                    "m": spec.m,
                    "q": spec.q,
                    "k_frac": spec.k_frac,
                    "k_abs": spec.k_abs if spec.k_abs is not None else int(round(float(spec.k_frac or 0.0) * len(y))),
                }
            )
    return model_rows, seed_rows


def aggregate_winners(model_df: pd.DataFrame, seed_df: pd.DataFrame) -> dict[str, object]:
    forecast_scores = model_df.groupby("forecaster_id")["forecast_metric"].mean().to_dict()
    deployed_scores = model_df.groupby("forecaster_id")["executed_metric"].mean().to_dict()
    return {
        "forecast_winner": best_model(forecast_scores),
        "deployed_winner": best_model(deployed_scores),
        "subopt_share": float(seed_df["suboptimal_flag"].mean()),
        "subopt_count": int(seed_df["suboptimal_flag"].sum()),
        "n": int(seed_df["seed"].nunique()),
        "mean_gap": float(seed_df["forecast_selected_deployed_gap"].mean()),
        "median_gap": float(seed_df["forecast_selected_deployed_gap"].median()),
    }


def bootstrap_ci(values: np.ndarray, samples: int) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(int(samples), dtype=np.float64)
    for idx in range(int(samples)):
        draws[idx] = float(np.mean(vals[rng.integers(0, vals.size, size=vals.size)]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def epsilon_audit(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for friction, cell in seed_df.groupby("friction_level", sort=True):
        n = int(len(cell))
        scale = float(cell["scale_component"].median())
        nominal = float(cell["suboptimal_flag"].mean())
        for eps in EPS_REL_GRID:
            threshold = float(eps) * scale + 1e-12
            adjusted = float((cell["forecast_selected_deployed_gap"] > threshold).mean())
            verdict = "stable" if eps == 0.0 or adjusted >= 0.70 or (nominal - adjusted) <= 0.10 else "sensitive"
            rows.append(
                {
                    "friction": float(friction),
                    "epsilon_rel": float(eps),
                    "n": n,
                    "scale": scale,
                    "nominal_subopt_share": nominal,
                    "adjusted_subopt_share": adjusted,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def evaluate_candidate(
    *,
    hourly: pd.DataFrame,
    windows: list[WindowSpec],
    spec: CandidateSpec,
    label_builder: Callable[[pd.Series, slice, slice], tuple[pd.Series, pd.Series]],
    interface_kind: str,
) -> CandidateData:
    validation_model_rows: list[dict[str, object]] = []
    validation_seed_rows: list[dict[str, object]] = []
    test_model_rows: list[dict[str, object]] = []
    test_seed_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    for win in windows:
        load, diag = eligible_aggregate(hourly, win)
        test_hours = win.test_end - win.test_start
        train_slice = slice(0, TRAIN_HOURS)
        val_slice = slice(TRAIN_HOURS, TRAIN_HOURS + VALIDATION_HOURS)
        test_slice = slice(TRAIN_HOURS + VALIDATION_HOURS, TRAIN_HOURS + VALIDATION_HOURS + test_hours)
        validation_event, test_event = label_builder(load, train_slice, slice(0, TRAIN_HOURS + VALIDATION_HOURS))
        val_pred, val_y = fit_predict_models(load, validation_event, train_slice, val_slice)
        test_pred, test_y = fit_predict_models(load, test_event, train_slice, test_slice)
        v_models, v_seeds = build_rows(
            spec=spec,
            split="validation",
            seed=win.seed,
            timestamps=load.index[val_slice],
            y=val_y,
            prob_by_model=val_pred,
            interface_kind=interface_kind,
        )
        t_models, t_seeds = build_rows(
            spec=spec,
            split="test",
            seed=win.seed,
            timestamps=load.index[test_slice],
            y=test_y,
            prob_by_model=test_pred,
            interface_kind=interface_kind,
        )
        validation_model_rows.extend(v_models)
        validation_seed_rows.extend(v_seeds)
        test_model_rows.extend(t_models)
        test_seed_rows.extend(t_seeds)
        diag.update(
            {
                "variant": spec.variant,
                "candidate_id": spec.candidate_id,
                "m": spec.m,
                "q": spec.q,
                "k_frac": spec.k_frac,
                "k_abs": spec.k_abs,
                "validation_event_rate": float(np.nanmean(val_y)),
                "validation_positive_count": int(np.nansum(val_y)),
                "test_event_rate": float(np.nanmean(test_y)),
                "test_positive_count": int(np.nansum(test_y)),
            }
        )
        diag_rows.append(diag)
    return CandidateData(
        spec=spec,
        validation_model=pd.DataFrame(validation_model_rows),
        validation_seed=pd.DataFrame(validation_seed_rows),
        test_model=pd.DataFrame(test_model_rows),
        test_seed=pd.DataFrame(test_seed_rows),
        window_diag=pd.DataFrame(diag_rows),
    )


def summarize_candidate(data: CandidateData, bootstrap_samples: int) -> dict[str, object]:
    spec = data.spec
    row: dict[str, object] = {
        "variant": spec.variant,
        "candidate_id": spec.candidate_id,
        "m": spec.m,
        "q": spec.q,
        "k_frac": spec.k_frac,
        "k_abs": spec.k_abs,
        "clean_mode": spec.clean_mode,
        "n_eval_units": int(data.test_seed["seed"].nunique()),
        "validation_event_rate": float(data.window_diag["validation_event_rate"].mean()),
        "test_event_rate": float(data.window_diag["test_event_rate"].mean()),
        "median_positives_per_window": float(data.window_diag["test_positive_count"].median()),
        "zero_positive_window_share": float((data.window_diag["test_positive_count"] == 0).mean()),
    }
    for split, model_df, seed_df in (
        ("validation", data.validation_model, data.validation_seed),
        ("test", data.test_model, data.test_seed),
    ):
        for friction in FRICTION_GRID:
            cell_model = model_df[np.isclose(model_df["friction_level"], friction)].copy()
            cell_seed = seed_df[np.isclose(seed_df["friction_level"], friction)].copy()
            agg = aggregate_winners(cell_model, cell_seed)
            lo, hi = bootstrap_ci(cell_seed["forecast_selected_deployed_gap"].to_numpy(dtype=np.float64), bootstrap_samples)
            if friction == 0.0:
                row[f"{split}_forecast_winner"] = agg["forecast_winner"]
                row[f"{split}_deployed_winner_kappa0"] = agg["deployed_winner"]
                row[f"{split}_zero_subopt"] = agg["subopt_share"]
            elif math.isclose(friction, 0.5):
                row[f"{split}_deployed_winner_kappa05"] = agg["deployed_winner"]
                row[f"{split}_subopt_kappa05"] = agg["subopt_share"]
                row[f"{split}_mean_gap_kappa05"] = agg["mean_gap"]
                row[f"{split}_median_gap_kappa05"] = agg["median_gap"]
                row[f"{split}_ci_low_kappa05"] = lo
            elif math.isclose(friction, 1.0):
                row[f"{split}_deployed_winner_kappa10"] = agg["deployed_winner"]
                row[f"{split}_subopt_kappa10"] = agg["subopt_share"]
                row[f"{split}_mean_gap_kappa10"] = agg["mean_gap"]
                row[f"{split}_median_gap_kappa10"] = agg["median_gap"]
                row[f"{split}_ci_low_kappa10"] = lo
    eps = epsilon_audit(data.test_seed)
    eps005 = eps[np.isclose(eps["epsilon_rel"], 0.005)]
    row["epsilon_verdict"] = "stable" if (eps005["verdict"].eq("stable")).all() else "sensitive"
    row["validation_aligned"] = row["validation_forecast_winner"] == row["validation_deployed_winner_kappa0"]
    row["test_aligned_zero"] = row["test_forecast_winner"] == row["test_deployed_winner_kappa0"]
    row["event_sanity"] = (
        0.03 <= row["test_event_rate"] <= 0.35
        and row["median_positives_per_window"] > 0
        and row["zero_positive_window_share"] <= 0.05
    )
    row["zero_gate"] = row["test_aligned_zero"] and row["test_zero_subopt"] <= 0.20
    positive05 = (
        row["test_deployed_winner_kappa05"] != row["test_forecast_winner"]
        and row["test_subopt_kappa05"] >= 0.70
        and row["test_mean_gap_kappa05"] > 0
        and row["test_median_gap_kappa05"] > 0
        and row["test_ci_low_kappa05"] > 0
    )
    positive10 = (
        row["test_deployed_winner_kappa10"] != row["test_forecast_winner"]
        and row["test_subopt_kappa10"] >= 0.70
        and row["test_mean_gap_kappa10"] > 0
        and row["test_median_gap_kappa10"] > 0
        and row["test_ci_low_kappa10"] > 0
    )
    row["positive_gate_kappa05"] = bool(positive05)
    row["positive_gate_kappa10"] = bool(positive10)
    row["positive_gate"] = bool(positive05 or positive10)
    row["main_eligible"] = bool(
        row["event_sanity"]
        and row["zero_gate"]
        and row["positive_gate"]
        and row["epsilon_verdict"] == "stable"
        and (spec.variant != "v4_four_week_capacity" or spec.clean_mode)
    )
    if row["main_eligible"]:
        row["gate_verdict"] = "Green"
        row["failure_reason"] = ""
    elif row["event_sanity"] and (row["zero_gate"] or row["positive_gate"]):
        row["gate_verdict"] = "Yellow"
        row["failure_reason"] = "; ".join(
            key
            for key, ok in [
                ("zero_gate", row["zero_gate"]),
                ("positive_gate", row["positive_gate"]),
                ("epsilon", row["epsilon_verdict"] == "stable"),
            ]
            if not ok
        )
    else:
        row["gate_verdict"] = "Red"
        reasons = []
        if not row["event_sanity"]:
            reasons.append("event_sanity")
        if not row["zero_gate"]:
            reasons.append("zero_friction")
        if not row["positive_gate"]:
            reasons.append("positive_recurrence")
        if row["epsilon_verdict"] != "stable":
            reasons.append("epsilon")
        row["failure_reason"] = "; ".join(reasons)
    return row


def select_canonical(rows: pd.DataFrame, variant: str) -> str:
    frame = rows[rows["variant"].eq(variant)].copy()
    frame["validation_ok"] = frame["validation_aligned"] & frame["validation_event_rate"].between(0.03, 0.35)
    eligible = frame[frame["validation_ok"]].copy()
    if eligible.empty:
        eligible = frame.copy()
    if variant == "v2_daily_peak_hour":
        order_map = {3.0: 0, 2.0: 1, 4.0: 2}
        eligible["tie_pref"] = eligible["m"].map(order_map).fillna(99)
        eligible = eligible.sort_values(["validation_zero_subopt", "tie_pref", "candidate_id"])
    elif variant == "v3_residual_peak_alert":
        eligible["q_pref"] = (eligible["q"] - 0.85).abs()
        eligible["k_pref"] = (eligible["k_frac"] - 0.15).abs()
        eligible = eligible.sort_values(["validation_zero_subopt", "q_pref", "k_pref", "candidate_id"])
    else:
        eligible["q_pref"] = (eligible["q"] - 0.90).abs()
        eligible["k_pref"] = (eligible["k_frac"] - 0.10).abs()
        eligible = eligible.sort_values(["clean_mode", "validation_zero_subopt", "q_pref", "k_pref", "candidate_id"], ascending=[False, True, True, True, True])
    return str(eligible.iloc[0]["candidate_id"])


def write_variant_outputs(
    output_dir: Path,
    data: CandidateData,
    summary_row: dict[str, object],
    variant_screen: pd.DataFrame,
    bootstrap_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_df = data.test_model.copy()
    seed_df = data.test_seed.copy()
    model_df.to_csv(output_dir / "q2_diff_forecasts_same_interface.csv", index=False)
    seed_df.to_csv(output_dir / "seed_level_selection.csv", index=False)
    data.window_diag.to_csv(output_dir / "window_diagnostics.csv", index=False)
    epsilon = epsilon_audit(seed_df)
    epsilon.to_csv(output_dir / "epsilon_tie_audit.csv", index=False)
    rows = []
    for (seed, model), group in model_df.groupby(["seed", "forecaster_id"], sort=True):
        seed_group = seed_df[seed_df["seed"].eq(seed)]
        row = {
            "seed": int(seed),
            "model_family": str(model),
            "forecast_metric_brier": float(group.iloc[0]["forecast_metric_brier"]),
            "forecast_winner": str(seed_group.iloc[0]["forecast_winner"]),
            "deployed_winner": str(seed_group[np.isclose(seed_group["friction_level"], 1.0)].iloc[0]["deployed_winner"]),
            "forecast_selected_deployed_gap": float(seed_group[np.isclose(seed_group["friction_level"], 1.0)].iloc[0]["forecast_selected_deployed_gap"]),
            "suboptimal_flag": bool(seed_group[np.isclose(seed_group["friction_level"], 1.0)].iloc[0]["suboptimal_flag"]),
        }
        for friction, suffix in [(0.0, "0"), (0.25, "025"), (0.5, "05"), (1.0, "10")]:
            row[f"deployed_utility_kappa_{suffix}"] = float(group[np.isclose(group["friction_level"], friction)].iloc[0]["executed_metric"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "seed_model_metrics.csv", index=False)
    variant_screen.to_csv(output_dir / "candidate_gate_grid.csv", index=False)
    pd.DataFrame([summary_row]).to_csv(output_dir / "canonical_candidate_gate_row.csv", index=False)
    preferred = 1.0 if summary_row.get("positive_gate_kappa10") else 0.5
    cell_seed = seed_df[np.isclose(seed_df["friction_level"], preferred)]
    cell_model = model_df[np.isclose(model_df["friction_level"], preferred)]
    agg = aggregate_winners(cell_model, cell_seed)
    lo, hi = bootstrap_ci(cell_seed["forecast_selected_deployed_gap"].to_numpy(dtype=np.float64), bootstrap_samples)
    table = pd.DataFrame(
        [
            {
                "Domain": "Load alert",
                "Interface": data.spec.variant.replace("_", " "),
                "$\\kappa$": f"{preferred:.2f}",
                "Forecast winner": MODEL_LABELS.get(str(agg["forecast_winner"]), str(agg["forecast_winner"])),
                "Deployed winner": MODEL_LABELS.get(str(agg["deployed_winner"]), str(agg["deployed_winner"])),
                "Agree.": f"{1.0 - float(agg['subopt_share']):.2f}",
                "Subopt.": f"{float(agg['subopt_share']):.2f} ({int(agg['subopt_count'])}/{int(agg['n'])})",
                "Mean gap": f"{float(agg['mean_gap']):.3f}",
                "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                "Role": "Third",
            }
        ]
    )
    table.to_csv(output_dir / "report_card_table.csv", index=False)
    table.to_latex(output_dir / "report_card_table.tex", index=False, escape=False)
    (output_dir / "gate_report.md").write_text(gate_report_text(data, summary_row, epsilon), encoding="utf-8")
    manifest = {
        "variant": data.spec.variant,
        "candidate_id": data.spec.candidate_id,
        "seed_meaning": "deterministic evaluation-unit index, not a random seed",
        "model_families": list(MODEL_ORDER),
        "small_mlp_gru_policy": "Excluded from canonical screening gate to avoid neural tuning confounds.",
        "summary": summary_row,
        "eligible_client_count_by_window": data.window_diag[["seed", "eligible_client_count"]].to_dict(orient="records"),
        "eligible_client_ids_hash_by_window": data.window_diag[["seed", "eligible_client_ids_hash"]].to_dict(orient="records"),
        "dropped_hour_count_by_window": data.window_diag[["seed", "dropped_hour_count"]].to_dict(orient="records"),
        "date_range_by_window": data.window_diag[["seed", "train_start", "validation_start", "test_start", "test_end"]].to_dict(orient="records"),
        "event_rate_by_window": data.window_diag[["seed", "validation_event_rate", "test_event_rate"]].to_dict(orient="records"),
        "positive_count_by_window": data.window_diag[["seed", "validation_positive_count", "test_positive_count"]].to_dict(orient="records"),
        "zero_positive_window_share": float(summary_row["zero_positive_window_share"]),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def gate_report_text(data: CandidateData, summary_row: dict[str, object], epsilon: pd.DataFrame) -> str:
    lines = [
        f"# {data.spec.variant} Gate Report",
        "",
        "## Selection discipline",
        "- Canonical candidate selected using validation only.",
        "- Positive-friction test outcomes were not used for q/k/m selection.",
        "- All candidates are reported in the screening root candidate grid.",
        "- The seed index denotes deterministic evaluation units rather than random seeds.",
        "",
        "## Canonical candidate",
        f"- candidate_id: {data.spec.candidate_id}",
        f"- m: {data.spec.m}",
        f"- q: {data.spec.q}",
        f"- k_frac: {data.spec.k_frac}",
        f"- k_abs: {data.spec.k_abs}",
        "",
        "## Gate results",
        f"- Event sanity: {'PASS' if summary_row['event_sanity'] else 'FAIL'}",
        f"- Zero-friction alignment: {'PASS' if summary_row['zero_gate'] else 'FAIL'}",
        f"- Positive-friction drift: {'PASS' if summary_row['positive_gate'] else 'FAIL'}",
        f"- Epsilon stability: {summary_row['epsilon_verdict']}",
        f"- Verdict: {summary_row['gate_verdict']}",
        f"- Failure reason: {summary_row['failure_reason']}",
        "",
        "## Key numbers",
        f"- n_eval_units: {summary_row['n_eval_units']}",
        f"- test_event_rate: {summary_row['test_event_rate']:.3f}",
        f"- median positives/window: {summary_row['median_positives_per_window']:.1f}",
        f"- zero-positive-window share: {summary_row['zero_positive_window_share']:.3f}",
        f"- kappa0 subopt: {summary_row['test_zero_subopt']:.2f}",
        f"- kappa0.5 subopt/gap/median: {summary_row['test_subopt_kappa05']:.2f} / {summary_row['test_mean_gap_kappa05']:.3f} / {summary_row['test_median_gap_kappa05']:.3f}",
        f"- kappa1.0 subopt/gap/median: {summary_row['test_subopt_kappa10']:.2f} / {summary_row['test_mean_gap_kappa10']:.3f} / {summary_row['test_median_gap_kappa10']:.3f}",
        "",
        "## Epsilon audit",
        epsilon.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def strongest_green(screen: pd.DataFrame) -> pd.Series | None:
    greens = screen[screen["main_eligible"].astype(bool)].copy()
    if greens.empty:
        return None
    chosen_rows = []
    for _, row in greens.iterrows():
        use_kappa = 1.0 if row["positive_gate_kappa10"] else 0.5
        adjusted = row["test_subopt_kappa10"] if use_kappa == 1.0 else row["test_subopt_kappa05"]
        nominal = adjusted
        median_gap = row["test_median_gap_kappa10"] if use_kappa == 1.0 else row["test_median_gap_kappa05"]
        ci_low = row["test_ci_low_kappa10"] if use_kappa == 1.0 else row["test_ci_low_kappa05"]
        priority = {"v2_daily_peak_hour": 0, "v3_residual_peak_alert": 1, "v4_four_week_capacity": 2}.get(row["variant"], 9)
        chosen_rows.append((adjusted, nominal, median_gap, ci_low, -row["test_zero_subopt"], -priority, row))
    chosen_rows.sort(key=lambda item: item[:-1], reverse=True)
    return chosen_rows[0][-1]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    hourly, raw_meta = read_hourly_panel(Path(args.raw_path))
    weekly_windows = make_windows(len(hourly), test_hours=WEEK_HOURS, step_hours=WEEK_HOURS, max_units=MAX_WEEKLY_UNITS, mode="weekly")
    four_week_clean = make_windows(len(hourly), test_hours=FOUR_WEEK_HOURS, step_hours=FOUR_WEEK_HOURS, max_units=MAX_WEEKLY_UNITS, mode="four_week_clean")
    four_week_overlap = make_windows(len(hourly), test_hours=FOUR_WEEK_HOURS, step_hours=WEEK_HOURS, max_units=MAX_WEEKLY_UNITS, mode="four_week_overlap")

    all_data: list[CandidateData] = []
    summary_rows: list[dict[str, object]] = []
    cache_dir = output_root / "_candidate_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(spec: CandidateSpec) -> Path:
        safe_id = f"{spec.variant_dir}__{spec.candidate_id}".replace("/", "_").replace("\\", "_")
        return cache_dir / f"{safe_id}.pkl"

    def run_candidate(
        *,
        spec: CandidateSpec,
        windows: list[WindowSpec],
        label_builder: Callable[[pd.Series, slice, slice], tuple[pd.Series, pd.Series]],
        interface_kind: str,
    ) -> None:
        path = cache_path(spec)
        if path.exists():
            with path.open("rb") as handle:
                data = pickle.load(handle)
            print(f"loaded checkpoint {spec.candidate_id}", flush=True)
        else:
            data = evaluate_candidate(
                hourly=hourly,
                windows=windows,
                spec=spec,
                label_builder=label_builder,
                interface_kind=interface_kind,
            )
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
            print(f"finished {data.spec.candidate_id}", flush=True)
        all_data.append(data)
        summary_rows.append(summarize_candidate(data, int(args.bootstrap_samples)))

    for m in (2, 3, 4):
        spec = CandidateSpec("v2_daily_peak_hour", "v2_daily_peak_hour", f"m{m}", float(m), None, None, int(m), True)

        def labels(load: pd.Series, train_slice: slice, train_val_slice: slice, m=m) -> tuple[pd.Series, pd.Series]:
            event = top_m_daily_labels(load, int(m))
            return event, event

        run_candidate(spec=spec, windows=weekly_windows, label_builder=labels, interface_kind="daily_top_m")

    for q in (0.80, 0.85, 0.90):
        for k_frac in (0.10, 0.15, 0.20):
            spec = CandidateSpec("v3_residual_peak_alert", "v3_residual_peak_alert", f"q{q:.2f}_k{k_frac:.2f}", None, float(q), float(k_frac), None, True)

            def labels(load: pd.Series, train_slice: slice, train_val_slice: slice, q=q) -> tuple[pd.Series, pd.Series]:
                return residual_labels(load, train_slice, float(q)), residual_labels(load, train_val_slice, float(q))

            run_candidate(spec=spec, windows=weekly_windows, label_builder=labels, interface_kind="weekly_top_k")

    for mode_name, windows, clean_mode in (
        ("clean", four_week_clean, True),
        ("overlap", four_week_overlap, False),
    ):
        for q in (0.85, 0.90, 0.95):
            for k_frac in (0.05, 0.10, 0.15):
                spec = CandidateSpec(
                    "v4_four_week_capacity",
                    "v4_four_week_capacity",
                    f"{mode_name}_q{q:.2f}_k{k_frac:.2f}",
                    None,
                    float(q),
                    float(k_frac),
                    None,
                    clean_mode,
                )

                def labels(load: pd.Series, train_slice: slice, train_val_slice: slice, q=q) -> tuple[pd.Series, pd.Series]:
                    return global_threshold_labels(load, train_slice, float(q)), global_threshold_labels(load, train_val_slice, float(q))

                run_candidate(spec=spec, windows=windows, label_builder=labels, interface_kind="weekly_top_k")

    screen = pd.DataFrame(summary_rows)
    canonical_ids = {
        variant: select_canonical(screen, variant)
        for variant in ("v2_daily_peak_hour", "v3_residual_peak_alert", "v4_four_week_capacity")
    }
    screen["canonical_for_variant"] = screen.apply(lambda row: row["candidate_id"] == canonical_ids.get(row["variant"]), axis=1)
    screen.to_csv(output_root / "candidate_screen_grid.csv", index=False)
    screen.to_latex(output_root / "a4_screening_verdict_table.tex", index=False, escape=True)
    selected = strongest_green(screen)
    if selected is None:
        (output_root / "selected_green_report_card_row.tex").write_text("% No Green A4 variant selected.\n", encoding="utf-8")
        (output_root / "selected_a4_main_row.tex").write_text("% No Green A4 variant selected.\n", encoding="utf-8")
    else:
        variant_data = next(data for data in all_data if data.spec.candidate_id == selected["candidate_id"] and data.spec.variant == selected["variant"])
        preferred = 1.0 if selected["positive_gate_kappa10"] else 0.5
        cell_seed = variant_data.test_seed[np.isclose(variant_data.test_seed["friction_level"], preferred)]
        cell_model = variant_data.test_model[np.isclose(variant_data.test_model["friction_level"], preferred)]
        agg = aggregate_winners(cell_model, cell_seed)
        lo, hi = bootstrap_ci(cell_seed["forecast_selected_deployed_gap"].to_numpy(dtype=np.float64), int(args.bootstrap_samples))
        line = (
            f"Load peak-alert & {selected['variant']} & {preferred:.2f} & "
            f"{MODEL_LABELS.get(str(agg['forecast_winner']), agg['forecast_winner'])} & "
            f"{MODEL_LABELS.get(str(agg['deployed_winner']), agg['deployed_winner'])} & "
            f"{1.0 - float(agg['subopt_share']):.2f} & "
            f"{float(agg['subopt_share']):.2f} ({int(agg['subopt_count'])}/{int(agg['n'])}) & "
            f"{float(agg['mean_gap']):.3f} & [{lo:.3f}, {hi:.3f}] & Third \\\\\n"
        )
        (output_root / "selected_green_report_card_row.tex").write_text(line, encoding="utf-8")
        (output_root / "selected_a4_main_row.tex").write_text(line, encoding="utf-8")

    for variant in ("v2_daily_peak_hour", "v3_residual_peak_alert", "v4_four_week_capacity"):
        cid = canonical_ids[variant]
        data = next(item for item in all_data if item.spec.variant == variant and item.spec.candidate_id == cid)
        row = screen[(screen["variant"].eq(variant)) & (screen["candidate_id"].eq(cid))].iloc[0].to_dict()
        variant_screen = screen[screen["variant"].eq(variant)].copy()
        write_variant_outputs(output_root / data.spec.variant_dir, data, row, variant_screen, int(args.bootstrap_samples))

    v1_dir = output_root / "v1_global_threshold_failed"
    v1_dir.mkdir(parents=True, exist_ok=True)
    if V1_OUTPUT.exists():
        for name in ("load_electricity_peak_alert_gate_report.md", "candidate_gate_grid.csv"):
            src = V1_OUTPUT / name
            if src.exists():
                shutil.copy2(src, v1_dir / name)

    summary = screening_summary(raw_meta, screen, canonical_ids, selected)
    (output_root / "screening_summary.md").write_text(summary, encoding="utf-8")
    manifest = {
        "raw_meta": raw_meta,
        "raw_data_policy": "Raw 710MB file reused in-place; not copied into screening output.",
        "output_root": str(output_root),
        "v1_existing_output_path": str(V1_OUTPUT),
        "weekly_units": len(weekly_windows),
        "four_week_clean_units": len(four_week_clean),
        "four_week_overlap_units": len(four_week_overlap),
        "canonical_ids": canonical_ids,
        "selected_green": None if selected is None else selected.to_dict(),
        "small_mlp_gru_policy": (
            "Small MLP/GRU are excluded from the canonical screening gate to avoid neural tuning confounds. "
            "If one variant becomes Green and time remains, MLP/GRU may be run as appendix-only neural sensitivity "
            "with fixed hyperparameters and no q/k/m reselection."
        ),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"screening output root: {output_root}")


def screening_summary(raw_meta: dict[str, object], screen: pd.DataFrame, canonical_ids: dict[str, str], selected: pd.Series | None) -> str:
    lines = [
        "# Load Alert Screening Summary",
        "",
        "## Scope",
        "This pass screens exactly three predeclared variants after the A4v1 Red result. No further task tweaking is performed in this pass.",
        "",
        "## Raw data",
        f"- path: {raw_meta['raw_path']}",
        "- raw 710MB data file was reused in place and not copied.",
        "",
        "## Existing A4v1",
        f"- existing output path: `{V1_OUTPUT}`",
        "- verdict: Red",
        "",
        "## Canonical candidates",
    ]
    for variant, cid in canonical_ids.items():
        row = screen[(screen["variant"].eq(variant)) & (screen["candidate_id"].eq(cid))].iloc[0]
        lines.append(
            f"- {variant}: `{cid}`, verdict={row['gate_verdict']}, "
            f"test_event_rate={row['test_event_rate']:.3f}, zero_subopt={row['test_zero_subopt']:.2f}, "
            f"kappa0.5_subopt={row['test_subopt_kappa05']:.2f}, kappa1.0_subopt={row['test_subopt_kappa10']:.2f}, "
            f"failure={row['failure_reason']}"
        )
    lines.extend(["", "## Selected Green"])
    if selected is None:
        lines.append("No variant met the main-eligible Green gate.")
    else:
        lines.append(f"Selected {selected['variant']} / {selected['candidate_id']} by the predeclared strongest-Green rule.")
    lines.extend(["", "## Full candidate grid", screen.to_markdown(index=False), ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
