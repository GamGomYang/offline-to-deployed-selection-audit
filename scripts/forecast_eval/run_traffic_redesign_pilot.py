#!/usr/bin/env python3
"""Traffic Hourly family redesign pilot runner.

This runner intentionally keeps the Traffic redesign self-contained: the old
Traffic Hourly exceedance implementation is no longer present in this branch,
so ingestion, labels, pooled models, top-k deployment, and promotion gates live
here while reusing the repository's Q2 ranking conventions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_result_row, prepare_results_frame, save_results  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "extensions" / "q2_pivot_revision_20260423" / "new_reruns" / "traffic_redesign"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "_cache"
TRAFFIC_ZIP_URL = "https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip?download=1"
TRAFFIC_DOI = "10.5281/zenodo.4656132"
TRAFFIC_REFERENCE_MD5 = "1cf694f99f95700217845078b467fb24"
TRAFFIC_REFERENCE_FILE_SIZE = 20228280
EXPECTED_N_SERIES = 862
EXPECTED_LENGTH = 17544
EXPECTED_FREQUENCY = "hourly"

SCENARIO_IDS = (
    "traffic_topk_alert_q2_v1",
    "traffic_relative_rank_q2_v1",
    "traffic_surge_onset_q2_v1",
)
FAMILY_IDS = (
    "reactive_short",
    "calibrated_baseline",
    "lagged_smoother",
    "linear_ar_head",
)
FRICTION_ONE = 1.0
FRICTION_MID = 0.5
FRICTION_ZERO = 0.0
TIE_ABS_FLOOR = 1e-10
TIE_REL_SCALE = 1e-8
FP_PENALTY = 0.25
FULL_REPLICATES = 100
LOGLOSS_EPS = 1e-6


FAMILY_SPECS: dict[str, dict[str, list[int]]] = {
    "reactive_short": {"lags": [1, 2, 3, 6, 12, 24], "diffs": [1, 24], "rolls": []},
    "calibrated_baseline": {"lags": [1, 24, 168], "diffs": [], "rolls": [24, 168]},
    "lagged_smoother": {"lags": [24, 48, 72, 168], "diffs": [], "rolls": [24, 168]},
    "linear_ar_head": {"lags": [1, 2, 3, 6, 12, 24, 48, 72, 168], "diffs": [], "rolls": []},
}


@dataclass(frozen=True)
class TrafficPanel:
    values: np.ndarray
    frequency: str
    metadata: dict[str, Any]
    source_lock: dict[str, Any]


@dataclass(frozen=True)
class SplitSpec:
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class TaskMaterialization:
    scenario_id: str
    variant_id: str
    labels: np.ndarray
    label_target_indices: np.ndarray
    validation_event_rate: float
    k_ref: int
    k_values: tuple[int, ...]
    params: dict[str, Any]
    provenance: dict[str, Any]


@dataclass
class FittedFamily:
    family_id: str
    model: LogisticRegression | None
    calibrator: IsotonicRegression | None
    constant_probability: float | None
    constant_predictor_fallback: bool
    feature_mean_by_series: np.ndarray
    feature_std_by_series: np.ndarray
    spec: dict[str, list[int]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Traffic Hourly redesign family pilots.")
    parser.add_argument("--scenario-id", choices=(*SCENARIO_IDS, "all"), default="all")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--frictions", default="0.00,0.50,1.00")
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--run-full-if-strong", action="store_true")
    parser.add_argument("--skip-summary-refresh", action="store_true", help="Compatibility no-op; v1 does not refresh summaries.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--source-zip", default=None, help="Optional local Traffic Hourly zip, primarily for tests.")
    parser.add_argument("--allow-noncanonical-panel", action="store_true", help="Allow tiny fixture panels in tests.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=50_000,
        help="Deterministic cap on pooled train rows used to fit each logistic model.",
    )
    return parser.parse_args()


def parse_frictions(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("At least one friction value is required.")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - reference metadata only, not a security boundary.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def download_or_copy_source(
    *,
    cache_dir: Path,
    source_zip: Path | None,
    skip_download: bool,
) -> tuple[Path, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "traffic_hourly_dataset.zip"
    headers: dict[str, Any] = {}

    if source_zip is not None:
        shutil.copy2(source_zip, zip_path)
    elif not zip_path.exists():
        if skip_download:
            raise FileNotFoundError(f"Traffic Hourly zip missing at {zip_path} and --skip-download was set.")
        response = requests.get(TRAFFIC_ZIP_URL, stream=True, timeout=120)
        response.raise_for_status()
        headers = dict(response.headers)
        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    elif skip_download:
        headers = {}

    return zip_path, headers


def _parse_tsf_value(value: str) -> float:
    value = value.strip()
    if value in {"?", "NaN", "nan", ""}:
        return float("nan")
    return float(value)


def parse_tsf_text(text: str) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    data_rows: list[list[float]] = []
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower == "@data":
            in_data = True
            continue
        if not in_data:
            if line.startswith("@"):
                parts = line.split(maxsplit=1)
                key = parts[0][1:].lower()
                metadata[key] = parts[1].strip() if len(parts) > 1 else ""
            continue

        series_text = line.rsplit(":", maxsplit=1)[-1]
        data_rows.append([_parse_tsf_value(value) for value in series_text.split(",")])

    if not data_rows:
        raise ValueError("No @data rows were found in TSF content.")
    lengths = {len(row) for row in data_rows}
    if len(lengths) != 1:
        raise ValueError(f"TSF rows are not equal length: {sorted(lengths)}")
    values = np.asarray(data_rows, dtype=np.float64)
    return values, metadata


def load_tsf_from_zip(zip_path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        tsf_names = [name for name in archive.namelist() if name.lower().endswith(".tsf")]
        if len(tsf_names) != 1:
            raise ValueError(f"Expected exactly one TSF file in {zip_path}, found {tsf_names}")
        tsf_name = tsf_names[0]
        raw_bytes = archive.read(tsf_name)
    values, metadata = parse_tsf_text(raw_bytes.decode("utf-8"))
    extracted = {
        "tsf_name": tsf_name,
        "tsf_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "tsf_size_bytes": str(len(raw_bytes)),
    }
    return values, metadata, extracted


def assert_panel(values: np.ndarray, metadata: dict[str, Any], *, allow_noncanonical: bool) -> dict[str, Any]:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D panel, got shape {values.shape}")
    if np.isnan(values).any():
        raise ValueError("Traffic Hourly panel contains missing values.")
    lengths_ok = bool(values.shape[1] > 0)
    frequency = str(metadata.get("frequency", "")).strip().lower()

    assertions = {
        "n_series": int(values.shape[0]),
        "series_length": int(values.shape[1]),
        "frequency": frequency,
        "no_missing": bool(not np.isnan(values).any()),
        "equal_length": lengths_ok,
        "canonical_panel": bool(
            values.shape == (EXPECTED_N_SERIES, EXPECTED_LENGTH) and frequency == EXPECTED_FREQUENCY
        ),
    }
    if not allow_noncanonical:
        if values.shape != (EXPECTED_N_SERIES, EXPECTED_LENGTH):
            raise ValueError(f"Expected canonical Traffic Hourly shape {(EXPECTED_N_SERIES, EXPECTED_LENGTH)}, got {values.shape}")
        if frequency != EXPECTED_FREQUENCY:
            raise ValueError(f"Expected frequency={EXPECTED_FREQUENCY}, got {frequency!r}")
    return assertions


def load_traffic_panel(
    *,
    cache_dir: Path,
    source_zip: Path | None = None,
    skip_download: bool = False,
    allow_noncanonical_panel: bool = False,
) -> TrafficPanel:
    zip_path, headers = download_or_copy_source(cache_dir=cache_dir, source_zip=source_zip, skip_download=skip_download)
    values, metadata, extracted = load_tsf_from_zip(zip_path)
    assertions = assert_panel(values, metadata, allow_noncanonical=allow_noncanonical_panel)
    content_length = headers.get("content-length") or headers.get("Content-Length")
    file_size = zip_path.stat().st_size
    source_lock = {
        "source_url": TRAFFIC_ZIP_URL,
        "doi": TRAFFIC_DOI,
        "downloaded_zip_path": str(zip_path),
        "downloaded_zip_sha256": sha256_file(zip_path),
        "downloaded_zip_md5_reference_observed": md5_file(zip_path),
        "reference_md5": TRAFFIC_REFERENCE_MD5,
        "reference_file_size": TRAFFIC_REFERENCE_FILE_SIZE,
        "observed_file_size": int(file_size),
        "http_content_length_observed": int(content_length) if content_length and str(content_length).isdigit() else None,
        "content_length_policy": "warning_only",
        "hard_lock_policy": "downloaded_zip_sha256_plus_parsed_panel_assertions",
        "created_utc": utc_now_iso(),
        "parsed_assertions": assertions,
        **extracted,
    }
    return TrafficPanel(values=values, frequency=str(metadata.get("frequency", "")), metadata=metadata, source_lock=source_lock)


def build_split(n_time: int) -> SplitSpec:
    if n_time <= 337:
        raise ValueError(f"Need more than 337 timesteps for fixed train/validation/test split, got {n_time}.")
    train_end = int(n_time - 336)
    validation_start = train_end
    validation_end = int(train_end + 168)
    test_start = validation_end
    test_end = int(n_time)
    return SplitSpec(
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
    )


def _seasonal_features(target_indices: np.ndarray) -> np.ndarray:
    source_idx = np.asarray(target_indices, dtype=np.int64) - 1
    day = 2.0 * np.pi * (source_idx % 24) / 24.0
    week = 2.0 * np.pi * (source_idx % 168) / 168.0
    return np.column_stack([np.sin(day), np.cos(day), np.sin(week), np.cos(week)]).astype(np.float64)


def continuous_features_for_series(values: np.ndarray, target_indices: np.ndarray, spec: dict[str, list[int]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    source_idx = np.asarray(target_indices, dtype=np.int64) - 1
    for lag in spec.get("lags", []):
        rows.append(values[source_idx - int(lag)])
    for diff in spec.get("diffs", []):
        rows.append(values[source_idx] - values[source_idx - int(diff)])
    for window in spec.get("rolls", []):
        width = int(window)
        cumsum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
        starts = source_idx - width + 1
        stops = source_idx + 1
        rows.append((cumsum[stops] - cumsum[starts]) / float(width))
    if not rows:
        return np.empty((len(target_indices), 0), dtype=np.float64)
    return np.column_stack(rows).astype(np.float64)


def target_indices_for_split(split: SplitSpec, max_lookback: int, split_name: str) -> np.ndarray:
    if split_name == "train":
        start, end = max(int(max_lookback) + 1, 1), split.train_end
    elif split_name == "validation":
        start, end = split.validation_start, split.validation_end
    elif split_name == "test":
        start, end = split.test_start, split.test_end
    else:
        raise ValueError(f"Unknown split_name: {split_name}")
    if start >= end:
        raise ValueError(f"Split {split_name} has no usable rows after max_lookback={max_lookback}.")
    return np.arange(start, end, dtype=np.int64)


def max_lookback(spec: dict[str, list[int]]) -> int:
    values = [0]
    for key in ("lags", "diffs", "rolls"):
        values.extend(int(v) for v in spec.get(key, []))
    return max(values)


def fit_family(
    panel: np.ndarray,
    labels: np.ndarray,
    split: SplitSpec,
    family_id: str,
    *,
    max_train_rows: int,
    seed: int,
) -> FittedFamily:
    spec = FAMILY_SPECS[family_id]
    train_idx = target_indices_for_split(split, max_lookback(spec), "train")
    val_idx = target_indices_for_split(split, max_lookback(spec), "validation")
    n_series = panel.shape[0]

    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    family_seed = int(hashlib.sha256(str(family_id).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(int(seed) + family_seed % 10_000)
    total_train_rows = int(n_series * len(train_idx))
    max_rows = min(int(max_train_rows), total_train_rows)
    if max_rows < total_train_rows:
        sampled_flat = np.sort(rng.choice(total_train_rows, size=max_rows, replace=False))
        sampled_by_series: dict[int, np.ndarray] = {}
        for flat_idx in sampled_flat:
            series_idx = int(flat_idx // len(train_idx))
            local_idx = int(flat_idx % len(train_idx))
            sampled_by_series.setdefault(series_idx, []).append(local_idx)
        sampled_by_series = {
            series_idx: np.asarray(local_indices, dtype=np.int64)
            for series_idx, local_indices in sampled_by_series.items()
        }
    else:
        sampled_by_series = {series_idx: np.arange(len(train_idx), dtype=np.int64) for series_idx in range(n_series)}

    for series_idx in range(n_series):
        cont = continuous_features_for_series(panel[series_idx], train_idx, spec)
        mean = cont.mean(axis=0) if cont.shape[1] else np.empty(0, dtype=np.float64)
        std = cont.std(axis=0) if cont.shape[1] else np.empty(0, dtype=np.float64)
        std = np.where(std <= 1e-12, 1.0, std)
        means.append(mean)
        stds.append(std)
        local_indices = sampled_by_series.get(series_idx)
        if local_indices is None or local_indices.size == 0:
            continue
        sampled_train_idx = train_idx[local_indices]
        sampled_cont = cont[local_indices]
        scaled = (sampled_cont - mean) / std if sampled_cont.shape[1] else sampled_cont
        train_x_parts.append(np.column_stack([scaled, _seasonal_features(sampled_train_idx)]))
        train_y_parts.append(labels[series_idx, sampled_train_idx])

    x_train = np.vstack(train_x_parts)
    y_train = np.concatenate(train_y_parts).astype(np.int64)
    positive_rate = float(y_train.mean()) if y_train.size else 0.0

    model: LogisticRegression | None = None
    calibrator: IsotonicRegression | None = None
    constant: float | None = None
    fallback = bool(np.unique(y_train).size < 2)
    if fallback:
        constant = positive_rate
    else:
        model = LogisticRegression(solver="liblinear", max_iter=200, C=1.0)
        model.fit(x_train, y_train)

    if family_id == "calibrated_baseline":
        val_probs = predict_family_probs(
            panel,
            FittedFamily(
                family_id=family_id,
                model=model,
                calibrator=None,
                constant_probability=constant,
                constant_predictor_fallback=fallback,
                feature_mean_by_series=np.vstack(means),
                feature_std_by_series=np.vstack(stds),
                spec=spec,
            ),
            val_idx,
        )
        val_y = labels[:, val_idx].reshape(-1)
        if np.unique(val_y).size >= 2:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_probs.reshape(-1), val_y.astype(np.int64))

    return FittedFamily(
        family_id=family_id,
        model=model,
        calibrator=calibrator,
        constant_probability=constant,
        constant_predictor_fallback=fallback,
        feature_mean_by_series=np.vstack(means),
        feature_std_by_series=np.vstack(stds),
        spec=spec,
    )


def predict_family_probs(panel: np.ndarray, fitted: FittedFamily, target_indices: np.ndarray) -> np.ndarray:
    n_series = panel.shape[0]
    season = _seasonal_features(target_indices)
    predictions: list[np.ndarray] = []
    for series_idx in range(n_series):
        cont = continuous_features_for_series(panel[series_idx], target_indices, fitted.spec)
        if cont.shape[1]:
            cont = (cont - fitted.feature_mean_by_series[series_idx]) / fitted.feature_std_by_series[series_idx]
        x = np.column_stack([cont, season])
        if fitted.constant_predictor_fallback:
            probs = np.full(x.shape[0], float(fitted.constant_probability), dtype=np.float64)
        elif fitted.model is not None:
            probs = fitted.model.predict_proba(x)[:, 1]
        else:
            raise RuntimeError(f"Family {fitted.family_id} has neither a model nor a constant fallback.")
        if fitted.calibrator is not None:
            probs = fitted.calibrator.predict(probs)
        predictions.append(np.clip(probs, 0.0, 1.0))
    return np.vstack(predictions)


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities, dtype=np.float64) - np.asarray(labels, dtype=np.float64)) ** 2))


def logloss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), LOGLOSS_EPS, 1.0 - LOGLOSS_EPS)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean(-(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))))


def c1_labels(panel: np.ndarray, split: SplitSpec) -> tuple[np.ndarray, dict[str, Any]]:
    train_values = panel[:, : split.train_end]
    q = 0.70
    thresholds = np.quantile(train_values, q, axis=1)
    labels = (panel > thresholds[:, None]).astype(np.int64)
    validation_rate_q70 = float(labels[:, split.validation_start : split.validation_end].mean())
    fallback = bool(validation_rate_q70 < 0.08)
    if fallback:
        q = 0.60
        thresholds = np.quantile(train_values, q, axis=1)
        labels = (panel > thresholds[:, None]).astype(np.int64)
    provenance = {
        "q": float(q),
        "effective_threshold_quantile": float(q),
        "fallback_triggered": fallback,
        "validation_event_rate_at_q70": validation_rate_q70,
    }
    return labels, provenance


def c2_labels(panel: np.ndarray, m: float) -> tuple[np.ndarray, dict[str, Any]]:
    n_series, n_time = panel.shape
    k = max(1, int(math.ceil(float(m) * n_series)))
    labels = np.zeros((n_series, n_time), dtype=np.int64)
    for t in range(n_time):
        ordered = np.argsort(-panel[:, t], kind="mergesort")[:k]
        labels[ordered, t] = 1
    return labels, {"m": float(m), "panel_top_count": int(k)}


def c3_labels(panel: np.ndarray, split: SplitSpec, q_delta: float) -> tuple[np.ndarray, dict[str, Any]]:
    deltas = np.diff(panel, prepend=panel[:, :1], axis=1)
    thresholds = np.zeros(panel.shape[0], dtype=np.float64)
    train_deltas = deltas[:, 1 : split.train_end]
    for series_idx in range(panel.shape[0]):
        positive = train_deltas[series_idx][train_deltas[series_idx] > 0.0]
        source = positive if positive.size else train_deltas[series_idx]
        thresholds[series_idx] = float(np.quantile(source, float(q_delta)))
    labels = (deltas > thresholds[:, None]).astype(np.int64)
    return labels, {"q_delta": float(q_delta)}


def pooled_validation_mean_positive_count(labels: np.ndarray, split: SplitSpec) -> float:
    validation_labels = labels[:, split.validation_start : split.validation_end]
    return float(validation_labels.sum(axis=0).mean())


def materialize_tasks(panel: np.ndarray, split: SplitSpec, scenario_id: str) -> list[TaskMaterialization]:
    tasks: list[TaskMaterialization] = []
    label_target_indices = np.arange(split.test_start, split.test_end, dtype=np.int64)
    if scenario_id == "traffic_topk_alert_q2_v1":
        labels, provenance = c1_labels(panel, split)
        mean_count = pooled_validation_mean_positive_count(labels, split)
        k_ref = max(1, int(round(mean_count)))
        k_values = tuple(sorted({max(1, int(math.floor(0.75 * k_ref))), k_ref, max(1, int(math.ceil(1.25 * k_ref)))}))
        tasks.append(
            TaskMaterialization(
                scenario_id=scenario_id,
                variant_id=f"q{provenance['effective_threshold_quantile']:.2f}_kgrid",
                labels=labels,
                label_target_indices=label_target_indices,
                validation_event_rate=float(labels[:, split.validation_start : split.validation_end].mean()),
                k_ref=k_ref,
                k_values=k_values,
                params={"q": float(provenance["effective_threshold_quantile"]), "k_values": list(k_values)},
                provenance={**provenance, "validation_mean_positive_count": mean_count},
            )
        )
    elif scenario_id == "traffic_relative_rank_q2_v1":
        for m in (0.10, 0.15):
            labels, provenance = c2_labels(panel, m)
            mean_count = pooled_validation_mean_positive_count(labels, split)
            k_ref = max(1, int(round(mean_count)))
            tasks.append(
                TaskMaterialization(
                    scenario_id=scenario_id,
                    variant_id=f"m{m:.2f}",
                    labels=labels,
                    label_target_indices=label_target_indices,
                    validation_event_rate=float(labels[:, split.validation_start : split.validation_end].mean()),
                    k_ref=k_ref,
                    k_values=(k_ref,),
                    params={"m": float(m), "k": int(k_ref)},
                    provenance={**provenance, "validation_mean_positive_count": mean_count},
                )
            )
    elif scenario_id == "traffic_surge_onset_q2_v1":
        for q_delta in (0.70, 0.75):
            labels, provenance = c3_labels(panel, split, q_delta)
            mean_count = pooled_validation_mean_positive_count(labels, split)
            k_ref = max(1, int(round(mean_count)))
            tasks.append(
                TaskMaterialization(
                    scenario_id=scenario_id,
                    variant_id=f"qdelta{q_delta:.2f}",
                    labels=labels,
                    label_target_indices=label_target_indices,
                    validation_event_rate=float(labels[:, split.validation_start : split.validation_end].mean()),
                    k_ref=k_ref,
                    k_values=(k_ref,),
                    params={"q_delta": float(q_delta), "k": int(k_ref)},
                    provenance={**provenance, "validation_mean_positive_count": mean_count},
                )
            )
    else:
        raise ValueError(f"Unsupported scenario_id: {scenario_id}")
    return tasks


def top_k_actions(probabilities: np.ndarray, k: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    k_eff = max(1, min(int(k), probs.shape[0]))
    actions = np.zeros(probs.shape, dtype=np.int64)
    for t in range(probs.shape[1]):
        ordered = np.argsort(-probs[:, t], kind="mergesort")[:k_eff]
        actions[ordered, t] = 1
    return actions


def set_switch_sizes(actions: np.ndarray) -> np.ndarray:
    action_array = np.asarray(actions, dtype=np.int64)
    if action_array.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    switches = np.zeros(action_array.shape[1], dtype=np.float64)
    previous = np.zeros(action_array.shape[0], dtype=np.int64)
    for t in range(action_array.shape[1]):
        current = action_array[:, t]
        switches[t] = float(np.not_equal(current, previous).sum())
        previous = current
    return switches


def evaluate_top_k(probabilities: np.ndarray, labels: np.ndarray, *, k: int, friction: float) -> dict[str, float]:
    actions = top_k_actions(probabilities, k)
    y = np.asarray(labels, dtype=np.int64)
    tp = ((actions == 1) & (y == 1)).sum(axis=0).astype(np.float64)
    fp = ((actions == 1) & (y == 0)).sum(axis=0).astype(np.float64)
    switch_sizes = set_switch_sizes(actions)
    utility = tp - FP_PENALTY * fp - float(friction) * switch_sizes
    n_units = max(actions.shape[0], 1)
    return {
        "deployed_utility": float(utility.mean()) if utility.size else 0.0,
        "switch_rate": float(switch_sizes.mean()) if switch_sizes.size else 0.0,
        "alert_or_set_size_rate": float(actions.sum(axis=0).mean() / n_units) if actions.shape[1] else 0.0,
    }


def scores_tied(a: float, b: float) -> bool:
    tol = max(TIE_ABS_FLOOR, TIE_REL_SCALE * max(abs(float(a)), abs(float(b)), 1.0))
    return abs(float(a) - float(b)) <= tol


def top_set(scores: dict[str, float], *, higher_is_better: bool) -> tuple[str, ...]:
    if not scores:
        return ()
    best = max(scores.values()) if higher_is_better else min(scores.values())
    winners = [key for key, value in scores.items() if scores_tied(value, best)]
    return tuple(sorted(winners))


def representative(winners: Iterable[str], tiebreak_scores: dict[str, float], *, higher_is_better: bool) -> str:
    values = list(winners)
    if not values:
        raise ValueError("Cannot choose representative from empty winner set.")
    return sorted(values, key=lambda key: ((-1 if higher_is_better else 1) * float(tiebreak_scores[key]), key))[0]


def bootstrap_indices(n_series: int, n_replicates: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return {rep: rng.integers(0, int(n_series), size=int(n_series), endpoint=False) for rep in range(int(n_replicates))}


def modal_winner(values: pd.Series, score_df: pd.DataFrame, score_column: str, *, higher_is_better: bool) -> str:
    counts = values.astype(str).value_counts()
    if counts.empty:
        return ""
    max_count = int(counts.max())
    candidates = sorted(counts[counts == max_count].index.tolist())
    mean_scores = score_df.groupby("family_id")[score_column].mean().to_dict()
    return sorted(candidates, key=lambda key: ((-1 if higher_is_better else 1) * float(mean_scores.get(key, 0.0)), key))[0]


def evaluate_task(
    *,
    panel: np.ndarray,
    task: TaskMaterialization,
    fitted: dict[str, FittedFamily],
    frictions: tuple[float, ...],
    replicates: int,
    seed: int,
    prefix: str,
) -> dict[str, Any]:
    test_idx = task.label_target_indices
    full_labels = task.labels[:, test_idx]
    full_probs = {family_id: predict_family_probs(panel, fitted[family_id], test_idx) for family_id in FAMILY_IDS}
    bootstraps = bootstrap_indices(panel.shape[0], replicates, seed)

    seed_rows: list[dict[str, Any]] = []
    q2_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for variant_k in task.k_values:
        for replicate_id, sampled_idx in bootstraps.items():
            labels = full_labels[sampled_idx, :]
            forecast_scores_brier = {
                family_id: brier(full_probs[family_id][sampled_idx, :], labels) for family_id in FAMILY_IDS
            }
            forecast_scores_logloss = {
                family_id: logloss(full_probs[family_id][sampled_idx, :], labels) for family_id in FAMILY_IDS
            }
            forecast_top = top_set(forecast_scores_brier, higher_is_better=False)
            forecast_winner = representative(forecast_top, forecast_scores_brier, higher_is_better=False)

            for friction in frictions:
                deployed_scores: dict[str, float] = {}
                deployment_cache: dict[str, dict[str, float]] = {}
                for family_id in FAMILY_IDS:
                    metrics = evaluate_top_k(full_probs[family_id][sampled_idx, :], labels, k=int(variant_k), friction=float(friction))
                    deployed_scores[family_id] = float(metrics["deployed_utility"])
                    deployment_cache[family_id] = metrics

                deployed_top = top_set(deployed_scores, higher_is_better=True)
                deployed_winner = representative(deployed_top, deployed_scores, higher_is_better=True)
                tie_involved = bool(len(forecast_top) > 1 or len(deployed_top) > 1)
                agreement = bool(set(forecast_top).intersection(deployed_top))
                gap = float(deployed_scores[deployed_winner] - deployed_scores[forecast_winner])

                selection_rows.append(
                    {
                        "scenario_id": task.scenario_id,
                        "variant_id": task.variant_id,
                        "k": int(variant_k),
                        "replicate_id": int(replicate_id),
                        "friction": float(friction),
                        "forecast_winner": forecast_winner,
                        "deployed_winner": deployed_winner,
                        "agreement": agreement,
                        "gap": gap,
                        "tie_involved": tie_involved,
                    }
                )

                for family_id in FAMILY_IDS:
                    metrics = deployment_cache[family_id]
                    seed_rows.append(
                        {
                            "scenario_id": task.scenario_id,
                            "variant_id": task.variant_id,
                            "replicate_id": int(replicate_id),
                            "friction": float(friction),
                            "family_id": family_id,
                            "k": int(variant_k),
                            "forecast_metric_brier": float(forecast_scores_brier[family_id]),
                            "forecast_metric_logloss": float(forecast_scores_logloss[family_id]),
                            "deployed_utility": float(metrics["deployed_utility"]),
                            "switch_rate": float(metrics["switch_rate"]),
                            "alert_or_set_size_rate": float(metrics["alert_or_set_size_rate"]),
                            "forecast_winner_flag": bool(family_id in forecast_top),
                            "deployed_winner_flag": bool(family_id in deployed_top),
                            "tie_involved": tie_involved,
                        }
                    )
                    q2_rows.append(
                        build_result_row(
                            question_id="Q2",
                            scenario_id=f"{task.scenario_id}:{task.variant_id}:k{int(variant_k)}",
                            domain="traffic_hourly",
                            seed=int(replicate_id),
                            forecaster_id=family_id,
                            interface_id="top_k_set",
                            friction_level=float(friction),
                            forecast_metric=-float(forecast_scores_brier[family_id]),
                            target_metric=float(metrics["deployed_utility"]),
                            executed_metric=float(metrics["deployed_utility"]),
                            realized_cost=float(float(friction) * metrics["switch_rate"]),
                            realized_turnover_or_adjustment=float(metrics["switch_rate"]),
                        )
                    )

    seed_df = pd.DataFrame(seed_rows)
    selection_seed_df = pd.DataFrame(selection_rows)
    summary_df = build_selection_summary(selection_seed_df, seed_df)
    zero_df = build_zero_row_diagnostics(selection_seed_df, seed_df)
    pilot_table = build_pilot_table(task, summary_df, seed_df)
    gates = evaluate_gates(task.scenario_id, summary_df, seed_df, task, fitted)
    pilot_table["verdict"] = gates["verdict"]
    q2_df = prepare_results_frame(q2_rows)

    return {
        "seed_level_metrics": seed_df,
        "selection_summary_by_friction": summary_df,
        "zero_row_diagnostics": zero_df,
        "pilot_table": pilot_table,
        "q2_raw": q2_df,
        "gates": gates,
    }


def build_selection_summary(selection_seed_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant_id, k, friction), group in selection_seed_df.groupby(["variant_id", "k", "friction"], sort=True):
        score_slice = seed_df[(seed_df["variant_id"] == variant_id) & (seed_df["k"] == k) & np.isclose(seed_df["friction"], float(friction))]
        rows.append(
            {
                "variant_id": str(variant_id),
                "k": int(k),
                "friction": float(friction),
                "forecast_winner": modal_winner(group["forecast_winner"], score_slice, "forecast_metric_brier", higher_is_better=False),
                "deployed_winner": modal_winner(group["deployed_winner"], score_slice, "deployed_utility", higher_is_better=True),
                "agreement": float(group["agreement"].mean()),
                "mean_gap": float(group["gap"].mean()),
                "median_gap": float(group["gap"].median()),
                "deployed_suboptimal_share": float((~group["agreement"].astype(bool)).mean()),
                "modal_winner_divergence": bool(
                    modal_winner(group["forecast_winner"], score_slice, "forecast_metric_brier", higher_is_better=False)
                    != modal_winner(group["deployed_winner"], score_slice, "deployed_utility", higher_is_better=True)
                ),
                "mean_set_switch_rate": float(score_slice["switch_rate"].mean()),
                "tie_involved_fraction": float(group["tie_involved"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant_id", "k", "friction"]).reset_index(drop=True)


def build_zero_row_diagnostics(selection_seed_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    zero_selection = selection_seed_df[np.isclose(selection_seed_df["friction"], 0.0)].copy()
    for row in zero_selection.itertuples(index=False):
        score_slice = seed_df[
            (seed_df["variant_id"] == row.variant_id)
            & (seed_df["k"] == row.k)
            & (seed_df["replicate_id"] == row.replicate_id)
            & np.isclose(seed_df["friction"], 0.0)
        ]
        rows.append(
            {
                "scenario_id": str(row.scenario_id),
                "variant_id": str(row.variant_id),
                "k": int(row.k),
                "replicate_id": int(row.replicate_id),
                "forecast_winner": str(row.forecast_winner),
                "deployed_winner": str(row.deployed_winner),
                "agreement_zero": bool(row.agreement),
                "mean_gap_zero": float(row.gap),
                "median_gap_zero": float(row.gap),
                "tie_involved_zero": bool(row.tie_involved),
                "set_switch_rate_zero": float(score_slice["switch_rate"].mean()),
                "alert_or_set_size_zero": float(score_slice["alert_or_set_size_rate"].mean()),
                "verdict_zero_explainable": "yes" if bool(row.agreement) or abs(float(row.gap)) <= 0.005 else "no",
            }
        )
    return pd.DataFrame(rows).sort_values(["variant_id", "k", "replicate_id"]).reset_index(drop=True)


def _row_at(summary_df: pd.DataFrame, variant_id: str, k: int, friction: float) -> pd.Series:
    matched = summary_df[
        (summary_df["variant_id"] == str(variant_id))
        & (summary_df["k"].astype(int) == int(k))
        & np.isclose(summary_df["friction"], float(friction))
    ]
    if matched.empty:
        raise KeyError(f"Missing summary row for variant={variant_id}, k={k}, friction={friction}.")
    return matched.iloc[0]


def _any_divergence(summary_df: pd.DataFrame, variant_id: str, k: int) -> bool:
    for friction in (FRICTION_MID, FRICTION_ONE):
        try:
            if bool(_row_at(summary_df, variant_id, k, friction)["modal_winner_divergence"]):
                return True
        except KeyError:
            continue
    return False


def _base_gate_rows(summary_df: pd.DataFrame, task: TaskMaterialization) -> tuple[pd.Series, pd.Series]:
    k = int(task.k_ref if task.k_ref in task.k_values else task.k_values[0])
    return _row_at(summary_df, task.variant_id, k, FRICTION_ZERO), _row_at(summary_df, task.variant_id, k, FRICTION_ONE)


def c1_gate_d(summary_df: pd.DataFrame, seed_df: pd.DataFrame, task: TaskMaterialization, fitted: dict[str, FittedFamily]) -> bool:
    if fitted["reactive_short"].constant_predictor_fallback:
        return False
    k = int(task.k_ref if task.k_ref in task.k_values else task.k_values[0])
    base = seed_df[(seed_df["variant_id"] == task.variant_id) & (seed_df["k"] == k)]
    switch = base.groupby("family_id")["switch_rate"].mean().to_dict()
    zero = _row_at(summary_df, task.variant_id, k, FRICTION_ZERO)
    candidates = [str(zero["forecast_winner"]), "reactive_short"]
    candidates = [family for family in candidates if family in switch and not fitted[family].constant_predictor_fallback]
    if not candidates:
        return False
    reactive_family = max(candidates, key=lambda family: float(switch[family]))
    return bool(
        float(switch[reactive_family]) > float(switch.get("calibrated_baseline", -np.inf))
        and float(switch[reactive_family]) > float(switch.get("lagged_smoother", -np.inf))
    )


def c2_gate_d(summary_df: pd.DataFrame, scenario_summary_df: pd.DataFrame, task: TaskMaterialization) -> bool:
    candidates = scenario_summary_df[scenario_summary_df["variant_id"] != task.variant_id]
    candidate_groups = [(str(row.variant_id), int(row.k)) for row in candidates[["variant_id", "k"]].drop_duplicates().itertuples(index=False)]
    candidate_groups.append((task.variant_id, int(task.k_ref)))
    for variant_id, k in candidate_groups:
        try:
            one = _row_at(scenario_summary_df, variant_id, k, FRICTION_ONE)
        except KeyError:
            continue
        score = 0
        score += int(float(one["deployed_suboptimal_share"]) > 0.50)
        score += int(float(one["mean_gap"]) > 0.0)
        score += int(float(one["median_gap"]) > 0.0)
        score += int(_any_divergence(scenario_summary_df, variant_id, k))
        if score >= 3:
            return True
    return False


def evaluate_gates(
    scenario_id: str,
    summary_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    task: TaskMaterialization,
    fitted: dict[str, FittedFamily],
    *,
    scenario_summary_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    zero, one = _base_gate_rows(summary_df, task)
    k = int(task.k_ref if task.k_ref in task.k_values else task.k_values[0])
    divergence = _any_divergence(summary_df, task.variant_id, k)

    if scenario_id == "traffic_topk_alert_q2_v1":
        gate_a = bool(
            (
                float(zero["agreement"]) >= 0.40
                or (abs(float(zero["mean_gap"])) <= 0.005 and abs(float(zero["median_gap"])) <= 0.005)
            )
            and float(zero["tie_involved_fraction"]) < 0.25
        )
        gate_b = bool(
            float(one["deployed_suboptimal_share"]) >= 0.50
            and float(one["mean_gap"]) > 0.0
            and float(one["median_gap"]) > 0.0
        )
        gate_c = divergence
        gate_d = c1_gate_d(summary_df, seed_df, task, fitted)
        weak = bool(gate_a and gate_b and gate_c)
    elif scenario_id == "traffic_relative_rank_q2_v1":
        gate_a = bool(float(zero["agreement"]) >= 0.50 or abs(float(zero["mean_gap"])) <= 0.003)
        gate_b = bool(
            float(one["deployed_suboptimal_share"]) >= 0.40
            and float(one["mean_gap"]) > 0.0
            and float(one["median_gap"]) > 0.0
        )
        gate_c = divergence
        gate_d = c2_gate_d(summary_df, scenario_summary_df if scenario_summary_df is not None else summary_df, task)
        one_divergence = bool(_row_at(summary_df, task.variant_id, k, FRICTION_ONE)["modal_winner_divergence"])
        weak = bool(gate_a and float(one["deployed_suboptimal_share"]) >= 0.40 and float(one["mean_gap"]) > 0.0 and one_divergence)
    elif scenario_id == "traffic_surge_onset_q2_v1":
        gate_a = bool(float(zero["agreement"]) >= 0.40 or abs(float(zero["mean_gap"])) <= 0.005)
        gate_b = bool(
            float(one["deployed_suboptimal_share"]) >= 0.50
            and float(one["mean_gap"]) > 0.0
            and float(one["median_gap"]) > 0.0
        )
        gate_c = divergence
        gate_d = bool(float(task.validation_event_rate) >= 0.05)
        weak = bool(gate_a and gate_d and float(one["deployed_suboptimal_share"]) >= 0.50 and float(one["mean_gap"]) > 0.0 and gate_c)
    else:
        raise ValueError(f"Unsupported scenario_id: {scenario_id}")

    strong = bool(gate_a and gate_b and gate_c and gate_d)
    verdict = "strong" if strong else ("weak" if weak else "fail")
    return {
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "gate_d": gate_d,
        "strong": strong,
        "weak": weak,
        "verdict": verdict,
        "variant_id": task.variant_id,
        "k": k,
    }


def build_pilot_table(task: TaskMaterialization, summary_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary_df.itertuples(index=False):
        metric_slice = seed_df[
            (seed_df["variant_id"] == row.variant_id)
            & (seed_df["k"] == row.k)
            & np.isclose(seed_df["friction"], float(row.friction))
        ]
        base = {
            "candidate_id": task.scenario_id,
            "variant_id": str(row.variant_id),
            "q": task.params.get("q", ""),
            "m": task.params.get("m", ""),
            "q_delta": task.params.get("q_delta", ""),
            "k": int(row.k),
            "validation_event_rate": float(task.validation_event_rate),
            "friction": float(row.friction),
            "forecast_winner": str(row.forecast_winner),
            "deployed_winner": str(row.deployed_winner),
            "agreement": float(row.agreement),
            "mean_gap": float(row.mean_gap),
            "median_gap": float(row.median_gap),
            "deployed_suboptimal_share": float(row.deployed_suboptimal_share),
            "modal_winner_divergence": bool(row.modal_winner_divergence),
            "mean_set_switch_rate": float(metric_slice["switch_rate"].mean()),
            "tie_involved_fraction": float(row.tie_involved_fraction),
        }
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["variant_id", "k", "friction"]).reset_index(drop=True)


def write_bundle(output_dir: Path, prefix: str, payload: dict[str, Any], ledger: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload["selection_summary_by_friction"].to_csv(output_dir / f"{prefix}_selection_summary_by_friction.csv", index=False)
    payload["seed_level_metrics"].to_csv(output_dir / f"{prefix}_seed_level_metrics.csv", index=False)
    if prefix == "pilot":
        payload["zero_row_diagnostics"].to_csv(output_dir / "pilot_zero_row_diagnostics.csv", index=False)
        payload["pilot_table"].to_csv(output_dir / "pilot_table.csv", index=False)
        write_json(output_dir / "pilot_report.json", payload["report"])
        write_json(output_dir / "pilot_candidate_ledger.json", ledger)
    else:
        write_json(output_dir / "full_report.json", payload["report"])
        write_json(output_dir / "full_candidate_ledger.json", ledger)
        payload["selection_summary_by_friction"].to_csv(output_dir / "full_robustness_canonical.csv", index=False)
    save_results(payload["q2_raw"], output_dir / f"{prefix}_q2_diff_forecasts_same_interface.csv")


def merge_task_payloads(task_payloads: list[dict[str, Any]], scenario_id: str, prefix: str) -> dict[str, Any]:
    seed_df = pd.concat([p["seed_level_metrics"] for p in task_payloads], ignore_index=True)
    summary_df = pd.concat([p["selection_summary_by_friction"] for p in task_payloads], ignore_index=True)
    zero_df = pd.concat([p["zero_row_diagnostics"] for p in task_payloads], ignore_index=True)
    table_df = pd.concat([p["pilot_table"] for p in task_payloads], ignore_index=True)
    q2_df = prepare_results_frame(pd.concat([p["q2_raw"] for p in task_payloads], ignore_index=True))
    gates = [p["gates"] for p in task_payloads]
    verdict_order = {"strong": 2, "weak": 1, "fail": 0}
    best = max(gates, key=lambda item: verdict_order[str(item["verdict"])])
    report = {
        "scenario_id": scenario_id,
        "prefix": prefix,
        "created_utc": utc_now_iso(),
        "verdict": best["verdict"],
        "best_variant_id": best["variant_id"],
        "best_k": best["k"],
        "gates": gates,
    }
    return {
        "seed_level_metrics": seed_df,
        "selection_summary_by_friction": summary_df,
        "zero_row_diagnostics": zero_df,
        "pilot_table": table_df,
        "q2_raw": q2_df,
        "report": report,
        "gates": gates,
    }


def run_scenario(
    *,
    scenario_id: str,
    panel: TrafficPanel,
    split: SplitSpec,
    output_root: Path,
    frictions: tuple[float, ...],
    replicates: int,
    seed: int,
    prefix: str,
    max_train_rows: int,
) -> dict[str, Any]:
    tasks = materialize_tasks(panel.values, split, scenario_id)
    fitted_by_variant: dict[str, dict[str, FittedFamily]] = {}
    payloads: list[dict[str, Any]] = []
    for task in tasks:
        print(f"[traffic-redesign] {prefix} {scenario_id} task={task.variant_id} fit_start", flush=True)
        fitted = {}
        for family_id in FAMILY_IDS:
            print(f"[traffic-redesign] {prefix} {scenario_id} task={task.variant_id} fitting {family_id}", flush=True)
            fitted[family_id] = fit_family(
                panel.values,
                task.labels,
                split,
                family_id,
                max_train_rows=int(max_train_rows),
                seed=int(seed),
            )
        print(f"[traffic-redesign] {prefix} {scenario_id} task={task.variant_id} eval_start", flush=True)
        fitted_by_variant[task.variant_id] = fitted
        payloads.append(
            evaluate_task(
                panel=panel.values,
                task=task,
                fitted=fitted,
                frictions=frictions,
                replicates=replicates,
                seed=seed,
                prefix=prefix,
            )
        )
        print(f"[traffic-redesign] {prefix} {scenario_id} task={task.variant_id} eval_done", flush=True)

    merged = merge_task_payloads(payloads, scenario_id, prefix)
    # Re-evaluate C2 gate D with the full scenario summary available.
    if scenario_id == "traffic_relative_rank_q2_v1":
        updated_gates = []
        for task in tasks:
            task_summary = merged["selection_summary_by_friction"][merged["selection_summary_by_friction"]["variant_id"] == task.variant_id].copy()
            task_seed = merged["seed_level_metrics"][merged["seed_level_metrics"]["variant_id"] == task.variant_id].copy()
            updated_gates.append(
                evaluate_gates(
                    scenario_id,
                    task_summary,
                    task_seed,
                    task,
                    fitted_by_variant[task.variant_id],
                    scenario_summary_df=merged["selection_summary_by_friction"],
                )
            )
        merged["gates"] = updated_gates
        verdict_order = {"strong": 2, "weak": 1, "fail": 0}
        best = max(updated_gates, key=lambda item: verdict_order[str(item["verdict"])])
        merged["report"]["gates"] = updated_gates
        merged["report"]["verdict"] = best["verdict"]
        merged["report"]["best_variant_id"] = best["variant_id"]
        merged["report"]["best_k"] = best["k"]
        for gate in updated_gates:
            mask = merged["pilot_table"]["variant_id"] == str(gate["variant_id"])
            merged["pilot_table"].loc[mask, "verdict"] = str(gate["verdict"])

    ledger = {
        "scenario_id": scenario_id,
        "prefix": prefix,
        "created_utc": utc_now_iso(),
        "source_lock": panel.source_lock,
        "split": split.__dict__,
        "families": {
            variant: {
                family_id: {
                    "constant_predictor_fallback": fitted.constant_predictor_fallback,
                    "feature_spec": fitted.spec,
                }
                for family_id, fitted in families.items()
            }
            for variant, families in fitted_by_variant.items()
        },
        "tasks": [
            {
                "variant_id": task.variant_id,
                "params": task.params,
                "provenance": task.provenance,
                "validation_event_rate": task.validation_event_rate,
                "k_ref": task.k_ref,
                "k_values": list(task.k_values),
            }
            for task in tasks
        ],
        "replicates": int(replicates),
        "replicate_seed": int(seed),
        "max_train_rows": int(max_train_rows),
        "frictions": list(frictions),
        "gates": merged["gates"],
    }
    write_bundle(output_root / scenario_id, prefix, merged, ledger)
    return merged["report"]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_root / "_cache"
    source_zip = Path(args.source_zip).resolve() if args.source_zip else None
    frictions = parse_frictions(args.frictions)
    panel = load_traffic_panel(
        cache_dir=cache_dir,
        source_zip=source_zip,
        skip_download=bool(args.skip_download),
        allow_noncanonical_panel=bool(args.allow_noncanonical_panel),
    )
    write_json(cache_dir / "traffic_hourly_source_lock.json", panel.source_lock)
    split = build_split(panel.values.shape[1])

    scenarios = SCENARIO_IDS if args.scenario_id == "all" else (str(args.scenario_id),)
    reports = []
    strong_scenarios: list[str] = []
    for scenario_id in scenarios:
        report = run_scenario(
            scenario_id=scenario_id,
            panel=panel,
            split=split,
            output_root=output_root,
            frictions=frictions,
            replicates=int(args.replicates),
            seed=int(args.seed),
            prefix="pilot",
            max_train_rows=int(args.max_train_rows),
        )
        reports.append(report)
        if str(report["verdict"]) == "strong":
            strong_scenarios.append(scenario_id)

    if args.run_full_if_strong:
        for scenario_id in strong_scenarios:
            reports.append(
                run_scenario(
                    scenario_id=scenario_id,
                    panel=panel,
                    split=split,
                    output_root=output_root,
                    frictions=frictions,
                    replicates=FULL_REPLICATES,
                    seed=int(args.seed),
                    prefix="full",
                    max_train_rows=int(args.max_train_rows),
                )
            )

    write_json(output_root / "traffic_redesign_run_report.json", {"created_utc": utc_now_iso(), "reports": reports})
    for report in reports:
        print(f"[traffic-redesign] {report['prefix']} {report['scenario_id']} verdict={report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
