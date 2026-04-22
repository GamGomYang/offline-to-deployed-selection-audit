#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
except ModuleNotFoundError as exc:  # pragma: no cover - explicit operator guidance
    raise SystemExit(
        "scikit-learn is required for inventory Q2 stronger baselines. "
        "Install requirements-forecast-eval.txt before running this script."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_same_interface_rank_summary import build_domain_rank_summary, write_summary_outputs  # noqa: E402
from common import build_result_row, mae_score, prepare_results_frame, save_results  # noqa: E402
import run_inventory as inventory_v2  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "forecast_eval" / "inventory_q2_stronger_baselines"
DEFAULT_CALIBRATION_LOG = ROOT / "outputs" / "forecast_eval" / "inventory" / "inventory_v2_calibration_log.csv"
DEFAULT_PROTOCOL_NOTE = DEFAULT_OUTPUT_DIR / "run_note.md"

TUNE_TRAIN_END = 84
VALIDATION_START = 84
VALIDATION_END = inventory_v2.TRAIN_END
FINAL_EVAL_START = inventory_v2.TRAIN_END
STAGE1_SEEDS = (0, 1)
STAGE2_SEEDS = (0, 1, 2)
FINAL_SEEDS = tuple(range(10))
DEFAULT_STAGE2_TOP_K = 3
FINAL_SCENARIO_ID = "inventory_control_v2_q2_stronger_baselines"
PATIENCE = 15
MAX_EPOCHS = inventory_v2.MLP_EPOCHS

FIXED_HEURISTIC_FAMILIES = ("naive_last", "moving_average_7")
TUNABLE_FAMILIES = (
    "linear_ar_ridge",
    "mlp_small",
    "gru_small",
    "reg_linear_lag_search",
    "gbrt_lagged",
    "mlp_large",
    "gru_variant",
)
FINAL_FORECASTER_IDS = FIXED_HEURISTIC_FAMILIES + TUNABLE_FAMILIES

DISPLAY_NAMES = {
    "naive_last": "Naive persistence",
    "moving_average_7": "Moving average (7)",
    "linear_ar_ridge": "Linear AR",
    "mlp_small": "Small MLP",
    "gru_small": "Small GRU",
    "reg_linear_lag_search": "Regularized linear + lag search",
    "gbrt_lagged": "Gradient-boosted trees",
    "mlp_large": "Large MLP",
    "gru_variant": "GRU variant",
}
INPUT_FORMS = {
    "naive_last": "fixed heuristic",
    "moving_average_7": "fixed heuristic",
    "linear_ar_ridge": "legacy tabular",
    "mlp_small": "legacy tabular",
    "gru_small": "sequence (lookback 28)",
    "reg_linear_lag_search": "lagged tabular",
    "gbrt_lagged": "lagged tabular",
    "mlp_large": "legacy tabular",
    "gru_variant": "sequence",
}


@dataclass(frozen=True)
class CandidateConfig:
    family: str
    config_id: str
    order_index: int
    params_json: str
    params: dict[str, Any]


class TunableMLP(nn.Module):
    def __init__(self, input_dim: int, *, hidden_width: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_width)))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_width)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TunableGRU(nn.Module):
    def __init__(self, *, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(int(hidden_size), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _output, hidden = self.gru(x)
        return self.head(hidden[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inventory Q2 stronger-baseline robustness evaluation.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calibration-log", default=str(DEFAULT_CALIBRATION_LOG))
    parser.add_argument("--horizon", type=int, default=inventory_v2.DEFAULT_HORIZON)
    parser.add_argument("--train-end", type=int, default=inventory_v2.TRAIN_END)
    parser.add_argument("--protocol-note", default=str(DEFAULT_PROTOCOL_NOTE))
    parser.add_argument("--search-profile", choices=("v1", "reactive_v2"), default="v1")
    parser.add_argument("--stage2-top-k", type=int, default=DEFAULT_STAGE2_TOP_K)
    parser.add_argument("--neural-loss", choices=("mse", "mae", "huber"), default="mse")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return parser.parse_args()


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _config_id(family: str, params: dict[str, Any]) -> str:
    param_bits = [f"{key}={params[key]}" for key in sorted(params)]
    return f"{family}__{'__'.join(param_bits)}"


def _candidate(family: str, order_index: int, **params: Any) -> CandidateConfig:
    params_json = _json_dumps(params)
    return CandidateConfig(
        family=str(family),
        config_id=_config_id(str(family), params),
        order_index=int(order_index),
        params_json=params_json,
        params=dict(params),
    )


def _candidate_lists(profile: str) -> dict[str, list[CandidateConfig]]:
    candidates: dict[str, list[CandidateConfig]] = {}

    order = 0
    candidates["linear_ar_ridge"] = []
    for alpha in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        candidates["linear_ar_ridge"].append(_candidate("linear_ar_ridge", order, alpha=float(alpha)))
        order += 1

    order = 0
    candidates["mlp_small"] = []
    mlp_small_lrs = (3e-4, 1e-3) if profile == "v1" else (1e-4, 3e-4, 1e-3)
    mlp_small_wds = (0.0, 1e-4) if profile == "v1" else (0.0, 1e-5, 1e-4)
    for lr in mlp_small_lrs:
        for weight_decay in mlp_small_wds:
            for batch_size in (64, 32):
                candidates["mlp_small"].append(
                    _candidate(
                        "mlp_small",
                        order,
                        lr=float(lr),
                        weight_decay=float(weight_decay),
                        batch_size=int(batch_size),
                    )
                )
                order += 1

    order = 0
    candidates["gru_small"] = []
    gru_small_lrs = (3e-4, 1e-3) if profile == "v1" else (1e-4, 3e-4, 1e-3)
    for lr in gru_small_lrs:
        for batch_size in (64, 32):
            candidates["gru_small"].append(
                _candidate("gru_small", order, lr=float(lr), batch_size=int(batch_size))
            )
            order += 1

    order = 0
    candidates["reg_linear_lag_search"] = []
    reg_lags = (14, 28, 56) if profile == "v1" else (7, 14, 28, 56)
    reg_penalties = ("ridge", "elasticnet") if profile == "v1" else ("ridge", "lasso", "elasticnet")
    for lag in reg_lags:
        for penalty in reg_penalties:
            for alpha in (1e-1, 1e-2, 1e-3):
                params: dict[str, Any] = {
                    "lag": int(lag),
                    "penalty": str(penalty),
                    "alpha": float(alpha),
                }
                if penalty == "elasticnet":
                    params["l1_ratio"] = 0.5
                candidates["reg_linear_lag_search"].append(_candidate("reg_linear_lag_search", order, **params))
                order += 1

    order = 0
    candidates["gbrt_lagged"] = []
    gbrt_lags = (28, 56) if profile == "v1" else (14, 28, 56)
    gbrt_estimators = (100, 300) if profile == "v1" else (100, 300, 500)
    gbrt_depths = (2, 3) if profile == "v1" else (2, 3, 5)
    for lag in gbrt_lags:
        for n_estimators in gbrt_estimators:
            for max_depth in gbrt_depths:
                for learning_rate in (0.05, 0.1):
                    for subsample in (1.0, 0.7):
                        for min_samples_leaf in (30, 10):
                            candidates["gbrt_lagged"].append(
                                _candidate(
                                    "gbrt_lagged",
                                    order,
                                    lag=int(lag),
                                    n_estimators=int(n_estimators),
                                    max_depth=int(max_depth),
                                    learning_rate=float(learning_rate),
                                    subsample=float(subsample),
                                    min_samples_leaf=int(min_samples_leaf),
                                )
                            )
                            order += 1

    order = 0
    candidates["mlp_large"] = []
    mlp_large_widths = (64, 128) if profile == "v1" else (64, 128, 256)
    mlp_large_depths = (2, 3) if profile == "v1" else (2, 3, 4)
    mlp_large_lrs = (3e-4, 1e-3) if profile == "v1" else (1e-4, 3e-4, 1e-3)
    for hidden_width in mlp_large_widths:
        for depth in mlp_large_depths:
            for dropout in (0.0, 0.1):
                for weight_decay in (1e-5, 1e-4):
                    for lr in mlp_large_lrs:
                        candidates["mlp_large"].append(
                            _candidate(
                                "mlp_large",
                                order,
                                hidden_width=int(hidden_width),
                                depth=int(depth),
                                dropout=float(dropout),
                                weight_decay=float(weight_decay),
                                lr=float(lr),
                                batch_size=64,
                            )
                        )
                        order += 1

    order = 0
    candidates["gru_variant"] = []
    gru_lengths = (28, 56) if profile == "v1" else (14, 28, 56)
    gru_hidden_sizes = (32, 64) if profile == "v1" else (32, 64, 128)
    gru_lrs = (3e-4, 1e-3) if profile == "v1" else (1e-4, 3e-4, 1e-3)
    for sequence_length in gru_lengths:
        for hidden_size in gru_hidden_sizes:
            for num_layers in (1, 2):
                dropout_values = (0.0,) if num_layers == 1 else (0.0, 0.1)
                for dropout in dropout_values:
                    for lr in gru_lrs:
                        candidates["gru_variant"].append(
                            _candidate(
                                "gru_variant",
                                order,
                                sequence_length=int(sequence_length),
                                hidden_size=int(hidden_size),
                                num_layers=int(num_layers),
                                dropout=float(dropout),
                                lr=float(lr),
                                batch_size=64,
                            )
                        )
                        order += 1

    return candidates


def _protocol_rows(stage1_candidates: dict[str, list[CandidateConfig]], *, stage2_top_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in FINAL_FORECASTER_IDS:
        tunable = family in TUNABLE_FAMILIES
        rows.append(
            {
                "family": family,
                "display_name": DISPLAY_NAMES[family],
                "tunable": bool(tunable),
                "input_form": INPUT_FORMS[family],
                "stage1_candidate_count": int(len(stage1_candidates.get(family, []))),
                "stage2_top_k": int(stage2_top_k if tunable else 0),
                "stage1_seeds": "|".join(str(seed) for seed in STAGE1_SEEDS) if tunable else "",
                "stage2_seeds": "|".join(str(seed) for seed in STAGE2_SEEDS) if tunable else "",
                "final_eval_seeds": "|".join(str(seed) for seed in FINAL_SEEDS),
                "validation_metric_name": "negative_mae" if tunable else "fixed_ex_ante",
                "selection_rule": (
                    "mean_validation_metric_then_median_then_preregistered_simpler_config_then_lexicographic"
                    if tunable
                    else "fixed_ex_ante"
                ),
            }
        )
    return rows


def _load_selected_config(calibration_log_path: Path) -> tuple[float, float, float]:
    calibration_df = pd.read_csv(calibration_log_path)
    selected = calibration_df[calibration_df["selected_flag"] == True]  # noqa: E712
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one selected config in {calibration_log_path}, found {len(selected)}.")
    row = selected.iloc[0]
    return float(row["burst_amp"]), float(row["safety_stock"]), float(row["stockout_w"])


def _legacy_supervised_arrays(
    demand: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(max(7, int(start_idx)), int(end_idx), dtype=np.int64)
    x = np.vstack([inventory_v2._feature_row(demand, int(idx)) for idx in indices]).astype(np.float64)
    y = demand[indices].astype(np.float64)
    return x, y


def _lagged_feature_row(demand: np.ndarray, idx: int, lag: int) -> np.ndarray:
    values = [float(demand[idx - step]) for step in range(1, int(lag) + 1)]
    values.extend(
        [
            float(demand[idx - 7 : idx].mean()),
            float(np.sin(2.0 * np.pi * idx / 7.0)),
            float(np.cos(2.0 * np.pi * idx / 7.0)),
        ]
    )
    return np.asarray(values, dtype=np.float64)


def _lagged_supervised_arrays(
    demand: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    begin = max(int(lag), int(start_idx))
    indices = np.arange(begin, int(end_idx), dtype=np.int64)
    x = np.vstack([_lagged_feature_row(demand, int(idx), int(lag)) for idx in indices]).astype(np.float64)
    y = demand[indices].astype(np.float64)
    return x, y


def _sequence_arrays(
    demand: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    begin = max(int(lookback), int(start_idx))
    indices = np.arange(begin, int(end_idx), dtype=np.int64)
    x = np.stack([demand[idx - lookback : idx] for idx in indices], axis=0).astype(np.float64)
    y = demand[indices].astype(np.float64)
    return x, y


def _heuristic_predictions(demand: np.ndarray, *, family: str, start_idx: int, end_idx: int) -> np.ndarray:
    indices = np.arange(int(start_idx), int(end_idx), dtype=np.int64)
    if family == "naive_last":
        return np.clip(demand[indices - 1], 0.0, None)
    if family == "moving_average_7":
        return np.asarray([demand[idx - 7 : idx].mean() for idx in indices], dtype=np.float64)
    raise ValueError(f"Unknown heuristic family: {family}")


def _normalize_features(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_train_scaled = (x_train - x_mean) / x_std
    x_other_scaled = (x_other - x_mean) / x_std
    return x_train_scaled, x_other_scaled, x_mean, x_std


def _neural_loss(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unsupported neural loss: {loss_name}")


def _fit_predict_linear_ar_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    model = inventory_v2._fit_linear_ar_ridge(x_train, y_train, ridge_penalty=float(alpha))
    return inventory_v2._predict_linear_ar(model, x_pred)


def _fit_predict_reg_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    penalty: str,
    alpha: float,
    l1_ratio: float | None,
) -> np.ndarray:
    x_train_scaled, x_pred_scaled, _x_mean, _x_std = _normalize_features(x_train, x_pred)
    if penalty == "ridge":
        model = Ridge(alpha=float(alpha), fit_intercept=True)
    elif penalty == "lasso":
        model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=10_000)
    elif penalty == "elasticnet":
        model = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), fit_intercept=True, max_iter=10_000)
    else:
        raise ValueError(f"Unsupported regularized linear penalty: {penalty}")
    model.fit(x_train_scaled, y_train)
    return np.clip(np.asarray(model.predict(x_pred_scaled), dtype=np.float64), 0.0, None)


def _fit_predict_gbrt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    model = GradientBoostingRegressor(
        random_state=61_000 + int(seed),
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
    )
    model.fit(x_train, y_train)
    return np.clip(np.asarray(model.predict(x_pred), dtype=np.float64), 0.0, None)


def _torch_seed(seed: int, config_id: str, offset: int) -> int:
    config_hash = int(hashlib.sha256(str(config_id).encode("utf-8")).hexdigest()[:8], 16)
    return int(offset + seed * 10_000 + (config_hash % 10_000))


def _tabular_mlp_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    seed: int,
    config_id: str,
    hidden_width: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    loss_name: str,
    max_epochs: int,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> np.ndarray:
    torch.manual_seed(_torch_seed(seed, config_id, 71_000))
    x_train_scaled, x_pred_scaled, x_mean, x_std = _normalize_features(x_train, x_pred)
    if x_val is not None:
        x_val_scaled = (x_val - x_mean) / x_std
    else:
        x_val_scaled = None

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y_std = 1.0 if y_std < 1e-8 else y_std

    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(((y_train - y_mean) / y_std)[:, None], dtype=torch.float32)
    x_pred_t = torch.tensor(x_pred_scaled, dtype=torch.float32)
    x_val_t = torch.tensor(x_val_scaled, dtype=torch.float32) if x_val_scaled is not None else None

    model = TunableMLP(
        x_train.shape[1],
        hidden_width=int(hidden_width),
        depth=int(depth),
        dropout=float(dropout),
    ).cpu()
    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = _neural_loss(loss_name)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_mae = float("inf")
    patience_counter = 0
    generator = torch.Generator().manual_seed(_torch_seed(seed, config_id, 72_000))

    for _epoch in range(int(max_epochs)):
        permutation = torch.randperm(x_train_t.shape[0], generator=generator)
        model.train()
        for start in range(0, len(permutation), int(batch_size)):
            batch_idx = permutation[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train_t[batch_idx])
            loss = loss_fn(pred, y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

        if x_val_t is None or y_val is None:
            continue

        model.eval()
        with torch.no_grad():
            val_scaled = model(x_val_t).squeeze(-1).cpu().numpy()
        val_pred = np.clip(val_scaled * y_std + y_mean, 0.0, None)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        if val_mae + 1e-12 < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_pred_t).squeeze(-1).cpu().numpy()
    return np.clip(y_pred_scaled * y_std + y_mean, 0.0, None)


def _gru_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    *,
    seed: int,
    config_id: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    loss_name: str,
    max_epochs: int,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> np.ndarray:
    torch.manual_seed(_torch_seed(seed, config_id, 81_000))
    train_mean = float(x_train.mean())
    train_std = float(x_train.std())
    train_std = 1.0 if train_std < 1e-8 else train_std

    x_train_t = torch.tensor(((x_train - train_mean) / train_std)[:, :, None], dtype=torch.float32)
    y_train_t = torch.tensor(((y_train - train_mean) / train_std)[:, None], dtype=torch.float32)
    x_pred_t = torch.tensor(((x_pred - train_mean) / train_std)[:, :, None], dtype=torch.float32)
    x_val_t = (
        torch.tensor(((x_val - train_mean) / train_std)[:, :, None], dtype=torch.float32) if x_val is not None else None
    )

    model = TunableGRU(
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=float(dropout),
    ).cpu()
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = _neural_loss(loss_name)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_mae = float("inf")
    patience_counter = 0
    generator = torch.Generator().manual_seed(_torch_seed(seed, config_id, 82_000))

    for _epoch in range(int(max_epochs)):
        permutation = torch.randperm(x_train_t.shape[0], generator=generator)
        model.train()
        for start in range(0, len(permutation), int(batch_size)):
            batch_idx = permutation[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_train_t[batch_idx])
            loss = loss_fn(pred, y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

        if x_val_t is None or y_val is None:
            continue

        model.eval()
        with torch.no_grad():
            val_scaled = model(x_val_t).squeeze(-1).cpu().numpy()
        val_pred = np.clip(val_scaled * train_std + train_mean, 0.0, None)
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        if val_mae + 1e-12 < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_pred_t).squeeze(-1).cpu().numpy()
    return np.clip(y_pred_scaled * train_std + train_mean, 0.0, None)


def _predict_for_family(
    *,
    family: str,
    params: dict[str, Any] | None,
    demand: np.ndarray,
    fit_start: int,
    fit_end: int,
    pred_start: int,
    pred_end: int,
    seed: int,
    config_id: str,
    neural_loss: str = "mse",
    max_epochs: int = MAX_EPOCHS,
    validation_start: int | None = None,
    validation_end: int | None = None,
) -> np.ndarray:
    if family in FIXED_HEURISTIC_FAMILIES:
        return _heuristic_predictions(demand, family=family, start_idx=pred_start, end_idx=pred_end)

    if family == "linear_ar_ridge":
        x_train, y_train = _legacy_supervised_arrays(demand, start_idx=fit_start, end_idx=fit_end)
        x_pred, _y_dummy = _legacy_supervised_arrays(demand, start_idx=pred_start, end_idx=pred_end)
        return _fit_predict_linear_ar_ridge(x_train, y_train, x_pred, alpha=float(params["alpha"]))

    if family == "reg_linear_lag_search":
        lag = int(params["lag"])
        x_train, y_train = _lagged_supervised_arrays(demand, start_idx=fit_start, end_idx=fit_end, lag=lag)
        x_pred, _y_dummy = _lagged_supervised_arrays(demand, start_idx=pred_start, end_idx=pred_end, lag=lag)
        return _fit_predict_reg_linear(
            x_train,
            y_train,
            x_pred,
            penalty=str(params["penalty"]),
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]) if "l1_ratio" in params else None,
        )

    if family == "gbrt_lagged":
        lag = int(params["lag"])
        x_train, y_train = _lagged_supervised_arrays(demand, start_idx=fit_start, end_idx=fit_end, lag=lag)
        x_pred, _y_dummy = _lagged_supervised_arrays(demand, start_idx=pred_start, end_idx=pred_end, lag=lag)
        return _fit_predict_gbrt(x_train, y_train, x_pred, params=params, seed=seed)

    if family in {"mlp_small", "mlp_large"}:
        x_train, y_train = _legacy_supervised_arrays(demand, start_idx=fit_start, end_idx=fit_end)
        x_pred, _y_dummy = _legacy_supervised_arrays(demand, start_idx=pred_start, end_idx=pred_end)
        x_val = y_val = None
        if validation_start is not None and validation_end is not None:
            x_val, y_val = _legacy_supervised_arrays(demand, start_idx=validation_start, end_idx=validation_end)
        if family == "mlp_small":
            return _tabular_mlp_prediction(
                x_train,
                y_train,
                x_pred,
                seed=seed,
                config_id=config_id,
                hidden_width=16,
                depth=1,
                dropout=0.0,
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
                batch_size=int(params["batch_size"]),
                loss_name=neural_loss,
                max_epochs=int(max_epochs),
                x_val=x_val,
                y_val=y_val,
            )
        return _tabular_mlp_prediction(
            x_train,
            y_train,
            x_pred,
            seed=seed,
            config_id=config_id,
            hidden_width=int(params["hidden_width"]),
            depth=int(params["depth"]),
            dropout=float(params["dropout"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            batch_size=int(params["batch_size"]),
            loss_name=neural_loss,
            max_epochs=int(max_epochs),
            x_val=x_val,
            y_val=y_val,
        )

    if family in {"gru_small", "gru_variant"}:
        lookback = inventory_v2.GRU_LOOKBACK if family == "gru_small" else int(params["sequence_length"])
        x_train, y_train = _sequence_arrays(demand, start_idx=fit_start, end_idx=fit_end, lookback=lookback)
        x_pred, _y_dummy = _sequence_arrays(demand, start_idx=pred_start, end_idx=pred_end, lookback=lookback)
        x_val = y_val = None
        if validation_start is not None and validation_end is not None:
            x_val, y_val = _sequence_arrays(demand, start_idx=validation_start, end_idx=validation_end, lookback=lookback)
        if family == "gru_small":
            return _gru_prediction(
                x_train,
                y_train,
                x_pred,
                seed=seed,
                config_id=config_id,
                hidden_size=inventory_v2.GRU_HIDDEN,
                num_layers=1,
                dropout=0.0,
                lr=float(params["lr"]),
                batch_size=int(params["batch_size"]),
                loss_name=neural_loss,
                max_epochs=int(max_epochs),
                x_val=x_val,
                y_val=y_val,
            )
        return _gru_prediction(
            x_train,
            y_train,
            x_pred,
            seed=seed,
            config_id=config_id,
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            lr=float(params["lr"]),
            batch_size=int(params["batch_size"]),
            loss_name=neural_loss,
            max_epochs=int(max_epochs),
            x_val=x_val,
            y_val=y_val,
        )

    raise ValueError(f"Unsupported family: {family}")


def _stage_rows_for_candidate(
    *,
    family: str,
    candidate: CandidateConfig,
    seeds: tuple[int, ...],
    demand_cache: dict[int, np.ndarray],
    fit_start: int,
    fit_end: int,
    pred_start: int,
    pred_end: int,
    stage_name: str,
    neural_loss: str,
    max_epochs: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        demand = demand_cache[int(seed)]
        predictions = _predict_for_family(
            family=family,
            params=candidate.params,
            demand=demand,
            fit_start=fit_start,
            fit_end=fit_end,
            pred_start=pred_start,
            pred_end=pred_end,
            seed=int(seed),
            config_id=candidate.config_id,
            neural_loss=neural_loss,
            max_epochs=int(max_epochs),
            validation_start=pred_start,
            validation_end=pred_end,
        )
        truth = demand[pred_start:pred_end]
        rows.append(
            {
                "stage": stage_name,
                "family": family,
                "display_name": DISPLAY_NAMES[family],
                "config_id": candidate.config_id,
                "order_index": int(candidate.order_index),
                "params_json": candidate.params_json,
                "seed": int(seed),
                "validation_negative_mae": float(mae_score(predictions, truth)),
                "train_start": int(fit_start),
                "train_end": int(fit_end) - 1,
                "validation_start": int(pred_start),
                "validation_end": int(pred_end) - 1,
            }
        )
    return rows


def _aggregate_stage(stage_df: pd.DataFrame) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "config_id",
                "order_index",
                "params_json",
                "mean_validation_negative_mae",
                "median_validation_negative_mae",
                "seed_count",
            ]
        )
    grouped = (
        stage_df.groupby(["family", "display_name", "config_id", "order_index", "params_json"], as_index=False)
        .agg(
            mean_validation_negative_mae=("validation_negative_mae", "mean"),
            median_validation_negative_mae=("validation_negative_mae", "median"),
            seed_count=("seed", "count"),
        )
        .sort_values(
            [
                "family",
                "mean_validation_negative_mae",
                "median_validation_negative_mae",
                "order_index",
                "config_id",
            ],
            ascending=[True, False, False, True, True],
        )
        .reset_index(drop=True)
    )
    return grouped


def _top_k_configs(stage_agg: pd.DataFrame, family: str, *, k: int) -> list[str]:
    subset = stage_agg[stage_agg["family"] == family].copy()
    if subset.empty:
        return []
    return subset.head(int(k))["config_id"].tolist()


def _representative_row(stage2_agg: pd.DataFrame, family: str) -> pd.Series:
    subset = stage2_agg[stage2_agg["family"] == family].copy()
    if subset.empty:
        raise RuntimeError(f"No stage-2 rows found for family {family}.")
    return subset.iloc[0]


def _validation_metrics_for_fixed_heuristic(
    *,
    family: str,
    demand_cache: dict[int, np.ndarray],
) -> tuple[float, float]:
    metrics: list[float] = []
    for seed in STAGE2_SEEDS:
        demand = demand_cache[int(seed)]
        predictions = _heuristic_predictions(
            demand,
            family=family,
            start_idx=VALIDATION_START,
            end_idx=VALIDATION_END,
        )
        metrics.append(float(mae_score(predictions, demand[VALIDATION_START:VALIDATION_END])))
    return float(np.mean(metrics)), float(np.median(metrics))


def _run_final_q2(
    *,
    demand_cache: dict[int, np.ndarray],
    representatives: pd.DataFrame,
    safety_stock: float,
    stockout_w: float,
    train_end: int,
    horizon: int,
    neural_loss: str,
    max_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q2_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    rep_map = {
        str(row.family): {
            "config_id": str(row.selected_config_id),
            "params": {} if pd.isna(row.selected_params_json) or row.selected_params_json == "" else json.loads(str(row.selected_params_json)),
        }
        for row in representatives.itertuples(index=False)
    }

    for seed in FINAL_SEEDS:
        demand = demand_cache[int(seed)]
        eval_truth = demand[FINAL_EVAL_START:horizon]
        initial_inventory = float(demand[:train_end].mean() + float(safety_stock))
        for family in FINAL_FORECASTER_IDS:
            rep = rep_map[family]
            predictions = _predict_for_family(
                family=family,
                params=rep["params"],
                demand=demand,
                fit_start=0,
                fit_end=train_end,
                pred_start=FINAL_EVAL_START,
                pred_end=horizon,
                seed=int(seed),
                config_id=str(rep["config_id"]),
                neural_loss=neural_loss,
                max_epochs=int(max_epochs),
            )
            forecast_metric = float(mae_score(predictions, eval_truth))
            for friction_level in inventory_v2.FRICTION_GRID:
                live_result = inventory_v2._run_live_inventory(
                    demand_eval=eval_truth,
                    forecasts_eval=np.asarray(predictions, dtype=np.float64),
                    safety_stock=float(safety_stock),
                    stockout_w=float(stockout_w),
                    friction_level=float(friction_level),
                    interface_id="responsive",
                    initial_inventory=initial_inventory,
                    initial_prev_order=float(inventory_v2.DEFAULT_INITIAL_PREV_ORDER),
                )
                q2_rows.append(
                    build_result_row(
                        question_id="Q2",
                        scenario_id=FINAL_SCENARIO_ID,
                        domain="inventory",
                        seed=int(seed),
                        forecaster_id=family,
                        interface_id="responsive",
                        friction_level=float(friction_level),
                        forecast_metric=forecast_metric,
                        target_metric=float(live_result["score"]),
                        executed_metric=float(live_result["score"]),
                        realized_cost=float(live_result["mean_change_cost"]),
                        realized_turnover_or_adjustment=float(live_result["mean_order_adjustment"]),
                    )
                )
                diagnostics_rows.append(
                    {
                        "question_id": "Q2",
                        "scenario_id": FINAL_SCENARIO_ID,
                        "domain": "inventory",
                        "seed": int(seed),
                        "forecaster_id": family,
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

    q2_df = prepare_results_frame(q2_rows)
    diagnostics_df = (
        pd.DataFrame(diagnostics_rows)
        .sort_values(["seed", "friction_level", "forecaster_id"])
        .reset_index(drop=True)
    )
    return q2_df, diagnostics_df


def _write_note(path: Path, *, search_profile: str, stage2_top_k: int, neural_loss: str, max_epochs: int) -> None:
    lines = [
        "# Inventory Q2 stronger baselines",
        "",
        "- Purpose: defense-oriented robustness layer for inventory Q2.",
        "- Locked evidence retained: same environment, interface, friction grid, held-out window, and final 10-seed evaluation.",
        f"- Search profile: {search_profile}.",
        f"- Standardized tuning procedure: stage 1 screening on seeds 0 and 1; stage 2 confirmation on the top {int(stage2_top_k)} configs per tunable family using seeds 0, 1, and 2.",
        "- Family representatives are selected by validation negative MAE only.",
        "- Held-out forecast-side ranking in the final Q2 raw schema uses the same held-out forecast metric as the locked inventory Q2 pipeline.",
        "- Fixed heuristics remain fixed ex ante and are excluded from search-budget claims.",
        f"- Neural training objective: {neural_loss}.",
        f"- Neural max epochs: {int(max_epochs)}.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(args.horizon)
    train_end = int(args.train_end)
    if horizon != inventory_v2.DEFAULT_HORIZON or train_end != inventory_v2.TRAIN_END:
        raise ValueError("This robustness runner is locked to the current inventory Q2 horizon and train_end.")

    burst_amp, safety_stock, stockout_w = _load_selected_config(Path(args.calibration_log).resolve())
    stage1_candidates = _candidate_lists(args.search_profile)
    all_seeds = sorted(set(STAGE1_SEEDS) | set(STAGE2_SEEDS) | set(FINAL_SEEDS))
    demand_cache = {
        int(seed): inventory_v2._generate_demand(int(seed), horizon=horizon, burst_amp=burst_amp) for seed in all_seeds
    }

    protocol_df = pd.DataFrame(_protocol_rows(stage1_candidates, stage2_top_k=int(args.stage2_top_k)))
    protocol_df["search_profile"] = str(args.search_profile)
    protocol_df["neural_loss"] = str(args.neural_loss)
    protocol_df["max_epochs"] = int(args.max_epochs)
    protocol_df.to_csv(output_dir / "protocol_summary.csv", index=False)
    _write_note(
        Path(args.protocol_note).resolve(),
        search_profile=str(args.search_profile),
        stage2_top_k=int(args.stage2_top_k),
        neural_loss=str(args.neural_loss),
        max_epochs=int(args.max_epochs),
    )

    stage1_rows: list[dict[str, object]] = []
    stage2_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    for family in TUNABLE_FAMILIES:
        candidate_list = stage1_candidates[family]
        for candidate in candidate_list:
            stage1_rows.extend(
                _stage_rows_for_candidate(
                    family=family,
                    candidate=candidate,
                    seeds=STAGE1_SEEDS,
                    demand_cache=demand_cache,
                    fit_start=0,
                    fit_end=TUNE_TRAIN_END,
                    pred_start=VALIDATION_START,
                    pred_end=VALIDATION_END,
                    stage_name="stage1",
                    neural_loss=str(args.neural_loss),
                    max_epochs=int(args.max_epochs),
                )
            )

        stage1_family_df = pd.DataFrame([row for row in stage1_rows if row["family"] == family])
        stage1_family_agg = _aggregate_stage(stage1_family_df)
        top_stage1_ids = set(_top_k_configs(stage1_family_agg, family, k=int(args.stage2_top_k)))
        top_stage1 = [candidate for candidate in candidate_list if candidate.config_id in top_stage1_ids]
        for candidate in top_stage1:
            stage2_rows.extend(
                _stage_rows_for_candidate(
                    family=family,
                    candidate=candidate,
                    seeds=STAGE2_SEEDS,
                    demand_cache=demand_cache,
                    fit_start=0,
                    fit_end=TUNE_TRAIN_END,
                    pred_start=VALIDATION_START,
                    pred_end=VALIDATION_END,
                    stage_name="stage2",
                    neural_loss=str(args.neural_loss),
                    max_epochs=int(args.max_epochs),
                )
            )

        stage2_family_df = pd.DataFrame([row for row in stage2_rows if row["family"] == family])
        stage2_family_agg = _aggregate_stage(stage2_family_df)
        representative = _representative_row(stage2_family_agg, family)
        representative_rows.append(
            {
                "family": family,
                "display_name": DISPLAY_NAMES[family],
                "tunable": True,
                "selected_config_id": str(representative["config_id"]),
                "selected_params_json": str(representative["params_json"]),
                "selection_metric_name": "validation_negative_mae",
                "validation_mean_metric": float(representative["mean_validation_negative_mae"]),
                "validation_median_metric": float(representative["median_validation_negative_mae"]),
                "stage1_candidate_count": int(len(candidate_list)),
                "stage2_candidate_count": int(len(top_stage1_ids)),
                "selection_rule": "mean_then_median_then_preregistered_simpler_config_then_lexicographic",
                "input_form": INPUT_FORMS[family],
            }
        )

    for family in FIXED_HEURISTIC_FAMILIES:
        val_mean, val_median = _validation_metrics_for_fixed_heuristic(family=family, demand_cache=demand_cache)
        representative_rows.append(
            {
                "family": family,
                "display_name": DISPLAY_NAMES[family],
                "tunable": False,
                "selected_config_id": "fixed_ex_ante",
                "selected_params_json": "",
                "selection_metric_name": "fixed_ex_ante",
                "validation_mean_metric": float(val_mean),
                "validation_median_metric": float(val_median),
                "stage1_candidate_count": 0,
                "stage2_candidate_count": 0,
                "selection_rule": "fixed_ex_ante",
                "input_form": INPUT_FORMS[family],
            }
        )

    stage1_df = pd.DataFrame(stage1_rows).sort_values(["family", "order_index", "seed"]).reset_index(drop=True)
    stage2_df = pd.DataFrame(stage2_rows).sort_values(["family", "order_index", "seed"]).reset_index(drop=True)
    representatives_df = (
        pd.DataFrame(representative_rows)
        .sort_values(["tunable", "family"], ascending=[False, True])
        .reset_index(drop=True)
    )

    stage1_df.to_csv(output_dir / "stage1_screening.csv", index=False)
    stage2_df.to_csv(output_dir / "stage2_validation.csv", index=False)
    representatives_df.to_csv(output_dir / "family_representatives.csv", index=False)

    q2_df, diagnostics_df = _run_final_q2(
        demand_cache=demand_cache,
        representatives=representatives_df,
        safety_stock=safety_stock,
        stockout_w=stockout_w,
        train_end=train_end,
        horizon=horizon,
        neural_loss=str(args.neural_loss),
        max_epochs=int(args.max_epochs),
    )
    save_results(q2_df, output_dir / "q2_diff_forecasts_same_interface.csv")
    diagnostics_df.to_csv(output_dir / "inventory_q2_stronger_baselines_diagnostics.csv", index=False)

    rank_outputs, _meta = build_domain_rank_summary(
        q2_df,
        domain="inventory",
        expected_interface_id="responsive",
    )
    write_summary_outputs(rank_outputs, output_dir)

    print(
        "[inventory-q2-stronger-baselines] "
        f"config=(burst_amp={burst_amp}, safety_stock={safety_stock}, stockout_w={stockout_w})"
    )
    print(f"[inventory-q2-stronger-baselines] wrote stage1 screening to {output_dir / 'stage1_screening.csv'}")
    print(f"[inventory-q2-stronger-baselines] wrote stage2 validation to {output_dir / 'stage2_validation.csv'}")
    print(f"[inventory-q2-stronger-baselines] wrote family representatives to {output_dir / 'family_representatives.csv'}")
    print(f"[inventory-q2-stronger-baselines] wrote Q2 raw rows={len(q2_df)} to {output_dir / 'q2_diff_forecasts_same_interface.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
