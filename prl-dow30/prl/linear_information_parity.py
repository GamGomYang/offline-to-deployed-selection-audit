from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import (
    DEFAULT_HISTORY_MIN,
    DEFAULT_LOOKBACK,
    DEFAULT_MEAN_VARIANCE_RISK_AVERSION,
    mean_variance_weights,
)
from .eval import trace_dict_to_frame
from .metrics import PortfolioMetrics, compute_metrics, turnover_l1


@dataclass
class LBIPConfig:
    window_size: int
    ridge_alpha: float = 10.0
    fit_passes: int = 2
    training_eta: float = 0.082
    covariance_lookback: int = DEFAULT_LOOKBACK
    covariance_history_min: int = DEFAULT_HISTORY_MIN
    mean_variance_risk_aversion: float = DEFAULT_MEAN_VARIANCE_RISK_AVERSION
    include_prev_weights: bool = True
    target_mode: str = "mean_variance"
    anchor_strength: float = 0.0
    equal_weight_shrink: float = 0.0
    log_clip: float = 1e-8
    eps: float = 1e-12


@dataclass
class LBIPFitResult:
    intercept: np.ndarray
    coef: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    asset_names: list[str]
    signal_names: list[str]
    obs_dim: int
    train_rows: int
    fit_passes: int
    ridge_alpha: float
    training_eta: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["intercept"] = self.intercept.tolist()
        payload["coef"] = self.coef.tolist()
        payload["feature_mean"] = self.feature_mean.tolist()
        payload["feature_std"] = self.feature_std.tolist()
        return payload


def _align_frames(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    signal_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    vol_clean = volatility.dropna(how="any")
    idx = returns.index.intersection(vol_clean.index)
    signal_aligned: pd.DataFrame | None = None
    if signal_features is not None:
        signal_clean = signal_features.dropna(how="any")
        idx = idx.intersection(signal_clean.index)
        signal_aligned = signal_clean.loc[idx]
    returns_aligned = returns.loc[idx]
    vol_aligned = vol_clean.loc[idx]
    return returns_aligned, vol_aligned, signal_aligned


def _stable_feature_scale(std: np.ndarray, eps: float) -> np.ndarray:
    out = np.asarray(std, dtype=np.float64).copy()
    out[~np.isfinite(out)] = 1.0
    out[np.abs(out) <= eps] = 1.0
    return out


def _normalize_simplex(weights: np.ndarray, eps: float) -> np.ndarray:
    out = np.asarray(weights, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, 0.0, None)
    total = float(out.sum())
    if not np.isfinite(total) or total <= eps:
        return np.full_like(out, 1.0 / out.size, dtype=np.float64)
    return (out / total).astype(np.float64)


def _build_state_vector(
    returns_arr: np.ndarray,
    vol_arr: np.ndarray,
    signal_arr: np.ndarray | None,
    *,
    step_idx: int,
    window_size: int,
    prev_weights: np.ndarray,
    include_prev_weights: bool,
) -> np.ndarray:
    start = step_idx - window_size
    end = step_idx
    returns_flat = returns_arr[start:end].reshape(-1)
    vol_vector = vol_arr[step_idx - 1]
    parts = [returns_flat, vol_vector]
    if include_prev_weights:
        parts.append(prev_weights)
    if signal_arr is not None:
        parts.append(signal_arr[step_idx - 1])
    return np.concatenate(parts, dtype=np.float64)


def _fit_ridge_closed_form(x: np.ndarray, y: np.ndarray, *, alpha: float, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    feature_mean = np.nanmean(x, axis=0)
    feature_std = _stable_feature_scale(np.nanstd(x, axis=0), eps)
    x_scaled = (np.nan_to_num(x, nan=0.0) - feature_mean) / feature_std

    xtx = x_scaled.T @ x_scaled
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    rhs = x_scaled.T @ y
    try:
        coef = np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx + reg) @ rhs

    intercept = np.nanmean(y, axis=0)
    return intercept.astype(np.float64), coef.astype(np.float64), feature_mean.astype(np.float64), feature_std.astype(np.float64)


def predict_expected_returns(model: LBIPFitResult, state_vector: np.ndarray) -> np.ndarray:
    x = np.asarray(state_vector, dtype=np.float64)
    x_scaled = (np.nan_to_num(x, nan=0.0) - model.feature_mean) / model.feature_std
    pred = model.intercept + x_scaled @ model.coef
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred.astype(np.float64)


def _anchored_mean_variance_weights(
    mean_return: np.ndarray,
    cov: np.ndarray,
    prev_weights: np.ndarray,
    *,
    risk_aversion: float,
    anchor_strength: float,
    equal_weight_shrink: float,
    eps: float,
) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    mu = np.asarray(mean_return, dtype=np.float64)
    prev = _normalize_simplex(prev_weights, eps)
    gamma = float(risk_aversion)
    gamma = gamma if np.isfinite(gamma) and gamma > 0.0 else DEFAULT_MEAN_VARIANCE_RISK_AVERSION
    tau = max(float(anchor_strength), 0.0)
    dim = cov.shape[0]
    cov_reg = cov.copy()
    if cov_reg.ndim != 2 or cov_reg.shape[0] != cov_reg.shape[1]:
        return prev
    trace = float(np.trace(cov_reg))
    ridge = 1e-6 * (trace / dim if np.isfinite(trace) and trace > 0.0 else 1.0)
    a = gamma * cov_reg + (tau + ridge) * np.eye(dim, dtype=np.float64)
    ones = np.ones(dim, dtype=np.float64)
    rhs = mu + tau * prev
    try:
        solve_rhs = np.linalg.solve(a, rhs)
        solve_one = np.linalg.solve(a, ones)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(a)
        solve_rhs = pinv @ rhs
        solve_one = pinv @ ones
    denom = float(ones @ solve_one)
    if not np.isfinite(denom) or abs(denom) <= eps:
        out = prev
    else:
        nu = float((ones @ solve_rhs - 1.0) / denom)
        out = solve_rhs - nu * solve_one
        out = _normalize_simplex(out, eps)
    beta = float(equal_weight_shrink)
    if np.isfinite(beta) and beta > 0.0:
        beta = min(beta, 1.0)
        eq = np.full(dim, 1.0 / dim, dtype=np.float64)
        out = _normalize_simplex((1.0 - beta) * out + beta * eq, eps)
    return out


def _target_weights_from_prediction(
    mu_hat: np.ndarray,
    arithmetic_returns_arr: np.ndarray,
    *,
    step_idx: int,
    lookback: int,
    history_min: int,
    risk_aversion: float,
    prev_weights: np.ndarray,
    target_mode: str,
    anchor_strength: float,
    equal_weight_shrink: float,
    eps: float,
) -> np.ndarray:
    start = max(0, int(step_idx) - int(lookback))
    history = arithmetic_returns_arr[start:step_idx]
    num_assets = arithmetic_returns_arr.shape[1]
    equal_weight = np.full(num_assets, 1.0 / num_assets, dtype=np.float64)
    if int(history.shape[0]) < int(history_min):
        return equal_weight
    cov = np.cov(history, rowvar=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    if cov.shape != (num_assets, num_assets):
        return equal_weight
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    mu_hat = np.nan_to_num(np.asarray(mu_hat, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mode = str(target_mode).strip().lower()
    if mode == "anchored_mean_variance":
        return _anchored_mean_variance_weights(
            mu_hat,
            cov,
            prev_weights,
            risk_aversion=float(risk_aversion),
            anchor_strength=float(anchor_strength),
            equal_weight_shrink=float(equal_weight_shrink),
            eps=eps,
        )
    return mean_variance_weights(mu_hat, cov, risk_aversion=float(risk_aversion), eps=eps)


def fit_lbip_model(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    signal_features: pd.DataFrame | None = None,
    config: LBIPConfig,
) -> LBIPFitResult:
    returns_aligned, vol_aligned, signal_aligned = _align_frames(returns, volatility, signal_features)
    if len(returns_aligned) <= int(config.window_size):
        raise ValueError("Not enough aligned rows to fit LBIP.")

    returns_arr = returns_aligned.to_numpy(dtype=np.float64, copy=False)
    vol_arr = vol_aligned.to_numpy(dtype=np.float64, copy=False)
    signal_arr = signal_aligned.to_numpy(dtype=np.float64, copy=False) if signal_aligned is not None else None
    arithmetic_returns_arr = np.expm1(returns_arr)

    num_assets = returns_arr.shape[1]
    equal_weight = np.full(num_assets, 1.0 / num_assets, dtype=np.float64)
    fit_result: LBIPFitResult | None = None

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for pass_idx in range(int(config.fit_passes)):
        x_rows = []
        y_rows = []
        prev_weights = equal_weight.copy()
        for step_idx in range(int(config.window_size), returns_arr.shape[0]):
            state_vector = _build_state_vector(
                returns_arr,
                vol_arr,
                signal_arr,
                step_idx=step_idx,
                window_size=int(config.window_size),
                prev_weights=prev_weights,
                include_prev_weights=bool(config.include_prev_weights),
            )
            x_rows.append(state_vector)
            y_rows.append(arithmetic_returns_arr[step_idx])

            if fit_result is None:
                continue

            mu_hat = predict_expected_returns(fit_result, state_vector)
            w_target = _target_weights_from_prediction(
                mu_hat,
                arithmetic_returns_arr,
                step_idx=step_idx,
                lookback=int(config.covariance_lookback),
                history_min=int(config.covariance_history_min),
                risk_aversion=float(config.mean_variance_risk_aversion),
                prev_weights=prev_weights,
                target_mode=str(config.target_mode),
                anchor_strength=float(config.anchor_strength),
                equal_weight_shrink=float(config.equal_weight_shrink),
                eps=float(config.eps),
            )
            prev_weights = _normalize_simplex(
                (1.0 - float(config.training_eta)) * prev_weights + float(config.training_eta) * w_target,
                float(config.eps),
            )

        x = np.vstack(x_rows)
        y = np.vstack(y_rows)
        intercept, coef, feature_mean, feature_std = _fit_ridge_closed_form(
            x,
            y,
            alpha=float(config.ridge_alpha),
            eps=float(config.eps),
        )
        fit_result = LBIPFitResult(
            intercept=intercept,
            coef=coef,
            feature_mean=feature_mean,
            feature_std=feature_std,
            asset_names=list(returns_aligned.columns),
            signal_names=list(pd.Index(signal_aligned.columns.get_level_values(0)).unique()) if signal_aligned is not None else [],
            obs_dim=int(x.shape[1]),
            train_rows=int(x.shape[0]),
            fit_passes=int(pass_idx) + 1,
            ridge_alpha=float(config.ridge_alpha),
            training_eta=float(config.training_eta),
        )

    if fit_result is None:
        raise RuntimeError("LBIP fit failed to produce a model.")
    return fit_result


def evaluate_lbip_eta(
    model: LBIPFitResult,
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    eta: float,
    transaction_cost: float,
    signal_features: pd.DataFrame | None = None,
    config: LBIPConfig,
    eval_id: str,
    run_id: str,
    seed: int = 0,
) -> tuple[PortfolioMetrics, pd.DataFrame]:
    returns_aligned, vol_aligned, signal_aligned = _align_frames(returns, volatility, signal_features)
    if len(returns_aligned) <= int(config.window_size):
        raise ValueError("Not enough aligned rows to evaluate LBIP.")

    returns_arr = returns_aligned.to_numpy(dtype=np.float64, copy=False)
    vol_arr = vol_aligned.to_numpy(dtype=np.float64, copy=False)
    signal_arr = signal_aligned.to_numpy(dtype=np.float64, copy=False) if signal_aligned is not None else None
    arithmetic_returns_arr = np.expm1(returns_arr)

    num_assets = returns_arr.shape[1]
    equal_weight = np.full(num_assets, 1.0 / num_assets, dtype=np.float64)
    prev_weights = equal_weight.copy()

    rewards: list[float] = []
    portfolio_returns: list[float] = []
    portfolio_returns_target: list[float] = []
    turnovers_exec: list[float] = []
    turnovers_target: list[float] = []
    dates: list[pd.Timestamp] = []
    costs: list[float] = []
    costs_target: list[float] = []
    net_returns_exp: list[float] = []
    net_returns_lin: list[float] = []
    net_returns_lin_target: list[float] = []
    log_returns_gross: list[float] = []
    log_returns_gross_target: list[float] = []
    log_returns_net: list[float] = []
    log_returns_net_target: list[float] = []
    eta_ts: list[float] = []
    lambda_ts: list[float] = []
    tracking_errors: list[float] = []
    collapse_flags: list[bool] = []
    collapse_reasons: list[str | None] = []

    for step_idx in range(int(config.window_size), returns_arr.shape[0]):
        state_vector = _build_state_vector(
            returns_arr,
            vol_arr,
            signal_arr,
            step_idx=step_idx,
            window_size=int(config.window_size),
            prev_weights=prev_weights,
            include_prev_weights=bool(config.include_prev_weights),
        )
        mu_hat = predict_expected_returns(model, state_vector)
        w_target = _target_weights_from_prediction(
            mu_hat,
            arithmetic_returns_arr,
            step_idx=step_idx,
            lookback=int(config.covariance_lookback),
            history_min=int(config.covariance_history_min),
            risk_aversion=float(config.mean_variance_risk_aversion),
            prev_weights=prev_weights,
            target_mode=str(config.target_mode),
            anchor_strength=float(config.anchor_strength),
            equal_weight_shrink=float(config.equal_weight_shrink),
            eps=float(config.eps),
        )
        w_exec = _normalize_simplex((1.0 - float(eta)) * prev_weights + float(eta) * w_target, float(config.eps))

        arithmetic_returns = arithmetic_returns_arr[step_idx]
        portfolio_return = float(np.dot(w_exec, arithmetic_returns))
        portfolio_return_target = float(np.dot(w_target, arithmetic_returns))
        turnover_exec = turnover_l1(prev_weights, w_exec)
        turnover_target = turnover_l1(prev_weights, w_target)
        cost_exec = float(transaction_cost) * turnover_exec
        cost_target = float(transaction_cost) * turnover_target
        net_return_lin_exec = portfolio_return - cost_exec
        net_return_lin_target_val = portfolio_return_target - cost_target
        tracking_error_l2 = float(np.linalg.norm(w_exec - w_target, ord=2))

        raw_log_argument = 1.0 + portfolio_return
        raw_log_argument_target = 1.0 + portfolio_return_target
        collapse_flag = False
        collapse_reason = None
        if not np.isfinite(raw_log_argument):
            collapse_flag = True
            collapse_reason = "log_argument_non_finite"
        if not np.isfinite(raw_log_argument_target):
            collapse_flag = True
            collapse_reason = collapse_reason or "target_log_argument_non_finite"
        log_argument = max(raw_log_argument if np.isfinite(raw_log_argument) else float(config.log_clip), float(config.log_clip))
        log_argument_target = max(
            raw_log_argument_target if np.isfinite(raw_log_argument_target) else float(config.log_clip),
            float(config.log_clip),
        )

        log_return_gross_val = math.log(log_argument)
        log_return_gross_target_val = math.log(log_argument_target)
        log_return_net_val = log_return_gross_val - cost_exec
        log_return_net_target_val = log_return_gross_target_val - cost_target
        reward = log_return_net_val

        rewards.append(reward)
        portfolio_returns.append(portfolio_return)
        portfolio_returns_target.append(portfolio_return_target)
        turnovers_exec.append(turnover_exec)
        turnovers_target.append(turnover_target)
        dates.append(pd.Timestamp(returns_aligned.index[step_idx]))
        costs.append(cost_exec)
        costs_target.append(cost_target)
        net_returns_exp.append(math.exp(log_return_net_val) - 1.0)
        net_returns_lin.append(net_return_lin_exec)
        net_returns_lin_target.append(net_return_lin_target_val)
        log_returns_gross.append(log_return_gross_val)
        log_returns_gross_target.append(log_return_gross_target_val)
        log_returns_net.append(log_return_net_val)
        log_returns_net_target.append(log_return_net_target_val)
        eta_ts.append(float(eta))
        lambda_ts.append(np.nan)
        tracking_errors.append(tracking_error_l2)
        collapse_flags.append(collapse_flag)
        collapse_reasons.append(collapse_reason)

        prev_weights = w_exec

    metrics = compute_metrics(
        rewards,
        portfolio_returns,
        turnovers_exec,
        turnovers_target=turnovers_target,
        net_returns_exp=net_returns_exp,
        net_returns_lin=net_returns_lin,
    )
    trace = {
        "dates": dates,
        "rewards": rewards,
        "portfolio_returns": portfolio_returns,
        "portfolio_returns_target": portfolio_returns_target,
        "turnovers": turnovers_exec,
        "turnovers_exec": turnovers_exec,
        "turnovers_target": turnovers_target,
        "turnover_target_changes": turnovers_target,
        "costs": costs,
        "costs_target": costs_target,
        "net_returns_exp": net_returns_exp,
        "net_returns_lin": net_returns_lin,
        "net_returns_lin_target": net_returns_lin_target,
        "log_returns_gross": log_returns_gross,
        "log_returns_gross_target": log_returns_gross_target,
        "log_returns_net": log_returns_net,
        "log_returns_net_target": log_returns_net_target,
        "eta_t": eta_ts,
        "lambda_t": lambda_ts,
        "tracking_error_l2": tracking_errors,
        "collapse_flag": collapse_flags,
        "collapse_reason": collapse_reasons,
    }
    trace_df = trace_dict_to_frame(trace, eval_id=eval_id, run_id=run_id, model_type="lbip", seed=int(seed))
    return metrics, trace_df


def fit_summary_dict(model: LBIPFitResult) -> dict[str, Any]:
    return {
        "asset_names": list(model.asset_names),
        "signal_names": list(model.signal_names),
        "obs_dim": int(model.obs_dim),
        "train_rows": int(model.train_rows),
        "fit_passes": int(model.fit_passes),
        "ridge_alpha": float(model.ridge_alpha),
        "training_eta": float(model.training_eta),
    }


def save_lbip_model(model: LBIPFitResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        intercept=model.intercept,
        coef=model.coef,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        asset_names=np.asarray(model.asset_names, dtype=object),
        signal_names=np.asarray(model.signal_names, dtype=object),
        obs_dim=np.asarray([model.obs_dim], dtype=np.int64),
        train_rows=np.asarray([model.train_rows], dtype=np.int64),
        fit_passes=np.asarray([model.fit_passes], dtype=np.int64),
        ridge_alpha=np.asarray([model.ridge_alpha], dtype=np.float64),
        training_eta=np.asarray([model.training_eta], dtype=np.float64),
    )


def save_lbip_summary(model: LBIPFitResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fit_summary_dict(model), indent=2))
