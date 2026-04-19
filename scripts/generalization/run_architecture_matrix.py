#!/usr/bin/env python3
"""
Support-only architecture-matrix runner for the forecasting workshop generalization package.

Assumptions documented here on purpose:

1. The runner writes raw final-period architecture comparisons for the current fixed U27 setup.
   Validation is used only where a support architecture needs an internal selection step
   (for example, choosing a positive tau or threshold).
2. "Forecast source fixed" is exact for the RL-stream architectures:
   `arch_rl_selected`, `arch_rule_eta_fixed`, and the optional
   `arch_threshold_rebalance` all replay the same frozen RL policy stream for a given seed.
3. `arch_linear_prox` follows the narrower wording already documented in
   `architecture_construction_note.md`: it freezes a family-internal linear forecast map fit once
   on the training split, then compares decision layers inside that family. It is included as a
   non-RL support architecture to reduce RL-only artifact concerns, not as proof of a single
   universal forecast source shared literally across every family.
4. The default seed grid is `[0]`. This stage targets execution-interface robustness rather than
   training-seed robustness, and the linear/prox support arm is deterministic under the frozen
   cached data path.
5. Under the current YAML specs, `arch_rule_eta_fixed` intentionally replays the same fixed-eta
   convex-combination rule over the same frozen RL target stream used by `arch_rl_selected`.
   Small numerical differences would indicate an implementation bug; close agreement is expected.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRL_ROOT = REPO_ROOT / "prl-dow30"
GENERALIZATION_SCRIPT_DIR = Path(__file__).resolve().parent

for candidate in (str(GENERALIZATION_SCRIPT_DIR), str(PRL_ROOT / "scripts"), str(PRL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from build_universes import load_universe_specs
from prl.data import slice_frame
from prl.envs import EnvConfig, stable_softmax
from prl.eval import load_model, trace_dict_to_frame
from prl.linear_information_parity import LBIPConfig, evaluate_lbip_eta, fit_lbip_model
from prl.metrics import turnover_l1
from prl.train import build_env_for_range, build_signal_features, create_scheduler, prepare_market_and_features
from step6_sanity import build_eval_context, run_eval_case


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "generalization" / "architecture_matrix.yaml"


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    forecast_source: str
    decision_rule: str
    params: dict[str, Any]
    compare_arm: str
    evaluation_role: str
    notes: list[str]
    path: Path


@dataclass(frozen=True)
class PreparedLBIPData:
    config_path: Path
    cfg: dict[str, Any]
    period: str
    train_returns: pd.DataFrame
    train_volatility: pd.DataFrame
    train_signals: pd.DataFrame | None
    eval_returns: pd.DataFrame
    eval_volatility: pd.DataFrame
    eval_signals: pd.DataFrame | None
    base_config: LBIPConfig
    signal_spec: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run architecture-matrix execution-interface support comparisons.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the architecture-matrix YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without executing them.")
    parser.add_argument(
        "--include-optional-threshold",
        action="store_true",
        help="Include appendix-only optional architectures such as arch_threshold_rebalance.",
    )
    return parser.parse_args()


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = (config_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _compute_sharpe(returns: pd.Series) -> float:
    arr = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    std = float(arr.std(ddof=0))
    if std <= 1e-8:
        return 0.0
    return float((arr.mean() / std) * np.sqrt(252.0))


def _compute_max_drawdown(equity: pd.Series) -> float:
    arr = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    run_max = np.maximum.accumulate(arr)
    drawdown = arr / run_max - 1.0
    return float(np.min(drawdown))


def _compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    arr = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    final = float(arr[-1])
    if final <= 0.0:
        return float("nan")
    years = float(arr.size) / float(periods_per_year)
    if years <= 0.0:
        return float("nan")
    return float(final ** (1.0 / years) - 1.0)


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _safe_sum(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def _safe_last_gap(exec_series: pd.Series, target_series: pd.Series) -> float | None:
    exec_values = pd.to_numeric(exec_series, errors="coerce").dropna()
    target_values = pd.to_numeric(target_series, errors="coerce").dropna()
    if exec_values.empty or target_values.empty:
        return None
    return float(abs(exec_values.iloc[-1] - target_values.iloc[-1]))


def _result_files(result_dir: Path, *, save_trace: bool) -> dict[str, Path]:
    files = {
        "json": result_dir / "result.json",
        "csv": result_dir / "result.csv",
        "meta": result_dir / "meta.json",
    }
    if save_trace:
        files["selected_trace"] = result_dir / "selected_trace.parquet"
        files["reference_trace"] = result_dir / "reference_trace.parquet"
    return files


def _result_complete(result_dir: Path, *, save_trace: bool) -> bool:
    required = list(_result_files(result_dir, save_trace=save_trace).values())
    return all(path.exists() for path in required)


def _write_result_bundle(
    result_dir: Path,
    row: dict[str, Any],
    *,
    selected_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    meta: dict[str, Any],
    save_trace: bool,
) -> None:
    _ensure_dir(result_dir)
    files = _result_files(result_dir, save_trace=save_trace)
    files["json"].write_text(json.dumps(row, indent=2))
    with files["csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    files["meta"].write_text(json.dumps(meta, indent=2))
    if save_trace:
        selected_df.to_parquet(files["selected_trace"], index=False)
        reference_df.to_parquet(files["reference_trace"], index=False)


def _derive_disagreement_type(
    *,
    delta_exec: float | None,
    delta_target: float | None,
    near_flat_threshold: float,
    suppression_ratio: float,
) -> str | None:
    if delta_exec is None or delta_target is None:
        return None
    if not np.isfinite(delta_exec) or not np.isfinite(delta_target):
        return None
    exec_mag = abs(float(delta_exec))
    target_mag = abs(float(delta_target))
    if exec_mag <= 1e-12 and target_mag <= 1e-12:
        return "aligned_flat"
    exec_sign = 0 if exec_mag <= 1e-12 else int(np.sign(delta_exec))
    target_sign = 0 if target_mag <= 1e-12 else int(np.sign(delta_target))
    if exec_sign != 0 and target_sign != 0 and exec_sign != target_sign:
        return "sign_flip"
    if exec_mag >= float(near_flat_threshold) and target_mag <= float(suppression_ratio) * exec_mag:
        return "target_suppression"
    if exec_sign != 0 and target_sign == 0 and exec_mag >= float(near_flat_threshold):
        return "target_suppression"
    return "aligned"


def _build_result_row(
    *,
    architecture_spec: ArchitectureSpec,
    period: str,
    seed: int,
    kappa: float,
    selected_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    selected_arm_label: str,
    reference_arm_label: str,
    result_dir: Path,
    model_path: str | None,
    near_flat_threshold: float,
    suppression_ratio: float,
    selection_payload_path: str | None = None,
) -> dict[str, Any]:
    sharpe_exec_selected = _compute_sharpe(selected_df["net_return_lin"])
    sharpe_exec_reference = _compute_sharpe(reference_df["net_return_lin"])
    sharpe_target_selected = _compute_sharpe(selected_df["net_return_lin_target"])
    sharpe_target_reference = _compute_sharpe(reference_df["net_return_lin_target"])
    delta_exec = float(sharpe_exec_selected - sharpe_exec_reference)
    delta_target = float(sharpe_target_selected - sharpe_target_reference)
    disagreement_type = _derive_disagreement_type(
        delta_exec=delta_exec,
        delta_target=delta_target,
        near_flat_threshold=float(near_flat_threshold),
        suppression_ratio=float(suppression_ratio),
    )

    row = {
        "architecture": architecture_spec.name,
        "family": architecture_spec.family,
        "evaluation_role": architecture_spec.evaluation_role,
        "compare_arm": architecture_spec.compare_arm,
        "period": period,
        "seed": int(seed),
        "kappa": float(kappa),
        "selected_arm": selected_arm_label,
        "reference_arm": reference_arm_label,
        "sharpe_exec_net": sharpe_exec_selected,
        "sharpe_target_net": sharpe_target_selected,
        "turnover_exec": _safe_mean(selected_df["turnover_exec"]),
        "turnover_target": _safe_mean(selected_df["turnover_target"]),
        "disagreement_type": disagreement_type,
        "delta_vs_reference_exec": delta_exec,
        "delta_vs_reference_target": delta_target,
        "zero_cost_near_flat_flag": bool(abs(delta_exec) <= float(near_flat_threshold)) if float(kappa) == 0.0 else None,
        "reference_sharpe_exec_net": sharpe_exec_reference,
        "reference_sharpe_target_net": sharpe_target_reference,
        "reference_turnover_exec": _safe_mean(reference_df["turnover_exec"]),
        "reference_turnover_target": _safe_mean(reference_df["turnover_target"]),
        "tracking_error_l2": _safe_mean(selected_df["tracking_error_l2"]) if "tracking_error_l2" in selected_df.columns else None,
        "final_path_gap": (
            _safe_last_gap(selected_df["equity_net_lin"], selected_df["equity_net_lin_target"])
            if "equity_net_lin" in selected_df.columns and "equity_net_lin_target" in selected_df.columns
            else None
        ),
        "cost_exec": _safe_sum(selected_df["cost"]) if "cost" in selected_df.columns else None,
        "cost_target": _safe_sum(selected_df["cost_target"]) if "cost_target" in selected_df.columns else None,
        "cagr_exec": _compute_cagr(selected_df["equity_net_lin"]) if "equity_net_lin" in selected_df.columns else None,
        "mdd_exec": _compute_max_drawdown(selected_df["equity_net_lin"]) if "equity_net_lin" in selected_df.columns else None,
        "steps": int(len(selected_df)),
        "collapse_flag_any": bool(selected_df.get("collapse_flag", pd.Series(False, index=selected_df.index)).astype(bool).any()),
        "result_dir": str(result_dir.resolve()),
        "selected_trace_path": str((result_dir / "selected_trace.parquet").resolve()),
        "reference_trace_path": str((result_dir / "reference_trace.parquet").resolve()),
        "model_path": model_path,
        "selection_payload_path": selection_payload_path,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
    }
    return row


def _load_architecture_specs(spec_dir: Path) -> dict[str, ArchitectureSpec]:
    specs: dict[str, ArchitectureSpec] = {}
    required_fields = (
        "name",
        "family",
        "forecast_source",
        "decision_rule",
        "params",
        "compare_arm",
        "evaluation_role",
        "notes",
    )
    for path in sorted(spec_dir.glob("*.yaml")):
        raw = _load_yaml(path)
        missing = [field for field in required_fields if field not in raw]
        if missing:
            raise ValueError(f"Architecture spec {path} missing fields: {missing}")
        name = str(raw["name"])
        if name in specs:
            raise ValueError(f"Duplicate architecture spec name detected: {name}")
        specs[name] = ArchitectureSpec(
            name=name,
            family=str(raw["family"]),
            forecast_source=str(raw["forecast_source"]),
            decision_rule=str(raw["decision_rule"]),
            params=dict(raw.get("params") or {}),
            compare_arm=str(raw["compare_arm"]),
            evaluation_role=str(raw["evaluation_role"]),
            notes=[str(note) for note in (raw.get("notes") or [])],
            path=path,
        )
    return specs


def _resolve_architecture_names(cfg: dict[str, Any], *, include_optional_threshold: bool) -> list[str]:
    names = [str(name) for name in (cfg.get("architectures") or [])]
    if include_optional_threshold or bool(cfg.get("include_optional_threshold", False)):
        names.extend(str(name) for name in (cfg.get("optional_architectures") or []))
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    if not ordered:
        raise ValueError("Config must define at least one architecture.")
    return ordered


def _parse_rule_vol_env(env_cfg: dict[str, Any]) -> tuple[int, float, float, float]:
    rule_vol = env_cfg.get("rule_vol", {}) or {}
    window = int(rule_vol.get("window", int(EnvConfig.rule_vol_window)))
    a = float(rule_vol.get("a", float(EnvConfig.rule_vol_a)))
    eta_clip = rule_vol.get("eta_clip")
    if eta_clip is None:
        eta_clip_min = float(EnvConfig.eta_clip_min)
        eta_clip_max = float(EnvConfig.eta_clip_max)
    else:
        if not isinstance(eta_clip, (list, tuple)) or len(eta_clip) != 2:
            raise ValueError("env.rule_vol.eta_clip must be [min, max].")
        eta_clip_min = float(eta_clip[0])
        eta_clip_max = float(eta_clip[1])
    return window, a, eta_clip_min, eta_clip_max


def _build_rl_eval_env(ctx: Any, *, transaction_cost: float, eta_mode: str, rebalance_eta: float | None):
    env_cfg = ctx.env_cfg
    rule_vol_window, rule_vol_a, eta_clip_min, eta_clip_max = _parse_rule_vol_env(env_cfg)
    signal_payload = ctx.signal_features if ctx.signal_state else None
    return build_env_for_range(
        market=ctx.market,
        features=ctx.features,
        start=ctx.eval_start,
        end=ctx.eval_end,
        window_size=ctx.window_size,
        c_tc=float(transaction_cost),
        seed=ctx.seed,
        logit_scale=float(env_cfg["logit_scale"]),
        random_reset=False,
        risk_lambda=float(env_cfg.get("risk_lambda", 0.0)),
        risk_penalty_type=str(env_cfg.get("risk_penalty_type", "r2")),
        rebalance_eta=rebalance_eta,
        eta_mode=str(eta_mode),
        rule_vol_window=rule_vol_window,
        rule_vol_a=rule_vol_a,
        eta_clip_min=eta_clip_min,
        eta_clip_max=eta_clip_max,
        signal_features=signal_payload,
    )


def _load_rl_model(ctx: Any, env) -> Any:
    scheduler = None
    if ctx.model_type == "prl":
        scheduler = create_scheduler(ctx.prl_cfg, ctx.window_size, ctx.market.returns.shape[1], ctx.features.stats_path)
    return load_model(ctx.model_path, ctx.model_type, env, scheduler=scheduler)


def _manual_rl_replay(
    ctx: Any,
    *,
    model: Any,
    transaction_cost: float,
    eval_tag: str,
    fixed_eta: float | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    if (fixed_eta is None) == (threshold is None):
        raise ValueError("Provide exactly one of fixed_eta or threshold for manual RL replay.")

    env = _build_rl_eval_env(ctx, transaction_cost=float(transaction_cost), eta_mode="fixed", rebalance_eta=1.0)
    try:
        obs = env.reset()
        base_env = env.envs[0]
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

        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action, dtype=np.float64)
            if action_arr.ndim == 2:
                action_arr = action_arr[0]
            z = np.clip(action_arr, base_env.action_space.low, base_env.action_space.high)
            w_target = stable_softmax(z, scale=base_env.cfg.logit_scale).astype(np.float64)

            prev_weights = base_env.prev_weights.astype(np.float64)
            turnover_target = turnover_l1(prev_weights, w_target)
            if threshold is not None:
                trigger = bool(turnover_target > float(threshold))
                w_exec = w_target if trigger else prev_weights.copy()
                eta_t = 1.0 if trigger else 0.0
                lambda_t = float(threshold)
            else:
                eta_t = float(fixed_eta)
                w_exec = (1.0 - eta_t) * prev_weights + eta_t * w_target
                w_exec = base_env._safe_normalize_weights(w_exec)
                lambda_t = np.nan

            returns_t = base_env.returns.iloc[base_env.current_step].to_numpy(copy=False)
            step_date = pd.Timestamp(base_env.returns.index[base_env.current_step])
            arithmetic_returns = np.expm1(returns_t)

            portfolio_return = float(np.dot(w_exec, arithmetic_returns))
            portfolio_return_target = float(np.dot(w_target, arithmetic_returns))
            turnover_exec = turnover_l1(prev_weights, w_exec)
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

            log_argument = max(raw_log_argument if np.isfinite(raw_log_argument) else float(base_env.cfg.log_clip), float(base_env.cfg.log_clip))
            log_argument_target = max(
                raw_log_argument_target if np.isfinite(raw_log_argument_target) else float(base_env.cfg.log_clip),
                float(base_env.cfg.log_clip),
            )
            log_return_gross_val = math.log(log_argument)
            log_return_gross_target_val = math.log(log_argument_target)
            log_return_net_val = log_return_gross_val - cost_exec
            log_return_net_target_val = log_return_gross_target_val - cost_target

            rewards.append(log_return_net_val)
            portfolio_returns.append(portfolio_return)
            portfolio_returns_target.append(portfolio_return_target)
            turnovers_exec.append(turnover_exec)
            turnovers_target.append(turnover_target)
            dates.append(step_date)
            costs.append(cost_exec)
            costs_target.append(cost_target)
            net_returns_exp.append(math.exp(log_return_net_val) - 1.0)
            net_returns_lin.append(net_return_lin_exec)
            net_returns_lin_target.append(net_return_lin_target_val)
            log_returns_gross.append(log_return_gross_val)
            log_returns_gross_target.append(log_return_gross_target_val)
            log_returns_net.append(log_return_net_val)
            log_returns_net_target.append(log_return_net_target_val)
            eta_ts.append(float(eta_t))
            lambda_ts.append(float(lambda_t) if np.isfinite(lambda_t) else np.nan)
            tracking_errors.append(tracking_error_l2)
            collapse_flags.append(collapse_flag)
            collapse_reasons.append(collapse_reason)

            base_env.prev_weights = w_exec.astype(np.float32)
            base_env.current_step += 1
            done = base_env.current_step >= len(base_env.returns)
            if done:
                break
            obs = np.expand_dims(base_env._get_observation(), axis=0).astype(np.float32)

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
        return trace_dict_to_frame(trace, eval_id=f"{ctx.run_id}__{eval_tag}", run_id=ctx.run_id, model_type=ctx.model_type, seed=ctx.seed)
    finally:
        env.close()


def _align_returns_vol(
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
    return returns.loc[idx], vol_clean.loc[idx], signal_aligned


def _resolve_talibp_config(cfg: dict[str, Any]) -> LBIPConfig:
    env_cfg = cfg.get("env", {}) or {}
    talibp_cfg = cfg.get("talibp", {}) or {}
    return LBIPConfig(
        window_size=int(env_cfg["L"]),
        ridge_alpha=float(talibp_cfg.get("ridge_alpha", 30.0)),
        fit_passes=int(talibp_cfg.get("fit_passes", 2)),
        training_eta=1.0,
        covariance_lookback=int(talibp_cfg.get("covariance_lookback", 252)),
        covariance_history_min=int(talibp_cfg.get("covariance_history_min", 30)),
        mean_variance_risk_aversion=float(talibp_cfg.get("mean_variance_risk_aversion", 10.0)),
        include_prev_weights=bool(talibp_cfg.get("include_prev_weights", True)),
        target_mode="anchored_mean_variance",
        anchor_strength=0.0,
        equal_weight_shrink=float(talibp_cfg.get("equal_weight_shrink", 0.0)),
        log_clip=float(talibp_cfg.get("log_clip", 1e-8)),
        eps=float(talibp_cfg.get("eps", 1e-12)),
    )


def _prepare_lbip_data(config_path: Path, *, period: str) -> PreparedLBIPData:
    cfg = _load_yaml(config_path)
    cfg["config_path"] = str(config_path)
    dates = cfg.get("dates", {}) or {}
    env_cfg = cfg.get("env", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    paper_mode = bool(data_cfg.get("paper_mode", False))
    require_cache_cfg = bool(data_cfg.get("require_cache", False))
    offline_cfg = bool(data_cfg.get("offline", False))
    offline = bool(offline_cfg or paper_mode or require_cache_cfg)
    require_cache = bool(require_cache_cfg or paper_mode or offline)
    cache_only = bool(paper_mode or require_cache_cfg or offline_cfg or offline)

    market, features = prepare_market_and_features(
        cfg,
        lv=int(env_cfg["Lv"]),
        force_refresh=bool(data_cfg.get("force_refresh", True)),
        offline=offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        session_opts=data_cfg.get("session_opts"),
        cache_only=cache_only,
    )
    signal_features, signal_spec = build_signal_features(market, config=cfg)

    train_returns = slice_frame(market.returns, dates["train_start"], dates["train_end"])
    train_volatility = slice_frame(features.volatility, dates["train_start"], dates["train_end"])
    train_signals = slice_frame(signal_features, dates["train_start"], dates["train_end"]) if signal_features is not None else None
    train_returns, train_volatility, train_signals = _align_returns_vol(train_returns, train_volatility, train_signals)

    eval_returns = slice_frame(market.returns, dates["test_start"], dates["test_end"])
    eval_volatility = slice_frame(features.volatility, dates["test_start"], dates["test_end"])
    eval_signals = slice_frame(signal_features, dates["test_start"], dates["test_end"]) if signal_features is not None else None
    eval_returns, eval_volatility, eval_signals = _align_returns_vol(eval_returns, eval_volatility, eval_signals)

    return PreparedLBIPData(
        config_path=config_path,
        cfg=cfg,
        period=period,
        train_returns=train_returns,
        train_volatility=train_volatility,
        train_signals=train_signals,
        eval_returns=eval_returns,
        eval_volatility=eval_volatility,
        eval_signals=eval_signals,
        base_config=_resolve_talibp_config(cfg),
        signal_spec=signal_spec,
    )


def _evaluate_tau_grid(
    prepared: PreparedLBIPData,
    *,
    taus: list[float],
    kappas: list[float],
    experiment_name: str,
    seed: int,
) -> dict[float, dict[float, pd.DataFrame]]:
    outputs: dict[float, dict[float, pd.DataFrame]] = {}
    for tau in taus:
        tau_cfg = LBIPConfig(**{**prepared.base_config.__dict__, "anchor_strength": float(tau)})
        model = fit_lbip_model(
            prepared.train_returns,
            prepared.train_volatility,
            signal_features=prepared.train_signals,
            config=tau_cfg,
        )
        outputs[float(tau)] = {}
        for kappa in kappas:
            _, trace_df = evaluate_lbip_eta(
                model,
                prepared.eval_returns,
                prepared.eval_volatility,
                eta=1.0,
                transaction_cost=float(kappa),
                signal_features=prepared.eval_signals,
                config=tau_cfg,
                eval_id=f"{experiment_name}_{prepared.period}_tau{tau:g}_kappa{kappa:g}_seed{seed}",
                run_id=f"{experiment_name}_{prepared.period}_tau{tau:g}_seed{seed}",
                seed=int(seed),
            )
            outputs[float(tau)][float(kappa)] = trace_df
    return outputs


def _select_tau(validation_results: dict[float, dict[float, pd.DataFrame]], *, kappas: list[float]) -> dict[str, Any]:
    pos_kappas = [float(value) for value in kappas if float(value) > 0.0]
    rows: list[dict[str, Any]] = []
    tau_values = sorted(validation_results.keys())
    baseline_tau = 0.0 if 0.0 in tau_values else float(tau_values[0])

    for tau in tau_values:
        row: dict[str, Any] = {"tau": float(tau)}
        scores: list[float] = []
        for kappa in kappas:
            df = validation_results[float(tau)][float(kappa)]
            row[f"sharpe_exec_kappa_{kappa:g}"] = _compute_sharpe(df["net_return_lin"])
            row[f"sharpe_target_kappa_{kappa:g}"] = _compute_sharpe(df["net_return_lin_target"])
            row[f"turnover_exec_kappa_{kappa:g}"] = _safe_mean(df["turnover_exec"])
            if float(kappa) > 0.0:
                scores.append(float(row[f"sharpe_exec_kappa_{kappa:g}"]))
        row["score_mean_sharpe_pos_kappa"] = float(np.mean(scores)) if scores else float("nan")
        rows.append(row)

    rows_df = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    best_score = float(rows_df["score_mean_sharpe_pos_kappa"].max())
    if best_score > 0.0:
        threshold = 0.95 * best_score
    elif best_score < 0.0:
        threshold = best_score / 0.95
    else:
        threshold = 0.0
    rows_df["qualifies"] = rows_df["score_mean_sharpe_pos_kappa"] >= threshold
    qualifying = rows_df[rows_df["qualifies"]].copy()
    if qualifying.empty:
        selected_tau = float(rows_df.loc[rows_df["score_mean_sharpe_pos_kappa"].idxmax(), "tau"])
    else:
        positive_qualifying = qualifying[qualifying["tau"] > 0.0]
        if positive_qualifying.empty:
            selected_tau = float(qualifying.sort_values(["tau", "score_mean_sharpe_pos_kappa"], ascending=[True, False]).iloc[0]["tau"])
        else:
            selected_tau = float(positive_qualifying.sort_values(["tau", "score_mean_sharpe_pos_kappa"], ascending=[True, False]).iloc[0]["tau"])

    rows_df["selected"] = rows_df["tau"] == selected_tau
    return {
        "selection_rule": "smallest_positive_tau_within_95pct_of_best_positive_cost_score",
        "baseline_tau": float(baseline_tau),
        "best_score": float(best_score),
        "threshold": float(threshold),
        "selected_tau": float(selected_tau),
        "rows": rows_df.to_dict(orient="records"),
    }


def _select_threshold(
    validation_results: dict[float, dict[float, pd.DataFrame]],
    *,
    kappas: list[float],
) -> dict[str, Any]:
    pos_kappas = [float(value) for value in kappas if float(value) > 0.0]
    rows: list[dict[str, Any]] = []
    threshold_values = sorted(validation_results.keys())
    baseline_threshold = 0.0 if 0.0 in threshold_values else float(threshold_values[0])

    for threshold_value in threshold_values:
        row: dict[str, Any] = {"threshold": float(threshold_value)}
        scores: list[float] = []
        for kappa in kappas:
            df = validation_results[float(threshold_value)][float(kappa)]
            row[f"sharpe_exec_kappa_{kappa:g}"] = _compute_sharpe(df["net_return_lin"])
            row[f"sharpe_target_kappa_{kappa:g}"] = _compute_sharpe(df["net_return_lin_target"])
            row[f"turnover_exec_kappa_{kappa:g}"] = _safe_mean(df["turnover_exec"])
            if float(kappa) > 0.0:
                scores.append(float(row[f"sharpe_exec_kappa_{kappa:g}"]))
        row["score_mean_sharpe_pos_kappa"] = float(np.mean(scores)) if scores else float("nan")
        rows.append(row)

    rows_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_score = float(rows_df["score_mean_sharpe_pos_kappa"].max())
    if best_score > 0.0:
        threshold_cutoff = 0.95 * best_score
    elif best_score < 0.0:
        threshold_cutoff = best_score / 0.95
    else:
        threshold_cutoff = 0.0
    rows_df["qualifies"] = rows_df["score_mean_sharpe_pos_kappa"] >= threshold_cutoff
    qualifying = rows_df[rows_df["qualifies"]].copy()
    if qualifying.empty:
        selected_threshold = float(rows_df.loc[rows_df["score_mean_sharpe_pos_kappa"].idxmax(), "threshold"])
    else:
        positive_qualifying = qualifying[qualifying["threshold"] > 0.0]
        if positive_qualifying.empty:
            selected_threshold = float(qualifying.sort_values(["threshold", "score_mean_sharpe_pos_kappa"], ascending=[True, False]).iloc[0]["threshold"])
        else:
            selected_threshold = float(positive_qualifying.sort_values(["threshold", "score_mean_sharpe_pos_kappa"], ascending=[True, False]).iloc[0]["threshold"])

    rows_df["selected"] = rows_df["threshold"] == selected_threshold
    return {
        "selection_rule": "smallest_positive_threshold_within_95pct_of_best_positive_cost_score",
        "baseline_threshold": float(baseline_threshold),
        "best_score": float(best_score),
        "threshold_cutoff": float(threshold_cutoff),
        "selected_threshold": float(selected_threshold),
        "rows": rows_df.to_dict(orient="records"),
    }


def _run_arch_rl_selected(
    architecture_spec: ArchitectureSpec,
    *,
    seed: int,
    kappas: list[float],
    raw_root: Path,
    rl_final_config: Path,
    rl_model_root: Path,
    offline: bool,
    save_trace: bool,
    near_flat_threshold: float,
    suppression_ratio: float,
) -> None:
    selected_eta = float(architecture_spec.params.get("selected_eta", 0.5))
    baseline_eta = float(architecture_spec.params.get("baseline_eta", 1.0))
    ctx = build_eval_context(
        config_path=str(rl_final_config),
        model_type="prl",
        seed=int(seed),
        model_root=str(rl_model_root),
        offline=offline,
        max_steps=0,
        prefer_metadata_config=False,
    )
    model_path = str(ctx.model_path.resolve())

    for kappa in kappas:
        result_dir = raw_root / architecture_spec.name / f"kappa_{kappa:g}" / f"seed_{int(seed)}"
        if _result_complete(result_dir, save_trace=save_trace):
            LOGGER.info("Skipping %s seed=%s kappa=%s; result already exists.", architecture_spec.name, seed, kappa)
            continue
        LOGGER.info("Running %s seed=%s kappa=%s via exact RL evaluation path.", architecture_spec.name, seed, kappa)
        _, _, selected_df, _ = run_eval_case(
            ctx,
            eta_mode="fixed",
            rebalance_eta=float(selected_eta),
            transaction_cost=float(kappa),
            eval_tag=f"{architecture_spec.name}__selected__eta_{selected_eta:g}__kappa_{kappa:g}",
        )
        _, _, reference_df, _ = run_eval_case(
            ctx,
            eta_mode="fixed",
            rebalance_eta=float(baseline_eta),
            transaction_cost=float(kappa),
            eval_tag=f"{architecture_spec.name}__reference__eta_{baseline_eta:g}__kappa_{kappa:g}",
        )
        row = _build_result_row(
            architecture_spec=architecture_spec,
            period="final",
            seed=int(seed),
            kappa=float(kappa),
            selected_df=selected_df,
            reference_df=reference_df,
            selected_arm_label=f"eta_{selected_eta:g}",
            reference_arm_label=f"eta_{baseline_eta:g}",
            result_dir=result_dir,
            model_path=model_path,
            near_flat_threshold=float(near_flat_threshold),
            suppression_ratio=float(suppression_ratio),
        )
        meta = {
            "architecture_spec_path": str(architecture_spec.path.resolve()),
            "evaluation_impl": "step6_sanity.run_eval_case",
            "forecast_source_mode": "exact_frozen_rl_policy_stream",
        }
        _write_result_bundle(result_dir, row, selected_df=selected_df, reference_df=reference_df, meta=meta, save_trace=save_trace)


def _run_arch_rule_eta_fixed(
    architecture_spec: ArchitectureSpec,
    *,
    seed: int,
    kappas: list[float],
    raw_root: Path,
    rl_final_config: Path,
    rl_model_root: Path,
    offline: bool,
    save_trace: bool,
    near_flat_threshold: float,
    suppression_ratio: float,
) -> None:
    selected_eta = float(architecture_spec.params.get("selected_eta", 0.5))
    baseline_eta = float(architecture_spec.params.get("baseline_eta", 1.0))
    ctx = build_eval_context(
        config_path=str(rl_final_config),
        model_type="prl",
        seed=int(seed),
        model_root=str(rl_model_root),
        offline=offline,
        max_steps=0,
        prefer_metadata_config=False,
    )
    base_env_for_model = _build_rl_eval_env(ctx, transaction_cost=0.0, eta_mode="fixed", rebalance_eta=1.0)
    try:
        model = _load_rl_model(ctx, base_env_for_model)
    finally:
        base_env_for_model.close()
    model_path = str(ctx.model_path.resolve())

    for kappa in kappas:
        result_dir = raw_root / architecture_spec.name / f"kappa_{kappa:g}" / f"seed_{int(seed)}"
        if _result_complete(result_dir, save_trace=save_trace):
            LOGGER.info("Skipping %s seed=%s kappa=%s; result already exists.", architecture_spec.name, seed, kappa)
            continue
        LOGGER.info("Running %s seed=%s kappa=%s via frozen-target replay.", architecture_spec.name, seed, kappa)
        selected_df = _manual_rl_replay(
            ctx,
            model=model,
            transaction_cost=float(kappa),
            eval_tag=f"{architecture_spec.name}__selected__eta_{selected_eta:g}__kappa_{kappa:g}",
            fixed_eta=float(selected_eta),
        )
        reference_df = _manual_rl_replay(
            ctx,
            model=model,
            transaction_cost=float(kappa),
            eval_tag=f"{architecture_spec.name}__reference__eta_{baseline_eta:g}__kappa_{kappa:g}",
            fixed_eta=float(baseline_eta),
        )
        row = _build_result_row(
            architecture_spec=architecture_spec,
            period="final",
            seed=int(seed),
            kappa=float(kappa),
            selected_df=selected_df,
            reference_df=reference_df,
            selected_arm_label=f"eta_{selected_eta:g}",
            reference_arm_label=f"eta_{baseline_eta:g}",
            result_dir=result_dir,
            model_path=model_path,
            near_flat_threshold=float(near_flat_threshold),
            suppression_ratio=float(suppression_ratio),
        )
        meta = {
            "architecture_spec_path": str(architecture_spec.path.resolve()),
            "evaluation_impl": "manual_rl_replay",
            "forecast_source_mode": "same_frozen_rl_target_stream",
        }
        _write_result_bundle(result_dir, row, selected_df=selected_df, reference_df=reference_df, meta=meta, save_trace=save_trace)


def _run_arch_threshold_rebalance(
    architecture_spec: ArchitectureSpec,
    *,
    seed: int,
    kappas: list[float],
    raw_root: Path,
    runtime_root: Path,
    rl_validation_config: Path,
    rl_final_config: Path,
    rl_model_root: Path,
    offline: bool,
    save_trace: bool,
    near_flat_threshold: float,
    suppression_ratio: float,
) -> None:
    threshold_grid = [0.0] + [float(value) for value in (architecture_spec.params.get("threshold_grid") or [])]
    threshold_values = sorted(set(float(value) for value in threshold_grid))
    validation_ctx = build_eval_context(
        config_path=str(rl_validation_config),
        model_type="prl",
        seed=int(seed),
        model_root=str(rl_model_root),
        offline=offline,
        max_steps=0,
        prefer_metadata_config=False,
    )
    final_ctx = build_eval_context(
        config_path=str(rl_final_config),
        model_type="prl",
        seed=int(seed),
        model_root=str(rl_model_root),
        offline=offline,
        max_steps=0,
        prefer_metadata_config=False,
    )
    base_env_for_model = _build_rl_eval_env(validation_ctx, transaction_cost=0.0, eta_mode="fixed", rebalance_eta=1.0)
    try:
        model = _load_rl_model(validation_ctx, base_env_for_model)
    finally:
        base_env_for_model.close()

    selection_dir = runtime_root / architecture_spec.name / "selection"
    _ensure_dir(selection_dir)
    selection_payload_path = selection_dir / f"validation_threshold_selection_seed{int(seed)}.json"

    validation_results: dict[float, dict[float, pd.DataFrame]] = {}
    for threshold_value in threshold_values:
        validation_results[float(threshold_value)] = {}
        for kappa in kappas:
            validation_results[float(threshold_value)][float(kappa)] = _manual_rl_replay(
                validation_ctx,
                model=model,
                transaction_cost=float(kappa),
                eval_tag=f"{architecture_spec.name}__validation__threshold_{threshold_value:g}__kappa_{kappa:g}",
                threshold=float(threshold_value),
            )

    selection_payload = _select_threshold(validation_results, kappas=kappas)
    selection_payload["seed"] = int(seed)
    selection_payload["architecture"] = architecture_spec.name
    selection_payload_path.write_text(json.dumps(selection_payload, indent=2))
    selected_threshold = float(selection_payload["selected_threshold"])

    for kappa in kappas:
        result_dir = raw_root / architecture_spec.name / f"kappa_{kappa:g}" / f"seed_{int(seed)}"
        if _result_complete(result_dir, save_trace=save_trace):
            LOGGER.info("Skipping %s seed=%s kappa=%s; result already exists.", architecture_spec.name, seed, kappa)
            continue
        LOGGER.info("Running %s seed=%s kappa=%s with selected threshold=%s.", architecture_spec.name, seed, kappa, selected_threshold)
        selected_df = _manual_rl_replay(
            final_ctx,
            model=model,
            transaction_cost=float(kappa),
            eval_tag=f"{architecture_spec.name}__selected__threshold_{selected_threshold:g}__kappa_{kappa:g}",
            threshold=float(selected_threshold),
        )
        reference_df = _manual_rl_replay(
            final_ctx,
            model=model,
            transaction_cost=float(kappa),
            eval_tag=f"{architecture_spec.name}__reference__threshold_0__kappa_{kappa:g}",
            threshold=0.0,
        )
        row = _build_result_row(
            architecture_spec=architecture_spec,
            period="final",
            seed=int(seed),
            kappa=float(kappa),
            selected_df=selected_df,
            reference_df=reference_df,
            selected_arm_label=f"threshold_{selected_threshold:g}",
            reference_arm_label="threshold_0",
            result_dir=result_dir,
            model_path=str(final_ctx.model_path.resolve()),
            near_flat_threshold=float(near_flat_threshold),
            suppression_ratio=float(suppression_ratio),
            selection_payload_path=str(selection_payload_path.resolve()),
        )
        meta = {
            "architecture_spec_path": str(architecture_spec.path.resolve()),
            "evaluation_impl": "manual_rl_replay_threshold",
            "forecast_source_mode": "same_frozen_rl_target_stream",
        }
        _write_result_bundle(result_dir, row, selected_df=selected_df, reference_df=reference_df, meta=meta, save_trace=save_trace)


def _run_arch_linear_prox(
    architecture_spec: ArchitectureSpec,
    *,
    seed: int,
    kappas: list[float],
    tau_grid: list[float],
    raw_root: Path,
    runtime_root: Path,
    talibp_validation_config: Path,
    talibp_final_config: Path,
    save_trace: bool,
    near_flat_threshold: float,
    suppression_ratio: float,
) -> None:
    runtime_arch_root = runtime_root / architecture_spec.name
    selection_dir = runtime_arch_root / "selection"
    _ensure_dir(selection_dir)
    selection_payload_path = selection_dir / f"validation_tau_selection_seed{int(seed)}.json"

    LOGGER.info("Preparing validation data for %s seed=%s.", architecture_spec.name, seed)
    validation_prepared = _prepare_lbip_data(talibp_validation_config, period="validation")
    validation_results = _evaluate_tau_grid(
        validation_prepared,
        taus=tau_grid,
        kappas=kappas,
        experiment_name="architecture_matrix_linear_prox",
        seed=int(seed),
    )
    selection_payload = _select_tau(validation_results, kappas=kappas)
    selection_payload["seed"] = int(seed)
    selection_payload["architecture"] = architecture_spec.name
    selection_payload_path.write_text(json.dumps(selection_payload, indent=2))
    selected_tau = float(selection_payload["selected_tau"])
    final_taus = sorted(set([0.0, float(selected_tau)]))

    LOGGER.info("Preparing final data for %s seed=%s with selected tau=%s.", architecture_spec.name, seed, selected_tau)
    final_prepared = _prepare_lbip_data(talibp_final_config, period="final")
    final_results = _evaluate_tau_grid(
        final_prepared,
        taus=final_taus,
        kappas=kappas,
        experiment_name="architecture_matrix_linear_prox",
        seed=int(seed),
    )

    for kappa in kappas:
        result_dir = raw_root / architecture_spec.name / f"kappa_{kappa:g}" / f"seed_{int(seed)}"
        if _result_complete(result_dir, save_trace=save_trace):
            LOGGER.info("Skipping %s seed=%s kappa=%s; result already exists.", architecture_spec.name, seed, kappa)
            continue
        selected_df = final_results[float(selected_tau)][float(kappa)]
        reference_df = final_results[0.0][float(kappa)]
        row = _build_result_row(
            architecture_spec=architecture_spec,
            period="final",
            seed=int(seed),
            kappa=float(kappa),
            selected_df=selected_df,
            reference_df=reference_df,
            selected_arm_label=f"tau_{selected_tau:g}",
            reference_arm_label="tau_0",
            result_dir=result_dir,
            model_path=None,
            near_flat_threshold=float(near_flat_threshold),
            suppression_ratio=float(suppression_ratio),
            selection_payload_path=str(selection_payload_path.resolve()),
        )
        meta = {
            "architecture_spec_path": str(architecture_spec.path.resolve()),
            "evaluation_impl": "linear_information_parity.evaluate_lbip_eta",
            "forecast_source_mode": "family_internal_frozen_linear_forecast_map",
            "signal_spec": final_prepared.signal_spec,
        }
        _write_result_bundle(result_dir, row, selected_df=selected_df, reference_df=reference_df, meta=meta, save_trace=save_trace)


def _summarize_plan(
    *,
    architecture_names: list[str],
    seeds: list[int],
    kappas: list[float],
    raw_output_root: Path,
    runtime_root: Path,
    tau_grid: list[float],
) -> dict[str, Any]:
    threshold_arch_count = 1 if "arch_threshold_rebalance" in architecture_names else 0
    rl_like_count = len([name for name in architecture_names if name in {"arch_rl_selected", "arch_rule_eta_fixed"}])
    linear_like_count = 1 if "arch_linear_prox" in architecture_names else 0
    eval_runs = len(seeds) * len(kappas) * (2 * rl_like_count + 2 * linear_like_count + 2 * threshold_arch_count)
    validation_sweeps = 0
    if linear_like_count:
        validation_sweeps += len(seeds) * len(kappas) * len(tau_grid)
    if threshold_arch_count:
        validation_sweeps += len(seeds) * len(kappas) * 4
    return {
        "status": "dry_run",
        "architecture_count": len(architecture_names),
        "architectures": architecture_names,
        "seed_count": len(seeds),
        "kappa_count": len(kappas),
        "final_pairwise_evals": eval_runs,
        "validation_sweep_evals": validation_sweeps,
        "raw_output_root": str(raw_output_root.resolve()),
        "runtime_root": str(runtime_root.resolve()),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    spec_dir = _resolve_path(config_path, str(cfg.get("architecture_spec_dir")))
    architecture_specs = _load_architecture_specs(spec_dir)
    architecture_names = _resolve_architecture_names(cfg, include_optional_threshold=bool(args.include_optional_threshold))
    missing_specs = [name for name in architecture_names if name not in architecture_specs]
    if missing_specs:
        raise ValueError(f"Unknown architecture specs requested: {missing_specs}")

    paths_cfg = cfg.get("paths", {}) or {}
    raw_output_root = _resolve_path(config_path, str(paths_cfg.get("raw_output_root")))
    runtime_root = _resolve_path(config_path, str(paths_cfg.get("runtime_root")))
    rl_validation_config = _resolve_path(config_path, str(paths_cfg.get("rl_validation_config")))
    rl_final_config = _resolve_path(config_path, str(paths_cfg.get("rl_final_config")))
    rl_model_root = _resolve_path(config_path, str(paths_cfg.get("rl_model_root")))
    talibp_validation_config = _resolve_path(config_path, str(paths_cfg.get("talibp_validation_config")))
    talibp_final_config = _resolve_path(config_path, str(paths_cfg.get("talibp_final_config")))

    execution_cfg = cfg.get("execution", {}) or {}
    offline = bool(execution_cfg.get("offline", True))
    seeds = [int(seed) for seed in (execution_cfg.get("seeds") or [0])]
    kappas = [float(kappa) for kappa in (execution_cfg.get("kappas") or [0.0, 0.0005, 0.001])]
    near_flat_threshold = float(execution_cfg.get("near_flat_threshold", 0.005))
    suppression_ratio = float(execution_cfg.get("suppression_ratio", 0.25))
    save_trace = bool(execution_cfg.get("save_trace", True))
    tau_grid = [float(value) for value in ((cfg.get("talibp", {}) or {}).get("taus") or [0.0, 0.03, 0.1, 0.3, 1.0])]

    if args.dry_run:
        print(
            json.dumps(
                _summarize_plan(
                    architecture_names=architecture_names,
                    seeds=seeds,
                    kappas=kappas,
                    raw_output_root=raw_output_root,
                    runtime_root=runtime_root,
                    tau_grid=tau_grid,
                ),
                indent=2,
            )
        )
        return 0

    _ensure_dir(raw_output_root)
    _ensure_dir(runtime_root)

    for seed in seeds:
        for name in architecture_names:
            architecture_spec = architecture_specs[name]
            LOGGER.info("=== Architecture %s | seed=%s ===", architecture_spec.name, seed)
            if architecture_spec.name == "arch_rl_selected":
                _run_arch_rl_selected(
                    architecture_spec,
                    seed=int(seed),
                    kappas=kappas,
                    raw_root=raw_output_root,
                    rl_final_config=rl_final_config,
                    rl_model_root=rl_model_root,
                    offline=offline,
                    save_trace=save_trace,
                    near_flat_threshold=near_flat_threshold,
                    suppression_ratio=suppression_ratio,
                )
            elif architecture_spec.name == "arch_rule_eta_fixed":
                _run_arch_rule_eta_fixed(
                    architecture_spec,
                    seed=int(seed),
                    kappas=kappas,
                    raw_root=raw_output_root,
                    rl_final_config=rl_final_config,
                    rl_model_root=rl_model_root,
                    offline=offline,
                    save_trace=save_trace,
                    near_flat_threshold=near_flat_threshold,
                    suppression_ratio=suppression_ratio,
                )
            elif architecture_spec.name == "arch_linear_prox":
                _run_arch_linear_prox(
                    architecture_spec,
                    seed=int(seed),
                    kappas=kappas,
                    tau_grid=tau_grid,
                    raw_root=raw_output_root,
                    runtime_root=runtime_root,
                    talibp_validation_config=talibp_validation_config,
                    talibp_final_config=talibp_final_config,
                    save_trace=save_trace,
                    near_flat_threshold=near_flat_threshold,
                    suppression_ratio=suppression_ratio,
                )
            elif architecture_spec.name == "arch_threshold_rebalance":
                _run_arch_threshold_rebalance(
                    architecture_spec,
                    seed=int(seed),
                    kappas=kappas,
                    raw_root=raw_output_root,
                    runtime_root=runtime_root,
                    rl_validation_config=rl_validation_config,
                    rl_final_config=rl_final_config,
                    rl_model_root=rl_model_root,
                    offline=offline,
                    save_trace=save_trace,
                    near_flat_threshold=near_flat_threshold,
                    suppression_ratio=suppression_ratio,
                )
            else:
                raise ValueError(f"Unsupported architecture name: {architecture_spec.name}")

    LOGGER.info("Architecture-matrix raw run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
