#!/usr/bin/env python3
"""
Independent non-RL deadband-partial comparator runner.

Assumptions documented here on purpose:

1. This runner does not use RL target replay. It consumes the shared deterministic target mapping
   built by `scripts/generalization/build_shared_targets.py`.
2. The frozen forecast source and target construction are identical across all deadband candidates.
   Only the execution layer changes.
3. This step writes the full validation and final candidate grid plus a full-rebalance reference.
   Champion selection is intentionally deferred to a later validation-only step.
4. The script reuses the repo's existing accounting and trace schema where possible:
   `prepare_market_and_features`, `slice_frame`, `turnover_l1`, `compute_metrics`, and
   `trace_dict_to_frame`.
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

for candidate in (str(GENERALIZATION_SCRIPT_DIR), str(PRL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from build_shared_targets import build_shared_target_bundle
from prl.data import slice_frame
from prl.eval import trace_dict_to_frame
from prl.metrics import compute_metrics, turnover_l1
from prl.train import prepare_market_and_features


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "generalization" / "arch_deadband_partial.yaml"


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
class PreparedPeriodData:
    period: str
    eval_returns: pd.DataFrame
    eval_targets: pd.DataFrame
    metadata: dict[str, Any]
    template_config_path: Path
    eval_start: str
    eval_end: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent non-RL deadband-partial comparator grid.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the deadband comparator YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned candidate grid without executing it.")
    parser.add_argument(
        "--period",
        action="append",
        choices=["validation", "final"],
        help="Optional period subset. May be passed multiple times.",
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


def _load_architecture_spec(path: Path) -> ArchitectureSpec:
    raw = _load_yaml(path)
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
    missing = [field for field in required_fields if field not in raw]
    if missing:
        raise ValueError(f"Architecture spec {path} missing fields: {missing}")
    return ArchitectureSpec(
        name=str(raw["name"]),
        family=str(raw["family"]),
        forecast_source=str(raw["forecast_source"]),
        decision_rule=str(raw["decision_rule"]),
        params=dict(raw.get("params") or {}),
        compare_arm=str(raw["compare_arm"]),
        evaluation_role=str(raw["evaluation_role"]),
        notes=[str(note) for note in (raw.get("notes") or [])],
        path=path,
    )


def _load_template_config(shared_target_config_path: Path, *, period: str) -> tuple[Path, dict[str, Any]]:
    shared_cfg = _load_yaml(shared_target_config_path)
    templates = shared_cfg.get("template_configs", {}) or {}
    if period not in templates:
        raise ValueError(f"shared_target_mapping template_configs missing period={period}")
    template_path = _resolve_path(shared_target_config_path, str(templates[period]))
    cfg = _load_yaml(template_path)
    cfg["config_path"] = str(template_path.resolve())
    return template_path, cfg


def _prepare_period_data(shared_target_config_path: Path, *, period: str, offline: bool) -> PreparedPeriodData:
    bundle = build_shared_target_bundle(shared_target_config_path, period=period)
    template_path, template_cfg = _load_template_config(shared_target_config_path, period=period)

    dates = template_cfg.get("dates", {}) or {}
    if "test_start" not in dates or "test_end" not in dates:
        raise ValueError(f"Template config {template_path} must define dates.test_start and dates.test_end.")

    data_cfg = template_cfg.get("data", {}) or {}
    env_cfg = template_cfg.get("env", {}) or {}
    paper_mode = bool(data_cfg.get("paper_mode", False))
    require_cache_cfg = bool(data_cfg.get("require_cache", False))
    offline_cfg = bool(data_cfg.get("offline", False))
    resolved_offline = bool(offline or offline_cfg or paper_mode or require_cache_cfg)
    require_cache = bool(require_cache_cfg or paper_mode or resolved_offline)
    cache_only = bool(paper_mode or require_cache_cfg or offline_cfg or resolved_offline)

    market, _features = prepare_market_and_features(
        template_cfg,
        lv=int(env_cfg["Lv"]),
        force_refresh=bool(data_cfg.get("force_refresh", True)),
        offline=resolved_offline,
        require_cache=require_cache,
        paper_mode=paper_mode,
        session_opts=data_cfg.get("session_opts"),
        cache_only=cache_only,
    )

    eval_returns = slice_frame(market.returns, dates["test_start"], dates["test_end"]).copy()
    eval_targets = slice_frame(bundle.target_frame, dates["test_start"], dates["test_end"]).copy()

    if eval_returns.empty:
        raise ValueError(f"No evaluation returns available for period={period} in {template_path}.")
    if eval_targets.empty:
        raise ValueError(f"No evaluation targets available for period={period} in {shared_target_config_path}.")

    idx = eval_returns.index.intersection(eval_targets.index)
    eval_returns = eval_returns.loc[idx]
    eval_targets = eval_targets.loc[idx]

    missing_assets = [ticker for ticker in eval_returns.columns if ticker not in eval_targets.columns]
    extra_assets = [ticker for ticker in eval_targets.columns if ticker not in eval_returns.columns]
    if missing_assets or extra_assets:
        raise ValueError(
            "Shared target columns must match evaluation return columns exactly: "
            f"missing={missing_assets}, extra={extra_assets}"
        )
    eval_targets = eval_targets.loc[:, eval_returns.columns]

    metadata = {
        **bundle.metadata,
        "template_config_path": str(template_path.resolve()),
        "eval_start": str(dates["test_start"]),
        "eval_end": str(dates["test_end"]),
        "n_eval_rows": int(len(eval_returns)),
        "n_assets": int(eval_returns.shape[1]),
    }
    return PreparedPeriodData(
        period=period,
        eval_returns=eval_returns,
        eval_targets=eval_targets,
        metadata=metadata,
        template_config_path=template_path,
        eval_start=str(dates["test_start"]),
        eval_end=str(dates["test_end"]),
    )


def _normalize_simplex(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= eps:
        return np.full(arr.shape[0], 1.0 / arr.shape[0], dtype=np.float64)
    return arr / total


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
    return all(path.exists() for path in _result_files(result_dir, save_trace=save_trace).values())


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


def _format_kappa(kappa: float) -> str:
    return np.format_float_positional(float(kappa), trim="-")


def _format_delta(value: float) -> str:
    return f"{float(value):.2f}"


def _format_eta(value: float) -> str:
    return np.format_float_positional(float(value), trim="-", precision=6)


def _candidate_label(*, delta: float, eta_db: float) -> str:
    return f"deadband_partial_delta{_format_delta(delta)}_eta{_format_eta(eta_db)}"


def _simulate_execution_rule(
    period_data: PreparedPeriodData,
    *,
    transaction_cost: float,
    seed: int,
    mode: str,
    delta: float | None,
    eta_db: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns_df = period_data.eval_returns
    target_df = period_data.eval_targets
    arithmetic_returns = np.expm1(returns_df.to_numpy(dtype=np.float64))
    target_arr = target_df.to_numpy(dtype=np.float64)

    num_assets = returns_df.shape[1]
    prev_weights = np.full(num_assets, 1.0 / num_assets, dtype=np.float64)

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
    gap_ts: list[float] = []
    trade_flags: list[bool] = []

    for step_idx, date in enumerate(returns_df.index):
        w_target = _normalize_simplex(target_arr[step_idx])
        gap_t = turnover_l1(prev_weights, w_target)

        if mode == "full_rebalance":
            w_exec = w_target.copy()
            eta_t = 1.0
            lambda_t = 0.0
            traded = True
        elif mode == "deadband_partial":
            if delta is None or eta_db is None:
                raise ValueError("deadband_partial mode requires delta and eta_db")
            if gap_t <= float(delta):
                w_exec = prev_weights.copy()
                eta_t = 0.0
                traded = False
            else:
                w_exec = _normalize_simplex((1.0 - float(eta_db)) * prev_weights + float(eta_db) * w_target)
                eta_t = float(eta_db)
                traded = True
            lambda_t = float(delta)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        arithmetic_returns_t = arithmetic_returns[step_idx]
        portfolio_return = float(np.dot(w_exec, arithmetic_returns_t))
        portfolio_return_target = float(np.dot(w_target, arithmetic_returns_t))
        turnover_exec = turnover_l1(prev_weights, w_exec)
        turnover_target = gap_t
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

        log_argument = max(raw_log_argument if np.isfinite(raw_log_argument) else 1e-8, 1e-8)
        log_argument_target = max(raw_log_argument_target if np.isfinite(raw_log_argument_target) else 1e-8, 1e-8)
        log_return_gross_val = math.log(log_argument)
        log_return_gross_target_val = math.log(log_argument_target)
        log_return_net_val = log_return_gross_val - cost_exec
        log_return_net_target_val = log_return_gross_target_val - cost_target

        rewards.append(log_return_net_val)
        portfolio_returns.append(portfolio_return)
        portfolio_returns_target.append(portfolio_return_target)
        turnovers_exec.append(turnover_exec)
        turnovers_target.append(turnover_target)
        dates.append(pd.Timestamp(date))
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
        lambda_ts.append(float(lambda_t))
        tracking_errors.append(tracking_error_l2)
        collapse_flags.append(collapse_flag)
        collapse_reasons.append(collapse_reason)
        gap_ts.append(float(gap_t))
        trade_flags.append(bool(traded))

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
    trace_df = trace_dict_to_frame(
        trace,
        eval_id=f"{period_data.period}__{mode}",
        run_id=f"deadband_partial__{period_data.period}",
        model_type="deadband_partial",
        seed=int(seed),
    )
    extras = {
        "metrics": metrics.to_dict(),
        "mean_gap_l1": float(np.mean(gap_ts)) if gap_ts else None,
        "trade_rate": float(np.mean(trade_flags)) if trade_flags else None,
    }
    return trace_df, extras


def _build_result_row(
    *,
    architecture_spec: ArchitectureSpec,
    period: str,
    seed: int,
    kappa: float,
    delta: float,
    eta_db: float,
    selected_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    selected_arm_label: str,
    reference_arm_label: str,
    result_dir: Path,
    near_flat_threshold: float,
    suppression_ratio: float,
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

    return {
        "architecture": architecture_spec.name,
        "family": architecture_spec.family,
        "evaluation_role": architecture_spec.evaluation_role,
        "compare_arm": architecture_spec.compare_arm,
        "period": period,
        "seed": int(seed),
        "kappa": float(kappa),
        "delta": float(delta),
        "eta_db": float(eta_db),
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
        "tracking_error_l2": _safe_mean(selected_df["tracking_error_l2"]),
        "final_path_gap": _safe_last_gap(selected_df["equity_net_lin"], selected_df["equity_net_lin_target"]),
        "cost_exec": _safe_sum(selected_df["cost"]),
        "cost_target": _safe_sum(selected_df["cost_target"]),
        "cagr_exec": _compute_cagr(selected_df["equity_net_lin"]),
        "mdd_exec": _compute_max_drawdown(selected_df["equity_net_lin"]),
        "steps": int(len(selected_df)),
        "selection_status": "grid_only_no_champion",
        "result_dir": str(result_dir.resolve()),
        "selected_trace_path": str((result_dir / "selected_trace.parquet").resolve()),
        "reference_trace_path": str((result_dir / "reference_trace.parquet").resolve()),
        "shared_target_independent_from_rl_replay": True,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _iter_periods(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    configured = [str(period) for period in (cfg.get("periods") or [])]
    if args.period:
        requested = [str(period) for period in args.period]
        return [period for period in configured if period in requested]
    return configured


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    architecture_spec_path = _resolve_path(config_path, str(cfg["architecture_spec"]))
    architecture_spec = _load_architecture_spec(architecture_spec_path)
    shared_target_config_path = _resolve_path(config_path, str(cfg["shared_target_config"]))

    execution_cfg = cfg.get("execution", {}) or {}
    reference_cfg = cfg.get("reference", {}) or {}
    raw_output_root = _resolve_path(config_path, str((cfg.get("paths", {}) or {})["raw_output_root"]))

    periods = _iter_periods(cfg, args)
    seeds = [int(seed) for seed in (execution_cfg.get("seeds") or [0])]
    kappas = [float(kappa) for kappa in (execution_cfg.get("kappas") or [])]
    delta_grid = [float(delta) for delta in (execution_cfg.get("delta_grid") or [])]
    eta_grid = [float(eta) for eta in (execution_cfg.get("eta_db_grid") or [])]
    save_trace = bool(execution_cfg.get("save_trace", True))
    near_flat_threshold = float(execution_cfg.get("near_flat_threshold", 0.005))
    suppression_ratio = float(execution_cfg.get("suppression_ratio", 0.25))
    offline = bool(execution_cfg.get("offline", True))
    reference_arm_label = str(reference_cfg.get("arm_label", "full_rebalance_baseline"))

    if not periods:
        raise ValueError("Config must define at least one period.")
    if not kappas or not delta_grid or not eta_grid:
        raise ValueError("Config must define non-empty kappas, delta_grid, and eta_db_grid.")

    period_data_map = {}
    for period in periods:
        period_data_map[period] = _prepare_period_data(shared_target_config_path, period=period, offline=offline)

    plan_count = len(periods) * len(seeds) * len(kappas) * len(delta_grid) * len(eta_grid)
    LOGGER.info(
        "Prepared deadband partial plan with %d candidate runs across periods=%s, kappas=%s, delta_grid=%s, eta_grid=%s",
        plan_count,
        periods,
        kappas,
        delta_grid,
        eta_grid,
    )

    if args.dry_run:
        for period in periods:
            period_data = period_data_map[period]
            LOGGER.info(
                "dry-run | period=%s | eval_window=%s:%s | n_rows=%d | template=%s",
                period,
                period_data.eval_start,
                period_data.eval_end,
                len(period_data.eval_returns),
                period_data.template_config_path,
            )
        return 0

    for period in periods:
        period_data = period_data_map[period]
        for seed in seeds:
            for kappa in kappas:
                reference_df, reference_extras = _simulate_execution_rule(
                    period_data,
                    transaction_cost=float(kappa),
                    seed=seed,
                    mode="full_rebalance",
                    delta=None,
                    eta_db=None,
                )
                for delta in delta_grid:
                    for eta_db in eta_grid:
                        candidate_label = _candidate_label(delta=delta, eta_db=eta_db)
                        result_dir = (
                            raw_output_root
                            / period
                            / f"kappa_{_format_kappa(kappa)}"
                            / f"delta_{_format_delta(delta)}__eta_{_format_eta(eta_db)}"
                            / f"seed_{seed}"
                        )
                        if _result_complete(result_dir, save_trace=save_trace):
                            LOGGER.info(
                                "skip existing | period=%s kappa=%s delta=%.2f eta_db=%.2f seed=%d",
                                period,
                                _format_kappa(kappa),
                                delta,
                                eta_db,
                                seed,
                            )
                            continue

                        LOGGER.info(
                            "run | period=%s kappa=%s delta=%.2f eta_db=%.2f seed=%d",
                            period,
                            _format_kappa(kappa),
                            delta,
                            eta_db,
                            seed,
                        )
                        selected_df, selected_extras = _simulate_execution_rule(
                            period_data,
                            transaction_cost=float(kappa),
                            seed=seed,
                            mode="deadband_partial",
                            delta=float(delta),
                            eta_db=float(eta_db),
                        )
                        row = _build_result_row(
                            architecture_spec=architecture_spec,
                            period=period,
                            seed=seed,
                            kappa=float(kappa),
                            delta=float(delta),
                            eta_db=float(eta_db),
                            selected_df=selected_df,
                            reference_df=reference_df,
                            selected_arm_label=candidate_label,
                            reference_arm_label=reference_arm_label,
                            result_dir=result_dir,
                            near_flat_threshold=near_flat_threshold,
                            suppression_ratio=suppression_ratio,
                        )
                        meta = {
                            "experiment_name": str(cfg.get("experiment_name", "forecasting_workshop_arch_deadband_partial")),
                            "architecture_spec_path": str(architecture_spec_path.resolve()),
                            "shared_target_config_path": str(shared_target_config_path.resolve()),
                            "template_config_path": str(period_data.template_config_path.resolve()),
                            "period": period,
                            "eval_start": period_data.eval_start,
                            "eval_end": period_data.eval_end,
                            "seed": int(seed),
                            "kappa": float(kappa),
                            "delta": float(delta),
                            "eta_db": float(eta_db),
                            "selection_policy": dict(cfg.get("selection_policy") or {}),
                            "shared_target_metadata": period_data.metadata,
                            "reference_behavior": reference_arm_label,
                            "candidate_behavior": candidate_label,
                            "reference_summary": reference_extras,
                            "candidate_summary": selected_extras,
                        }
                        _write_result_bundle(
                            result_dir,
                            row,
                            selected_df=selected_df,
                            reference_df=reference_df,
                            meta=meta,
                            save_trace=save_trace,
                        )

    LOGGER.info("deadband partial candidate grid complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
