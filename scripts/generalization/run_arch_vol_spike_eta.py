#!/usr/bin/env python3
"""
Independent non-RL volatility-spike-eta comparator runner.

Assumptions documented here on purpose:

1. This runner does not use RL target replay. It consumes the shared deterministic target mapping
   built by `scripts/generalization/build_shared_targets.py`.
2. The frozen forecast source and target construction are identical across all volatility-spike
   candidates. Only the execution layer changes.
3. The state variable is a deterministic relative volatility spike score, not a learned signal:
   `spike_t = sigma_t / (sigma_ref_t + eps)` where `sigma_t` is target-weighted rolling volatility
   and `sigma_ref_t` is a rolling median reference.
4. This redesigned family uses a triggered fixed eta gate:
   if `spike_t <= trigger`, execute full rebalance; otherwise execute a fixed partial step `eta_low`.
5. This step writes the full validation and final candidate grid plus a full-rebalance reference.
   Champion selection is intentionally deferred to a later validation-only step.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_arch_deadband_partial as deadband


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = deadband.REPO_ROOT / "configs" / "generalization" / "arch_vol_spike_eta.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent non-RL volatility-spike-eta comparator grid.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the vol-spike comparator YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned candidate grid without executing it.")
    parser.add_argument(
        "--period",
        action="append",
        choices=["validation", "final"],
        help="Optional period subset. May be passed multiple times.",
    )
    return parser.parse_args()


def _format_trigger(value: float) -> str:
    return f"{float(value):.2f}"


def _format_eta_low(value: float) -> str:
    return f"{float(value):.3f}"


def _candidate_label(*, trigger: float, eta_low: float, lookback_sigma: int, lookback_ref: int) -> str:
    return (
        f"vol_spike_trigger{_format_trigger(trigger)}_etaLow{_format_eta_low(eta_low)}"
        f"_lb{int(lookback_sigma)}_ref{int(lookback_ref)}"
    )


def _compute_volatility_state(
    period_data: deadband.PreparedPeriodData,
    *,
    lookback_sigma: int,
    lookback_ref: int,
    eps: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if int(lookback_sigma) <= 1:
        raise ValueError("lookback_sigma must be greater than 1.")
    if int(lookback_ref) < int(lookback_sigma):
        raise ValueError("lookback_ref must be greater than or equal to lookback_sigma.")

    arithmetic_returns_df = np.expm1(period_data.eval_returns).astype(np.float64)
    rolling_asset_vol = arithmetic_returns_df.rolling(window=int(lookback_sigma), min_periods=int(lookback_sigma)).std(ddof=0)
    rolling_asset_vol = rolling_asset_vol.bfill().ffill()
    if rolling_asset_vol.isna().any().any():
        raise ValueError("Volatility proxy contains NaN values after deterministic backfill/ffill.")

    sigma_t = (period_data.eval_targets * rolling_asset_vol).sum(axis=1).astype(np.float64)
    sigma_t = sigma_t.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if sigma_t.isna().any():
        raise ValueError("Scalar volatility proxy contains NaN values after deterministic fill.")

    ref_min_periods = max(int(lookback_sigma), int(lookback_ref) // 3)
    sigma_ref_t = sigma_t.rolling(window=int(lookback_ref), min_periods=ref_min_periods).median().bfill().ffill()
    sigma_ref_t = sigma_ref_t.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if sigma_ref_t.isna().any():
        raise ValueError("Rolling sigma reference contains NaN values after deterministic fill.")

    spike_t = sigma_t / (sigma_ref_t + float(eps))
    spike_t = spike_t.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if spike_t.isna().any():
        raise ValueError("Relative volatility spike contains NaN values after deterministic fill.")

    return rolling_asset_vol, sigma_t, sigma_ref_t, spike_t


def _simulate_execution_rule(
    period_data: deadband.PreparedPeriodData,
    *,
    transaction_cost: float,
    seed: int,
    mode: str,
    trigger: float | None,
    eta_low: float | None,
    lookback_sigma: int,
    lookback_ref: int,
    eps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns_df = period_data.eval_returns
    target_df = period_data.eval_targets
    arithmetic_returns = np.expm1(returns_df.to_numpy(dtype=np.float64))
    target_arr = target_df.to_numpy(dtype=np.float64)
    _, sigma_series, sigma_ref_series, spike_series = _compute_volatility_state(
        period_data,
        lookback_sigma=int(lookback_sigma),
        lookback_ref=int(lookback_ref),
        eps=float(eps),
    )
    sigma_arr = sigma_series.to_numpy(dtype=np.float64)
    sigma_ref_arr = sigma_ref_series.to_numpy(dtype=np.float64)
    spike_arr = spike_series.to_numpy(dtype=np.float64)

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
    sigma_ts: list[float] = []
    sigma_ref_ts: list[float] = []
    spike_ts: list[float] = []
    intervention_flags: list[bool] = []

    for step_idx, date in enumerate(returns_df.index):
        w_target = deadband._normalize_simplex(target_arr[step_idx])
        turnover_target = deadband.turnover_l1(prev_weights, w_target)

        if mode == "full_rebalance":
            eta_t = 1.0
            lambda_t = float(sigma_arr[step_idx])
            w_exec = w_target.copy()
            intervened = True
        elif mode == "vol_spike_eta":
            if trigger is None or eta_low is None:
                raise ValueError("vol_spike_eta mode requires trigger and eta_low")
            spike_t = max(float(spike_arr[step_idx]), 0.0)
            if spike_t <= float(trigger):
                eta_t = 1.0
                intervened = False
            else:
                eta_t = float(eta_low)
                intervened = eta_t < 1.0 - 1e-12
            w_exec = deadband._normalize_simplex((1.0 - eta_t) * prev_weights + eta_t * w_target)
            lambda_t = spike_t
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        arithmetic_returns_t = arithmetic_returns[step_idx]
        portfolio_return = float(np.dot(w_exec, arithmetic_returns_t))
        portfolio_return_target = float(np.dot(w_target, arithmetic_returns_t))
        turnover_exec = deadband.turnover_l1(prev_weights, w_exec)
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
        sigma_ts.append(float(sigma_arr[step_idx]))
        sigma_ref_ts.append(float(sigma_ref_arr[step_idx]))
        spike_ts.append(float(spike_arr[step_idx]))
        intervention_flags.append(bool(intervened))

        prev_weights = w_exec

    metrics = deadband.compute_metrics(
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
    trace_df = deadband.trace_dict_to_frame(
        trace,
        eval_id=f"{period_data.period}__{mode}",
        run_id=f"vol_spike_eta__{period_data.period}",
        model_type="vol_spike_eta",
        seed=int(seed),
    )
    extras = {
        "metrics": metrics.to_dict(),
        "mean_sigma_proxy": float(np.mean(sigma_ts)) if sigma_ts else None,
        "mean_sigma_ref": float(np.mean(sigma_ref_ts)) if sigma_ref_ts else None,
        "mean_spike": float(np.mean(spike_ts)) if spike_ts else None,
        "mean_eta_t": float(np.mean(eta_ts)) if eta_ts else None,
        "activation_rate": float(np.mean(intervention_flags)) if intervention_flags else None,
        "lookback_sigma": int(lookback_sigma),
        "lookback_ref": int(lookback_ref),
    }
    return trace_df, extras


def _build_result_row(
    *,
    architecture_spec: deadband.ArchitectureSpec,
    period: str,
    seed: int,
    kappa: float,
    trigger: float,
    eta_low: float,
    lookback_sigma: int,
    lookback_ref: int,
    selected_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    selected_arm_label: str,
    reference_arm_label: str,
    result_dir: Path,
    near_flat_threshold: float,
    suppression_ratio: float,
    selected_extras: dict[str, Any],
) -> dict[str, Any]:
    sharpe_exec_selected = deadband._compute_sharpe(selected_df["net_return_lin"])
    sharpe_exec_reference = deadband._compute_sharpe(reference_df["net_return_lin"])
    sharpe_target_selected = deadband._compute_sharpe(selected_df["net_return_lin_target"])
    sharpe_target_reference = deadband._compute_sharpe(reference_df["net_return_lin_target"])
    delta_exec = float(sharpe_exec_selected - sharpe_exec_reference)
    delta_target = float(sharpe_target_selected - sharpe_target_reference)
    disagreement_type = deadband._derive_disagreement_type(
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
        "trigger": float(trigger),
        "eta_low": float(eta_low),
        "lookback_sigma": int(lookback_sigma),
        "lookback_ref": int(lookback_ref),
        "selected_arm": selected_arm_label,
        "reference_arm": reference_arm_label,
        "sharpe_exec_net": sharpe_exec_selected,
        "sharpe_target_net": sharpe_target_selected,
        "turnover_exec": deadband._safe_mean(selected_df["turnover_exec"]),
        "turnover_target": deadband._safe_mean(selected_df["turnover_target"]),
        "disagreement_type": disagreement_type,
        "delta_vs_reference_exec": delta_exec,
        "delta_vs_reference_target": delta_target,
        "zero_cost_near_flat_flag": bool(abs(delta_exec) <= float(near_flat_threshold)) if float(kappa) == 0.0 else None,
        "reference_sharpe_exec_net": sharpe_exec_reference,
        "reference_sharpe_target_net": sharpe_target_reference,
        "reference_turnover_exec": deadband._safe_mean(reference_df["turnover_exec"]),
        "reference_turnover_target": deadband._safe_mean(reference_df["turnover_target"]),
        "tracking_error_l2": deadband._safe_mean(selected_df["tracking_error_l2"]),
        "final_path_gap": deadband._safe_last_gap(selected_df["equity_net_lin"], selected_df["equity_net_lin_target"]),
        "cost_exec": deadband._safe_sum(selected_df["cost"]),
        "cost_target": deadband._safe_sum(selected_df["cost_target"]),
        "cagr_exec": deadband._compute_cagr(selected_df["equity_net_lin"]),
        "mdd_exec": deadband._compute_max_drawdown(selected_df["equity_net_lin"]),
        "mean_eta_t": selected_extras.get("mean_eta_t"),
        "mean_spike": selected_extras.get("mean_spike"),
        "activation_rate": selected_extras.get("activation_rate"),
        "mean_sigma_proxy": selected_extras.get("mean_sigma_proxy"),
        "steps": int(len(selected_df)),
        "selection_status": "grid_only_no_champion",
        "result_dir": str(result_dir.resolve()),
        "selected_trace_path": str((result_dir / "selected_trace.parquet").resolve()),
        "reference_trace_path": str((result_dir / "reference_trace.parquet").resolve()),
        "shared_target_independent_from_rl_replay": True,
        "run_completed_at": deadband.datetime.now(deadband.timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config_path = Path(args.config).resolve()
    cfg = deadband._load_yaml(config_path)

    architecture_spec_path = deadband._resolve_path(config_path, str(cfg["architecture_spec"]))
    architecture_spec = deadband._load_architecture_spec(architecture_spec_path)
    shared_target_config_path = deadband._resolve_path(config_path, str(cfg["shared_target_config"]))

    execution_cfg = cfg.get("execution", {}) or {}
    reference_cfg = cfg.get("reference", {}) or {}
    raw_output_root = deadband._resolve_path(config_path, str((cfg.get("paths", {}) or {})["raw_output_root"]))

    periods = deadband._iter_periods(cfg, args)
    seeds = [int(seed) for seed in (execution_cfg.get("seeds") or [0])]
    kappas = [float(kappa) for kappa in (execution_cfg.get("kappas") or [])]
    trigger_grid = [float(value) for value in (execution_cfg.get("trigger_grid") or [])]
    eta_low_grid = [float(value) for value in (execution_cfg.get("eta_low_grid") or [])]
    lookback_sigma = int(execution_cfg.get("lookback_sigma", 20))
    lookback_ref = int(execution_cfg.get("lookback_ref", 60))
    eps = float(execution_cfg.get("eps", 1e-8))
    save_trace = bool(execution_cfg.get("save_trace", True))
    near_flat_threshold = float(execution_cfg.get("near_flat_threshold", 0.005))
    suppression_ratio = float(execution_cfg.get("suppression_ratio", 0.25))
    offline = bool(execution_cfg.get("offline", True))
    reference_arm_label = str(reference_cfg.get("arm_label", "full_rebalance_baseline"))

    if not periods:
        raise ValueError("Config must define at least one period.")
    if not kappas or not trigger_grid or not eta_low_grid:
        raise ValueError("Config must define non-empty kappas, trigger_grid, and eta_low_grid.")

    period_data_map = {}
    for period in periods:
        period_data_map[period] = deadband._prepare_period_data(shared_target_config_path, period=period, offline=offline)

    plan_count = len(periods) * len(seeds) * len(kappas) * len(trigger_grid) * len(eta_low_grid)
    LOGGER.info(
        "Prepared volatility-spike eta plan with %d candidate runs across periods=%s, kappas=%s, trigger_grid=%s, eta_low_grid=%s",
        plan_count,
        periods,
        kappas,
        trigger_grid,
        eta_low_grid,
    )

    if args.dry_run:
        for period in periods:
            period_data = period_data_map[period]
            LOGGER.info(
                "dry-run | period=%s | eval_window=%s:%s | n_rows=%d | template=%s | lookback_sigma=%d | lookback_ref=%d",
                period,
                period_data.eval_start,
                period_data.eval_end,
                len(period_data.eval_returns),
                period_data.template_config_path,
                lookback_sigma,
                lookback_ref,
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
                    trigger=None,
                    eta_low=None,
                    lookback_sigma=lookback_sigma,
                    lookback_ref=lookback_ref,
                    eps=eps,
                )
                for trigger in trigger_grid:
                    for eta_low in eta_low_grid:
                        candidate_label = _candidate_label(
                            trigger=trigger,
                            eta_low=eta_low,
                            lookback_sigma=lookback_sigma,
                            lookback_ref=lookback_ref,
                        )
                        result_dir = (
                            raw_output_root
                            / period
                            / f"kappa_{deadband._format_kappa(kappa)}"
                            / f"trigger_{_format_trigger(trigger)}__etaLow_{_format_eta_low(eta_low)}"
                            / f"seed_{seed}"
                        )
                        if deadband._result_complete(result_dir, save_trace=save_trace):
                            LOGGER.info(
                                "skip existing | period=%s kappa=%s trigger=%.2f eta_low=%.3f seed=%d",
                                period,
                                deadband._format_kappa(kappa),
                                trigger,
                                eta_low,
                                seed,
                            )
                            continue

                        LOGGER.info(
                            "run | period=%s kappa=%s trigger=%.2f eta_low=%.3f seed=%d",
                            period,
                            deadband._format_kappa(kappa),
                            trigger,
                            eta_low,
                            seed,
                        )
                        selected_df, selected_extras = _simulate_execution_rule(
                            period_data,
                            transaction_cost=float(kappa),
                            seed=seed,
                            mode="vol_spike_eta",
                            trigger=float(trigger),
                            eta_low=float(eta_low),
                            lookback_sigma=lookback_sigma,
                            lookback_ref=lookback_ref,
                            eps=eps,
                        )
                        row = _build_result_row(
                            architecture_spec=architecture_spec,
                            period=period,
                            seed=seed,
                            kappa=float(kappa),
                            trigger=float(trigger),
                            eta_low=float(eta_low),
                            lookback_sigma=lookback_sigma,
                            lookback_ref=lookback_ref,
                            selected_df=selected_df,
                            reference_df=reference_df,
                            selected_arm_label=candidate_label,
                            reference_arm_label=reference_arm_label,
                            result_dir=result_dir,
                            near_flat_threshold=near_flat_threshold,
                            suppression_ratio=suppression_ratio,
                            selected_extras=selected_extras,
                        )
                        meta = {
                            "experiment_name": str(cfg.get("experiment_name", "forecasting_workshop_arch_vol_spike_eta")),
                            "architecture_spec_path": str(architecture_spec_path.resolve()),
                            "shared_target_config_path": str(shared_target_config_path.resolve()),
                            "template_config_path": str(period_data.template_config_path.resolve()),
                            "period": period,
                            "eval_start": period_data.eval_start,
                            "eval_end": period_data.eval_end,
                            "seed": int(seed),
                            "kappa": float(kappa),
                            "trigger": float(trigger),
                            "eta_low": float(eta_low),
                            "lookback_sigma": int(lookback_sigma),
                            "lookback_ref": int(lookback_ref),
                            "eps": float(eps),
                            "selection_policy": dict(cfg.get("selection_policy") or {}),
                            "shared_target_metadata": period_data.metadata,
                            "volatility_proxy_rule": "target_weighted_mean_of_rolling_asset_volatility",
                            "relative_spike_rule": "sigma_over_rolling_median_sigma",
                            "reference_behavior": reference_arm_label,
                            "candidate_behavior": candidate_label,
                            "reference_summary": reference_extras,
                            "candidate_summary": selected_extras,
                        }
                        deadband._write_result_bundle(
                            result_dir,
                            row,
                            selected_df=selected_df,
                            reference_df=reference_df,
                            meta=meta,
                            save_trace=save_trace,
                        )

    LOGGER.info("volatility-spike eta candidate grid complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
