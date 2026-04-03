#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prl.data import slice_frame
from prl.linear_information_parity import (
    LBIPConfig,
    evaluate_lbip_eta,
    fit_lbip_model,
    fit_summary_dict,
    save_lbip_model,
    save_lbip_summary,
)
from prl.train import build_signal_features, prepare_market_and_features


DEFAULT_ETAS = (1.0, 0.5, 0.2, 0.1, 0.082, 0.05, 0.02)
DEFAULT_KAPPAS = (0.0, 0.0005, 0.001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Linear Baseline for Information Parity (LBIP) eta-frontier validation.")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    parser.add_argument("--out", type=str, required=True, help="Output root.")
    parser.add_argument("--kappas", nargs="+", type=float, default=list(DEFAULT_KAPPAS), help="Kappa grid.")
    parser.add_argument("--etas", nargs="+", type=float, default=list(DEFAULT_ETAS), help="Eta grid.")
    parser.add_argument("--offline", action="store_true", help="Use cache only / offline mode.")
    parser.add_argument("--skip-selection", action="store_true", help="Skip selection artifact build.")
    return parser.parse_args()


def _format_kappa(value: float) -> str:
    return f"{float(value):g}"


def _format_eta(value: float) -> str:
    return f"{float(value):g}"


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


def _compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    arr = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    final = float(arr[-1])
    if final <= 0.0:
        return float("nan")
    years = float(arr.size) / float(periods_per_year)
    if years <= 0.0:
        return 0.0
    return float(final ** (1.0 / years) - 1.0)


def _metric_row(
    *,
    experiment_name: str,
    kappa: float,
    eta: float,
    run_id: str,
    arm_dir: str,
    trace_path: Path,
    trace_df: pd.DataFrame,
    metrics: Any,
) -> dict[str, Any]:
    collapse_series = trace_df.get("collapse_flag")
    collapse_flag_any = bool(pd.Series(collapse_series).fillna(False).astype(bool).any()) if collapse_series is not None else False
    collapse_count = int(pd.Series(collapse_series).fillna(False).astype(bool).sum()) if collapse_series is not None else 0
    misalignment = pd.to_numeric(trace_df["net_return_lin"], errors="coerce") - pd.to_numeric(
        trace_df["net_return_lin_target"], errors="coerce"
    )
    return {
        "experiment_name": experiment_name,
        "kappa": float(kappa),
        "seed": 0,
        "arm": "eta_sweep",
        "arm_dir": arm_dir,
        "run_id": run_id,
        "model_type": "lbip",
        "eta_mode": "fixed",
        "eta_requested": float(eta),
        "eta": float(eta),
        "rule_vol_a": np.nan,
        "n_steps": int(len(trace_df)),
        "sharpe_net_lin": float(metrics.sharpe_net_lin) if metrics.sharpe_net_lin is not None else float("nan"),
        "cagr": _compute_cagr(trace_df["equity_net_lin"]),
        "maxdd": float(metrics.max_drawdown_net_lin) if metrics.max_drawdown_net_lin is not None else float("nan"),
        "avg_turnover_exec": float(metrics.avg_turnover_exec) if metrics.avg_turnover_exec is not None else float("nan"),
        "avg_turnover_target": float(metrics.avg_turnover_target) if metrics.avg_turnover_target is not None else float("nan"),
        "tracking_error_l2_mean": float(pd.to_numeric(trace_df["tracking_error_l2"], errors="coerce").mean()),
        "misalignment_gap_mean": float(pd.to_numeric(misalignment, errors="coerce").mean()),
        "collapse_flag_any": collapse_flag_any,
        "collapse_count": collapse_count,
        "trace_path": str(trace_path),
    }


def _resolve_lbip_config(cfg: dict[str, Any]) -> LBIPConfig:
    env_cfg = cfg.get("env", {}) or {}
    lbip_cfg = cfg.get("lbip", {}) or {}
    return LBIPConfig(
        window_size=int(env_cfg["L"]),
        ridge_alpha=float(lbip_cfg.get("ridge_alpha", 10.0)),
        fit_passes=int(lbip_cfg.get("fit_passes", 2)),
        training_eta=float(lbip_cfg.get("training_eta", 0.082)),
        covariance_lookback=int(lbip_cfg.get("covariance_lookback", 252)),
        covariance_history_min=int(lbip_cfg.get("covariance_history_min", 30)),
        mean_variance_risk_aversion=float(lbip_cfg.get("mean_variance_risk_aversion", 10.0)),
        include_prev_weights=bool(lbip_cfg.get("include_prev_weights", True)),
        target_mode=str(lbip_cfg.get("target_mode", "mean_variance")),
        anchor_strength=float(lbip_cfg.get("anchor_strength", 0.0)),
        equal_weight_shrink=float(lbip_cfg.get("equal_weight_shrink", 0.0)),
        log_clip=float(lbip_cfg.get("log_clip", 1e-8)),
        eps=float(lbip_cfg.get("eps", 1e-12)),
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text())
    cfg["config_path"] = str(config_path)
    dates = cfg.get("dates", {}) or {}
    env_cfg = cfg.get("env", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    paper_mode = bool(data_cfg.get("paper_mode", False))
    require_cache_cfg = bool(data_cfg.get("require_cache", False))
    offline_cfg = bool(data_cfg.get("offline", False))
    offline = bool(args.offline or offline_cfg or paper_mode or require_cache_cfg)
    require_cache = bool(require_cache_cfg or paper_mode or offline)
    cache_only = bool(paper_mode or require_cache_cfg or offline_cfg or args.offline)

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

    lbip_config = _resolve_lbip_config(cfg)
    model = fit_lbip_model(
        train_returns,
        train_volatility,
        signal_features=train_signals,
        config=lbip_config,
    )

    fit_dir = out_root / "fit"
    save_lbip_model(model, fit_dir / "lbip_model_seed0.npz")
    save_lbip_summary(model, fit_dir / "lbip_summary_seed0.json")
    (fit_dir / "signal_spec_seed0.json").write_text(json.dumps(signal_spec, indent=2))

    experiment_name = str(output_cfg.get("experiment_name", config_path.stem))
    kappas = [float(v) for v in args.kappas]
    etas = [float(v) for v in args.etas]
    run_id = f"{experiment_name}_lbip_seed0"

    metrics_rows: list[dict[str, Any]] = []
    for kappa in kappas:
        for eta in etas:
            arm_dir = f"eta_{_format_eta(eta)}"
            seed_dir = out_root / f"kappa_{_format_kappa(kappa)}" / arm_dir / "seed_0"
            seed_dir.mkdir(parents=True, exist_ok=True)
            eval_id = f"{run_id}__kappa{kappa:.6g}__eta{eta:.6g}"
            metrics, trace_df = evaluate_lbip_eta(
                model,
                eval_returns,
                eval_volatility,
                eta=float(eta),
                transaction_cost=float(kappa),
                signal_features=eval_signals,
                config=lbip_config,
                eval_id=eval_id,
                run_id=run_id,
                seed=0,
            )
            trace_path = seed_dir / "trace.parquet"
            trace_df.to_parquet(trace_path, index=False)
            row = _metric_row(
                experiment_name=experiment_name,
                kappa=float(kappa),
                eta=float(eta),
                run_id=run_id,
                arm_dir=arm_dir,
                trace_path=trace_path,
                trace_df=trace_df,
                metrics=metrics,
            )
            pd.DataFrame([row]).to_csv(seed_dir / "metrics.csv", index=False)
            metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["kappa", "eta"]).reset_index(drop=True)
        metrics_df.to_csv(out_root / "metrics.csv", index=False)

    target_mode = str(lbip_config.target_mode).strip().lower()
    if target_mode == "anchored_mean_variance":
        baseline_name = "Anchored Linear Baseline for Information Parity"
        baseline_short_name = "ALBIP"
    else:
        baseline_name = "Linear Baseline for Information Parity"
        baseline_short_name = "LBIP"

    fit_summary = fit_summary_dict(model)
    fit_summary.update({
        "target_mode": str(lbip_config.target_mode),
        "anchor_strength": float(lbip_config.anchor_strength),
        "equal_weight_shrink": float(lbip_config.equal_weight_shrink),
    })

    protocol = {
        "config": str(config_path),
        "output_root": str(out_root),
        "experiment_name": experiment_name,
        "eval_window": {"start": dates["test_start"], "end": dates["test_end"]},
        "kappas": kappas,
        "etas": etas,
        "baseline_name": baseline_name,
        "baseline_short_name": baseline_short_name,
        "fit_summary": fit_summary,
        "signal_spec": signal_spec,
        "matched_definitions": {
            "primary_metric": "executed_path_sharpe_net_lin",
            "cost_definition": "kappa * executed_turnover",
            "long_only_fully_invested": True,
        },
    }
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2))

    subprocess.run([sys.executable, str(ROOT / "scripts" / "step6_build_reports.py"), "--root", str(out_root)], check=True)
    if not args.skip_selection:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "select_eta_from_validation.py"),
                "--root",
                str(out_root),
                "--output-dir",
                str(out_root / "selection"),
                "--baseline-eta",
                "1.0",
                "--positive-kappas",
                ",".join(str(v) for v in kappas if float(v) > 0.0),
                "--relative-threshold",
                "0.95",
            ],
            check=True,
        )

    print(f"WROTE_PROTOCOL={out_root / 'protocol.json'}")
    print(f"WROTE_AGGREGATE={out_root / 'aggregate.csv'}")
    if not args.skip_selection:
        print(f"WROTE_SELECTION={out_root / 'selection' / 'validation_eta_selection.json'}")


if __name__ == "__main__":
    main()
