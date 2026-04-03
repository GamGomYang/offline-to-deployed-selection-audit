#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    save_lbip_model,
)
from prl.train import build_signal_features, prepare_market_and_features

DEFAULT_KAPPAS = (0.0, 0.0005, 0.001)
DEFAULT_CS = (0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cost-Calibrated TA-LBIP (CC-TA-LBIP).")
    parser.add_argument("--config", type=str, required=True, help="YAML config.")
    parser.add_argument("--out", type=str, required=True, help="Output root.")
    parser.add_argument("--kappas", nargs="+", type=float, default=list(DEFAULT_KAPPAS), help="Kappa grid.")
    parser.add_argument("--cs", nargs="+", type=float, default=list(DEFAULT_CS), help="Cost-scale c grid.")
    parser.add_argument("--offline", action="store_true", help="Use cache only / offline mode.")
    parser.add_argument("--skip-selection", action="store_true", help="Skip validation c-selection artifact.")
    return parser.parse_args()


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


def _resolve_fit_config(cfg: dict[str, Any]) -> LBIPConfig:
    env_cfg = cfg.get("env", {}) or {}
    base_cfg = cfg.get("cctalibp", {}) or {}
    return LBIPConfig(
        window_size=int(env_cfg["L"]),
        ridge_alpha=float(base_cfg.get("ridge_alpha", 30.0)),
        fit_passes=int(base_cfg.get("fit_passes", 2)),
        training_eta=1.0,
        covariance_lookback=int(base_cfg.get("covariance_lookback", 252)),
        covariance_history_min=int(base_cfg.get("covariance_history_min", 30)),
        mean_variance_risk_aversion=float(base_cfg.get("mean_variance_risk_aversion", 10.0)),
        include_prev_weights=bool(base_cfg.get("include_prev_weights", True)),
        target_mode="mean_variance",
        anchor_strength=0.0,
        equal_weight_shrink=float(base_cfg.get("equal_weight_shrink", 0.0)),
        log_clip=float(base_cfg.get("log_clip", 1e-8)),
        eps=float(base_cfg.get("eps", 1e-12)),
    )


def _resolve_eval_config(base_config: LBIPConfig, *, c_value: float, kappa: float) -> LBIPConfig:
    return LBIPConfig(
        **{
            **base_config.__dict__,
            "target_mode": "anchored_mean_variance",
            "anchor_strength": float(c_value) * float(kappa),
            "training_eta": 1.0,
        }
    )


def _metric_row(
    *,
    experiment_name: str,
    kappa: float,
    c_value: float,
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
        "arm": "c_sweep",
        "arm_dir": arm_dir,
        "run_id": run_id,
        "model_type": "cc_ta_lbip",
        "c_value": float(c_value),
        "effective_penalty": float(c_value) * float(kappa),
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


def _write_aggregate(metrics_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    grouped = (
        metrics_df.groupby(["kappa", "c_value"], as_index=False)
        .agg(
            n_runs=("sharpe_net_lin", "count"),
            median_sharpe=("sharpe_net_lin", "median"),
            iqr_sharpe=("sharpe_net_lin", lambda x: float(pd.Series(x).quantile(0.75) - pd.Series(x).quantile(0.25))),
            median_turnover_exec=("avg_turnover_exec", "median"),
            collapse_rate=("collapse_flag_any", lambda x: float(pd.Series(x).mean())),
        )
        .sort_values(["kappa", "c_value"])
        .reset_index(drop=True)
    )
    grouped.to_csv(out_root / "aggregate.csv", index=False)
    return grouped


def _write_selection(aggregate_df: pd.DataFrame, out_root: Path, kappas: list[float]) -> dict[str, Any]:
    pos_kappas = [float(v) for v in kappas if float(v) > 0.0]
    rows = []
    c_values = sorted(pd.unique(aggregate_df["c_value"]).tolist())
    baseline_c = 0.0 if 0.0 in c_values else float(c_values[0])
    for c_value in c_values:
        row = {"c_value": float(c_value)}
        score_vals = []
        for kappa in pos_kappas:
            cur = aggregate_df[(aggregate_df["c_value"] == c_value) & (aggregate_df["kappa"] == kappa)].iloc[0]
            base = aggregate_df[(aggregate_df["c_value"] == baseline_c) & (aggregate_df["kappa"] == kappa)].iloc[0]
            score_vals.append(float(cur["median_sharpe"]))
            row[f"sharpe_kappa_{kappa:g}"] = float(cur["median_sharpe"])
            row[f"delta_sharpe_vs_c0_kappa_{kappa:g}"] = float(cur["median_sharpe"] - base["median_sharpe"])
            row[f"turnover_kappa_{kappa:g}"] = float(cur["median_turnover_exec"])
            row[f"effective_penalty_kappa_{kappa:g}"] = float(c_value) * float(kappa)
        row["score_mean_sharpe_pos_kappa"] = float(np.mean(score_vals)) if score_vals else float("nan")
        if (aggregate_df["kappa"] == 0.0).any():
            row["kappa0_sharpe"] = float(aggregate_df[(aggregate_df["c_value"] == c_value) & (aggregate_df["kappa"] == 0.0)].iloc[0]["median_sharpe"])
        else:
            row["kappa0_sharpe"] = float("nan")
        rows.append(row)
    rows_df = pd.DataFrame(rows).sort_values("c_value").reset_index(drop=True)
    best_score = float(rows_df["score_mean_sharpe_pos_kappa"].max())
    threshold = 0.95 * best_score if best_score > 0 else best_score / 0.95 if best_score < 0 else 0.0
    rows_df["qualifies"] = rows_df["score_mean_sharpe_pos_kappa"] >= threshold
    qualifying = rows_df[rows_df["qualifies"]].copy()
    if qualifying.empty:
        selected_c = float(rows_df.loc[rows_df["score_mean_sharpe_pos_kappa"].idxmax(), "c_value"])
    else:
        positive_qualifying = qualifying[qualifying["c_value"] > 0.0]
        if positive_qualifying.empty:
            selected_c = float(qualifying.sort_values(["c_value", "score_mean_sharpe_pos_kappa"], ascending=[True, False]).iloc[0]["c_value"])
        else:
            selected_c = float(positive_qualifying.sort_values(["c_value", "score_mean_sharpe_pos_kappa"], ascending=[True, False]).iloc[0]["c_value"])
    rows_df["selected"] = rows_df["c_value"] == selected_c
    payload = {
        "selection_rule": "smallest positive c within 95% of best positive-cost validation score",
        "baseline_c": baseline_c,
        "best_score": best_score,
        "threshold": threshold,
        "selected_c": selected_c,
        "rows": rows_df.to_dict(orient="records"),
    }
    sel_dir = out_root / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "validation_c_selection.json").write_text(json.dumps(payload, indent=2))
    rows_df.to_csv(sel_dir / "validation_c_selection.csv", index=False)
    return payload


def _write_selected_vs_c0(aggregate_df: pd.DataFrame, selected_c: float, out_root: Path) -> None:
    rows = []
    sel = aggregate_df[aggregate_df["c_value"] == selected_c].set_index("kappa")
    base = aggregate_df[aggregate_df["c_value"] == 0.0].set_index("kappa")
    for kappa in sorted(pd.unique(aggregate_df["kappa"]).tolist()):
        rows.append({
            "kappa": float(kappa),
            "selected_c": float(selected_c),
            "selected_sharpe": float(sel.loc[kappa, "median_sharpe"]),
            "baseline_c0_sharpe": float(base.loc[kappa, "median_sharpe"]),
            "delta_sharpe": float(sel.loc[kappa, "median_sharpe"] - base.loc[kappa, "median_sharpe"]),
            "selected_turnover": float(sel.loc[kappa, "median_turnover_exec"]),
            "baseline_c0_turnover": float(base.loc[kappa, "median_turnover_exec"]),
            "effective_penalty": float(selected_c) * float(kappa),
        })
    pd.DataFrame(rows).to_csv(out_root / "selected_vs_c0.csv", index=False)


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
    signal_features, _ = build_signal_features(market, config=cfg)

    train_returns = slice_frame(market.returns, dates["train_start"], dates["train_end"])
    train_volatility = slice_frame(features.volatility, dates["train_start"], dates["train_end"])
    train_signals = slice_frame(signal_features, dates["train_start"], dates["train_end"]) if signal_features is not None else None
    train_returns, train_volatility, train_signals = _align_returns_vol(train_returns, train_volatility, train_signals)

    eval_returns = slice_frame(market.returns, dates["test_start"], dates["test_end"])
    eval_volatility = slice_frame(features.volatility, dates["test_start"], dates["test_end"])
    eval_signals = slice_frame(signal_features, dates["test_start"], dates["test_end"]) if signal_features is not None else None
    eval_returns, eval_volatility, eval_signals = _align_returns_vol(eval_returns, eval_volatility, eval_signals)

    fit_config = _resolve_fit_config(cfg)
    experiment_name = str(output_cfg.get("experiment_name", config_path.stem))
    kappas = [float(v) for v in args.kappas]
    c_values = [float(v) for v in args.cs]

    fit_root = out_root / "fit"
    fit_root.mkdir(parents=True, exist_ok=True)
    model = fit_lbip_model(
        train_returns,
        train_volatility,
        signal_features=train_signals,
        config=fit_config,
    )
    save_lbip_model(model, fit_root / "cc_talibp_model_seed0.npz")
    summary = {
        "target_mode_fit": str(fit_config.target_mode),
        "ridge_alpha": float(fit_config.ridge_alpha),
        "fit_passes": int(fit_config.fit_passes),
        "training_eta": float(fit_config.training_eta),
        "include_prev_weights": bool(fit_config.include_prev_weights),
        "mean_variance_risk_aversion": float(fit_config.mean_variance_risk_aversion),
        "equal_weight_shrink": float(fit_config.equal_weight_shrink),
        "obs_dim": int(model.obs_dim),
        "train_rows": int(model.train_rows),
    }
    (fit_root / "cc_talibp_summary_seed0.json").write_text(json.dumps(summary, indent=2))

    traces_root = out_root / "traces"
    traces_root.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []

    for c_value in c_values:
        c_tag = f"c_{float(c_value):g}"
        for kappa in kappas:
            eval_cfg = _resolve_eval_config(fit_config, c_value=float(c_value), kappa=float(kappa))
            run_id = f"{experiment_name}_cctalibp_c{c_value:g}_kappa{kappa:g}_seed0"
            trace_dir = traces_root / c_tag
            trace_dir.mkdir(parents=True, exist_ok=True)
            metrics, trace_df = evaluate_lbip_eta(
                model,
                eval_returns,
                eval_volatility,
                eta=1.0,
                transaction_cost=float(kappa),
                signal_features=eval_signals,
                config=eval_cfg,
                eval_id=f"kappa_{kappa:g}",
                run_id=run_id,
                seed=0,
            )
            trace_path = trace_dir / f"trace_kappa_{kappa:g}.csv"
            trace_df.to_csv(trace_path, index=False)
            metrics_rows.append(
                _metric_row(
                    experiment_name=experiment_name,
                    kappa=float(kappa),
                    c_value=float(c_value),
                    run_id=run_id,
                    arm_dir=c_tag,
                    trace_path=trace_path,
                    trace_df=trace_df,
                    metrics=metrics,
                )
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["kappa", "c_value"]).reset_index(drop=True)
    metrics_df.to_csv(out_root / "metrics.csv", index=False)
    aggregate_df = _write_aggregate(metrics_df, out_root)

    selected_c = None
    if not args.skip_selection:
        payload = _write_selection(aggregate_df, out_root, kappas)
        selected_c = float(payload["selected_c"])
        _write_selected_vs_c0(aggregate_df, selected_c, out_root)

    protocol = {
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "kappas": kappas,
        "c_grid": c_values,
        "baseline_short_name": "CC-TA-LBIP",
        "fit_config": summary,
        "selection_rule": "smallest positive c within 95% of best positive-cost validation score",
        "selected_c": selected_c,
    }
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2))

    print(f"WROTE_AGGREGATE={out_root / 'aggregate.csv'}")
    print(f"WROTE_METRICS={out_root / 'metrics.csv'}")
    if selected_c is not None:
        print(f"SELECTED_C={selected_c}")
        print(f"WROTE_SELECTION={out_root / 'selection' / 'validation_c_selection.json'}")
        print(f"WROTE_SELECTED_SUMMARY={out_root / 'selected_vs_c0.csv'}")


if __name__ == "__main__":
    main()
