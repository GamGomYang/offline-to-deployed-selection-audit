#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import subprocess
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

from build_universes import UniverseSpec, get_universe_spec, load_universe_specs
from prl.train import run_training
from step6_sanity import build_eval_context, run_eval_case


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "generalization" / "multi_universe.yaml"
DEFAULT_REQUIRED_RESULT_COLUMNS = (
    "universe_name",
    "period",
    "seed",
    "eta",
    "kappa",
    "sharpe_exec_net",
    "sharpe_target_net",
    "turnover_exec",
    "turnover_target",
    "cost_exec",
    "cost_target",
)


@dataclass(frozen=True)
class TemplateSet:
    train: Path
    validation: Path
    final: Path


@dataclass(frozen=True)
class UniversePaths:
    name: str
    raw_root: Path
    runtime_root: Path
    train_root: Path
    config_dir: Path
    cache_dir: Path
    train_config_path: Path
    validation_config_path: Path
    final_config_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-universe execution-aware support comparisons.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the multi-universe experiment YAML.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without executing them.")
    parser.add_argument(
        "--include-optional-random",
        action="store_true",
        help="Include optional appendix-only universes from the config, such as u27_random_seed17.",
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


def _to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


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


def _resolve_templates(config_path: Path, cfg: dict[str, Any]) -> TemplateSet:
    raw_templates = cfg.get("template_configs", {}) or {}
    missing = [key for key in ("train", "validation", "final") if key not in raw_templates]
    if missing:
        raise ValueError(f"template_configs must define train/validation/final, missing={missing}")
    return TemplateSet(
        train=_resolve_path(config_path, str(raw_templates["train"])),
        validation=_resolve_path(config_path, str(raw_templates["validation"])),
        final=_resolve_path(config_path, str(raw_templates["final"])),
    )


def _resolve_universe_names(cfg: dict[str, Any], *, include_optional_random: bool) -> list[str]:
    universes = [str(name) for name in (cfg.get("universes") or [])]
    if not universes:
        raise ValueError("Config must define at least one primary universe.")
    if include_optional_random or bool(cfg.get("include_optional_random", False)):
        universes.extend(str(name) for name in (cfg.get("optional_universes") or []))
    seen: set[str] = set()
    ordered: list[str] = []
    for name in universes:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _resolve_paths(cfg: dict[str, Any], config_path: Path, universe_name: str) -> UniversePaths:
    paths_cfg = cfg.get("paths", {}) or {}
    raw_output_root = _resolve_path(config_path, str(paths_cfg.get("raw_output_root")))
    runtime_root = _resolve_path(config_path, str(paths_cfg.get("runtime_root")))
    processed_root = _resolve_path(config_path, str(paths_cfg.get("processed_root")))

    raw_root = raw_output_root / universe_name
    runtime_universe_root = runtime_root / universe_name
    train_root = runtime_universe_root / "train_control"
    config_dir = raw_root / "configs"
    cache_dir = processed_root / universe_name
    return UniversePaths(
        name=universe_name,
        raw_root=raw_root,
        runtime_root=runtime_universe_root,
        train_root=train_root,
        config_dir=config_dir,
        cache_dir=cache_dir,
        train_config_path=config_dir / "train_snapshot.yaml",
        validation_config_path=config_dir / "validation_snapshot.yaml",
        final_config_path=config_dir / "final_snapshot.yaml",
    )


def _materialize_signal_path(cfg: dict[str, Any], template_path: Path) -> None:
    signals_cfg = cfg.get("signals", {}) or {}
    selected_path = signals_cfg.get("selected_signals_path")
    if not selected_path:
        return
    resolved = Path(selected_path)
    if not resolved.is_absolute():
        resolved = (template_path.resolve().parent / resolved).resolve()
    signals_cfg["selected_signals_path"] = str(resolved)
    cfg["signals"] = signals_cfg


def _apply_universe_to_template(
    template_cfg: dict[str, Any],
    *,
    template_path: Path,
    universe_spec: UniverseSpec,
    cache_dir: Path,
    stage_output_root: Path,
    preserve_default_processed_dir: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(template_cfg)
    data_cfg = cfg.setdefault("data", {})
    universe_cfg = cfg.setdefault("universe", {})
    output_cfg = cfg.setdefault("output", {})

    if preserve_default_processed_dir:
        processed_dir = Path(str(data_cfg.get("processed_dir", cache_dir)))
        if not processed_dir.is_absolute():
            processed_dir = (REPO_ROOT / processed_dir).resolve()
    else:
        processed_dir = cache_dir.resolve()

    data_cfg["processed_dir"] = str(processed_dir)
    data_cfg["universe_policy"] = "fixed_list"
    data_cfg["min_assets"] = int(universe_spec.ticker_count)

    universe_cfg["policy"] = "fixed_list"
    universe_cfg["min_assets"] = int(universe_spec.ticker_count)
    universe_cfg["fixed_asset_list"] = list(universe_spec.tickers)

    output_cfg["root"] = str(stage_output_root.resolve())
    cfg["config_path"] = str(template_path.resolve())

    _materialize_signal_path(cfg, template_path)
    return cfg


def _write_config_snapshot(path: Path, cfg: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))


def _cache_ready(cache_dir: Path) -> bool:
    required = ("prices.parquet", "returns.parquet", "data_manifest.json")
    return all((cache_dir / name).exists() for name in required)


def _build_cache(train_config_path: Path, train_cfg: dict[str, Any]) -> None:
    processed_dir = Path(str(train_cfg.get("data", {}).get("processed_dir", ""))).resolve()
    if _cache_ready(processed_dir):
        LOGGER.info("Cache already present for %s", processed_dir)
        return
    cmd = [
        sys.executable,
        str(PRL_ROOT / "scripts" / "build_cache.py"),
        "--config",
        str(train_config_path),
    ]
    test_end = str(train_cfg.get("dates", {}).get("test_end", "")).strip()
    if test_end:
        cmd.extend(["--end-date", test_end])
    LOGGER.info("Building cache with command: %s", " ".join(cmd))
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PRL_ROOT}{':' + existing_pythonpath if existing_pythonpath else ''}"
    )
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def _load_latest_run_metadata(model_root: Path, *, model_type: str, seed: int) -> dict[str, Any] | None:
    reports_dir = model_root / "reports"
    if not reports_dir.exists():
        return None
    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in reports_dir.glob("run_metadata_*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if str(data.get("model_type")) != str(model_type):
            continue
        if int(data.get("seed", -1)) != int(seed):
            continue
        candidates.append((str(data.get("created_at", "")), data))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_existing_model_path(model_root: Path, *, model_type: str, seed: int) -> Path | None:
    metadata = _load_latest_run_metadata(model_root, model_type=model_type, seed=seed)
    if metadata:
        artifact_paths = metadata.get("artifact_paths") or metadata.get("artifacts") or {}
        model_path = artifact_paths.get("model_path")
        if model_path:
            path = Path(str(model_path))
            if path.exists():
                return path
    candidate = model_root / "models" / f"{model_type}_seed{int(seed)}_final.zip"
    if candidate.exists():
        return candidate
    return None


def _ensure_model(
    *,
    train_cfg: dict[str, Any],
    train_config_path: Path,
    train_root: Path,
    model_type: str,
    seed: int,
    offline: bool,
) -> Path:
    existing = _resolve_existing_model_path(train_root, model_type=model_type, seed=seed)
    if existing is not None:
        LOGGER.info("Reusing existing model for seed=%s at %s", seed, existing)
        return existing

    LOGGER.info("Training control model for seed=%s", seed)
    model_path = run_training(
        config={**copy.deepcopy(train_cfg), "config_path": str(train_config_path.resolve())},
        model_type=model_type,
        seed=int(seed),
        output_dir=train_root / "models",
        reports_dir=train_root / "reports",
        logs_dir=train_root / "logs",
        force_refresh=False,
        offline=offline,
        cache_only=offline,
    )
    return model_path


def _result_files(result_dir: Path, *, save_trace: bool) -> tuple[Path, Path, Path, Path]:
    json_path = result_dir / "result.json"
    csv_path = result_dir / "result.csv"
    trace_path = result_dir / "trace.parquet"
    signature_path = result_dir / "env_signature.json"
    return json_path, csv_path, trace_path if save_trace else Path(""), signature_path


def _result_complete(result_dir: Path, *, save_trace: bool) -> bool:
    json_path, csv_path, trace_path, signature_path = _result_files(result_dir, save_trace=save_trace)
    required = [json_path, csv_path, signature_path]
    if save_trace:
        required.append(trace_path)
    return all(path.exists() for path in required)


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


def _build_result_row(
    *,
    universe_spec: UniverseSpec,
    period: str,
    seed: int,
    eta: float,
    kappa: float,
    metrics: Any,
    df: pd.DataFrame,
    result_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    sharpe_exec_net = (
        float(metrics.sharpe_net_lin) if getattr(metrics, "sharpe_net_lin", None) is not None else _compute_sharpe(df["net_return_lin"])
    )
    cagr_exec = (
        float(metrics.cagr_net_lin) if getattr(metrics, "cagr_net_lin", None) is not None else _compute_cagr(df["equity_net_lin"])
    )
    mdd_exec = (
        float(metrics.max_drawdown_net_lin)
        if getattr(metrics, "max_drawdown_net_lin", None) is not None
        else _compute_max_drawdown(df["equity_net_lin"])
    )
    row = {
        "universe_name": universe_spec.name,
        "evaluation_role": universe_spec.evaluation_role,
        "period": period,
        "seed": int(seed),
        "eta": float(eta),
        "kappa": float(kappa),
        "sharpe_exec_net": sharpe_exec_net,
        "sharpe_target_net": _compute_sharpe(df["net_return_lin_target"]),
        "turnover_exec": _safe_mean(df["turnover_exec"]),
        "turnover_target": _safe_mean(df["turnover_target"]),
        "tracking_error_l2": _safe_mean(df["tracking_error_l2"]) if "tracking_error_l2" in df.columns else None,
        "final_path_gap": (
            _safe_last_gap(df["equity_net_lin"], df["equity_net_lin_target"])
            if "equity_net_lin" in df.columns and "equity_net_lin_target" in df.columns
            else None
        ),
        "cost_exec": _safe_sum(df["cost"]) if "cost" in df.columns else None,
        "cost_target": _safe_sum(df["cost_target"]) if "cost_target" in df.columns else None,
        "cagr_exec": cagr_exec,
        "mdd_exec": mdd_exec,
        "steps": int(len(df)),
        "collapse_flag_any": bool(df.get("collapse_flag", pd.Series(False, index=df.index)).astype(bool).any()),
        "result_dir": str(result_dir.resolve()),
        "trace_path": str((result_dir / "trace.parquet").resolve()),
        "model_path": str(model_path.resolve()),
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
    }
    return row


def _write_result_row(result_dir: Path, row: dict[str, Any], *, df: pd.DataFrame, signature_info: dict[str, Any], save_trace: bool) -> None:
    _ensure_dir(result_dir)
    json_path, csv_path, trace_path, signature_path = _result_files(result_dir, save_trace=save_trace)
    json_path.write_text(json.dumps(row, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    if save_trace:
        df.to_parquet(trace_path, index=False)
    signature_path.write_text(json.dumps(signature_info, indent=2))


def _summarize_plan(
    *,
    universe_specs: list[UniverseSpec],
    seeds: list[int],
    etas: list[float],
    kappas: list[float],
    raw_output_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    eval_runs = len(universe_specs) * len(seeds) * len(etas) * len(kappas) * 2
    train_runs = len(universe_specs) * len(seeds)
    return {
        "status": "dry_run",
        "universe_count": len(universe_specs),
        "universes": [spec.name for spec in universe_specs],
        "seed_count": len(seeds),
        "eta_count": len(etas),
        "kappa_count": len(kappas),
        "train_runs": train_runs,
        "eval_runs": eval_runs,
        "raw_output_root": str(raw_output_root.resolve()),
        "runtime_root": str(runtime_root.resolve()),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    universe_spec_dir = _resolve_path(config_path, str(cfg.get("universe_spec_dir")))
    load_universe_specs(universe_spec_dir)

    templates = _resolve_templates(config_path, cfg)
    universe_names = _resolve_universe_names(cfg, include_optional_random=bool(args.include_optional_random))
    universe_specs = [get_universe_spec(name, universe_spec_dir) for name in universe_names]

    execution_cfg = cfg.get("execution", {}) or {}
    model_type = str(execution_cfg.get("model_type", "prl"))
    seeds = [int(seed) for seed in (execution_cfg.get("seeds") or [0, 1, 2])]
    etas = [float(eta) for eta in (execution_cfg.get("etas") or [1.0, 0.5])]
    kappas = [float(kappa) for kappa in (execution_cfg.get("kappas") or [0.0, 0.0005, 0.001])]
    offline = bool(execution_cfg.get("offline", True))
    max_steps = int(execution_cfg.get("max_steps", 0))
    save_trace = bool(execution_cfg.get("save_trace", True))

    base_train_cfg = _load_yaml(templates.train)
    paths_cfg = cfg.get("paths", {}) or {}
    raw_output_root = _resolve_path(config_path, str(paths_cfg.get("raw_output_root")))
    runtime_root = _resolve_path(config_path, str(paths_cfg.get("runtime_root")))
    preserve_default_for_current = bool(paths_cfg.get("preserve_default_processed_dir_for_current", True))

    if args.dry_run:
        print(
            json.dumps(
                _summarize_plan(
                    universe_specs=universe_specs,
                    seeds=seeds,
                    etas=etas,
                    kappas=kappas,
                    raw_output_root=raw_output_root,
                    runtime_root=runtime_root,
                ),
                indent=2,
            )
        )
        return 0

    for universe_spec in universe_specs:
        universe_paths = _resolve_paths(cfg, config_path, universe_spec.name)
        _ensure_dir(universe_paths.config_dir)
        _ensure_dir(universe_paths.raw_root)
        _ensure_dir(universe_paths.train_root)

        preserve_default_cache = preserve_default_for_current and universe_spec.name == "u27_current"
        train_cfg = _apply_universe_to_template(
            base_train_cfg,
            template_path=templates.train,
            universe_spec=universe_spec,
            cache_dir=universe_paths.cache_dir,
            stage_output_root=universe_paths.train_root,
            preserve_default_processed_dir=preserve_default_cache,
        )
        validation_cfg = _apply_universe_to_template(
            _load_yaml(templates.validation),
            template_path=templates.validation,
            universe_spec=universe_spec,
            cache_dir=universe_paths.cache_dir,
            stage_output_root=universe_paths.runtime_root / "validation",
            preserve_default_processed_dir=preserve_default_cache,
        )
        final_cfg = _apply_universe_to_template(
            _load_yaml(templates.final),
            template_path=templates.final,
            universe_spec=universe_spec,
            cache_dir=universe_paths.cache_dir,
            stage_output_root=universe_paths.runtime_root / "final",
            preserve_default_processed_dir=preserve_default_cache,
        )

        _write_config_snapshot(universe_paths.train_config_path, train_cfg)
        _write_config_snapshot(universe_paths.validation_config_path, validation_cfg)
        _write_config_snapshot(universe_paths.final_config_path, final_cfg)

        LOGGER.info("=== Universe %s ===", universe_spec.name)
        _build_cache(universe_paths.train_config_path, train_cfg)

        model_paths_by_seed: dict[int, Path] = {}
        for seed in seeds:
            model_paths_by_seed[int(seed)] = _ensure_model(
                train_cfg=train_cfg,
                train_config_path=universe_paths.train_config_path,
                train_root=universe_paths.train_root,
                model_type=model_type,
                seed=int(seed),
                offline=offline,
            )

        for period_name, eval_config_path in (
            ("validation", universe_paths.validation_config_path),
            ("final", universe_paths.final_config_path),
        ):
            for seed in seeds:
                missing_runs = []
                for kappa in kappas:
                    for eta in etas:
                        result_dir = (
                            universe_paths.raw_root
                            / period_name
                            / f"kappa_{kappa:g}"
                            / f"eta_{eta:g}"
                            / f"seed_{int(seed)}"
                        )
                        if not _result_complete(result_dir, save_trace=save_trace):
                            missing_runs.append((float(kappa), float(eta), result_dir))
                if not missing_runs:
                    LOGGER.info(
                        "Skipping %s period for universe=%s seed=%s; all eta/kappa results already exist.",
                        period_name,
                        universe_spec.name,
                        seed,
                    )
                    continue

                ctx = build_eval_context(
                    config_path=str(eval_config_path),
                    model_type=model_type,
                    seed=int(seed),
                    model_root=str(universe_paths.train_root),
                    offline=offline,
                    max_steps=max_steps,
                    model_path_arg=str(model_paths_by_seed[int(seed)]),
                    prefer_metadata_config=False,
                )

                for kappa, eta, result_dir in missing_runs:
                    LOGGER.info(
                        "Running universe=%s period=%s seed=%s eta=%s kappa=%s",
                        universe_spec.name,
                        period_name,
                        seed,
                        eta,
                        kappa,
                    )
                    metrics, _, df, signature_info = run_eval_case(
                        ctx,
                        eta_mode="fixed",
                        rebalance_eta=float(eta),
                        transaction_cost=float(kappa),
                        eval_tag=f"{universe_spec.name}__{period_name}__eta_{eta:g}__kappa_{kappa:g}",
                    )
                    row = _build_result_row(
                        universe_spec=universe_spec,
                        period=period_name,
                        seed=int(seed),
                        eta=float(eta),
                        kappa=float(kappa),
                        metrics=metrics,
                        df=df,
                        result_dir=result_dir,
                        model_path=model_paths_by_seed[int(seed)],
                    )
                    missing_columns = [column for column in DEFAULT_REQUIRED_RESULT_COLUMNS if column not in row]
                    if missing_columns:
                        raise RuntimeError(f"Result row missing required columns: {missing_columns}")
                    _write_result_row(result_dir, row, df=df, signature_info=signature_info, save_trace=save_trace)

    LOGGER.info("Multi-universe raw run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
