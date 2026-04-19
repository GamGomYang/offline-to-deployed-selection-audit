#!/usr/bin/env python3
"""
Shared deterministic target-construction module for the independent non-RL comparators.

Design intent:

1. The new comparators must *not* use RL target replay.
2. Both comparators must start from the same frozen forecast signal and the same deterministic
   target-construction rule.
3. Only the execution rule should differ across those comparators.

This module therefore reuses the existing frozen paper-control signal package when available,
builds the signal-state features deterministically from cached market data, aggregates those
signals into a single frozen score vector s_t, re-zscores that score cross-sectionally, and
maps it to a long-only fully-invested target via stable softmax.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRL_ROOT = REPO_ROOT / "prl-dow30"

for candidate in (str(PRL_ROOT),):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from prl.envs import stable_softmax
from prl.signals import cross_sectional_zscore
from prl.train import build_signal_features, prepare_market_and_features


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "generalization" / "shared_target_mapping.yaml"


@dataclass(frozen=True)
class SharedTargetBundle:
    period: str
    score_frame: pd.DataFrame
    target_frame: pd.DataFrame
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or describe the shared deterministic target mapping.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the shared-target YAML.")
    parser.add_argument(
        "--period",
        choices=["validation", "final"],
        default="final",
        help="Which template config period to build from.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print a short JSON description instead of materializing any external file.",
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


def _load_template_config(shared_cfg: dict[str, Any], config_path: Path, *, period: str) -> tuple[Path, dict[str, Any]]:
    templates = shared_cfg.get("template_configs", {}) or {}
    if period not in templates:
        raise ValueError(f"shared_target_mapping template_configs missing period={period}")
    template_path = _resolve_path(config_path, str(templates[period]))
    cfg = _load_yaml(template_path)
    cfg["config_path"] = str(template_path.resolve())
    return template_path, cfg


def _validate_forecast_source(shared_cfg: dict[str, Any], template_cfg: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    forecast_cfg = dict(shared_cfg.get("forecast_source", {}) or {})
    if not forecast_cfg:
        raise ValueError("shared_target_mapping.yaml must define forecast_source.")

    signal_names_cfg = [str(name) for name in (forecast_cfg.get("signal_names") or [])]
    snapshot_path = _resolve_path(config_path, str(forecast_cfg.get("signal_selection_snapshot")))
    snapshot_payload = json.loads(snapshot_path.read_text())
    snapshot_names = [str(name) for name in (snapshot_payload.get("selected_signals") or [])]

    template_signals = dict(template_cfg.get("signals", {}) or {})
    template_names = [str(name) for name in (template_signals.get("signal_names") or [])]

    if signal_names_cfg and snapshot_names and signal_names_cfg != snapshot_names:
        raise ValueError(
            "Configured signal_names do not match the frozen signal-selection snapshot: "
            f"config={signal_names_cfg}, snapshot={snapshot_names}"
        )
    if snapshot_names and template_names and snapshot_names != template_names:
        raise ValueError(
            "Template signal_names do not match the frozen signal-selection snapshot: "
            f"template={template_names}, snapshot={snapshot_names}"
        )
    if str(template_signals.get("selection_policy", "")).strip().lower() != str(forecast_cfg.get("selection_policy", "")).strip().lower():
        raise ValueError(
            "Template selection_policy must match the shared forecast source selection policy: "
            f"template={template_signals.get('selection_policy')}, shared={forecast_cfg.get('selection_policy')}"
        )

    resolved = dict(forecast_cfg)
    resolved["signal_selection_snapshot"] = str(snapshot_path.resolve())
    resolved["resolved_signal_names"] = snapshot_names or signal_names_cfg or template_names
    return resolved


def _prepare_signal_state(template_cfg: dict[str, Any], *, offline: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    signal_features, signal_spec = build_signal_features(market, config=template_cfg)
    if signal_features is None or signal_features.empty:
        raise ValueError("No signal-state features could be built for the shared deterministic target mapping.")
    return signal_features, signal_spec


def _aggregate_signal_scores(signal_features: pd.DataFrame) -> pd.DataFrame:
    signal_names = list(pd.Index(signal_features.columns.get_level_values(0)).unique())
    if not signal_names:
        raise ValueError("signal_features has no signal-name level to aggregate.")

    blocks = [signal_features.xs(signal_name, axis=1, level=0).astype(np.float64) for signal_name in signal_names]
    score_frame = sum(blocks) / float(len(blocks))
    return score_frame.astype(np.float64)


def _apply_score_processing(score_frame: pd.DataFrame, shared_cfg: dict[str, Any]) -> pd.DataFrame:
    proc_cfg = shared_cfg.get("score_processing", {}) or {}
    out = score_frame.astype(np.float64).copy()
    if bool(proc_cfg.get("cross_sectional_zscore", True)):
        out = cross_sectional_zscore(out)
    clip_abs = proc_cfg.get("clip_abs_zscore")
    if clip_abs is not None:
        out = out.clip(lower=-float(clip_abs), upper=float(clip_abs))
    return out


def _row_to_weights(values: pd.Series, *, softmax_scale: float) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return np.full(arr.shape[0], 1.0 / arr.shape[0], dtype=np.float64)

    safe = arr.copy()
    safe[~valid] = np.nanmedian(safe[valid]) if valid.any() else 0.0
    weights = stable_softmax(safe, scale=float(softmax_scale)).astype(np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(arr.shape[0], 1.0 / arr.shape[0], dtype=np.float64)
    return weights / total


def build_shared_target_bundle(shared_config_path: str | Path, *, period: str) -> SharedTargetBundle:
    config_path = Path(shared_config_path).resolve()
    shared_cfg = _load_yaml(config_path)
    _template_path, template_cfg = _load_template_config(shared_cfg, config_path, period=period)
    forecast_cfg = _validate_forecast_source(shared_cfg, template_cfg, config_path=config_path)
    signal_features, signal_spec = _prepare_signal_state(template_cfg, offline=bool(shared_cfg.get("execution", {}).get("offline", True)))

    raw_score_frame = _aggregate_signal_scores(signal_features)
    processed_score_frame = _apply_score_processing(raw_score_frame, shared_cfg)

    target_cfg = shared_cfg.get("target_mapping", {}) or {}
    if str(target_cfg.get("rule", "stable_softmax")).strip().lower() != "stable_softmax":
        raise ValueError("The current shared target builder only supports rule=stable_softmax.")
    softmax_scale = float(target_cfg.get("softmax_scale", 1.0))
    target_frame = pd.DataFrame(
        [_row_to_weights(processed_score_frame.loc[idx], softmax_scale=softmax_scale) for idx in processed_score_frame.index],
        index=processed_score_frame.index,
        columns=processed_score_frame.columns,
        dtype=np.float64,
    )

    metadata = {
        "shared_target_config_path": str(config_path),
        "period": period,
        "forecast_source_family": str(forecast_cfg.get("family")),
        "signal_selection_snapshot": str(forecast_cfg.get("signal_selection_snapshot")),
        "signal_names": list(forecast_cfg.get("resolved_signal_names") or []),
        "selection_policy": str(forecast_cfg.get("selection_policy")),
        "score_aggregation": str(forecast_cfg.get("score_aggregation")),
        "score_processing": dict(shared_cfg.get("score_processing", {}) or {}),
        "target_mapping": dict(target_cfg),
        "signal_spec": signal_spec,
        "n_rows": int(target_frame.shape[0]),
        "n_assets": int(target_frame.shape[1]),
        "independent_from_rl_target_replay": True,
    }
    return SharedTargetBundle(
        period=period,
        score_frame=processed_score_frame,
        target_frame=target_frame,
        metadata=metadata,
    )


def build_shared_score_frame(shared_config_path: str | Path, *, period: str) -> pd.DataFrame:
    return build_shared_target_bundle(shared_config_path, period=period).score_frame


def build_shared_target_frame(shared_config_path: str | Path, *, period: str) -> pd.DataFrame:
    return build_shared_target_bundle(shared_config_path, period=period).target_frame


def describe_bundle(bundle: SharedTargetBundle) -> dict[str, Any]:
    return {
        **bundle.metadata,
        "score_index_start": str(bundle.score_frame.index.min()),
        "score_index_end": str(bundle.score_frame.index.max()),
        "target_weight_sum_min": float(bundle.target_frame.sum(axis=1).min()),
        "target_weight_sum_max": float(bundle.target_frame.sum(axis=1).max()),
        "target_weight_min": float(bundle.target_frame.min().min()),
        "target_weight_max": float(bundle.target_frame.max().max()),
    }


def main() -> int:
    args = parse_args()
    bundle = build_shared_target_bundle(args.config, period=args.period)
    if args.describe:
        print(json.dumps(describe_bundle(bundle), indent=2))
    else:
        print(json.dumps(describe_bundle(bundle), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
