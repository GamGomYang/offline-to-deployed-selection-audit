#!/usr/bin/env python3
"""
Minimal domain-general toy process for target-vs-executed evaluation mismatch.

This appendix-only script keeps the setup intentionally small:

1. A latent desired action evolves over time.
2. A frozen forecast signal is formed by adding noise to that latent target.
3. Two deterministic target-construction rules map the same forecast to proposed actions.
4. A frictional execution layer limits how quickly realized actions can move.
5. We evaluate the same pair of arms twice:
   - target-based: as if the proposal were realized directly
   - executed-based: after the frictional execution rule

The goal is not to create a new empirical claim. The goal is only to show that
the evaluation-object mismatch can arise in a generic decision process, not only
in the main finance-accounting pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from target_exec_audit_utils import classify_pair  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "generalization" / "toy_example.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the appendix-only toy target-vs-executed example.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the toy-example YAML config.")
    return parser.parse_args()


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _build_desired_action(cfg: dict) -> np.ndarray:
    desired_cfg = cfg["process"]["desired_action"]
    horizon = int(cfg["time_horizon"])
    t = np.arange(horizon, dtype=np.float64)
    desired = np.full(horizon, float(desired_cfg.get("base_level", 0.5)), dtype=np.float64)
    for component in desired_cfg.get("components", []):
        amplitude = float(component["amplitude"])
        period = float(component["period"])
        phase = float(component.get("phase", 0.0))
        kind = str(component.get("kind", "sin")).strip().lower()
        angle = 2.0 * np.pi * t / period + phase
        if kind == "cos":
            desired += amplitude * np.cos(angle)
        else:
            desired += amplitude * np.sin(angle)
    return np.clip(
        desired,
        float(desired_cfg.get("lower_clip", 0.02)),
        float(desired_cfg.get("upper_clip", 0.98)),
    )


def _build_forecast_signal(cfg: dict, desired_action: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    noise_std = float(cfg["process"]["forecast_noise_std"])
    signal = desired_action + noise_std * rng.normal(size=desired_action.shape[0])
    return np.clip(signal, 0.0, 1.0)


def _build_target_action(forecast_signal: np.ndarray, *, gain: float) -> np.ndarray:
    neutral = 0.5
    target = neutral + float(gain) * (forecast_signal - neutral)
    return np.clip(target, 0.0, 1.0)


def _step_cap_for_friction(cfg: dict, friction: float) -> float:
    exec_cfg = cfg["execution"]
    if np.isclose(float(friction), 0.0):
        return float(exec_cfg["zero_friction_step_cap"])
    return float(exec_cfg["positive_friction_base_step_cap"]) / (
        1.0 + float(exec_cfg["friction_scale"]) * float(friction)
    )


def _execute_with_step_cap(target_action: np.ndarray, *, step_cap: float) -> np.ndarray:
    executed = np.empty_like(target_action, dtype=np.float64)
    executed[0] = float(target_action[0])
    for idx in range(1, target_action.shape[0]):
        delta = float(target_action[idx]) - float(executed[idx - 1])
        executed[idx] = float(executed[idx - 1]) + float(np.clip(delta, -step_cap, step_cap))
    return np.clip(executed, 0.0, 1.0)


def _score_action(action: np.ndarray, desired_action: np.ndarray) -> float:
    # Utility is generic squared-error tracking to the latent desired action.
    return float(-np.mean((action - desired_action) ** 2))


def _interpretation_from_sign(sign_label: str) -> str:
    if sign_label == "+":
        return "tempered_better"
    if sign_label == "-":
        return "responsive_better"
    return "tie"


def build_results(cfg: dict) -> pd.DataFrame:
    desired_action = _build_desired_action(cfg)
    forecast_signal = _build_forecast_signal(cfg, desired_action)

    arm_cfg = cfg["arms"]
    responsive_target = _build_target_action(forecast_signal, gain=float(arm_cfg["responsive"]["gain"]))
    tempered_target = _build_target_action(forecast_signal, gain=float(arm_cfg["tempered"]["gain"]))

    rows: list[dict[str, object]] = []
    for friction in cfg["friction_grid"]:
        friction_value = float(friction)
        step_cap = _step_cap_for_friction(cfg, friction_value)
        responsive_exec = _execute_with_step_cap(responsive_target, step_cap=step_cap)
        tempered_exec = _execute_with_step_cap(tempered_target, step_cap=step_cap)

        target_score_responsive = _score_action(responsive_target, desired_action)
        target_score_tempered = _score_action(tempered_target, desired_action)
        executed_score_responsive = _score_action(responsive_exec, desired_action)
        executed_score_tempered = _score_action(tempered_exec, desired_action)

        audit = classify_pair(
            metric_exec_a=executed_score_tempered,
            metric_exec_b=executed_score_responsive,
            metric_tgt_a=target_score_tempered,
            metric_tgt_b=target_score_responsive,
        )

        rows.append(
            {
                "friction": friction_value,
                "step_cap": step_cap,
                "target_score_responsive": target_score_responsive,
                "target_score_tempered": target_score_tempered,
                "delta_target_tempered_minus_responsive": audit.delta_tgt,
                "executed_score_responsive": executed_score_responsive,
                "executed_score_tempered": executed_score_tempered,
                "delta_executed_tempered_minus_responsive": audit.delta_exec,
                "mean_target_shift_responsive": float(np.mean(np.abs(np.diff(responsive_target)))),
                "mean_target_shift_tempered": float(np.mean(np.abs(np.diff(tempered_target)))),
                "mean_target_exec_gap_responsive": float(np.mean(np.abs(responsive_exec - responsive_target))),
                "mean_target_exec_gap_tempered": float(np.mean(np.abs(tempered_exec - tempered_target))),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": audit.disagreement_strength,
                "interpretation_target": _interpretation_from_sign(audit.sign_tgt),
                "interpretation_executed": _interpretation_from_sign(audit.sign_exec),
                "zero_friction_agreement_flag": "yes" if np.isclose(friction_value, 0.0) and np.isclose(audit.delta_exec, audit.delta_tgt) else "no",
                "appendix_only": "yes",
            }
        )

    return pd.DataFrame(rows).sort_values("friction").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    output_csv = _resolve_path(config_path, str(cfg["outputs"]["csv"]))

    results_df = build_results(cfg)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"[toy-example] wrote {len(results_df)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
