from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class EventMicroConfig:
    horizon: int = 800
    seeds: int = 20
    phi: float = 0.9
    latent_noise_std: float = 0.55
    shock_prob: float = 0.04
    shock_std: float = 0.9
    logit_bias: float = 0.0
    state_scale: float = 1.0
    shock_logit_scale: float = 0.85
    baseline_noise_std: float = 0.03
    reactive_coefficient: float = 0.35
    reactive_noise_std: float = 0.03
    smoother_lambda: float = 0.85
    smoother_noise_std: float = 0.03
    noisy_noise_std: float = 0.08
    threshold_tau: float = 0.5
    friction_grid: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.05, 0.1, 0.25, 0.5, 1.0))
    tp_reward: float = 1.0
    fp_penalty: float = 1.0
    fn_penalty: float = 0.0
    initial_action: int = 0
    logloss_eps: float = 1e-6
    scenario_id: str = "event_micro_q2_switching_v1"

    def seed_list(self) -> list[int]:
        return list(range(int(self.seeds)))


def _normalize_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def load_config(path: str | Path) -> EventMicroConfig:
    payload = yaml.safe_load(Path(path).read_text())
    payload = payload or {}
    normalized = dict(payload)
    if "friction_grid" in normalized:
        normalized["friction_grid"] = _normalize_tuple(normalized["friction_grid"])
    return EventMicroConfig(**normalized)


def generate_event_stream(horizon: int, seed: int, config: EventMicroConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    z = np.zeros(int(horizon), dtype=np.float64)
    shock_flags = np.zeros(int(horizon), dtype=np.int64)
    shock_values = np.zeros(int(horizon), dtype=np.float64)

    stationary_std = float(config.latent_noise_std) / max(np.sqrt(max(1.0 - config.phi**2, 1e-6)), 1e-6)
    z[0] = stationary_std * rng.normal()

    for idx in range(int(horizon)):
        if rng.random() < float(config.shock_prob):
            shock_flags[idx] = 1
            shock_values[idx] = float(config.shock_std) * rng.normal()
        if idx == 0:
            continue
        z[idx] = float(config.phi) * z[idx - 1] + float(config.latent_noise_std) * rng.normal()

    logits = float(config.logit_bias) + float(config.state_scale) * z + float(config.shock_logit_scale) * shock_values
    q_true = _sigmoid(logits)
    y = rng.binomial(1, q_true, size=int(horizon)).astype(np.int64)

    return {
        "z_state": z,
        "shock_flags": shock_flags,
        "shock_values": shock_values,
        "q_true": q_true,
        "y": y,
    }
