from __future__ import annotations

import numpy as np

from .data import EventMicroConfig


FORECASTER_IDS = (
    "calibrated_baseline",
    "reactive_sharp",
    "lagged_smoother",
    "noisy_heuristic",
)


def _clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def generate_forecasts(q_true: np.ndarray, *, seed: int, config: EventMicroConfig) -> dict[str, np.ndarray]:
    q_array = np.asarray(q_true, dtype=np.float64)
    rng = np.random.default_rng(10_000 + int(seed))

    baseline = _clip_probs(q_array + float(config.baseline_noise_std) * rng.normal(size=q_array.shape[0]))

    delta = np.diff(q_array, prepend=q_array[0])
    reactive = _clip_probs(
        q_array
        + float(config.reactive_coefficient) * delta
        + float(config.reactive_noise_std) * rng.normal(size=q_array.shape[0])
    )

    smoother = np.zeros_like(q_array, dtype=np.float64)
    smoother[0] = q_array[0]
    for idx in range(1, q_array.shape[0]):
        smoother[idx] = (
            float(config.smoother_lambda) * smoother[idx - 1]
            + (1.0 - float(config.smoother_lambda)) * q_array[idx]
            + float(config.smoother_noise_std) * rng.normal()
        )
    smoother = _clip_probs(smoother)

    noisy = _clip_probs(q_array + float(config.noisy_noise_std) * rng.normal(size=q_array.shape[0]))

    return {
        "calibrated_baseline": baseline,
        "reactive_sharp": reactive,
        "lagged_smoother": smoother,
        "noisy_heuristic": noisy,
    }
