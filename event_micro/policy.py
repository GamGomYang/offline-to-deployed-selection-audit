from __future__ import annotations

import numpy as np


def threshold_policy(probabilities: np.ndarray, tau: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    return (probs >= float(tau)).astype(np.int64)


def hysteresis_policy(probabilities: np.ndarray, tau: float, delta: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return np.zeros(0, dtype=np.int64)

    actions = np.zeros(probs.size, dtype=np.int64)
    actions[0] = int(probs[0] >= float(tau))
    upper = float(tau) + float(delta)
    lower = float(tau) - float(delta)

    for idx in range(1, probs.size):
        if actions[idx - 1] == 0 and probs[idx] > upper:
            actions[idx] = 1
        elif actions[idx - 1] == 1 and probs[idx] < lower:
            actions[idx] = 0
        else:
            actions[idx] = actions[idx - 1]
    return actions
