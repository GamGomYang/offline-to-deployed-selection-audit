from __future__ import annotations

import numpy as np


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    return float(np.mean((probs - y) ** 2))


def log_loss_score(probabilities: np.ndarray, outcomes: np.ndarray, *, eps: float = 1e-6) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), float(eps), 1.0 - float(eps))
    y = np.asarray(outcomes, dtype=np.float64)
    loss = -(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
    return float(np.mean(loss))


def switching_indicators(actions: np.ndarray, *, initial_action: int = 0) -> np.ndarray:
    action_array = np.asarray(actions, dtype=np.int64)
    if action_array.size == 0:
        return np.zeros(0, dtype=np.float64)
    previous = np.concatenate([[int(initial_action)], action_array[:-1]])
    return (action_array != previous).astype(np.float64)


def evaluate_actions(
    actions: np.ndarray,
    outcomes: np.ndarray,
    *,
    friction: float,
    tp_reward: float = 1.0,
    fp_penalty: float = 1.0,
    fn_penalty: float = 0.0,
    initial_action: int = 0,
) -> dict[str, float]:
    action_array = np.asarray(actions, dtype=np.int64)
    y = np.asarray(outcomes, dtype=np.int64)

    switches = switching_indicators(action_array, initial_action=int(initial_action))
    true_positive = (action_array == 1) & (y == 1)
    false_positive = (action_array == 1) & (y == 0)
    false_negative = (action_array == 0) & (y == 1)

    utility = (
        float(tp_reward) * true_positive.astype(np.float64)
        - float(fp_penalty) * false_positive.astype(np.float64)
        - float(fn_penalty) * false_negative.astype(np.float64)
        - float(friction) * switches
    )
    n_switches = int(switches.sum())
    horizon = int(action_array.shape[0])
    switch_rate = float(n_switches / max(horizon - 1, 1))
    return {
        "deployed_utility": float(utility.mean()),
        "mean_switch_cost": float(float(friction) * switches.mean()) if horizon else 0.0,
        "n_switches": n_switches,
        "switch_rate": switch_rate,
    }
