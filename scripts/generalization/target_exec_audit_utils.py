from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ZERO_COST_NEAR_FLAT_THRESHOLD = 0.005
ZERO_COST_YELLOW_THRESHOLD = 0.01
RANK_TIE_TOL = 0.001
SUPPRESSION_RATIO = 0.25
SUPPRESSION_MIN_EXEC_DELTA = 0.005


@dataclass(frozen=True)
class PairAudit:
    delta_exec: float
    delta_tgt: float
    rank_exec: str
    rank_tgt: str
    sign_exec: str
    sign_tgt: str
    disagreement_type: str
    disagreement_strength: int


def kappa_sort_key(value: float) -> float:
    return float(value)


def kappa_label(value: float) -> str:
    if np.isclose(float(value), 0.0):
        return "0"
    if np.isclose(float(value), 1e-4):
        return "1e-4"
    if np.isclose(float(value), 2e-4):
        return "2e-4"
    if np.isclose(float(value), 5e-4):
        return "5e-4"
    if np.isclose(float(value), 1e-3):
        return "1e-3"
    if np.isclose(float(value), 2e-3):
        return "2e-3"
    return f"{float(value):g}"


def latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def display_universe_name(name: str) -> str:
    mapping = {
        "u27_current": "Current",
        "u27_alt_largecap": "Alt-LargeCap",
        "u27_sector_balanced": "Sector-Balanced",
        "u27_random_seed17": "Random-17",
    }
    return mapping.get(name, str(name).replace("_", "-"))


def display_architecture_name(name: str) -> str:
    mapping = {
        "arch_rl_selected": "RL-Selected",
        "arch_deadband_partial_champion": "Deadband-Champion",
        "arch_deadband_partial_runnerup": "Deadband-RunnerUp",
        "arch_vol_spike_eta_champion": "VolScaledEta-Champion",
        "arch_vol_spike_eta_runnerup": "VolScaledEta-RunnerUp",
        "arch_rule_eta_fixed": "Rule-EtaFixed",
        "arch_linear_prox": "Linear-Prox",
        "arch_threshold_rebalance": "Threshold-Rebalance",
    }
    return mapping.get(name, str(name).replace("_", "-"))


def _rank_label(delta: float, *, tie_tol: float = RANK_TIE_TOL) -> str:
    if not np.isfinite(delta) or abs(float(delta)) <= float(tie_tol):
        return "tie"
    return "a>b" if float(delta) > 0.0 else "a<b"


def _sign_label(delta: float, *, tie_tol: float = RANK_TIE_TOL) -> str:
    if not np.isfinite(delta) or abs(float(delta)) <= float(tie_tol):
        return "0"
    return "+" if float(delta) > 0.0 else "-"


def classify_pair(
    *,
    metric_exec_a: float,
    metric_exec_b: float,
    metric_tgt_a: float,
    metric_tgt_b: float,
    tie_tol: float = RANK_TIE_TOL,
    suppression_ratio: float = SUPPRESSION_RATIO,
    suppression_min_exec_delta: float = SUPPRESSION_MIN_EXEC_DELTA,
) -> PairAudit:
    delta_exec = float(metric_exec_a) - float(metric_exec_b)
    delta_tgt = float(metric_tgt_a) - float(metric_tgt_b)

    rank_exec = _rank_label(delta_exec, tie_tol=tie_tol)
    rank_tgt = _rank_label(delta_tgt, tie_tol=tie_tol)
    sign_exec = _sign_label(delta_exec, tie_tol=tie_tol)
    sign_tgt = _sign_label(delta_tgt, tie_tol=tie_tol)

    disagreement_type = "none"
    disagreement_strength = 0

    if sign_exec != "0" and sign_tgt != "0" and sign_exec != sign_tgt:
        disagreement_type = "sign_flip"
        disagreement_strength = 3
    elif rank_exec != rank_tgt:
        disagreement_type = "ranking_mismatch"
        disagreement_strength = 2
    elif sign_exec == sign_tgt and sign_exec != "0":
        if abs(delta_exec) >= float(suppression_min_exec_delta) and abs(delta_tgt) <= float(suppression_ratio) * abs(delta_exec):
            disagreement_type = "magnitude_only"
            disagreement_strength = 1

    return PairAudit(
        delta_exec=delta_exec,
        delta_tgt=delta_tgt,
        rank_exec=rank_exec,
        rank_tgt=rank_tgt,
        sign_exec=sign_exec,
        sign_tgt=sign_tgt,
        disagreement_type=disagreement_type,
        disagreement_strength=disagreement_strength,
    )


def zero_cost_near_flat_override(
    audit: PairAudit,
    *,
    kappa: float,
    near_flat_threshold: float = ZERO_COST_NEAR_FLAT_THRESHOLD,
) -> PairAudit:
    if not np.isclose(float(kappa), 0.0):
        return audit
    if abs(float(audit.delta_exec)) > float(near_flat_threshold):
        return audit
    if abs(float(audit.delta_tgt)) > float(near_flat_threshold):
        return audit
    return PairAudit(
        delta_exec=audit.delta_exec,
        delta_tgt=audit.delta_tgt,
        rank_exec="tie",
        rank_tgt="tie",
        sign_exec="0",
        sign_tgt="0",
        disagreement_type="none",
        disagreement_strength=0,
    )
