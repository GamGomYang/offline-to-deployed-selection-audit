from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import build_result_row, mse_score, prepare_results_frame


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "forecast_eval" / "synthetic"
DEFAULT_CALIBRATION_REPORT_PATH = DEFAULT_OUTPUT_DIR / "calibration_report.csv"
DEFAULT_SELECTED_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "selected_config.json"

FRICTION_GRID = [0.0, 0.25, 0.5, 1.0]
CALIBRATION_SEEDS = list(range(0, 8))
REPORT_SEEDS = list(range(20, 40))
FORECASTER_IDS = [
    "naive_last",
    "moving_average",
    "linear_ar",
    "noisy_overreactive",
]
Q1_FORECASTER_ID = "noisy_overreactive"
Q2_INTERFACE_ID = "tempered"
LATENT_CLIP_LOWER = -1.5
LATENT_CLIP_UPPER = 1.5
DEFAULT_HORIZON = 240
LINEAR_AR_HISTORY = 50
FIXED_LATENT_JUMP_SCALE = 0.38
LATENT_JUMP_TIMESTEPS = (50, 100, 150, 200)
Q1_SCENARIO_ID = "synthetic_q1_clean_v2"
Q2_SCENARIO_ID = "synthetic_q2_stress_v2"
SELECTION_WARNING_NO_MONOTONE = "no_monotone_candidate_found_under_zero_gate"
SELECTION_WARNING_NO_GATE = "no_candidate_satisfied_zero_friction_gate"
SELECTION_WARNING_NO_STRICT_Q2_INCREASE = "no_strict_q2_disagreement_increase_found"


@dataclass(frozen=True)
class SyntheticConfig:
    moving_average_window: int
    overreactive_alpha: float
    overreactive_noise: float
    eta_base: float
    eta_friction_lambda: float = 0.0
    latent_process: str = "ar1_jumps"
    horizon: int = DEFAULT_HORIZON
    ar_phi: float = 0.88
    ar_noise_std: float = 0.10
    ar_clip_lower: float = LATENT_CLIP_LOWER
    ar_clip_upper: float = LATENT_CLIP_UPPER
    latent_jump_scale: float = FIXED_LATENT_JUMP_SCALE
    latent_jump_timesteps: tuple[int, ...] = LATENT_JUMP_TIMESTEPS
    block_levels_scale: float = 1.0
    block_noise_std: float = 0.08
    q1_forecaster_id: str = Q1_FORECASTER_ID
    q2_interface_id: str = Q2_INTERFACE_ID

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibration_grid() -> list[SyntheticConfig]:
    return [
        SyntheticConfig(
            moving_average_window=window,
            overreactive_alpha=alpha,
            overreactive_noise=noise,
            eta_base=eta_base,
            eta_friction_lambda=eta_friction_lambda,
        )
        for window, alpha, noise, eta_base, eta_friction_lambda in product(
            [5, 8, 12],
            [1.4, 1.8, 2.2, 2.6],
            [0.0, 0.03, 0.06],
            [0.20, 0.25, 0.30],
            [0.0, 0.5, 1.0, 1.5, 2.0],
        )
    ]


def q1_calibration_grid() -> list[SyntheticConfig]:
    return calibration_grid()


def q2_calibration_grid() -> list[SyntheticConfig]:
    return [
        SyntheticConfig(
            moving_average_window=window,
            overreactive_alpha=alpha,
            overreactive_noise=0.0,
            eta_base=eta_base,
            eta_friction_lambda=eta_friction_lambda,
            latent_process="block_levels",
            block_levels_scale=1.0,
            block_noise_std=0.08,
        )
        for window, alpha, eta_base, eta_friction_lambda in product(
            [5, 8],
            [1.1, 1.2, 1.4],
            [0.25, 0.30],
            [1.0, 1.5, 2.0],
        )
    ]


def config_id(config: SyntheticConfig) -> str:
    base = (
        f"proc{config.latent_process}_"
        f"w{config.moving_average_window}_"
        f"a{config.overreactive_alpha:.2f}_"
        f"n{config.overreactive_noise:.2f}_"
        f"eta{config.eta_base:.2f}_"
        f"lam{config.eta_friction_lambda:.2f}"
    )
    if config.latent_process == "block_levels":
        base = f"{base}_bs{config.block_levels_scale:.2f}_bn{config.block_noise_std:.2f}"
    return base


def config_from_dict(payload: dict[str, object]) -> SyntheticConfig:
    return SyntheticConfig(
        moving_average_window=int(payload["moving_average_window"]),
        overreactive_alpha=float(payload["overreactive_alpha"]),
        overreactive_noise=float(payload["overreactive_noise"]),
        eta_base=float(payload["eta_base"]),
        eta_friction_lambda=float(payload.get("eta_friction_lambda", 0.0)),
        latent_process=str(payload.get("latent_process", "ar1_jumps")),
        horizon=int(payload.get("horizon", DEFAULT_HORIZON)),
        ar_phi=float(payload.get("ar_phi", 0.88)),
        ar_noise_std=float(payload.get("ar_noise_std", 0.10)),
        ar_clip_lower=float(payload.get("ar_clip_lower", LATENT_CLIP_LOWER)),
        ar_clip_upper=float(payload.get("ar_clip_upper", LATENT_CLIP_UPPER)),
        latent_jump_scale=float(payload.get("latent_jump_scale", FIXED_LATENT_JUMP_SCALE)),
        latent_jump_timesteps=tuple(int(step) for step in payload.get("latent_jump_timesteps", LATENT_JUMP_TIMESTEPS)),
        block_levels_scale=float(payload.get("block_levels_scale", 1.0)),
        block_noise_std=float(payload.get("block_noise_std", 0.08)),
        q1_forecaster_id=str(payload.get("q1_forecaster_id", Q1_FORECASTER_ID)),
        q2_interface_id=str(payload.get("q2_interface_id", Q2_INTERFACE_ID)),
    )


def load_selected_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def save_selected_config(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def latent_state(seed: int, config: SyntheticConfig) -> np.ndarray:
    if config.latent_process == "block_levels":
        rng = np.random.default_rng(seed)
        levels = np.array([0.0, 0.55, -0.65, 0.85, -0.50, 0.15], dtype=np.float64) * float(config.block_levels_scale)
        block_size = max(1, config.horizon // len(levels))
        series = np.zeros(config.horizon, dtype=np.float64)
        for idx, level in enumerate(levels):
            start = idx * block_size
            end = config.horizon if idx == len(levels) - 1 else min((idx + 1) * block_size, config.horizon)
            series[start:end] = level + float(config.block_noise_std) * rng.normal(size=max(end - start, 0))
        return np.clip(series, config.ar_clip_lower, config.ar_clip_upper)

    rng = np.random.default_rng(seed)
    series = np.zeros(config.horizon, dtype=np.float64)
    jump_terms = np.zeros(config.horizon, dtype=np.float64)
    cumulative_jump = 0.0
    for step in config.latent_jump_timesteps:
        step_index = int(step)
        if step_index <= 0 or step_index >= config.horizon:
            continue
        direction = float(rng.choice([-1.0, 1.0]))
        magnitude = float(config.latent_jump_scale) * float(0.8 + 0.4 * rng.random())
        cumulative_jump += direction * magnitude
        jump_terms[step_index] = cumulative_jump
    for idx in range(1, config.horizon):
        series[idx] = config.ar_phi * series[idx - 1] + config.ar_noise_std * rng.normal() + jump_terms[idx]
    return np.clip(series, config.ar_clip_lower, config.ar_clip_upper)


def _fit_linear_ar(history: np.ndarray) -> tuple[float, float]:
    if history.shape[0] < 4:
        last_value = float(history[-1]) if history.shape[0] else 0.0
        return 0.0, last_value

    x = history[:-1]
    y = history[1:]
    denominator = float(np.dot(x, x))
    phi = 0.0 if denominator <= 1e-12 else float(np.dot(x, y) / denominator)
    intercept = float(np.mean(y - phi * x))
    return phi, intercept


def build_forecasts(latent: np.ndarray, seed: int, config: SyntheticConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    forecasts = {forecaster_id: np.zeros_like(latent, dtype=np.float64) for forecaster_id in FORECASTER_IDS}

    for idx in range(1, latent.shape[0]):
        history = latent[:idx]
        forecasts["naive_last"][idx] = history[-1]
        forecasts["moving_average"][idx] = float(history[max(0, idx - config.moving_average_window) :].mean())

        ar_history = history[-min(LINEAR_AR_HISTORY, history.shape[0]) :]
        phi, intercept = _fit_linear_ar(ar_history)
        forecasts["linear_ar"][idx] = intercept + phi * history[-1]

        prev_value = history[-2] if idx > 1 else 0.0
        overshoot = config.overreactive_alpha * (history[-1] - prev_value)
        forecasts["noisy_overreactive"][idx] = history[-1] + overshoot + config.overreactive_noise * rng.normal()

    return forecasts


def target_action(forecast: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    return np.clip(np.asarray(forecast, dtype=np.float64), config.ar_clip_lower, config.ar_clip_upper)


def responsive_execution(target: np.ndarray) -> np.ndarray:
    return np.array(target, copy=True)


def tempered_execution(target: np.ndarray, friction_level: float, config: SyntheticConfig) -> np.ndarray:
    target_array = np.asarray(target, dtype=np.float64)
    if np.isclose(float(friction_level), 0.0):
        return np.array(target_array, copy=True)

    effective_eta = float(config.eta_base) / (1.0 + float(config.eta_friction_lambda) * float(friction_level))
    effective_eta = float(np.clip(effective_eta, 0.0, 1.0))
    executed = np.zeros_like(target_array, dtype=np.float64)
    executed[0] = target_array[0]
    for idx in range(1, target_array.shape[0]):
        executed[idx] = (1.0 - effective_eta) * executed[idx - 1] + effective_eta * target_array[idx]
    return executed


def objective_metrics(path: np.ndarray, latent: np.ndarray, friction_level: float) -> tuple[float, float, float]:
    action_path = np.asarray(path, dtype=np.float64)
    latent_path = np.asarray(latent, dtype=np.float64)
    previous = np.concatenate([action_path[:1], action_path[:-1]])
    adjustments = np.abs(action_path - previous)
    penalties = float(friction_level) * adjustments
    losses = (action_path - latent_path) ** 2 + penalties
    return -float(np.mean(losses)), float(np.mean(penalties)), float(np.mean(adjustments))


def _scenario_id(question_id: str, config: SyntheticConfig) -> str:
    if question_id == "Q1" and config.latent_process == "ar1_jumps":
        return Q1_SCENARIO_ID
    if question_id == "Q2" and config.latent_process == "block_levels":
        return Q2_SCENARIO_ID
    return f"synthetic_{question_id.lower()}_{config.latent_process}_v2"


def build_q1_frame(config: SyntheticConfig, seeds: Iterable[int]) -> pd.DataFrame:
    q1_rows: list[dict[str, object]] = []
    scenario_id = _scenario_id("Q1", config)

    for seed in seeds:
        latent = latent_state(int(seed), config)
        forecast_map = build_forecasts(latent, int(seed), config)
        target_map = {forecaster_id: target_action(forecast, config) for forecaster_id, forecast in forecast_map.items()}

        q1_target = target_map[config.q1_forecaster_id]
        interface_targets = {
            "responsive": np.array(q1_target, copy=True),
            "tempered": np.array(q1_target, copy=True),
        }
        if len({array_hash(values) for values in interface_targets.values()}) != 1:
            raise AssertionError("Q1 forecast mismatch: interface arms must share the exact same target path.")

        q1_forecast_metric = mse_score(forecast_map[config.q1_forecaster_id], latent)
        for friction_level in FRICTION_GRID:
            target_metric, _, _ = objective_metrics(q1_target, latent, friction_level)
            execution_map = {
                "responsive": responsive_execution(q1_target),
                "tempered": tempered_execution(q1_target, friction_level, config),
            }
            for interface_id, executed in execution_map.items():
                executed_metric, realized_cost, realized_adjustment = objective_metrics(executed, latent, friction_level)
                q1_rows.append(
                    build_result_row(
                        question_id="Q1",
                        scenario_id=scenario_id,
                        domain="synthetic",
                        seed=int(seed),
                        forecaster_id=config.q1_forecaster_id,
                        interface_id=interface_id,
                        friction_level=float(friction_level),
                        forecast_metric=q1_forecast_metric,
                        target_metric=target_metric,
                        executed_metric=executed_metric,
                        realized_cost=realized_cost,
                        realized_turnover_or_adjustment=realized_adjustment,
                    )
                )

    return prepare_results_frame(q1_rows)


def build_q2_frame(config: SyntheticConfig, seeds: Iterable[int]) -> pd.DataFrame:
    q2_rows: list[dict[str, object]] = []
    scenario_id = _scenario_id("Q2", config)

    for seed in seeds:
        latent = latent_state(int(seed), config)
        forecast_map = build_forecasts(latent, int(seed), config)
        target_map = {forecaster_id: target_action(forecast, config) for forecaster_id, forecast in forecast_map.items()}

        for friction_level in FRICTION_GRID:
            for forecaster_id in FORECASTER_IDS:
                target = target_map[forecaster_id]
                executed = tempered_execution(target, friction_level, config)
                target_metric, _, _ = objective_metrics(target, latent, friction_level)
                executed_metric, realized_cost, realized_adjustment = objective_metrics(executed, latent, friction_level)
                q2_rows.append(
                    build_result_row(
                        question_id="Q2",
                        scenario_id=scenario_id,
                        domain="synthetic",
                        seed=int(seed),
                        forecaster_id=forecaster_id,
                        interface_id=config.q2_interface_id,
                        friction_level=float(friction_level),
                        forecast_metric=mse_score(forecast_map[forecaster_id], latent),
                        target_metric=target_metric,
                        executed_metric=executed_metric,
                        realized_cost=realized_cost,
                        realized_turnover_or_adjustment=realized_adjustment,
                    )
                )

    return prepare_results_frame(q2_rows)


def build_synthetic_frames(config: SyntheticConfig, seeds: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_q1_frame(config, seeds), build_q2_frame(config, seeds)


def _stderr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.shape[0] <= 1:
        return 0.0
    return float(numeric.std(ddof=1) / math.sqrt(numeric.shape[0]))


def _nondecreasing(values: list[float], *, tol: float = 1e-12) -> bool:
    return all(values[idx] <= values[idx + 1] + tol for idx in range(len(values) - 1))


def _nonincreasing(values: list[float], *, tol: float = 1e-12) -> bool:
    return all(values[idx] + tol >= values[idx + 1] for idx in range(len(values) - 1))


def _order_label(metric_a: float, metric_b: float, label_a: str, label_b: str, *, tol: float = 1e-12) -> str:
    delta = float(metric_a) - float(metric_b)
    if abs(delta) <= tol:
        return "tie"
    return f"{label_a}>{label_b}" if delta > 0.0 else f"{label_a}<{label_b}"


def _kendall_tau_from_ranks(forecast_ranks: np.ndarray, executed_ranks: np.ndarray) -> float:
    concordant = 0
    discordant = 0
    for idx, jdx in combinations(range(forecast_ranks.shape[0]), 2):
        sign_forecast = float(np.sign(forecast_ranks[idx] - forecast_ranks[jdx]))
        sign_executed = float(np.sign(executed_ranks[idx] - executed_ranks[jdx]))
        if sign_forecast == 0.0 and sign_executed == 0.0:
            concordant += 1
        elif sign_forecast == 0.0 or sign_executed == 0.0:
            continue
        elif sign_forecast == sign_executed:
            concordant += 1
        else:
            discordant += 1
    denominator = concordant + discordant
    if denominator == 0:
        return 1.0
    return float((concordant - discordant) / denominator)


def _spearman_rho_from_ranks(forecast_ranks: np.ndarray, executed_ranks: np.ndarray) -> float:
    if forecast_ranks.shape[0] <= 1:
        return 1.0
    if np.allclose(forecast_ranks, forecast_ranks[0]) or np.allclose(executed_ranks, executed_ranks[0]):
        return 1.0
    return float(np.corrcoef(forecast_ranks.astype(np.float64), executed_ranks.astype(np.float64))[0, 1])


def build_q1_gap_summary(q1_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_level = (
        q1_df.assign(abs_target_executed_gap=q1_df["target_executed_gap"].abs())
        .groupby(["seed", "friction_level"], as_index=False)["abs_target_executed_gap"]
        .mean()
        .rename(columns={"abs_target_executed_gap": "mean_abs_target_executed_gap"})
    )
    aggregate = (
        seed_level.groupby("friction_level", as_index=False)["mean_abs_target_executed_gap"]
        .agg(
            mean_abs_target_executed_gap="mean",
            median_abs_target_executed_gap="median",
            stderr_abs_target_executed_gap=_stderr,
            n_seeds="count",
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )
    return seed_level, aggregate


def build_q2_diagnostics(
    q2_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    disagreement_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    pairwise_flip_rows: list[dict[str, object]] = []

    for (seed, friction_level), group in q2_df.groupby(["seed", "friction_level"], sort=True):
        ordered = group.sort_values("forecaster_id").reset_index(drop=True)
        forecast_metrics = {
            str(row.forecaster_id): float(row.forecast_metric) for row in ordered.itertuples(index=False)
        }
        executed_metrics = {
            str(row.forecaster_id): float(row.executed_metric) for row in ordered.itertuples(index=False)
        }
        forecaster_ids = sorted(forecast_metrics)

        disagreement_count = 0
        total_pairs = 0
        for model_a, model_b in combinations(forecaster_ids, 2):
            forecast_order = _order_label(
                forecast_metrics[model_a],
                forecast_metrics[model_b],
                model_a,
                model_b,
            )
            executed_order = _order_label(
                executed_metrics[model_a],
                executed_metrics[model_b],
                model_a,
                model_b,
            )
            total_pairs += 1
            if forecast_order != executed_order:
                disagreement_count += 1
                pairwise_flip_rows.append(
                    {
                        "seed": int(seed),
                        "friction_level": float(friction_level),
                        "model_a": model_a,
                        "model_b": model_b,
                        "forecast_order": forecast_order,
                        "executed_order": executed_order,
                    }
                )

        disagreement_rows.append(
            {
                "seed": int(seed),
                "friction_level": float(friction_level),
                "pairwise_disagreement_rate": float(disagreement_count / total_pairs),
            }
        )

        forecast_ranks = ordered["rank_within_forecast_metric"].to_numpy(dtype=np.int64)
        executed_ranks = ordered["rank_within_executed_metric"].to_numpy(dtype=np.int64)
        correlation_rows.append(
            {
                "seed": int(seed),
                "friction_level": float(friction_level),
                "kendall_tau": _kendall_tau_from_ranks(forecast_ranks, executed_ranks),
                "spearman_rho": _spearman_rho_from_ranks(forecast_ranks, executed_ranks),
            }
        )

    disagreement_seed = pd.DataFrame(disagreement_rows).sort_values(["seed", "friction_level"]).reset_index(drop=True)
    disagreement_aggregate = (
        disagreement_seed.groupby("friction_level", as_index=False)["pairwise_disagreement_rate"]
        .agg(
            mean_pairwise_disagreement_rate="mean",
            median_pairwise_disagreement_rate="median",
            stderr_pairwise_disagreement_rate=_stderr,
            n_seeds="count",
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )

    correlation_seed = pd.DataFrame(correlation_rows).sort_values(["seed", "friction_level"]).reset_index(drop=True)
    correlation_aggregate = (
        correlation_seed.groupby("friction_level", as_index=False)
        .agg(
            mean_kendall_tau=("kendall_tau", "mean"),
            median_kendall_tau=("kendall_tau", "median"),
            stderr_kendall_tau=("kendall_tau", _stderr),
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
            stderr_spearman_rho=("spearman_rho", _stderr),
            n_seeds=("seed", "count"),
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )

    pairwise_flip_columns = [
        "seed",
        "friction_level",
        "model_a",
        "model_b",
        "forecast_order",
        "executed_order",
    ]
    pairwise_flips = (
        pd.DataFrame(pairwise_flip_rows, columns=pairwise_flip_columns)
        .sort_values(["friction_level", "seed", "model_a", "model_b"])
        .reset_index(drop=True)
    )
    return disagreement_seed, disagreement_aggregate, correlation_seed, correlation_aggregate, pairwise_flips


def _positive_friction_values(df: pd.DataFrame, value_column: str) -> list[float]:
    positive = (
        df.loc[df["friction_level"] > 0.0, ["friction_level", value_column]]
        .sort_values("friction_level")[value_column]
        .astype(float)
        .tolist()
    )
    return positive


def evaluate_candidate(config: SyntheticConfig, seeds: Iterable[int]) -> dict[str, object]:
    q1_df, q2_df = build_synthetic_frames(config, seeds)
    q1_seed, q1_agg = build_q1_gap_summary(q1_df)
    q2_dis_seed, q2_dis_agg, _, q2_corr_agg, _ = build_q2_diagnostics(q2_df)

    zero_q1_gap = float(
        q1_agg.loc[q1_agg["friction_level"] == 0.0, "mean_abs_target_executed_gap"].iloc[0]
    )
    zero_disagreement = float(
        q2_dis_agg.loc[q2_dis_agg["friction_level"] == 0.0, "mean_pairwise_disagreement_rate"].iloc[0]
    )
    zero_kendall = float(q2_corr_agg.loc[q2_corr_agg["friction_level"] == 0.0, "mean_kendall_tau"].iloc[0])
    zero_spearman = float(q2_corr_agg.loc[q2_corr_agg["friction_level"] == 0.0, "mean_spearman_rho"].iloc[0])

    zero_gate_pass = (
        np.isclose(zero_q1_gap, 0.0)
        and np.isclose(zero_disagreement, 0.0)
        and np.isclose(zero_kendall, 1.0)
        and np.isclose(zero_spearman, 1.0)
    )

    positive_disagreement_mean = float(
        q2_dis_seed.loc[q2_dis_seed["friction_level"] > 0.0, "pairwise_disagreement_rate"].mean()
    )
    positive_disagreement_min = float(
        q2_dis_agg.loc[q2_dis_agg["friction_level"] > 0.0, "mean_pairwise_disagreement_rate"].min()
    )
    positive_q1_gap_mean = float(
        q1_seed.loc[q1_seed["friction_level"] > 0.0, "mean_abs_target_executed_gap"].mean()
    )
    positive_mean_kendall_tau = float(
        q2_corr_agg.loc[q2_corr_agg["friction_level"] > 0.0, "mean_kendall_tau"].mean()
    )
    positive_mean_spearman_rho = float(
        q2_corr_agg.loc[q2_corr_agg["friction_level"] > 0.0, "mean_spearman_rho"].mean()
    )
    seed_std_positive_disagreement = float(
        q2_dis_seed.loc[q2_dis_seed["friction_level"] > 0.0, "pairwise_disagreement_rate"].std(ddof=1)
        if q2_dis_seed.loc[q2_dis_seed["friction_level"] > 0.0].shape[0] > 1
        else 0.0
    )

    monotone_q1_gap = _nondecreasing(
        _positive_friction_values(q1_agg, "mean_abs_target_executed_gap"),
    )
    monotone_q2_disagreement = _nondecreasing(
        _positive_friction_values(q2_dis_agg, "mean_pairwise_disagreement_rate"),
    )
    monotone_q2_kendall = _nonincreasing(
        _positive_friction_values(q2_corr_agg, "mean_kendall_tau"),
    )
    monotone_q2_spearman = _nonincreasing(
        _positive_friction_values(q2_corr_agg, "mean_spearman_rho"),
    )
    monotone_candidate = (
        monotone_q1_gap and monotone_q2_disagreement and monotone_q2_kendall and monotone_q2_spearman
    )

    return {
        "config_id": config_id(config),
        "moving_average_window": config.moving_average_window,
        "overreactive_alpha": config.overreactive_alpha,
        "overreactive_noise": config.overreactive_noise,
        "eta_base": config.eta_base,
        "eta_friction_lambda": config.eta_friction_lambda,
        "latent_jump_scale": config.latent_jump_scale,
        "zero_gate_pass": bool(zero_gate_pass),
        "zero_q1_gap": zero_q1_gap,
        "zero_q2_disagreement_rate": zero_disagreement,
        "zero_q2_kendall_tau": zero_kendall,
        "zero_q2_spearman_rho": zero_spearman,
        "positive_min_pairwise_disagreement_rate": positive_disagreement_min,
        "positive_mean_pairwise_disagreement_rate": positive_disagreement_mean,
        "positive_mean_kendall_tau": positive_mean_kendall_tau,
        "positive_mean_spearman_rho": positive_mean_spearman_rho,
        "positive_mean_abs_q1_gap": positive_q1_gap_mean,
        "positive_seed_std_pairwise_disagreement_rate": seed_std_positive_disagreement,
        "monotone_q1_gap": bool(monotone_q1_gap),
        "monotone_q2_disagreement": bool(monotone_q2_disagreement),
        "monotone_q2_kendall": bool(monotone_q2_kendall),
        "monotone_q2_spearman": bool(monotone_q2_spearman),
        "monotone_candidate": bool(monotone_candidate),
        "failed_benchmark": False,
        "failure_reason": "",
    }


def _selection_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["selection_priority"] = 99
    monotone_mask = ordered["zero_gate_pass"] & ordered["monotone_candidate"]
    gate_mask = ordered["zero_gate_pass"]
    ordered.loc[monotone_mask, "selection_priority"] = 0
    ordered.loc[gate_mask & ~monotone_mask, "selection_priority"] = 1
    return ordered


def select_candidate(report_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if report_df.empty:
        raise ValueError("Calibration report is empty; no synthetic candidates were evaluated.")

    zero_gate_pool = report_df.loc[report_df["zero_gate_pass"]].copy()
    selection_warning: str | None = None

    if zero_gate_pool.empty:
        ordered = _selection_sort_columns(report_df)
        selection_pool = ordered.copy()
        selection_warning = SELECTION_WARNING_NO_GATE
    else:
        selection_pool = zero_gate_pool.copy()

    selection_pool = selection_pool.sort_values(
        [
            "positive_min_pairwise_disagreement_rate",
            "positive_mean_pairwise_disagreement_rate",
            "positive_mean_kendall_tau",
            "positive_mean_spearman_rho",
            "monotone_candidate",
            "positive_mean_abs_q1_gap",
            "positive_seed_std_pairwise_disagreement_rate",
            "moving_average_window",
            "overreactive_alpha",
            "overreactive_noise",
            "eta_base",
            "eta_friction_lambda",
        ],
        ascending=[False, False, True, True, False, False, True, True, True, True, True, True],
    ).reset_index(drop=True)
    selected_row = selection_pool.iloc[0].to_dict()
    selected_config = SyntheticConfig(
        moving_average_window=int(selected_row["moving_average_window"]),
        overreactive_alpha=float(selected_row["overreactive_alpha"]),
        overreactive_noise=float(selected_row["overreactive_noise"]),
        eta_base=float(selected_row["eta_base"]),
        eta_friction_lambda=float(selected_row.get("eta_friction_lambda", 0.0)),
        latent_jump_scale=float(selected_row.get("latent_jump_scale", FIXED_LATENT_JUMP_SCALE)),
    )
    if selection_warning is None and not bool(selected_row["monotone_candidate"]):
        selection_warning = SELECTION_WARNING_NO_MONOTONE

    full_report = report_df.copy()
    full_report["selection_warning"] = ""
    full_report["selected"] = False
    selected_mask = full_report["config_id"] == config_id(selected_config)
    full_report.loc[selected_mask, "selected"] = True
    if selection_warning is not None:
        full_report.loc[selected_mask, "selection_warning"] = selection_warning

    payload = {
        "selected_config": selected_config.to_dict(),
        "selected_config_id": config_id(selected_config),
        "selection_warning": selection_warning,
        "selection_metrics": {
            "positive_min_pairwise_disagreement_rate": float(
                selected_row["positive_min_pairwise_disagreement_rate"]
            ),
            "positive_mean_pairwise_disagreement_rate": float(selected_row["positive_mean_pairwise_disagreement_rate"]),
            "positive_mean_kendall_tau": float(selected_row["positive_mean_kendall_tau"]),
            "positive_mean_spearman_rho": float(selected_row["positive_mean_spearman_rho"]),
            "positive_mean_abs_q1_gap": float(selected_row["positive_mean_abs_q1_gap"]),
            "positive_seed_std_pairwise_disagreement_rate": float(
                selected_row["positive_seed_std_pairwise_disagreement_rate"]
            ),
            "zero_gate_pass": bool(selected_row["zero_gate_pass"]),
            "monotone_candidate": bool(selected_row["monotone_candidate"]),
        },
        "calibration_seeds": CALIBRATION_SEEDS,
        "report_seeds": REPORT_SEEDS,
        "friction_grid": FRICTION_GRID,
    }
    return full_report, payload


def run_calibration(output_dir: str | Path, seeds: Iterable[int] = CALIBRATION_SEEDS) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = [evaluate_candidate(config, seeds) for config in calibration_grid()]
    report_df = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)
    report_df, payload = select_candidate(report_df)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path / DEFAULT_CALIBRATION_REPORT_PATH.name, index=False)
    save_selected_config(output_path / DEFAULT_SELECTED_CONFIG_PATH.name, payload)
    return report_df, payload


def evaluate_q1_candidate(config: SyntheticConfig, seeds: Iterable[int]) -> dict[str, object]:
    q1_df = build_q1_frame(config, seeds)
    q1_seed, q1_agg = build_q1_gap_summary(q1_df)
    zero_q1_gap = float(
        q1_agg.loc[q1_agg["friction_level"] == 0.0, "mean_abs_target_executed_gap"].iloc[0]
    )
    positive_mean_abs_q1_gap = float(
        q1_seed.loc[q1_seed["friction_level"] > 0.0, "mean_abs_target_executed_gap"].mean()
    )
    positive_seed_std_gap = float(
        q1_seed.loc[q1_seed["friction_level"] > 0.0, "mean_abs_target_executed_gap"].std(ddof=1)
        if q1_seed.loc[q1_seed["friction_level"] > 0.0].shape[0] > 1
        else 0.0
    )
    monotone_q1_gap = _nondecreasing(_positive_friction_values(q1_agg, "mean_abs_target_executed_gap"))
    positive_q1 = q1_df.loc[q1_df["friction_level"] > 0.0].copy()
    positive_pivot = positive_q1.pivot_table(
        index=["seed", "friction_level"],
        columns="interface_id",
        values="executed_metric",
    )
    positive_tempered_win_rate = float(
        (positive_pivot["tempered"] >= positive_pivot["responsive"]).mean() if not positive_pivot.empty else 0.0
    )
    return {
        "selection_target": "Q1",
        "config_id": config_id(config),
        "moving_average_window": config.moving_average_window,
        "overreactive_alpha": config.overreactive_alpha,
        "overreactive_noise": config.overreactive_noise,
        "eta_base": config.eta_base,
        "eta_friction_lambda": config.eta_friction_lambda,
        "latent_process": config.latent_process,
        "block_levels_scale": config.block_levels_scale,
        "block_noise_std": config.block_noise_std,
        "zero_gate_pass": bool(np.isclose(zero_q1_gap, 0.0)),
        "zero_q1_gap": zero_q1_gap,
        "positive_mean_abs_q1_gap": positive_mean_abs_q1_gap,
        "positive_seed_std_q1_gap": positive_seed_std_gap,
        "positive_tempered_win_rate": positive_tempered_win_rate,
        "monotone_q1_gap": bool(monotone_q1_gap),
        "monotone_candidate": bool(monotone_q1_gap),
        "failed_benchmark": False,
        "failure_reason": "",
    }


def select_q1_candidate(report_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    selection_pool = report_df.loc[report_df["zero_gate_pass"]].copy()
    selection_warning: str | None = None
    if selection_pool.empty:
        selection_pool = report_df.copy()
        selection_warning = SELECTION_WARNING_NO_GATE

    selection_pool = selection_pool.sort_values(
        [
            "monotone_q1_gap",
            "positive_tempered_win_rate",
            "positive_mean_abs_q1_gap",
            "positive_seed_std_q1_gap",
            "moving_average_window",
            "overreactive_alpha",
            "overreactive_noise",
            "eta_base",
            "eta_friction_lambda",
        ],
        ascending=[False, False, False, True, True, True, True, True, True],
    ).reset_index(drop=True)
    selected_row = selection_pool.iloc[0].to_dict()
    selected_config = config_from_dict(selected_row)
    if selection_warning is None and not bool(selected_row["monotone_q1_gap"]):
        selection_warning = SELECTION_WARNING_NO_MONOTONE

    full_report = report_df.copy()
    full_report["selection_warning"] = ""
    full_report["selected"] = False
    selected_mask = full_report["config_id"] == config_id(selected_config)
    full_report.loc[selected_mask, "selected"] = True
    if selection_warning is not None:
        full_report.loc[selected_mask, "selection_warning"] = selection_warning

    payload = {
        "selected_config": selected_config.to_dict(),
        "selected_config_id": config_id(selected_config),
        "selection_warning": selection_warning,
        "selection_metrics": {
            "zero_gate_pass": bool(selected_row["zero_gate_pass"]),
            "monotone_q1_gap": bool(selected_row["monotone_q1_gap"]),
            "positive_mean_abs_q1_gap": float(selected_row["positive_mean_abs_q1_gap"]),
            "positive_seed_std_q1_gap": float(selected_row["positive_seed_std_q1_gap"]),
            "positive_tempered_win_rate": float(selected_row["positive_tempered_win_rate"]),
        },
    }
    return full_report, payload


def evaluate_q2_candidate(config: SyntheticConfig, seeds: Iterable[int]) -> dict[str, object]:
    q2_df = build_q2_frame(config, seeds)
    q2_dis_seed, q2_dis_agg, _, q2_corr_agg, q2_pairwise_flips = build_q2_diagnostics(q2_df)

    zero_disagreement = float(
        q2_dis_agg.loc[q2_dis_agg["friction_level"] == 0.0, "mean_pairwise_disagreement_rate"].iloc[0]
    )
    zero_kendall = float(q2_corr_agg.loc[q2_corr_agg["friction_level"] == 0.0, "mean_kendall_tau"].iloc[0])
    zero_spearman = float(q2_corr_agg.loc[q2_corr_agg["friction_level"] == 0.0, "mean_spearman_rho"].iloc[0])
    zero_gate_pass = (
        np.isclose(zero_disagreement, 0.0)
        and np.isclose(zero_kendall, 1.0)
        and np.isclose(zero_spearman, 1.0)
    )

    positive_disagreement_values = _positive_friction_values(q2_dis_agg, "mean_pairwise_disagreement_rate")
    positive_mean_pairwise_disagreement_rate = float(
        q2_dis_seed.loc[q2_dis_seed["friction_level"] > 0.0, "pairwise_disagreement_rate"].mean()
    )
    positive_min_pairwise_disagreement_rate = float(
        q2_dis_agg.loc[q2_dis_agg["friction_level"] > 0.0, "mean_pairwise_disagreement_rate"].min()
    )
    positive_seed_std_pairwise_disagreement_rate = float(
        q2_dis_seed.loc[q2_dis_seed["friction_level"] > 0.0, "pairwise_disagreement_rate"].std(ddof=1)
        if q2_dis_seed.loc[q2_dis_seed["friction_level"] > 0.0].shape[0] > 1
        else 0.0
    )
    positive_mean_kendall_tau = float(
        q2_corr_agg.loc[q2_corr_agg["friction_level"] > 0.0, "mean_kendall_tau"].mean()
    )
    positive_mean_spearman_rho = float(
        q2_corr_agg.loc[q2_corr_agg["friction_level"] > 0.0, "mean_spearman_rho"].mean()
    )
    monotone_q2_disagreement = _nondecreasing(positive_disagreement_values)
    monotone_q2_kendall = _nonincreasing(_positive_friction_values(q2_corr_agg, "mean_kendall_tau"))
    monotone_q2_spearman = _nonincreasing(_positive_friction_values(q2_corr_agg, "mean_spearman_rho"))
    strict_q2_disagreement_increase = bool(
        any(
            positive_disagreement_values[idx] < positive_disagreement_values[idx + 1] - 1e-12
            for idx in range(len(positive_disagreement_values) - 1)
        )
    )
    if q2_pairwise_flips.empty:
        positive_max_pairwise_flip_seed_share = 0.0
    else:
        n_seeds = max(len(list(seeds)), 1)
        positive_flip_share = (
            q2_pairwise_flips.groupby(["friction_level", "model_a", "model_b"])["seed"]
            .nunique()
            .div(n_seeds)
        )
        positive_max_pairwise_flip_seed_share = float(positive_flip_share.max())

    monotone_candidate = bool(monotone_q2_disagreement and monotone_q2_kendall and monotone_q2_spearman)
    return {
        "selection_target": "Q2",
        "config_id": config_id(config),
        "moving_average_window": config.moving_average_window,
        "overreactive_alpha": config.overreactive_alpha,
        "overreactive_noise": config.overreactive_noise,
        "eta_base": config.eta_base,
        "eta_friction_lambda": config.eta_friction_lambda,
        "latent_process": config.latent_process,
        "block_levels_scale": config.block_levels_scale,
        "block_noise_std": config.block_noise_std,
        "zero_gate_pass": bool(zero_gate_pass),
        "zero_q2_disagreement_rate": zero_disagreement,
        "zero_q2_kendall_tau": zero_kendall,
        "zero_q2_spearman_rho": zero_spearman,
        "positive_min_pairwise_disagreement_rate": positive_min_pairwise_disagreement_rate,
        "positive_mean_pairwise_disagreement_rate": positive_mean_pairwise_disagreement_rate,
        "positive_seed_std_pairwise_disagreement_rate": positive_seed_std_pairwise_disagreement_rate,
        "positive_mean_kendall_tau": positive_mean_kendall_tau,
        "positive_mean_spearman_rho": positive_mean_spearman_rho,
        "positive_max_pairwise_flip_seed_share": positive_max_pairwise_flip_seed_share,
        "strict_q2_disagreement_increase": strict_q2_disagreement_increase,
        "monotone_q2_disagreement": bool(monotone_q2_disagreement),
        "monotone_q2_kendall": bool(monotone_q2_kendall),
        "monotone_q2_spearman": bool(monotone_q2_spearman),
        "monotone_candidate": monotone_candidate,
        "failed_benchmark": False,
        "failure_reason": "",
    }


def select_q2_candidate(report_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    selection_pool = report_df.loc[report_df["zero_gate_pass"]].copy()
    selection_warning: str | None = None
    if selection_pool.empty:
        selection_pool = report_df.copy()
        selection_warning = SELECTION_WARNING_NO_GATE

    selection_pool = selection_pool.sort_values(
        [
            "monotone_candidate",
            "strict_q2_disagreement_increase",
            "positive_min_pairwise_disagreement_rate",
            "positive_mean_pairwise_disagreement_rate",
            "positive_mean_kendall_tau",
            "positive_mean_spearman_rho",
            "positive_max_pairwise_flip_seed_share",
            "positive_seed_std_pairwise_disagreement_rate",
            "moving_average_window",
            "overreactive_alpha",
            "eta_base",
            "eta_friction_lambda",
        ],
        ascending=[False, False, False, False, True, True, False, True, True, True, True, True],
    ).reset_index(drop=True)
    selected_row = selection_pool.iloc[0].to_dict()
    selected_config = config_from_dict(selected_row)
    if selection_warning is None and not bool(selected_row["monotone_candidate"]):
        selection_warning = SELECTION_WARNING_NO_MONOTONE
    if selection_warning is None and not bool(selected_row["strict_q2_disagreement_increase"]):
        selection_warning = SELECTION_WARNING_NO_STRICT_Q2_INCREASE

    full_report = report_df.copy()
    full_report["selection_warning"] = ""
    full_report["selected"] = False
    selected_mask = full_report["config_id"] == config_id(selected_config)
    full_report.loc[selected_mask, "selected"] = True
    if selection_warning is not None:
        full_report.loc[selected_mask, "selection_warning"] = selection_warning

    payload = {
        "selected_config": selected_config.to_dict(),
        "selected_config_id": config_id(selected_config),
        "selection_warning": selection_warning,
        "selection_metrics": {
            "zero_gate_pass": bool(selected_row["zero_gate_pass"]),
            "monotone_candidate": bool(selected_row["monotone_candidate"]),
            "strict_q2_disagreement_increase": bool(selected_row["strict_q2_disagreement_increase"]),
            "positive_min_pairwise_disagreement_rate": float(selected_row["positive_min_pairwise_disagreement_rate"]),
            "positive_mean_pairwise_disagreement_rate": float(selected_row["positive_mean_pairwise_disagreement_rate"]),
            "positive_mean_kendall_tau": float(selected_row["positive_mean_kendall_tau"]),
            "positive_mean_spearman_rho": float(selected_row["positive_mean_spearman_rho"]),
            "positive_max_pairwise_flip_seed_share": float(selected_row["positive_max_pairwise_flip_seed_share"]),
            "positive_seed_std_pairwise_disagreement_rate": float(
                selected_row["positive_seed_std_pairwise_disagreement_rate"]
            ),
        },
    }
    return full_report, payload


def run_split_calibration(
    output_dir: str | Path,
    seeds: Iterable[int] = CALIBRATION_SEEDS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    q1_report = pd.DataFrame([evaluate_q1_candidate(config, seeds) for config in q1_calibration_grid()]).sort_values(
        "config_id"
    )
    q2_report = pd.DataFrame([evaluate_q2_candidate(config, seeds) for config in q2_calibration_grid()]).sort_values(
        "config_id"
    )
    q1_report, q1_payload = select_q1_candidate(q1_report.reset_index(drop=True))
    q2_report, q2_payload = select_q2_candidate(q2_report.reset_index(drop=True))
    combined_report = pd.concat([q1_report, q2_report], ignore_index=True, sort=False)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    combined_report.to_csv(output_path / DEFAULT_CALIBRATION_REPORT_PATH.name, index=False)
    payload = {
        "benchmark_variant": "split_q1_q2_v1",
        "q1_selected_config": q1_payload["selected_config"],
        "q1_selected_config_id": q1_payload["selected_config_id"],
        "q1_selection_warning": q1_payload["selection_warning"],
        "q1_selection_metrics": q1_payload["selection_metrics"],
        "q2_selected_config": q2_payload["selected_config"],
        "q2_selected_config_id": q2_payload["selected_config_id"],
        "q2_selection_warning": q2_payload["selection_warning"],
        "q2_selection_metrics": q2_payload["selection_metrics"],
        "calibration_seeds": CALIBRATION_SEEDS,
        "report_seeds": REPORT_SEEDS,
        "friction_grid": FRICTION_GRID,
    }
    save_selected_config(output_path / DEFAULT_SELECTED_CONFIG_PATH.name, payload)
    return combined_report, payload


def build_report_payload(config: SyntheticConfig, q1_df: pd.DataFrame, q2_df: pd.DataFrame) -> dict[str, object]:
    q1_seed, q1_agg = build_q1_gap_summary(q1_df)
    q2_dis_seed, q2_dis_agg, q2_corr_seed, q2_corr_agg, pairwise_flips = build_q2_diagnostics(q2_df)
    return {
        "q1_seed": q1_seed,
        "q1_aggregate": q1_agg,
        "q2_disagreement_seed": q2_dis_seed,
        "q2_disagreement_aggregate": q2_dis_agg,
        "q2_correlation_seed": q2_corr_seed,
        "q2_correlation_aggregate": q2_corr_agg,
        "q2_pairwise_flips": pairwise_flips,
        "config": config,
    }


def validate_report_payload(
    q1_df: pd.DataFrame,
    q2_df: pd.DataFrame,
    q1_gap_aggregate: pd.DataFrame,
    q2_disagreement_aggregate: pd.DataFrame,
    q2_correlation_aggregate: pd.DataFrame,
    q2_pairwise_flips: pd.DataFrame,
    *,
    report_seeds: Iterable[int],
) -> dict[str, object]:
    failures: list[str] = []
    report_seed_list = [int(seed) for seed in report_seeds]
    n_report_seeds = len(report_seed_list)

    zero_q1 = q1_df.loc[q1_df["friction_level"] == 0.0]
    if not np.allclose(zero_q1["target_metric"].to_numpy(), zero_q1["executed_metric"].to_numpy(), atol=1e-12):
        failures.append("q1_zero_friction_target_and_executed_metrics_diverged")

    zero_disagreement = float(
        q2_disagreement_aggregate.loc[
            q2_disagreement_aggregate["friction_level"] == 0.0,
            "mean_pairwise_disagreement_rate",
        ].iloc[0]
    )
    zero_kendall = float(
        q2_correlation_aggregate.loc[
            q2_correlation_aggregate["friction_level"] == 0.0,
            "mean_kendall_tau",
        ].iloc[0]
    )
    zero_spearman = float(
        q2_correlation_aggregate.loc[
            q2_correlation_aggregate["friction_level"] == 0.0,
            "mean_spearman_rho",
        ].iloc[0]
    )
    if not np.isclose(zero_disagreement, 0.0):
        failures.append("q2_zero_friction_pairwise_disagreement_nonzero")
    if not np.isclose(zero_kendall, 1.0):
        failures.append("q2_zero_friction_kendall_not_one")
    if not np.isclose(zero_spearman, 1.0):
        failures.append("q2_zero_friction_spearman_not_one")

    positive_q1_gaps = _positive_friction_values(q1_gap_aggregate, "mean_abs_target_executed_gap")
    if not _nondecreasing(positive_q1_gaps):
        failures.append("q1_positive_friction_gap_not_nondecreasing")

    positive_disagreement = q2_disagreement_aggregate.loc[
        q2_disagreement_aggregate["friction_level"] > 0.0,
        "mean_pairwise_disagreement_rate",
    ]
    if int((positive_disagreement > 0.0).sum()) < 2:
        failures.append("q2_positive_friction_disagreement_not_present_at_two_levels")

    if q2_pairwise_flips.empty:
        failures.append("q2_pairwise_flips_missing")
        max_flip_share = 0.0
    else:
        flip_share = (
            q2_pairwise_flips.groupby(["friction_level", "model_a", "model_b"])["seed"]
            .nunique()
            .div(max(n_report_seeds, 1))
        )
        max_flip_share = float(flip_share.max())
        if max_flip_share < 0.40:
            failures.append("q2_no_forecaster_pair_flipped_in_40_percent_of_report_seeds")

    return {
        "failed_benchmark": bool(failures),
        "failure_reasons": failures,
        "max_pairwise_flip_seed_share": max_flip_share,
    }


def update_calibration_report_failure(
    calibration_report_path: str | Path,
    selected_config_id: str | Iterable[str],
    validation_payload: dict[str, object],
) -> None:
    report_path = Path(calibration_report_path)
    if not report_path.exists():
        return

    report_df = pd.read_csv(report_path)
    if isinstance(selected_config_id, str):
        selected_config_ids = [selected_config_id]
    else:
        selected_config_ids = [str(value) for value in selected_config_id]
    mask = report_df["config_id"].isin(selected_config_ids)
    if "failed_benchmark" in report_df.columns:
        report_df["failed_benchmark"] = report_df["failed_benchmark"].fillna(False).astype(bool)
    else:
        report_df["failed_benchmark"] = False
    if "failure_reason" in report_df.columns:
        report_df["failure_reason"] = report_df["failure_reason"].fillna("").astype(str)
    else:
        report_df["failure_reason"] = ""
    report_df.loc[mask, "failed_benchmark"] = bool(validation_payload["failed_benchmark"])
    report_df.loc[mask, "failure_reason"] = "|".join(validation_payload["failure_reasons"])
    report_df.to_csv(report_path, index=False)
