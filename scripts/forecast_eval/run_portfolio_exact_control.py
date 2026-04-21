#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRL_ROOT = REPO_ROOT / "prl-dow30"

for candidate in (str(SCRIPT_DIR), str(PRL_ROOT / "scripts"), str(PRL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from common import build_result_row, save_results  # noqa: E402
from prl.envs import stable_softmax  # noqa: E402
from prl.metrics import compute_metrics, turnover_l1  # noqa: E402
from prl.train import create_scheduler  # noqa: E402
from step6_sanity import _build_eval_env, build_eval_context, load_model  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "generalization" / "multi_universe.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "portfolio_exact_control"
FINAL_PERIOD = "final"
REPLAY_INTERFACES = (
    ("eta_1_0", 1.0),
    ("eta_0_5", 0.5),
)
FRICTION_GRID = (0.0, 5e-4, 1e-3)
ZERO_COST_NEAR_FLAT_THRESHOLD = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact-control portfolio replay support on the final multi-universe PRL assets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the multi-universe YAML.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for exact-control artifacts.")
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_dumps(values: np.ndarray | list[float]) -> str:
    if isinstance(values, np.ndarray):
        payload = np.asarray(values, dtype=np.float64).round(12).tolist()
    else:
        payload = [float(value) for value in values]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _hash_array(values: np.ndarray) -> tuple[str, str]:
    json_text = _json_dumps(values)
    return json_text, _sha256_text(json_text)


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _paired_median(series_a: pd.Series, series_b: pd.Series) -> float:
    delta = pd.to_numeric(series_a, errors="coerce") - pd.to_numeric(series_b, errors="coerce")
    values = delta.dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _load_config(config_path: Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    execution_cfg = cfg.get("execution", {}) or {}
    return {
        "universes": [str(name) for name in (cfg.get("universes") or [])],
        "seeds": [int(seed) for seed in (execution_cfg.get("seeds") or [])],
        "runtime_root": _resolve_path(config_path, str((cfg.get("paths", {}) or {}).get("runtime_root"))),
        "raw_output_root": _resolve_path(config_path, str((cfg.get("paths", {}) or {}).get("raw_output_root"))),
        "offline": bool(execution_cfg.get("offline", True)),
        "max_steps": int(execution_cfg.get("max_steps", 0)),
        "model_type": str(execution_cfg.get("model_type", "prl")),
    }


def _source_rollout_id(universe_id: str, seed: int) -> str:
    return f"{universe_id}__{FINAL_PERIOD}__seed_{int(seed)}__source_eta1_kappa0"


def _load_model_and_env(ctx, *, eta: float, kappa: float):
    env = _build_eval_env(ctx, eta_mode="fixed", rebalance_eta=float(eta), transaction_cost=float(kappa))
    scheduler = None
    if ctx.model_type == "prl":
        scheduler = create_scheduler(ctx.prl_cfg, ctx.window_size, ctx.market.returns.shape[1], ctx.features.stats_path)
    model = load_model(ctx.model_path, ctx.model_type, env, scheduler=scheduler)
    return env, model


def _run_source_rollout(*, universe_id: str, seed: int, ctx) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env, model = _load_model_and_env(ctx, eta=1.0, kappa=0.0)
    base_env = env.envs[0]
    obs = env.reset()
    done = False
    step_index = 0
    source_rollout_id = _source_rollout_id(universe_id, seed)
    initial_prev_weights_source = base_env.prev_weights.astype(np.float64).copy()
    asset_ordering = [str(ticker) for ticker in base_env.returns.columns]
    source_rows: list[dict[str, Any]] = []

    while not done:
        prev_weights_source = base_env.prev_weights.astype(np.float64).copy()
        current_step = int(base_env.current_step)
        step_date = pd.Timestamp(base_env.returns.index[current_step])
        arithmetic_returns = np.expm1(base_env.returns.iloc[current_step].to_numpy(dtype=np.float64))

        action, _ = model.predict(obs, deterministic=True)
        action_vector = np.asarray(action, dtype=np.float64).reshape(-1)
        clipped_action = np.clip(action_vector, base_env.action_space.low, base_env.action_space.high).astype(np.float64)
        proposal_weights = stable_softmax(clipped_action, scale=float(base_env.cfg.logit_scale)).astype(np.float64)

        policy_output_raw_json, policy_output_hash = _hash_array(action_vector)
        proposal_weights_json, proposal_hash = _hash_array(proposal_weights)
        prev_weights_json, prev_weights_hash = _hash_array(prev_weights_source)
        returns_json, returns_hash = _hash_array(arithmetic_returns)

        obs, reward_vec, done_vec, info_list = env.step(action)
        done = bool(done_vec[0])
        info = info_list[0]
        executed_weights_source = proposal_weights.copy()
        executed_weights_json, executed_weights_hash = _hash_array(executed_weights_source)

        source_rows.append(
            {
                "date": step_date,
                "universe_id": universe_id,
                "seed": int(seed),
                "source_rollout_id": source_rollout_id,
                "step_index": int(step_index),
                "policy_output_raw_json": policy_output_raw_json,
                "policy_output_hash": policy_output_hash,
                "proposal_weights_json": proposal_weights_json,
                "proposal_hash": proposal_hash,
                "executed_weights_source_json": executed_weights_json,
                "executed_weights_source_hash": executed_weights_hash,
                "prev_weights_source_json": prev_weights_json,
                "prev_weights_source_hash": prev_weights_hash,
                "returns_vector_json": returns_json,
                "returns_vector_hash": returns_hash,
                "portfolio_return_source": _safe_float(info.get("portfolio_return")),
                "portfolio_return_target_source": _safe_float(info.get("portfolio_return_target")),
                "net_return_lin_source": _safe_float(info.get("net_return_lin_exec", info.get("portfolio_return", 0.0))),
                "net_return_lin_target_source": _safe_float(info.get("net_return_lin_target")),
                "tracking_error_l2_source": _safe_float(info.get("tracking_error_l2")),
                "realized_return_source": _safe_float(info.get("portfolio_return")),
            }
        )
        step_index += 1

    metadata = {
        "source_rollout_id": source_rollout_id,
        "universe_id": universe_id,
        "seed": int(seed),
        "period": FINAL_PERIOD,
        "initial_prev_weights_source": json.loads(_json_dumps(initial_prev_weights_source)),
        "logit_scale": float(base_env.cfg.logit_scale),
        "action_dim": int(base_env.action_space.shape[0]),
        "cash_included": False,
        "asset_ordering": asset_ordering,
        "ticker_ordering": asset_ordering,
        "action_transform": "stable_softmax",
        "action_low": np.asarray(base_env.action_space.low, dtype=np.float64).round(12).tolist(),
        "action_high": np.asarray(base_env.action_space.high, dtype=np.float64).round(12).tolist(),
        "config_path": str(Path(ctx.cfg["config_path"]).resolve()),
        "model_path": str(Path(ctx.model_path).resolve()),
        "model_type": str(ctx.model_type),
        "signal_state": bool(ctx.signal_state),
        "signal_names": [str(name) for name in (ctx.signal_names or [])],
        "eval_start": str(ctx.eval_start),
        "eval_end": str(ctx.eval_end),
    }
    return source_rows, metadata


def _replay_source_rollout(
    source_rows: list[dict[str, Any]],
    *,
    universe_id: str,
    seed: int,
    replay_interface_id: str,
    eta: float,
    kappa: float,
    initial_prev_weights_source: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev_weights = np.asarray(initial_prev_weights_source, dtype=np.float64).copy()
    replay_rows: list[dict[str, Any]] = []
    rewards_exec: list[float] = []
    portfolio_returns_exec: list[float] = []
    turnovers_exec: list[float] = []
    net_returns_exec: list[float] = []
    rewards_target: list[float] = []
    portfolio_returns_target: list[float] = []
    turnovers_target: list[float] = []
    net_returns_target: list[float] = []

    for row in source_rows:
        proposal_weights = np.asarray(json.loads(str(row["proposal_weights_json"])), dtype=np.float64)
        arithmetic_returns = np.asarray(json.loads(str(row["returns_vector_json"])), dtype=np.float64)
        if np.isclose(float(eta), 1.0, atol=1e-15):
            executed_weights = proposal_weights.copy()
        else:
            executed_weights = (1.0 - float(eta)) * prev_weights + float(eta) * proposal_weights
            total_weight = float(executed_weights.sum())
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                executed_weights = prev_weights.copy()
            else:
                executed_weights = executed_weights / total_weight

        turnover_exec = turnover_l1(prev_weights, executed_weights)
        turnover_target = turnover_l1(prev_weights, proposal_weights)
        cost_exec = float(kappa) * turnover_exec
        cost_target = float(kappa) * turnover_target
        portfolio_return_exec = float(np.dot(executed_weights, arithmetic_returns))
        portfolio_return_target = float(np.dot(proposal_weights, arithmetic_returns))
        net_return_exec = portfolio_return_exec - cost_exec
        net_return_target = portfolio_return_target - cost_target
        tracking_error_l2 = float(np.linalg.norm(executed_weights - proposal_weights, ord=2))

        executed_weights_json, executed_weights_hash = _hash_array(executed_weights)
        prev_weights_json, prev_weights_hash = _hash_array(prev_weights)

        replay_rows.append(
            {
                "date": row["date"],
                "universe_id": universe_id,
                "seed": int(seed),
                "source_rollout_id": row["source_rollout_id"],
                "replay_interface_id": replay_interface_id,
                "kappa": float(kappa),
                "step_index": int(row["step_index"]),
                "policy_output_hash": row["policy_output_hash"],
                "proposal_hash": row["proposal_hash"],
                "proposal_weights_json": row["proposal_weights_json"],
                "prev_weights_replay_json": prev_weights_json,
                "prev_weights_replay_hash": prev_weights_hash,
                "executed_weights_replay_json": executed_weights_json,
                "executed_weights_replay_hash": executed_weights_hash,
                "portfolio_return_exec": portfolio_return_exec,
                "portfolio_return_target": portfolio_return_target,
                "net_return_exec": net_return_exec,
                "net_return_target": net_return_target,
                "turnover_exec": turnover_exec,
                "turnover_target": turnover_target,
                "cost_exec": cost_exec,
                "cost_target": cost_target,
                "tracking_error_l2": tracking_error_l2,
            }
        )

        rewards_exec.append(net_return_exec)
        portfolio_returns_exec.append(portfolio_return_exec)
        turnovers_exec.append(turnover_exec)
        net_returns_exec.append(net_return_exec)

        rewards_target.append(net_return_target)
        portfolio_returns_target.append(portfolio_return_target)
        turnovers_target.append(turnover_target)
        net_returns_target.append(net_return_target)

        prev_weights = executed_weights

    metrics_exec = compute_metrics(
        rewards_exec,
        portfolio_returns_exec,
        turnovers_exec,
        net_returns_lin=net_returns_exec,
    )
    metrics_target = compute_metrics(
        rewards_target,
        portfolio_returns_target,
        turnovers_target,
        net_returns_lin=net_returns_target,
    )
    replay_df = pd.DataFrame(replay_rows)
    summary = {
        "universe_id": universe_id,
        "seed": int(seed),
        "source_rollout_id": str(source_rows[0]["source_rollout_id"]),
        "replay_interface_id": replay_interface_id,
        "kappa": float(kappa),
        "sharpe_exec_net": float(metrics_exec.sharpe_net_lin),
        "sharpe_target_net": float(metrics_target.sharpe_net_lin),
        "turnover_exec_mean": float(metrics_exec.avg_turnover_exec),
        "turnover_target_mean": float(metrics_target.avg_turnover_exec),
        "cost_exec_sum": float(replay_df["cost_exec"].sum()),
        "cost_target_sum": float(replay_df["cost_target"].sum()),
        "tracking_error_l2_mean": float(replay_df["tracking_error_l2"].mean()),
        "final_path_gap": float(
            abs(
                np.cumprod(1.0 + replay_df["net_return_exec"].to_numpy(dtype=np.float64))[-1]
                - np.cumprod(1.0 + replay_df["net_return_target"].to_numpy(dtype=np.float64))[-1]
            )
        ),
    }
    return replay_rows, summary


def _build_integrity_frames(source_df: pd.DataFrame, replay_df: pd.DataFrame, replay_summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []

    replay_counts = (
        replay_df.groupby(["universe_id", "seed", "source_rollout_id", "kappa", "step_index"], as_index=False)
        .size()
        .rename(columns={"size": "rows_in_group"})
    )
    pairing_failure_by_rollout = (
        replay_counts.assign(pairing_failure=lambda frame: frame["rows_in_group"] != len(REPLAY_INTERFACES))
        .groupby(["universe_id", "seed", "source_rollout_id"], as_index=False)["pairing_failure"]
        .sum()
        .rename(columns={"pairing_failure": "pairing_failure_count"})
    )

    unique_hashes = (
        replay_df.groupby(["universe_id", "seed", "source_rollout_id", "step_index"], as_index=False)
        .agg(
            forecast_hash_unique=("policy_output_hash", "nunique"),
            proposal_hash_unique=("proposal_hash", "nunique"),
        )
    )
    hash_rollup = unique_hashes.groupby(["universe_id", "seed", "source_rollout_id"], as_index=False).agg(
        n_steps=("step_index", "count"),
        forecast_hash_mismatch_count=("forecast_hash_unique", lambda values: int((pd.Series(values) != 1).sum())),
        proposal_hash_mismatch_count=("proposal_hash_unique", lambda values: int((pd.Series(values) != 1).sum())),
    )

    source_key_cols = ["universe_id", "seed", "source_rollout_id", "step_index"]
    source_vs_eta1 = replay_df[
        (replay_df["replay_interface_id"] == "eta_1_0") & np.isclose(replay_df["kappa"], 0.0, atol=1e-15)
    ].merge(
        source_df[source_key_cols + ["executed_weights_source_hash"]],
        on=source_key_cols,
        how="left",
    )
    zero_friction_match = (
        source_vs_eta1.assign(weight_match=source_vs_eta1["executed_weights_replay_hash"] == source_vs_eta1["executed_weights_source_hash"])
        .groupby(["universe_id", "seed", "source_rollout_id"], as_index=False)["weight_match"]
        .all()
    )
    source_summary = (
        source_df.groupby(["universe_id", "seed", "source_rollout_id"], as_index=False)
        .agg(
            source_sharpe_exec_net=("net_return_lin_source", lambda values: float(compute_metrics(values, values, [0.0] * len(pd.Series(values).dropna()), net_returns_lin=values).sharpe_net_lin)),
            source_cost_exec_sum=("net_return_lin_source", lambda _values: 0.0),
        )
    )
    replay_eta1_zero = replay_summary_df[
        (replay_summary_df["replay_interface_id"] == "eta_1_0") & np.isclose(replay_summary_df["kappa"], 0.0, atol=1e-15)
    ][["universe_id", "seed", "source_rollout_id", "sharpe_exec_net", "cost_exec_sum"]].rename(
        columns={"sharpe_exec_net": "replay_sharpe_exec_net", "cost_exec_sum": "replay_cost_exec_sum"}
    )
    zero_scalar_match = source_summary.merge(
        replay_eta1_zero,
        on=["universe_id", "seed", "source_rollout_id"],
        how="left",
    )
    zero_scalar_match["scalar_match"] = (
        np.isclose(zero_scalar_match["source_sharpe_exec_net"], zero_scalar_match["replay_sharpe_exec_net"], atol=1e-10)
        & np.isclose(zero_scalar_match["source_cost_exec_sum"], zero_scalar_match["replay_cost_exec_sum"], atol=1e-12)
    )

    merged = (
        hash_rollup.merge(pairing_failure_by_rollout, on=["universe_id", "seed", "source_rollout_id"], how="left")
        .merge(zero_friction_match, on=["universe_id", "seed", "source_rollout_id"], how="left")
        .merge(zero_scalar_match[["universe_id", "seed", "source_rollout_id", "scalar_match"]], on=["universe_id", "seed", "source_rollout_id"], how="left")
        .fillna({"pairing_failure_count": 0, "weight_match": False, "scalar_match": False})
    )
    merged["per_step_proposal_equality_failure_count"] = merged["proposal_hash_mismatch_count"]
    merged["zero_friction_replay_agreement_flag"] = merged["weight_match"] & merged["scalar_match"]
    merged["forecast_hash_identical_flag"] = merged["forecast_hash_mismatch_count"] == 0
    merged["proposal_hash_identical_flag"] = merged["proposal_hash_mismatch_count"] == 0

    forecast_rows = merged[
        [
            "universe_id",
            "seed",
            "source_rollout_id",
            "n_steps",
            "forecast_hash_mismatch_count",
            "forecast_hash_identical_flag",
        ]
    ].copy()
    proposal_rows = merged[
        [
            "universe_id",
            "seed",
            "source_rollout_id",
            "n_steps",
            "proposal_hash_mismatch_count",
            "proposal_hash_identical_flag",
            "per_step_proposal_equality_failure_count",
            "pairing_failure_count",
            "zero_friction_replay_agreement_flag",
        ]
    ].copy()
    return forecast_rows, proposal_rows


def _build_q1_rows(replay_summary_df: pd.DataFrame) -> list[dict[str, object]]:
    q1_rows: list[dict[str, object]] = []
    for (universe_id, seed, kappa, source_rollout_id), group in replay_summary_df.groupby(
        ["universe_id", "seed", "kappa", "source_rollout_id"],
        sort=True,
    ):
        target_metric = float(group["sharpe_target_net"].iloc[0])
        for row in group.itertuples(index=False):
            q1_rows.append(
                build_result_row(
                    question_id="Q1",
                    scenario_id=f"portfolio_exact_control_{universe_id}_v1",
                    domain="portfolio",
                    seed=int(seed),
                    forecaster_id="frozen_prl_policy_output",
                    interface_id=str(row.replay_interface_id),
                    friction_level=float(kappa),
                    forecast_metric=0.0,
                    target_metric=target_metric,
                    executed_metric=float(row.sharpe_exec_net),
                    realized_cost=float(row.cost_exec_sum),
                    realized_turnover_or_adjustment=float(row.turnover_exec_mean),
                )
            )
    return q1_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _refresh_master_summary() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "forecast_eval" / "build_summary.py")],
        cwd=str(REPO_ROOT),
        check=True,
    )


def _print_validation_summary(replay_summary_df: pd.DataFrame, proposal_hash_df: pd.DataFrame) -> None:
    recurrence_df = (
        replay_summary_df.pivot_table(
            index=["universe_id", "seed", "kappa"],
            columns="replay_interface_id",
            values="sharpe_exec_net",
            aggfunc="first",
        )
        .reset_index()
    )
    recurrence_df["delta_exec_eta05_minus_eta1"] = recurrence_df["eta_0_5"] - recurrence_df["eta_1_0"]
    positive = recurrence_df[recurrence_df["kappa"] > 0.0]
    positive_direction = (
        positive.groupby("universe_id", as_index=False)["delta_exec_eta05_minus_eta1"]
        .median()
        .assign(direction_reproduced=lambda frame: frame["delta_exec_eta05_minus_eta1"] > 0.0)
    )
    zero_cost = recurrence_df[np.isclose(recurrence_df["kappa"], 0.0, atol=1e-15)].copy()
    zero_cost["near_flat"] = zero_cost["delta_exec_eta05_minus_eta1"].abs() <= ZERO_COST_NEAR_FLAT_THRESHOLD
    print(
        "[portfolio-exact-control] zero-friction near-flat groups="
        f"{int(zero_cost['near_flat'].sum())}/{int(len(zero_cost))}"
    )
    print(
        "[portfolio-exact-control] positive-cost recurrence universes="
        f"{int(positive_direction['direction_reproduced'].sum())}/{int(len(positive_direction))}"
    )
    print(
        "[portfolio-exact-control] identity flags "
        f"forecast_all={bool(proposal_hash_df['proposal_hash_identical_flag'].all())} "
        f"pairing_failures={int(proposal_hash_df['pairing_failure_count'].sum())} "
        f"zero_friction_replay_agreement={bool(proposal_hash_df['zero_friction_replay_agreement_flag'].all())}"
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path)
    universe_ids = [universe_id for universe_id in cfg["universes"] if universe_id in {"u27_current", "u27_alt_largecap", "u27_sector_balanced"}]

    source_rows_all: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    replay_rows_all: list[dict[str, Any]] = []
    replay_summary_rows: list[dict[str, Any]] = []

    for universe_id in universe_ids:
        final_config_path = cfg["raw_output_root"] / universe_id / "configs" / "final_snapshot.yaml"
        model_root = cfg["runtime_root"] / universe_id / "train_control"
        if not final_config_path.exists():
            raise FileNotFoundError(f"Final snapshot config missing: {final_config_path}")
        if not model_root.exists():
            raise FileNotFoundError(f"Train-control model root missing: {model_root}")

        for seed in cfg["seeds"]:
            ctx = build_eval_context(
                config_path=str(final_config_path),
                model_type=cfg["model_type"],
                seed=int(seed),
                model_root=str(model_root),
                offline=cfg["offline"],
                max_steps=cfg["max_steps"],
                model_path_arg=None,
                prefer_metadata_config=False,
            )
            source_rows, metadata = _run_source_rollout(universe_id=universe_id, seed=int(seed), ctx=ctx)
            source_rows_all.extend(source_rows)
            source_metadata.append(metadata)
            initial_prev_weights_source = np.asarray(metadata["initial_prev_weights_source"], dtype=np.float64)

            for replay_interface_id, eta in REPLAY_INTERFACES:
                for kappa in FRICTION_GRID:
                    replay_rows, replay_summary = _replay_source_rollout(
                        source_rows,
                        universe_id=universe_id,
                        seed=int(seed),
                        replay_interface_id=replay_interface_id,
                        eta=float(eta),
                        kappa=float(kappa),
                        initial_prev_weights_source=initial_prev_weights_source,
                    )
                    replay_rows_all.extend(replay_rows)
                    replay_summary_rows.append(replay_summary)

    source_df = pd.DataFrame(source_rows_all).sort_values(["universe_id", "seed", "step_index"]).reset_index(drop=True)
    replay_df = pd.DataFrame(replay_rows_all).sort_values(
        ["universe_id", "seed", "kappa", "replay_interface_id", "step_index"]
    ).reset_index(drop=True)
    replay_summary_df = pd.DataFrame(replay_summary_rows).sort_values(
        ["universe_id", "seed", "kappa", "replay_interface_id"]
    ).reset_index(drop=True)

    forecast_hash_df, proposal_hash_df = _build_integrity_frames(source_df, replay_df, replay_summary_df)
    q1_df = save_results(_build_q1_rows(replay_summary_df), output_dir / "q1_same_forecast_diff_interface.csv")

    source_df.to_parquet(output_dir / "source_rollout_steps.parquet", index=False)
    replay_df.to_parquet(output_dir / "replay_steps.parquet", index=False)
    replay_summary_df.to_csv(output_dir / "portfolio_control_results.csv", index=False)
    forecast_hash_df.to_csv(output_dir / "forecast_hash_check.csv", index=False)
    proposal_hash_df.to_csv(output_dir / "proposal_hash_check.csv", index=False)
    _write_json(output_dir / "source_rollout_metadata.json", source_metadata)

    _refresh_master_summary()
    _print_validation_summary(replay_summary_df, proposal_hash_df)
    print(f"[portfolio-exact-control] wrote source steps to {output_dir / 'source_rollout_steps.parquet'}")
    print(f"[portfolio-exact-control] wrote replay steps to {output_dir / 'replay_steps.parquet'}")
    print(f"[portfolio-exact-control] wrote summary rows={len(replay_summary_df)}")
    print(f"[portfolio-exact-control] wrote shared-harness Q1 rows={len(q1_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
