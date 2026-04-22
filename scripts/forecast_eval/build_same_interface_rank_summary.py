#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIE_ABS_FLOOR = 1e-10
TIE_REL_SCALE = 1e-8


@dataclass(frozen=True)
class RankSummaryMeta:
    domain: str
    expected_interface_id: str
    observed_interface_ids: tuple[str, ...]
    n_rows: int
    n_seeds: int
    friction_levels: tuple[float, ...]
    forecaster_ids: tuple[str, ...]
    min_n_forecasters_per_seed_friction: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tie-aware same-interface rank summaries from a raw Q2 CSV.")
    parser.add_argument("--input", required=True, help="Input Q2 CSV.")
    parser.add_argument("--domain", required=True, help="Domain label for the output rows.")
    parser.add_argument("--expected-interface", required=True, help="Expected fixed interface_id for this domain.")
    parser.add_argument("--output-dir", required=True, help="Directory for derived summary CSVs.")
    return parser.parse_args()


def tie_tolerance(value_a: float, value_b: float) -> float:
    return max(TIE_ABS_FLOOR, TIE_REL_SCALE * max(abs(float(value_a)), abs(float(value_b)), 1.0))


def scores_tied(value_a: float, value_b: float) -> bool:
    return abs(float(value_a) - float(value_b)) <= tie_tolerance(value_a, value_b)


def _stderr(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size <= 1:
        return 0.0
    return float(array.std(ddof=1) / math.sqrt(array.size))


def _canonical_model_pair(model_a: str, model_b: str) -> tuple[str, str]:
    return tuple(sorted((str(model_a), str(model_b))))


def _cluster_sorted_scores(scores_desc: list[float]) -> list[list[int]]:
    groups: list[list[int]] = []
    for idx, score in enumerate(scores_desc):
        if not groups:
            groups.append([idx])
            continue
        current_group = groups[-1]
        if all(scores_tied(score, scores_desc[group_idx]) for group_idx in current_group):
            current_group.append(idx)
        else:
            groups.append([idx])
    return groups


def average_ranks_from_scores(scores_by_model: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores_by_model.items(), key=lambda item: (-float(item[1]), str(item[0])))
    sorted_scores = [float(score) for _model, score in ordered]
    clusters = _cluster_sorted_scores(sorted_scores)

    ranks: dict[str, float] = {}
    for cluster in clusters:
        start_rank = cluster[0] + 1
        end_rank = cluster[-1] + 1
        average_rank = float(start_rank + end_rank) / 2.0
        for cluster_idx in cluster:
            model_id = str(ordered[cluster_idx][0])
            ranks[model_id] = average_rank
    return ranks


def tied_top_set(scores_by_model: dict[str, float]) -> tuple[str, ...]:
    if not scores_by_model:
        return ()
    ordered = sorted(scores_by_model.items(), key=lambda item: (-float(item[1]), str(item[0])))
    best_score = float(ordered[0][1])
    winners = [str(model_id) for model_id, score in ordered if scores_tied(float(score), best_score)]
    return tuple(sorted(winners))


def _representative_from_best_set(
    best_set: tuple[str, ...],
    *,
    tiebreak_scores: dict[str, float],
) -> str:
    if not best_set:
        raise ValueError("best_set must not be empty.")
    ordered = sorted(
        best_set,
        key=lambda model_id: (-float(tiebreak_scores[str(model_id)]), str(model_id)),
    )
    return str(ordered[0])


def _best_membership_winner(
    members: dict[str, int],
    *,
    mean_scores: dict[str, float],
) -> str:
    if not members:
        return ""
    ordered = sorted(
        members,
        key=lambda model_id: (-int(members[str(model_id)]), -float(mean_scores[str(model_id)]), str(model_id)),
    )
    return str(ordered[0])


def _constant_like(values: np.ndarray) -> bool:
    if values.size <= 1:
        return True
    return bool(np.allclose(values, values[0], atol=1e-15, rtol=0.0))


def spearman_rho_from_average_ranks(forecast_ranks: dict[str, float], executed_ranks: dict[str, float]) -> float:
    models = sorted(forecast_ranks)
    forecast_values = np.asarray([forecast_ranks[model] for model in models], dtype=np.float64)
    executed_values = np.asarray([executed_ranks[model] for model in models], dtype=np.float64)
    if _constant_like(forecast_values) and _constant_like(executed_values):
        return 1.0
    if _constant_like(forecast_values) or _constant_like(executed_values):
        return 0.0
    return float(np.corrcoef(forecast_values, executed_values)[0, 1])


def _pair_sign(score_left: float, score_right: float) -> int:
    if scores_tied(score_left, score_right):
        return 0
    return 1 if float(score_left) > float(score_right) else -1


def kendall_tau_b_from_scores(forecast_scores: dict[str, float], executed_scores: dict[str, float]) -> float:
    concordant = 0
    discordant = 0
    tied_forecast_only = 0
    tied_executed_only = 0

    models = sorted(forecast_scores)
    for idx in range(len(models)):
        for jdx in range(idx + 1, len(models)):
            model_a = models[idx]
            model_b = models[jdx]
            sign_forecast = _pair_sign(forecast_scores[model_a], forecast_scores[model_b])
            sign_executed = _pair_sign(executed_scores[model_a], executed_scores[model_b])
            if sign_forecast == 0 and sign_executed == 0:
                continue
            if sign_forecast == 0:
                tied_forecast_only += 1
                continue
            if sign_executed == 0:
                tied_executed_only += 1
                continue
            if sign_forecast == sign_executed:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        float(concordant + discordant + tied_forecast_only) * float(concordant + discordant + tied_executed_only)
    )
    if denominator <= 1e-15:
        return 1.0
    return float((concordant - discordant) / denominator)


def build_domain_rank_summary(
    q2_df: pd.DataFrame,
    *,
    domain: str,
    expected_interface_id: str,
) -> tuple[dict[str, pd.DataFrame], RankSummaryMeta]:
    df = q2_df.copy()
    if df.empty:
        raise ValueError("Input Q2 DataFrame is empty.")

    observed_interfaces = tuple(sorted(str(value) for value in df["interface_id"].dropna().unique().tolist()))
    friction_levels = tuple(sorted(float(value) for value in df["friction_level"].dropna().unique().tolist()))
    forecaster_ids = tuple(sorted(str(value) for value in df["forecaster_id"].dropna().unique().tolist()))
    min_n_forecasters = int(
        df.groupby(["seed", "friction_level"], dropna=False)["forecaster_id"].nunique().min()
    )

    meta = RankSummaryMeta(
        domain=str(domain),
        expected_interface_id=str(expected_interface_id),
        observed_interface_ids=observed_interfaces,
        n_rows=int(len(df)),
        n_seeds=int(df["seed"].nunique()),
        friction_levels=friction_levels,
        forecaster_ids=forecaster_ids,
        min_n_forecasters_per_seed_friction=min_n_forecasters,
    )

    seed_rows: list[dict[str, object]] = []
    pairwise_event_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for (seed, friction_level), group in df.groupby(["seed", "friction_level"], sort=True):
        ordered = group.sort_values("forecaster_id").reset_index(drop=True)
        forecast_scores = {str(row.forecaster_id): float(row.forecast_metric) for row in ordered.itertuples(index=False)}
        executed_scores = {str(row.forecaster_id): float(row.executed_metric) for row in ordered.itertuples(index=False)}
        forecast_ranks = average_ranks_from_scores(forecast_scores)
        executed_ranks = average_ranks_from_scores(executed_scores)
        forecast_best_set = tied_top_set(forecast_scores)
        deployed_best_set = tied_top_set(executed_scores)
        agreement_flag = bool(set(forecast_best_set).intersection(deployed_best_set))
        forecast_selected_representative = _representative_from_best_set(
            forecast_best_set,
            tiebreak_scores=executed_scores,
        )
        deployed_selected_representative = _representative_from_best_set(
            deployed_best_set,
            tiebreak_scores=executed_scores,
        )
        deployed_gap_of_forecast_selected = float(
            executed_scores[deployed_selected_representative] - executed_scores[forecast_selected_representative]
        )

        models = sorted(forecast_scores)
        n_forecasters = len(models)
        n_possible_pairs = int(n_forecasters * (n_forecasters - 1) / 2)
        flip_count = 0
        forecast_tie_pair_count = 0
        executed_tie_pair_count = 0
        either_tie_pair_count = 0
        forecast_tied_models: set[str] = set()
        executed_tied_models: set[str] = set()
        n_comparable_pairs = 0

        for idx in range(n_forecasters):
            for jdx in range(idx + 1, n_forecasters):
                model_a = models[idx]
                model_b = models[jdx]
                forecast_tied = scores_tied(forecast_scores[model_a], forecast_scores[model_b])
                executed_tied = scores_tied(executed_scores[model_a], executed_scores[model_b])
                if forecast_tied:
                    forecast_tie_pair_count += 1
                    forecast_tied_models.update((model_a, model_b))
                if executed_tied:
                    executed_tie_pair_count += 1
                    executed_tied_models.update((model_a, model_b))
                if forecast_tied or executed_tied:
                    either_tie_pair_count += 1
                    continue

                n_comparable_pairs += 1
                forecast_order = f"{model_a}>{model_b}" if forecast_scores[model_a] > forecast_scores[model_b] else f"{model_a}<{model_b}"
                executed_order = f"{model_a}>{model_b}" if executed_scores[model_a] > executed_scores[model_b] else f"{model_a}<{model_b}"
                if (forecast_scores[model_a] > forecast_scores[model_b]) != (
                    executed_scores[model_a] > executed_scores[model_b]
                ):
                    flip_count += 1
                    canonical_a, canonical_b = _canonical_model_pair(model_a, model_b)
                    pairwise_event_rows.append(
                        {
                            "domain": str(domain),
                            "seed": int(seed),
                            "friction_level": float(friction_level),
                            "model_a": canonical_a,
                            "model_b": canonical_b,
                            "forecast_order": forecast_order,
                            "executed_order": executed_order,
                        }
                    )

        comparable_pair_fraction = float(n_comparable_pairs / n_possible_pairs) if n_possible_pairs else 0.0
        flip_rate = float(flip_count / n_comparable_pairs) if n_comparable_pairs else 0.0
        kendall_tau_b = kendall_tau_b_from_scores(forecast_scores, executed_scores)
        spearman_rho = spearman_rho_from_average_ranks(forecast_ranks, executed_ranks)

        seed_rows.append(
            {
                "domain": str(domain),
                "seed": int(seed),
                "friction_level": float(friction_level),
                "interface_id": str(expected_interface_id),
                "n_forecasters": int(n_forecasters),
                "n_possible_pairs": int(n_possible_pairs),
                "n_comparable_pairs": int(n_comparable_pairs),
                "comparable_pair_fraction": comparable_pair_fraction,
                "flip_count": int(flip_count),
                "flip_rate": flip_rate,
                "kendall_tau_b": kendall_tau_b,
                "spearman_rho": spearman_rho,
                "forecast_tie_pair_count": int(forecast_tie_pair_count),
                "executed_tie_pair_count": int(executed_tie_pair_count),
                "either_tie_pair_count": int(either_tie_pair_count),
                "forecast_tied_forecaster_count": int(len(forecast_tied_models)),
                "executed_tied_forecaster_count": int(len(executed_tied_models)),
            }
        )
        selection_rows.append(
            {
                "domain": str(domain),
                "seed": int(seed),
                "friction_level": float(friction_level),
                "interface_id": str(expected_interface_id),
                "n_forecasters": int(n_forecasters),
                "forecast_best_set": "|".join(forecast_best_set),
                "deployed_best_set": "|".join(deployed_best_set),
                "forecast_best_set_size": int(len(forecast_best_set)),
                "deployed_best_set_size": int(len(deployed_best_set)),
                "forecast_selected_representative": str(forecast_selected_representative),
                "deployed_selected_representative": str(deployed_selected_representative),
                "agreement_flag": bool(agreement_flag),
                "selection_disagreement_flag": bool(not agreement_flag),
                "deployed_gap_of_forecast_selected": deployed_gap_of_forecast_selected,
            }
        )

        for model_id in models:
            model_rows.append(
                {
                    "domain": str(domain),
                    "seed": int(seed),
                    "friction_level": float(friction_level),
                    "interface_id": str(expected_interface_id),
                    "forecaster_id": str(model_id),
                    "forecast_rank": float(forecast_ranks[model_id]),
                    "executed_rank": float(executed_ranks[model_id]),
                    "rank_gap": float(executed_ranks[model_id] - forecast_ranks[model_id]),
                }
            )

    seed_level = (
        pd.DataFrame(seed_rows)
        .sort_values(["domain", "friction_level", "seed"])
        .reset_index(drop=True)
    )

    rank_correlation_by_friction = (
        seed_level.groupby(["domain", "friction_level"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_flip_rate=("flip_rate", "mean"),
            median_flip_rate=("flip_rate", "median"),
            stderr_flip_rate=("flip_rate", _stderr),
            mean_kendall_tau_b=("kendall_tau_b", "mean"),
            median_kendall_tau_b=("kendall_tau_b", "median"),
            stderr_kendall_tau_b=("kendall_tau_b", _stderr),
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
            stderr_spearman_rho=("spearman_rho", _stderr),
            mean_n_comparable_pairs=("n_comparable_pairs", "mean"),
            min_n_comparable_pairs=("n_comparable_pairs", "min"),
            mean_comparable_pair_fraction=("comparable_pair_fraction", "mean"),
            mean_forecast_tie_pair_count=("forecast_tie_pair_count", "mean"),
            mean_executed_tie_pair_count=("executed_tie_pair_count", "mean"),
            mean_either_tie_pair_count=("either_tie_pair_count", "mean"),
            mean_forecast_tied_forecaster_count=("forecast_tied_forecaster_count", "mean"),
            mean_executed_tied_forecaster_count=("executed_tied_forecaster_count", "mean"),
        )
        .sort_values(["domain", "friction_level"])
        .reset_index(drop=True)
    )

    pairwise_events = pd.DataFrame(
        pairwise_event_rows,
        columns=["domain", "seed", "friction_level", "model_a", "model_b", "forecast_order", "executed_order"],
    )
    if pairwise_events.empty:
        pairwise_events = pd.DataFrame(
            columns=["domain", "seed", "friction_level", "model_a", "model_b", "forecast_order", "executed_order"]
        )
    pairwise_events = pairwise_events.sort_values(["domain", "friction_level", "model_a", "model_b", "seed"]).reset_index(drop=True)

    seed_count_map = (
        seed_level.groupby(["domain", "friction_level"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )
    if pairwise_events.empty:
        pairwise_by_friction = pd.DataFrame(
            columns=[
                "domain",
                "friction_level",
                "model_a",
                "model_b",
                "flip_seed_count",
                "flip_seed_share",
                "representative_seed",
                "representative_forecast_order",
                "representative_executed_order",
            ]
        )
    else:
        representative = (
            pairwise_events.sort_values(["domain", "friction_level", "model_a", "model_b", "seed"])
            .groupby(["domain", "friction_level", "model_a", "model_b"], as_index=False)
            .first()
            .rename(
                columns={
                    "seed": "representative_seed",
                    "forecast_order": "representative_forecast_order",
                    "executed_order": "representative_executed_order",
                }
            )
        )
        pairwise_by_friction = (
            pairwise_events.groupby(["domain", "friction_level", "model_a", "model_b"], as_index=False)["seed"]
            .nunique()
            .rename(columns={"seed": "flip_seed_count"})
            .merge(seed_count_map, on=["domain", "friction_level"], how="left")
            .merge(
                representative[
                    [
                        "domain",
                        "friction_level",
                        "model_a",
                        "model_b",
                        "representative_seed",
                        "representative_forecast_order",
                        "representative_executed_order",
                    ]
                ],
                on=["domain", "friction_level", "model_a", "model_b"],
                how="left",
            )
        )
        pairwise_by_friction["flip_seed_share"] = (
            pairwise_by_friction["flip_seed_count"] / pairwise_by_friction["n_seeds"].clip(lower=1)
        )
        pairwise_by_friction = pairwise_by_friction.drop(columns=["n_seeds"])
        pairwise_by_friction = pairwise_by_friction.sort_values(
            ["domain", "friction_level", "model_a", "model_b"]
        ).reset_index(drop=True)

    model_rank_summary = (
        pd.DataFrame(model_rows)
        .groupby(["domain", "friction_level", "forecaster_id"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_forecast_rank=("forecast_rank", "mean"),
            mean_executed_rank=("executed_rank", "mean"),
            mean_rank_gap=("rank_gap", "mean"),
        )
        .sort_values(["domain", "friction_level", "forecaster_id"])
        .reset_index(drop=True)
    )

    selection_seed_level = (
        pd.DataFrame(selection_rows)
        .sort_values(["domain", "friction_level", "seed"])
        .reset_index(drop=True)
    )
    selection_summary_rows: list[dict[str, object]] = []
    for (domain_value, friction_level), group in selection_seed_level.groupby(["domain", "friction_level"], sort=True):
        forecast_membership: dict[str, int] = {}
        deployed_membership: dict[str, int] = {}
        mean_forecast_scores = (
            df[
                np.isclose(df["friction_level"], float(friction_level), atol=1e-15)
                & (df["domain"] == str(domain_value))
            ]
            .groupby("forecaster_id", as_index=False)["forecast_metric"]
            .mean()
        )
        mean_executed_scores = (
            df[
                np.isclose(df["friction_level"], float(friction_level), atol=1e-15)
                & (df["domain"] == str(domain_value))
            ]
            .groupby("forecaster_id", as_index=False)["executed_metric"]
            .mean()
        )
        mean_forecast_map = {
            str(row.forecaster_id): float(row.forecast_metric) for row in mean_forecast_scores.itertuples(index=False)
        }
        mean_executed_map = {
            str(row.forecaster_id): float(row.executed_metric) for row in mean_executed_scores.itertuples(index=False)
        }

        for row in group.itertuples(index=False):
            for model_id in str(row.forecast_best_set).split("|"):
                if model_id:
                    forecast_membership[model_id] = forecast_membership.get(model_id, 0) + 1
            for model_id in str(row.deployed_best_set).split("|"):
                if model_id:
                    deployed_membership[model_id] = deployed_membership.get(model_id, 0) + 1

        selection_summary_rows.append(
            {
                "domain": str(domain_value),
                "friction_level": float(friction_level),
                "n_seeds": int(len(group)),
                "most_frequent_forecast_best": _best_membership_winner(
                    forecast_membership,
                    mean_scores=mean_forecast_map,
                ),
                "most_frequent_deployed_best": _best_membership_winner(
                    deployed_membership,
                    mean_scores=mean_executed_map,
                ),
                "agreement_rate": float(group["agreement_flag"].mean()),
                "disagreement_rate": float(group["selection_disagreement_flag"].mean()),
                "mean_deployed_gap_of_forecast_selected": float(group["deployed_gap_of_forecast_selected"].mean()),
                "median_deployed_gap_of_forecast_selected": float(group["deployed_gap_of_forecast_selected"].median()),
            }
        )

    selection_summary_by_friction = (
        pd.DataFrame(selection_summary_rows)
        .sort_values(["domain", "friction_level"])
        .reset_index(drop=True)
    )

    outputs = {
        "seed_level_rank_stats": seed_level,
        "seed_level_selection_stats": selection_seed_level,
        "rank_correlation_by_friction": rank_correlation_by_friction,
        "pairwise_flip_events": pairwise_events,
        "pairwise_flips_by_friction": pairwise_by_friction,
        "model_rank_summary": model_rank_summary,
        "selection_summary_by_friction": selection_summary_by_friction,
    }
    return outputs, meta


def validate_q2_source(
    q2_df: pd.DataFrame,
    *,
    expected_interface_id: str,
    min_forecasters_per_seed_friction: int = 4,
) -> list[str]:
    failures: list[str] = []
    question_ids = sorted(str(value) for value in q2_df["question_id"].dropna().unique().tolist())
    observed_interfaces = sorted(str(value) for value in q2_df["interface_id"].dropna().unique().tolist())
    forecaster_counts = q2_df.groupby(["seed", "friction_level"], dropna=False)["forecaster_id"].nunique()

    if question_ids != ["Q2"]:
        failures.append(f"expected_question_id_Q2_observed_{'|'.join(question_ids) or 'none'}")
    if observed_interfaces != [str(expected_interface_id)]:
        failures.append(
            f"expected_interface_{expected_interface_id}_observed_{'|'.join(observed_interfaces) or 'none'}"
        )
    if forecaster_counts.empty or int(forecaster_counts.min()) < int(min_forecasters_per_seed_friction):
        failures.append(f"min_forecasters_below_{int(min_forecasters_per_seed_friction)}")
    return failures


def write_summary_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_map = {
        "seed_level_rank_stats": "seed_level_rank_stats.csv",
        "seed_level_selection_stats": "seed_level_selection_stats.csv",
        "rank_correlation_by_friction": "rank_correlation_by_friction.csv",
        "pairwise_flip_events": "pairwise_flip_events.csv",
        "pairwise_flips_by_friction": "pairwise_flips_by_friction.csv",
        "model_rank_summary": "model_rank_summary.csv",
        "selection_summary_by_friction": "selection_summary_by_friction.csv",
    }
    for key, filename in filename_map.items():
        outputs[key].to_csv(output_dir / filename, index=False)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    q2_df = pd.read_csv(input_path)

    failures = validate_q2_source(q2_df, expected_interface_id=str(args.expected_interface))
    if failures:
        raise SystemExit(f"[same-interface-rank-summary] invalid input: {failures}")

    outputs, _meta = build_domain_rank_summary(
        q2_df,
        domain=str(args.domain),
        expected_interface_id=str(args.expected_interface),
    )
    write_summary_outputs(outputs, output_dir)
    for name in outputs:
        print(f"[same-interface-rank-summary] wrote {output_dir / f'{name}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
