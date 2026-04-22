#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_same_interface_rank_summary import (  # noqa: E402
    RankSummaryMeta,
    build_domain_rank_summary,
    validate_q2_source,
    write_summary_outputs,
)


DOMAIN_ORDER = {"synthetic": 0, "inventory": 1, "portfolio": 2}
STEP5_INTERPRETATION = (
    "Step 5 is a Q2 package: under a fixed deployed interface, forecast-metric ranking does not "
    "reliably determine deployed operational ranking as frictions increase."
)


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    domain_role: str
    source_path: Path
    expected_interface_id: str
    min_forecasters_per_seed_friction: int


CANONICAL_SPECS = [
    DomainSpec(
        domain="synthetic",
        domain_role="required",
        source_path=REPO_ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q2_diff_forecasts_same_interface.csv",
        expected_interface_id="tempered",
        min_forecasters_per_seed_friction=4,
    ),
    DomainSpec(
        domain="inventory",
        domain_role="required",
        source_path=REPO_ROOT
        / "outputs"
        / "forecast_eval"
        / "inventory_step4_seed_stability_locked"
        / "inventory_v2_seed_stability_q2.csv",
        expected_interface_id="responsive",
        min_forecasters_per_seed_friction=5,
    ),
    DomainSpec(
        domain="portfolio",
        domain_role="stretch",
        source_path=REPO_ROOT / "outputs" / "forecast_eval" / "portfolio" / "q2_diff_forecasts_same_interface.csv",
        expected_interface_id="tempered",
        min_forecasters_per_seed_friction=4,
    ),
]

PAPER_FORECASTER_LABELS = {
    "naive_last": "Naive persistence",
    "moving_average": "Moving average",
    "linear_ar": "Linear AR",
    "noisy_overreactive": "Reactive extrapolation heuristic",
    "moving_average_7": "Moving average (7)",
    "linear_ar_ridge": "Linear AR",
    "mlp_small": "Small MLP",
    "gru_small": "Small GRU",
    "ewma_20": "EWMA (20)",
    "rolling_mean_20": "Rolling mean (20)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Step 5 Q2 same-interface evidence package.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "forecast_eval" / "step5_same_interface"),
        help="Output directory for the derived Step 5 package.",
    )
    parser.add_argument(
        "--paper-results-dir",
        default=str(REPO_ROOT / "paper" / "forecasting_workshop" / "results"),
        help="Directory for paper-facing summary artifacts.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _paper_label(model_id: str) -> str:
    return PAPER_FORECASTER_LABELS.get(str(model_id), str(model_id))


def _paper_pair_label(pair_value: str) -> str:
    if not pair_value:
        return "--"
    return " / ".join(_paper_label(part) for part in str(pair_value).split("|") if part)


def _domain_sort_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_domain_order"] = out["domain"].map(DOMAIN_ORDER).fillna(999).astype(int)
    sort_columns = ["_domain_order"]
    if "friction_level" in out.columns:
        sort_columns.append("friction_level")
    if "model_a" in out.columns:
        sort_columns.extend(["model_a", "model_b"])
    if "seed" in out.columns:
        sort_columns.append("seed")
    if "forecaster_id" in out.columns:
        sort_columns.append("forecaster_id")
    out = out.sort_values(sort_columns).drop(columns="_domain_order").reset_index(drop=True)
    return out


def _collect_meta(df: pd.DataFrame, *, domain: str, expected_interface_id: str) -> RankSummaryMeta:
    if df.empty:
        return RankSummaryMeta(
            domain=str(domain),
            expected_interface_id=str(expected_interface_id),
            observed_interface_ids=(),
            n_rows=0,
            n_seeds=0,
            friction_levels=(),
            forecaster_ids=(),
            min_n_forecasters_per_seed_friction=0,
        )
    return RankSummaryMeta(
        domain=str(domain),
        expected_interface_id=str(expected_interface_id),
        observed_interface_ids=tuple(sorted(str(value) for value in df["interface_id"].dropna().unique().tolist())),
        n_rows=int(len(df)),
        n_seeds=int(df["seed"].nunique()),
        friction_levels=tuple(sorted(float(value) for value in df["friction_level"].dropna().unique().tolist())),
        forecaster_ids=tuple(sorted(str(value) for value in df["forecaster_id"].dropna().unique().tolist())),
        min_n_forecasters_per_seed_friction=int(
            df.groupby(["seed", "friction_level"], dropna=False)["forecaster_id"].nunique().min()
        ),
    )


def _best_positive_row(rank_corr: pd.DataFrame) -> pd.Series | None:
    positive = rank_corr[rank_corr["friction_level"] > 0.0].copy()
    if positive.empty:
        return None
    return (
        positive.sort_values(["mean_flip_rate", "friction_level"], ascending=[False, True])
        .reset_index(drop=True)
        .iloc[0]
    )


def _strongest_positive_pair(pairwise: pd.DataFrame) -> pd.Series | None:
    positive = pairwise[pairwise["friction_level"] > 0.0].copy()
    if positive.empty:
        return None
    return (
        positive.sort_values(
            ["flip_seed_share", "friction_level", "model_a", "model_b"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
        .iloc[0]
    )


def _required_gate(rank_corr: pd.DataFrame, pairwise: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    zero = rank_corr[np.isclose(rank_corr["friction_level"], 0.0, atol=1e-15)]
    positive = rank_corr[rank_corr["friction_level"] > 0.0]

    if zero.empty:
        return False, ["missing_zero_friction_row"]
    zero_row = zero.iloc[0]
    zero_flip = float(zero_row["mean_flip_rate"])
    zero_spearman = float(zero_row["mean_spearman_rho"])
    positive_flip_exists = bool((positive["mean_flip_rate"] > zero_flip).any())
    positive_spearman_drop_exists = bool((positive["mean_spearman_rho"] < zero_spearman).any())
    positive_pair_exists = bool((pairwise[pairwise["friction_level"] > 0.0]["flip_seed_share"] >= 0.50).any())

    if zero_flip > 0.10:
        reasons.append("zero_friction_mean_flip_rate_above_0.10")
    if zero_spearman < 0.80:
        reasons.append("zero_friction_mean_spearman_below_0.80")
    if not positive_flip_exists:
        reasons.append("no_positive_friction_flip_rate_increase")
    if not positive_spearman_drop_exists:
        reasons.append("no_positive_friction_spearman_drop")
    if not positive_pair_exists:
        reasons.append("no_positive_friction_pair_flip_share_ge_0.50")

    return len(reasons) == 0, reasons


def _stretch_gate(rank_corr: pd.DataFrame, pairwise: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    zero = rank_corr[np.isclose(rank_corr["friction_level"], 0.0, atol=1e-15)]
    positive = rank_corr[rank_corr["friction_level"] > 0.0]

    if zero.empty:
        return False, ["missing_zero_friction_row"]
    zero_row = zero.iloc[0]
    zero_flip = float(zero_row["mean_flip_rate"])
    zero_spearman = float(zero_row["mean_spearman_rho"])
    positive_flip_exists = bool((positive["mean_flip_rate"] > zero_flip).any())
    positive_spearman_drop_exists = bool((positive["mean_spearman_rho"] < zero_spearman).any())
    positive_pair_exists = bool((pairwise[pairwise["friction_level"] > 0.0]["flip_seed_share"] >= 0.50).any())

    if zero_flip > 0.10:
        reasons.append("zero_friction_mean_flip_rate_above_0.10")
    if zero_spearman < 0.50:
        reasons.append("zero_friction_mean_spearman_below_0.50")
    if not positive_flip_exists:
        reasons.append("no_positive_friction_flip_rate_increase")
    if not positive_spearman_drop_exists:
        reasons.append("no_positive_friction_spearman_drop")
    if not positive_pair_exists:
        reasons.append("no_positive_friction_pair_flip_share_ge_0.50")

    return len(reasons) == 0, reasons


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def _paper_table_from_verdicts(verdict_df: pd.DataFrame) -> pd.DataFrame:
    table = verdict_df.copy()
    table["Domain"] = table["domain"].map(
        {
            "synthetic": "Synthetic",
            "inventory": "Inventory",
            "portfolio": "Portfolio",
        }
    )
    table["Status"] = table["inclusion_status"]
    table["Role"] = table["domain_role"]
    table["Seeds"] = table["n_seeds"]
    table["Zero Flip Rate"] = table["zero_friction_mean_flip_rate"]
    table["Zero Spearman"] = table["zero_friction_mean_spearman_rho"]
    table["Best Positive Friction"] = table["best_positive_friction"]
    table["Best Positive Flip Rate"] = table["best_positive_flip_rate"]
    table["Strongest Flip Pair"] = table["strongest_flip_pair"].map(_paper_pair_label)
    table["Strongest Flip Share"] = table["strongest_flip_share"]
    table["Note"] = table["verdict_note"]
    table = table[
        [
            "Domain",
            "Role",
            "Status",
            "Seeds",
            "Zero Flip Rate",
            "Zero Spearman",
            "Best Positive Friction",
            "Best Positive Flip Rate",
            "Strongest Flip Pair",
            "Strongest Flip Share",
            "Note",
        ]
    ]
    return table


def _write_paper_table_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = table.copy()
    digits_by_column = {
        "Zero Flip Rate": 3,
        "Zero Spearman": 3,
        "Best Positive Friction": 4,
        "Best Positive Flip Rate": 3,
        "Strongest Flip Share": 3,
    }
    for column, digits in digits_by_column.items():
        formatted[column] = formatted[column].map(lambda value, digits=digits: _format_float(value, digits=digits))
    formatted = formatted.fillna("--")
    formatted.to_csv(path, index=False)


def _write_paper_table_tex(table: pd.DataFrame, path: Path) -> None:
    formatted = table.copy()
    formatted["Zero Flip Rate"] = formatted["Zero Flip Rate"].map(lambda value: _format_float(value, digits=3))
    formatted["Zero Spearman"] = formatted["Zero Spearman"].map(lambda value: _format_float(value, digits=3))
    formatted["Best Positive Friction"] = formatted["Best Positive Friction"].map(
        lambda value: _format_float(value, digits=4)
    )
    formatted["Best Positive Flip Rate"] = formatted["Best Positive Flip Rate"].map(
        lambda value: _format_float(value, digits=3)
    )
    formatted["Strongest Flip Share"] = formatted["Strongest Flip Share"].map(
        lambda value: _format_float(value, digits=3)
    )
    formatted["Strongest Flip Pair"] = formatted["Strongest Flip Pair"].replace(
        {"Linear AR / Reactive extrapolation heuristic": "Linear AR / Reactive heuristic"}
    )
    formatted["Note"] = formatted["Note"].replace(
        {
            "passed_step5_gate": "passed gate",
            "zero_friction_mean_flip_rate_above_0.10": "flip $>$ 0.10",
            "zero_friction_mean_spearman_below_0.50": "rho $<$ 0.50",
            "zero_friction_mean_flip_rate_above_0.10;zero_friction_mean_spearman_below_0.50": "flip $>$ 0.10; rho $<$ 0.50",
        }
    )
    formatted = formatted.fillna("--")
    lines = [
        "\\begin{tabularx}{\\textwidth}{@{}l l l r r r r r >{\\raggedright\\arraybackslash}X r >{\\raggedright\\arraybackslash}X@{}}",
        "\\toprule",
        "Domain & Role & Status & Seeds & Zero flip & Zero rho & Best +fric. & Best +flip & Strongest pair & Flip share & Note \\\\",
        "\\midrule",
    ]
    for row in formatted.itertuples(index=False):
        lines.append(
            f"{row.Domain} & {row.Role} & {row.Status} & {row.Seeds} & {row[4]} & {row[5]} & {row[6]} & {row[7]} & {row[8]} & {row[9]} & {row[10]} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabularx}"])
    tex = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)


def _selection_table_from_inventory_summary(
    selection_summary: pd.DataFrame,
    selection_seed_level: pd.DataFrame,
) -> pd.DataFrame:
    table = selection_summary.copy()
    suboptimal_counts = (
        selection_seed_level.assign(
            deployed_suboptimal_flag=selection_seed_level["deployed_gap_of_forecast_selected"].gt(1e-12)
        )
        .groupby("friction_level", as_index=False)
        .agg(
            deployed_suboptimal_seeds=("deployed_suboptimal_flag", "sum"),
            total_seeds=("seed", "count"),
        )
    )
    suboptimal_counts["Deployed-suboptimal seeds / total"] = suboptimal_counts.apply(
        lambda row: f"{int(row['deployed_suboptimal_seeds'])}/{int(row['total_seeds'])}",
        axis=1,
    )
    table = table.merge(
        suboptimal_counts[["friction_level", "Deployed-suboptimal seeds / total"]],
        on="friction_level",
        how="left",
    )
    table = table.loc[:, [
        "friction_level",
        "most_frequent_forecast_best",
        "most_frequent_deployed_best",
        "agreement_rate",
        "mean_deployed_gap_of_forecast_selected",
        "Deployed-suboptimal seeds / total",
    ]]
    table = table.rename(
        columns={
            "friction_level": "Friction",
            "most_frequent_forecast_best": "Forecast-side winner",
            "most_frequent_deployed_best": "Deployed winner",
            "agreement_rate": "Agreement rate",
            "mean_deployed_gap_of_forecast_selected": "Mean deployed gap",
        }
    )
    table["Forecast-side winner"] = table["Forecast-side winner"].map(_paper_label)
    table["Deployed winner"] = table["Deployed winner"].map(_paper_label)
    return table


def _write_selection_table_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = table.copy()
    formatted["Friction"] = formatted["Friction"].map(lambda value: _format_float(value, digits=2))
    formatted["Agreement rate"] = formatted["Agreement rate"].map(lambda value: _format_float(value, digits=2))
    formatted["Mean deployed gap"] = formatted["Mean deployed gap"].map(lambda value: _format_float(value, digits=3))
    formatted.to_csv(path, index=False)


def _write_selection_table_tex(table: pd.DataFrame, path: Path) -> None:
    formatted = table.copy()
    formatted["Friction"] = formatted["Friction"].map(lambda value: _format_float(value, digits=2))
    formatted["Agreement rate"] = formatted["Agreement rate"].map(lambda value: _format_float(value, digits=2))
    formatted["Mean deployed gap"] = formatted["Mean deployed gap"].map(lambda value: _format_float(value, digits=3))
    tex = formatted.to_latex(index=False, escape=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)


def _count_based_lines(verdict_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    required = verdict_df[verdict_df["domain_role"] == "required"].copy()
    valid_required = required[required["inclusion_status"] != "invalid_input"].copy()

    if len(valid_required) != len(required):
        lines.append(
            f"{len(valid_required)}/{len(required)} required domains had valid inputs; invalid inputs were excluded from count-based summaries."
        )

    if valid_required.empty:
        lines.append("0/0 valid required domains available for count-based Step 5 verdicts.")
        return lines

    stronger_mismatch = int(valid_required["positive_flip_rate_increase_flag"].sum())
    lower_corr = int(valid_required["positive_spearman_drop_flag"].sum())
    denominator_label = "required domains" if len(valid_required) == len(required) else "valid required domains"
    lines.append(
        f"{stronger_mismatch}/{len(valid_required)} {denominator_label} show stronger positive-friction ranking mismatch than at zero friction."
    )
    lines.append(
        f"{lower_corr}/{len(valid_required)} {denominator_label} show lower rank correlation as friction increases."
    )

    portfolio = verdict_df[verdict_df["domain"] == "portfolio"]
    if not portfolio.empty:
        portfolio_status = str(portfolio.iloc[0]["inclusion_status"])
        if portfolio_status == "excluded":
            lines.append("Portfolio omitted by stretch gate.")
        elif portfolio_status == "invalid_input":
            lines.append("Portfolio excluded because the stretch input was invalid.")
        elif portfolio_status == "included":
            lines.append("Portfolio passed the stretch gate and is retained as support-only evidence.")
    return lines


def _internal_verdict_lines(verdict_df: pd.DataFrame) -> list[str]:
    lines = ["# Step 5 Verdict", "", STEP5_INTERPRETATION, "", "## Domain Status"]
    for row in verdict_df.itertuples(index=False):
        lines.append(
            f"- {row.domain}: role={row.domain_role}, status={row.inclusion_status}, note={row.verdict_note}"
        )
    lines.extend(["", "## Count-Based Summary"])
    for line in _count_based_lines(verdict_df):
        lines.append(f"- {line}")
    return lines


def _paper_note_lines(verdict_df: pd.DataFrame) -> list[str]:
    lines = [
        "# Step 5: Same-Interface Q2 Summary",
        "",
        "Step 5 is a descriptive Q2 package rather than a new experiment. Its role is to summarize the same-interface ranking evidence after the fifth inventory baseline is added, while the main text now centers the inventory selection-drift consequence directly.",
        "",
        "## Domain Rows",
    ]
    for row in verdict_df.itertuples(index=False):
        label = {"synthetic": "Synthetic", "inventory": "Inventory", "portfolio": "Portfolio"}[row.domain]
        lines.append(
            f"- {label}: status={row.inclusion_status}, zero_flip_rate={_format_float(row.zero_friction_mean_flip_rate)}, zero_spearman={_format_float(row.zero_friction_mean_spearman_rho)}, strongest_flip_pair={_paper_pair_label(row.strongest_flip_pair)}"
        )
    lines.extend(["", "## Count-Based Verdict"])
    for line in _count_based_lines(verdict_df):
        lines.append(f"- {line}")
    return lines


def _selection_note_lines(selection_table: pd.DataFrame) -> list[str]:
    lines = [
        "# Inventory Q2 Selection Drift",
        "",
        "Table 1 reports the operational selection consequence in the required inventory domain under one fixed responsive replenishment interface.",
        "",
        "Winner columns report the most frequent seed-level best model at each friction level.",
        "Positive mean deployed gap means the forecast-selected model underperforms the deployed-selected model by that amount.",
        "Deployed-suboptimal seeds / total reports the number of seeds in which the forecast-selected model is not a deployed best model.",
        "",
        "## Friction Rows",
    ]
    for row in selection_table.itertuples(index=False):
        lines.append(
            f"- friction={_format_float(row.Friction, digits=2)}: forecast_winner={row[1]}, deployed_winner={row[2]}, agreement_rate={_format_float(row[3], digits=2)}, mean_deployed_gap={_format_float(row[4], digits=3)}, deployed_suboptimal={row[5]}"
        )
    return lines


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    paper_results_dir = Path(args.paper_results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_results_dir.mkdir(parents=True, exist_ok=True)

    manifest_sources: list[dict[str, object]] = []
    verdict_rows: list[dict[str, object]] = []
    domain_outputs_by_name: dict[str, dict[str, pd.DataFrame]] = {}
    portfolio_gate_definition = {
        "zero_friction_mean_flip_rate_max": 0.10,
        "zero_friction_mean_spearman_rho_min": 0.50,
        "requires_positive_flip_rate_increase": True,
        "requires_positive_spearman_drop": True,
        "requires_positive_pair_flip_seed_share_ge": 0.50,
    }

    for spec in CANONICAL_SPECS:
        q2_df = pd.read_csv(spec.source_path)
        meta = _collect_meta(q2_df, domain=spec.domain, expected_interface_id=spec.expected_interface_id)
        validation_failures = validate_q2_source(
            q2_df,
            expected_interface_id=spec.expected_interface_id,
            min_forecasters_per_seed_friction=spec.min_forecasters_per_seed_friction,
        )

        source_record = {
            "domain": spec.domain,
            "domain_role": spec.domain_role,
            "source_path": str(spec.source_path.resolve()),
            "source_sha256": _sha256_file(spec.source_path),
            "expected_interface_id": spec.expected_interface_id,
            "min_forecasters_per_seed_friction_required": spec.min_forecasters_per_seed_friction,
            "observed_interface_ids": list(meta.observed_interface_ids),
            "n_rows": meta.n_rows,
            "n_seeds": meta.n_seeds,
            "friction_levels": list(meta.friction_levels),
            "forecaster_ids": list(meta.forecaster_ids),
            "min_n_forecasters_per_seed_friction": meta.min_n_forecasters_per_seed_friction,
            "validation_failures": validation_failures,
        }
        manifest_sources.append(source_record)

        if validation_failures:
            verdict_rows.append(
                {
                    "domain": spec.domain,
                    "domain_role": spec.domain_role,
                    "source_path": str(spec.source_path.resolve()),
                    "source_sha256": source_record["source_sha256"],
                    "expected_interface_id": spec.expected_interface_id,
                    "observed_interface_id": "|".join(meta.observed_interface_ids),
                    "n_seeds": meta.n_seeds,
                    "zero_friction_mean_flip_rate": float("nan"),
                    "zero_friction_mean_kendall_tau_b": float("nan"),
                    "zero_friction_mean_spearman_rho": float("nan"),
                    "zero_friction_mean_n_comparable_pairs": float("nan"),
                    "zero_friction_mean_comparable_pair_fraction": float("nan"),
                    "best_positive_friction": float("nan"),
                    "best_positive_flip_rate": float("nan"),
                    "best_positive_kendall_drop": float("nan"),
                    "best_positive_spearman_drop": float("nan"),
                    "strongest_flip_pair": "",
                    "strongest_flip_share": float("nan"),
                    "positive_flip_rate_increase_flag": False,
                    "positive_spearman_drop_flag": False,
                    "required_domain_pass_flag": spec.domain_role == "required" and False,
                    "stretch_domain_pass_flag": spec.domain_role == "stretch" and False,
                    "inclusion_status": "invalid_input",
                    "verdict_note": ";".join(validation_failures),
                }
            )
            continue

        outputs, _ = build_domain_rank_summary(
            q2_df,
            domain=spec.domain,
            expected_interface_id=spec.expected_interface_id,
        )
        domain_outputs_by_name[spec.domain] = outputs
        domain_output_dir = output_dir / spec.domain
        write_summary_outputs({key: _domain_sort_values(value) for key, value in outputs.items()}, domain_output_dir)

        rank_corr = outputs["rank_correlation_by_friction"].copy()
        pairwise = outputs["pairwise_flips_by_friction"].copy()
        zero_row = rank_corr[np.isclose(rank_corr["friction_level"], 0.0, atol=1e-15)].iloc[0]
        best_positive = _best_positive_row(rank_corr)
        strongest_pair = _strongest_positive_pair(pairwise)
        positive_flip_increase = bool(
            (rank_corr[rank_corr["friction_level"] > 0.0]["mean_flip_rate"] > float(zero_row["mean_flip_rate"])).any()
        )
        positive_spearman_drop = bool(
            (
                rank_corr[rank_corr["friction_level"] > 0.0]["mean_spearman_rho"]
                < float(zero_row["mean_spearman_rho"])
            ).any()
        )

        if spec.domain_role == "required":
            gate_pass, gate_failures = _required_gate(rank_corr, pairwise)
            inclusion_status = "pass" if gate_pass else "fail"
            required_domain_pass_flag = bool(gate_pass)
            stretch_domain_pass_flag = False
        else:
            gate_pass, gate_failures = _stretch_gate(rank_corr, pairwise)
            inclusion_status = "included" if gate_pass else "excluded"
            required_domain_pass_flag = False
            stretch_domain_pass_flag = bool(gate_pass)

        best_positive_friction = float(best_positive["friction_level"]) if best_positive is not None else float("nan")
        best_positive_flip_rate = float(best_positive["mean_flip_rate"]) if best_positive is not None else float("nan")
        best_positive_kendall_drop = (
            float(zero_row["mean_kendall_tau_b"]) - float(best_positive["mean_kendall_tau_b"])
            if best_positive is not None
            else float("nan")
        )
        best_positive_spearman_drop = (
            float(zero_row["mean_spearman_rho"]) - float(best_positive["mean_spearman_rho"])
            if best_positive is not None
            else float("nan")
        )
        strongest_flip_pair = (
            f"{strongest_pair['model_a']}|{strongest_pair['model_b']}" if strongest_pair is not None else ""
        )
        strongest_flip_share = float(strongest_pair["flip_seed_share"]) if strongest_pair is not None else float("nan")

        verdict_note = "passed_step5_gate" if gate_pass else ";".join(gate_failures)
        verdict_rows.append(
            {
                "domain": spec.domain,
                "domain_role": spec.domain_role,
                "source_path": str(spec.source_path.resolve()),
                "source_sha256": source_record["source_sha256"],
                "expected_interface_id": spec.expected_interface_id,
                "observed_interface_id": "|".join(meta.observed_interface_ids),
                "n_seeds": meta.n_seeds,
                "zero_friction_mean_flip_rate": float(zero_row["mean_flip_rate"]),
                "zero_friction_mean_kendall_tau_b": float(zero_row["mean_kendall_tau_b"]),
                "zero_friction_mean_spearman_rho": float(zero_row["mean_spearman_rho"]),
                "zero_friction_mean_n_comparable_pairs": float(zero_row["mean_n_comparable_pairs"]),
                "zero_friction_mean_comparable_pair_fraction": float(zero_row["mean_comparable_pair_fraction"]),
                "best_positive_friction": best_positive_friction,
                "best_positive_flip_rate": best_positive_flip_rate,
                "best_positive_kendall_drop": best_positive_kendall_drop,
                "best_positive_spearman_drop": best_positive_spearman_drop,
                "strongest_flip_pair": strongest_flip_pair,
                "strongest_flip_share": strongest_flip_share,
                "positive_flip_rate_increase_flag": positive_flip_increase,
                "positive_spearman_drop_flag": positive_spearman_drop,
                "required_domain_pass_flag": required_domain_pass_flag if spec.domain_role == "required" else pd.NA,
                "stretch_domain_pass_flag": stretch_domain_pass_flag if spec.domain_role == "stretch" else pd.NA,
                "inclusion_status": inclusion_status,
                "verdict_note": verdict_note,
            }
        )

    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df["_domain_order"] = verdict_df["domain"].map(DOMAIN_ORDER).fillna(999).astype(int)
    verdict_df = verdict_df.sort_values("_domain_order").drop(columns="_domain_order").reset_index(drop=True)

    verdict_path = output_dir / "domain_step5_verdict.csv"
    verdict_df.to_csv(verdict_path, index=False)

    manifest = {
        "build_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "step5_interpretation": STEP5_INTERPRETATION,
        "rank_method": "descending_average_rank",
        "tie_policy": {
            "score_equality_rule": "abs(a-b) <= max(1e-10, 1e-8 * max(|a|, |b|, 1.0))",
            "pairwise_flip_rule": "only comparable pairs; ties on forecast or executed side are excluded from numerator and denominator",
        },
        "kendall_variant": "tau_b",
        "spearman_definition": "pearson_correlation_on_recomputed_average_ranks",
        "portfolio_gate_definition": portfolio_gate_definition,
        "sources": manifest_sources,
        "portfolio_inclusion_status": str(
            verdict_df.loc[verdict_df["domain"] == "portfolio", "inclusion_status"].iloc[0]
            if (verdict_df["domain"] == "portfolio").any()
            else "missing"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _write_markdown(output_dir / "step5_verdict.md", _internal_verdict_lines(verdict_df))

    paper_table = _paper_table_from_verdicts(verdict_df)
    _write_paper_table_csv(paper_table, paper_results_dir / "table_step5_same_interface_summary.csv")
    _write_paper_table_tex(paper_table, paper_results_dir / "table_step5_same_interface_summary.tex")
    _write_markdown(paper_results_dir / "step5_same_interface_note_v1.md", _paper_note_lines(verdict_df))

    inventory_selection_summary = domain_outputs_by_name["inventory"]["selection_summary_by_friction"].copy()
    inventory_selection_seed = domain_outputs_by_name["inventory"]["seed_level_selection_stats"].copy()
    selection_table = _selection_table_from_inventory_summary(inventory_selection_summary, inventory_selection_seed)
    _write_selection_table_csv(selection_table, paper_results_dir / "table_q2_selection_drift_inventory.csv")
    _write_selection_table_tex(selection_table, paper_results_dir / "table_q2_selection_drift_inventory.tex")
    _write_markdown(paper_results_dir / "selection_drift_note_v1.md", _selection_note_lines(selection_table))

    print(f"[step5-package] wrote {manifest_path}")
    print(f"[step5-package] wrote {verdict_path}")
    print(f"[step5-package] wrote {output_dir / 'step5_verdict.md'}")
    print(f"[step5-package] wrote {paper_results_dir / 'table_step5_same_interface_summary.csv'}")
    print(f"[step5-package] wrote {paper_results_dir / 'table_step5_same_interface_summary.tex'}")
    print(f"[step5-package] wrote {paper_results_dir / 'step5_same_interface_note_v1.md'}")
    print(f"[step5-package] wrote {paper_results_dir / 'table_q2_selection_drift_inventory.csv'}")
    print(f"[step5-package] wrote {paper_results_dir / 'table_q2_selection_drift_inventory.tex'}")
    print(f"[step5-package] wrote {paper_results_dir / 'selection_drift_note_v1.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
