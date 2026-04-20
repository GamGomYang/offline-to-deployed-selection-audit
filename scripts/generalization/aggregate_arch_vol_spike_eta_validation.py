#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from target_exec_audit_utils import classify_pair, format_float, kappa_sort_key, zero_cost_near_flat_override


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = (
    ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "architecture_matrix"
    / "raw_candidates_v2"
    / "arch_vol_spike_eta"
    / "validation"
)
DEFAULT_OUTPUT_CSV = (
    ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "arch_vol_spike_eta"
    / "validation_results.csv"
)
DEFAULT_OUTPUT_NOTE = (
    ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "notes"
    / "arch_vol_spike_eta_validation_note.md"
)
VALIDATION_NEAR_FLAT_THRESHOLD = 0.01
POSITIVE_KAPPAS = [5e-4, 1e-3]
CHAMPION_RELATIVE_FLOOR = 0.90
RUNNERUP_RELATIVE_FLOOR = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate validation-only volatility-spike eta results.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Validation raw result root.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Destination validation CSV.")
    parser.add_argument("--output-note", default=str(DEFAULT_OUTPUT_NOTE), help="Destination validation note.")
    return parser.parse_args()


def _format_trigger_key(value: float) -> str:
    return f"{float(value):.2f}"


def _format_eta_low_key(value: float) -> str:
    return f"{float(value):.3f}"


def _config_key(trigger: float, eta_low: float, lookback_sigma: int, lookback_ref: int) -> str:
    return (
        f"trigger_{_format_trigger_key(trigger)}__etaLow_{_format_eta_low_key(eta_low)}"
        f"__lb_{int(lookback_sigma)}__ref_{int(lookback_ref)}"
    )


def _load_rows(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result_path in sorted(raw_root.glob("kappa_*/trigger_*__etaLow_*/seed_*/result.json")):
        rows.append(json.loads(result_path.read_text()))
    if not rows:
        raise FileNotFoundError(f"No vol-spike validation result.json files found under {raw_root}")
    df = pd.DataFrame(rows)
    numeric_cols = [
        "seed",
        "kappa",
        "trigger",
        "eta_low",
        "lookback_sigma",
        "lookback_ref",
        "sharpe_exec_net",
        "sharpe_target_net",
        "turnover_exec",
        "turnover_target",
        "delta_vs_reference_exec",
        "delta_vs_reference_target",
        "reference_sharpe_exec_net",
        "reference_sharpe_target_net",
        "reference_turnover_exec",
        "reference_turnover_target",
        "tracking_error_l2",
        "final_path_gap",
        "cost_exec",
        "cost_target",
        "cagr_exec",
        "mdd_exec",
        "steps",
        "mean_eta_t",
        "mean_spike",
        "activation_rate",
        "mean_sigma_proxy",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _turnover_reduction_pct(turnover_selected: float, turnover_reference: float) -> float:
    if not np.isfinite(turnover_selected) or not np.isfinite(turnover_reference) or turnover_reference <= 0.0:
        return float("nan")
    return float(((turnover_reference - turnover_selected) / turnover_reference) * 100.0)


def _build_row_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for row in raw_df.itertuples(index=False):
        audit = classify_pair(
            metric_exec_a=float(row.sharpe_exec_net),
            metric_exec_b=float(row.reference_sharpe_exec_net),
            metric_tgt_a=float(row.sharpe_target_net),
            metric_tgt_b=float(row.reference_sharpe_target_net),
        )
        audit = zero_cost_near_flat_override(
            audit,
            kappa=float(row.kappa),
            near_flat_threshold=VALIDATION_NEAR_FLAT_THRESHOLD,
        )
        zero_cost_near_flat_flag = (
            "yes"
            if np.isclose(float(row.kappa), 0.0) and abs(float(audit.delta_exec)) <= VALIDATION_NEAR_FLAT_THRESHOLD
            else ("no" if np.isclose(float(row.kappa), 0.0) else "n/a")
        )
        positive_cost_direction_flag = (
            "yes" if float(row.kappa) > 0.0 and float(audit.delta_exec) > 0.0 else ("no" if float(row.kappa) > 0.0 else "n/a")
        )
        mean_eta_t = float(row.mean_eta_t) if np.isfinite(row.mean_eta_t) else float("nan")
        intervention_pct = float((1.0 - mean_eta_t) * 100.0) if np.isfinite(mean_eta_t) else float("nan")
        out_rows.append(
            {
                "config_key": _config_key(
                    float(row.trigger),
                    float(row.eta_low),
                    int(row.lookback_sigma),
                    int(row.lookback_ref),
                ),
                "trigger": float(row.trigger),
                "eta_low": float(row.eta_low),
                "lookback_sigma": int(row.lookback_sigma),
                "lookback_ref": int(row.lookback_ref),
                "kappa": float(row.kappa),
                "delta_sharpe_exec": float(audit.delta_exec),
                "delta_sharpe_tgt": float(audit.delta_tgt),
                "turnover_reduction_pct": _turnover_reduction_pct(float(row.turnover_exec), float(row.reference_turnover_exec)),
                "rank_exec": audit.rank_exec,
                "rank_tgt": audit.rank_tgt,
                "sign_exec": audit.sign_exec,
                "sign_tgt": audit.sign_tgt,
                "disagreement_type": audit.disagreement_type,
                "disagreement_strength": int(audit.disagreement_strength),
                "zero_cost_near_flat_flag": zero_cost_near_flat_flag,
                "positive_cost_direction_flag": positive_cost_direction_flag,
                "mean_eta_t": mean_eta_t,
                "mean_intervention_pct": intervention_pct,
                "mean_spike": float(row.mean_spike) if np.isfinite(row.mean_spike) else float("nan"),
                "activation_rate": float(row.activation_rate) if np.isfinite(row.activation_rate) else float("nan"),
                "selected_arm": str(row.selected_arm),
                "reference_arm": str(row.reference_arm),
            }
        )
    out_df = pd.DataFrame(out_rows)
    return out_df.sort_values(
        ["trigger", "eta_low", "lookback_sigma", "lookback_ref", "kappa"],
        key=lambda s: s.map(kappa_sort_key) if s.name == "kappa" else s,
    ).reset_index(drop=True)


def _attach_config_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_rows: list[dict[str, object]] = []
    for keys, group in df.groupby(["trigger", "eta_low", "lookback_sigma", "lookback_ref"], sort=True):
        trigger, eta_low, lookback_sigma, lookback_ref = keys
        group = group.sort_values("kappa", key=lambda s: s.map(kappa_sort_key))
        zero_row = group[np.isclose(group["kappa"], 0.0, atol=1e-15)]
        pos_rows = group[group["kappa"].isin(POSITIVE_KAPPAS)].copy()

        near_flat_pass = bool((zero_row["zero_cost_near_flat_flag"] == "yes").all()) if not zero_row.empty else False
        positive_pass = bool((pos_rows["delta_sharpe_exec"] > 0.0).all()) if len(pos_rows) == len(POSITIVE_KAPPAS) else False
        disagreement_pass = bool(pos_rows["disagreement_type"].isin(["ranking_mismatch", "sign_flip"]).any())
        eligibility_flag = bool(near_flat_pass and positive_pass and disagreement_pass)

        mean_positive_cost_delta = float(pos_rows["delta_sharpe_exec"].mean()) if not pos_rows.empty else float("nan")
        sum_positive_disagreement = int(pos_rows["disagreement_strength"].sum()) if not pos_rows.empty else 0
        mean_positive_turnover_reduction = float(pos_rows["turnover_reduction_pct"].mean()) if not pos_rows.empty else float("nan")
        mean_positive_intervention_pct = float(pos_rows["mean_intervention_pct"].mean()) if not pos_rows.empty else float("nan")
        mean_positive_activation_rate = float(pos_rows["activation_rate"].mean()) if not pos_rows.empty else float("nan")
        zero_cost_abs_delta = float(zero_row["delta_sharpe_exec"].abs().max()) if not zero_row.empty else float("nan")

        config_rows.append(
            {
                "config_key": _config_key(float(trigger), float(eta_low), int(lookback_sigma), int(lookback_ref)),
                "trigger": float(trigger),
                "eta_low": float(eta_low),
                "lookback_sigma": int(lookback_sigma),
                "lookback_ref": int(lookback_ref),
                "eligibility_flag": "yes" if eligibility_flag else "no",
                "zero_cost_near_flat_pass": "yes" if near_flat_pass else "no",
                "positive_cost_direction_pass": "yes" if positive_pass else "no",
                "positive_cost_disagreement_pass": "yes" if disagreement_pass else "no",
                "zero_cost_abs_delta_sharpe_exec": zero_cost_abs_delta,
                "mean_positive_cost_delta_sharpe_exec": mean_positive_cost_delta,
                "sum_positive_cost_disagreement_strength": sum_positive_disagreement,
                "mean_positive_cost_turnover_reduction_pct": mean_positive_turnover_reduction,
                "mean_positive_cost_intervention_pct": mean_positive_intervention_pct,
                "mean_positive_cost_activation_rate": mean_positive_activation_rate,
            }
        )

    config_df = pd.DataFrame(config_rows)
    config_df["champion_recommendation"] = "no"
    config_df["runnerup_recommendation"] = "no"
    eligible_df = config_df[config_df["eligibility_flag"] == "yes"].copy()
    if not eligible_df.empty:
        best_score = float(eligible_df["mean_positive_cost_delta_sharpe_exec"].max())
        champion_band = eligible_df[
            eligible_df["mean_positive_cost_delta_sharpe_exec"] >= CHAMPION_RELATIVE_FLOOR * best_score
        ].copy()
        champion_band = champion_band.sort_values(
            by=[
                "zero_cost_abs_delta_sharpe_exec",
                "mean_positive_cost_intervention_pct",
                "mean_positive_cost_activation_rate",
                "sum_positive_cost_disagreement_strength",
                "trigger",
                "eta_low",
            ],
            ascending=[True, True, True, False, False, False],
        ).reset_index(drop=True)
        champion_key = str(champion_band.iloc[0]["config_key"])
        config_df.loc[config_df["config_key"] == champion_key, "champion_recommendation"] = "yes"

        remaining = eligible_df[eligible_df["config_key"] != champion_key].copy()
        if not remaining.empty:
            runner_best = float(remaining["mean_positive_cost_delta_sharpe_exec"].max())
            runner_band = remaining[
                remaining["mean_positive_cost_delta_sharpe_exec"] >= RUNNERUP_RELATIVE_FLOOR * runner_best
            ].copy()
            runner_band = runner_band.sort_values(
                by=[
                    "zero_cost_abs_delta_sharpe_exec",
                    "mean_positive_cost_intervention_pct",
                    "mean_positive_cost_activation_rate",
                    "sum_positive_cost_disagreement_strength",
                    "trigger",
                    "eta_low",
                ],
                ascending=[True, True, True, False, False, False],
            ).reset_index(drop=True)
            runner_key = str(runner_band.iloc[0]["config_key"])
            config_df.loc[config_df["config_key"] == runner_key, "runnerup_recommendation"] = "yes"

    out_df = df.merge(
        config_df,
        on=["config_key", "trigger", "eta_low", "lookback_sigma", "lookback_ref"],
        how="left",
    )
    return out_df, config_df


def _write_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _write_note(config_df: pd.DataFrame, output_note: Path) -> None:
    output_note.parent.mkdir(parents=True, exist_ok=True)
    eligible_df = config_df[config_df["eligibility_flag"] == "yes"].copy()
    champion = config_df[config_df["champion_recommendation"] == "yes"].copy()
    runnerup = config_df[config_df["runnerup_recommendation"] == "yes"].copy()

    eligible_list: list[str] = []
    for row in eligible_df.sort_values(
        by=["mean_positive_cost_delta_sharpe_exec", "sum_positive_cost_disagreement_strength"],
        ascending=[False, False],
    ).itertuples(index=False):
        eligible_list.append(
            f"- `{row.config_key}`: mean positive-cost `ΔSharpe_exec={format_float(row.mean_positive_cost_delta_sharpe_exec)}`, "
            f"sum disagreement strength `{int(row.sum_positive_cost_disagreement_strength)}`, "
            f"mean turnover reduction `{format_float(row.mean_positive_cost_turnover_reduction_pct, digits=1)}%`, "
            f"mean intervention `{format_float(row.mean_positive_cost_intervention_pct, digits=3)}%`, "
            f"activation `{format_float(row.mean_positive_cost_activation_rate, digits=3)}`"
        )
    if not eligible_list:
        eligible_list = ["- None. This family remains a redesign candidate."]

    champion_text = "No champion recommendation was produced."
    if not champion.empty:
        row = champion.iloc[0]
        champion_text = (
            f"`{row['config_key']}` with mean positive-cost `ΔSharpe_exec={format_float(row['mean_positive_cost_delta_sharpe_exec'])}`, "
            f"zero-cost `|ΔSharpe_exec|={format_float(row['zero_cost_abs_delta_sharpe_exec'])}`, "
            f"mean intervention `{format_float(row['mean_positive_cost_intervention_pct'], digits=3)}%`, "
            f"activation `{format_float(row['mean_positive_cost_activation_rate'], digits=3)}`"
        )

    runnerup_text = "No runner-up recommendation was produced."
    if not runnerup.empty:
        row = runnerup.iloc[0]
        runnerup_text = (
            f"`{row['config_key']}` with mean positive-cost `ΔSharpe_exec={format_float(row['mean_positive_cost_delta_sharpe_exec'])}`, "
            f"zero-cost `|ΔSharpe_exec|={format_float(row['zero_cost_abs_delta_sharpe_exec'])}`, "
            f"mean intervention `{format_float(row['mean_positive_cost_intervention_pct'], digits=3)}%`, "
            f"activation `{format_float(row['mean_positive_cost_activation_rate'], digits=3)}`"
        )

    note = "\n".join(
        [
            "# Volatility-Spike Eta Validation Note",
            "",
            "This note summarizes the redesigned validation-only grid search for `arch_vol_spike_eta`.",
            "",
            "The validation eligibility rule uses the same documented near-flat threshold as the deadband comparator:",
            "",
            f"- `|ΔSharpe_exec(kappa=0)| <= {VALIDATION_NEAR_FLAT_THRESHOLD:.2f}`",
            "",
            "The remaining eligibility checks are also the same:",
            "",
            "- both positive-cost rows (`kappa in {5e-4, 1e-3}`) must have `ΔSharpe_exec > 0`",
            "- at least one positive-cost row must show `disagreement_type in {ranking_mismatch, sign_flip}`",
            "",
            "Selection is validation-based only. Within the eligible set, the champion is chosen from a high-score band and then filtered by stability-first tie-breaks:",
            "",
            "- smaller zero-cost `|ΔSharpe_exec|`",
            "- smaller mean intervention away from full rebalance",
            "- smaller activation rate",
            "- larger disagreement strength",
            "- simpler parameters",
            "",
            "Eligible configurations:",
            *eligible_list,
            "",
            f"Champion recommendation: {champion_text}",
            f"Runner-up recommendation: {runnerup_text}",
        ]
    )
    output_note.write_text(note + "\n")


def main() -> int:
    args = parse_args()
    raw_df = _load_rows(Path(args.raw_root).resolve())
    row_df = _build_row_metrics(raw_df)
    output_df, config_df = _attach_config_summary(row_df)
    _write_csv(output_df, Path(args.output_csv).resolve())
    _write_note(config_df, Path(args.output_note).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
