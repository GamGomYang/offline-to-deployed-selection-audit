#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from target_exec_audit_utils import classify_pair, format_float, kappa_label, kappa_sort_key, zero_cost_near_flat_override


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = (
    ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "architecture_matrix"
    / "raw_candidates_v3"
    / "arch_deadband_partial"
    / "validation"
)
DEFAULT_OUTPUT_CSV = (
    ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "outputs"
    / "arch_deadband_partial"
    / "validation_results.csv"
)
DEFAULT_OUTPUT_NOTE = (
    ROOT
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "notes"
    / "arch_deadband_partial_validation_note.md"
)
VALIDATION_NEAR_FLAT_THRESHOLD = 0.01
POSITIVE_KAPPAS = [5e-4, 1e-3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate validation-only deadband partial results.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Validation raw result root.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Destination validation CSV.")
    parser.add_argument("--output-note", default=str(DEFAULT_OUTPUT_NOTE), help="Destination validation note.")
    return parser.parse_args()


def _load_rows(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result_path in sorted(raw_root.glob("kappa_*/delta_*__eta_*/seed_*/result.json")):
        rows.append(json.loads(result_path.read_text()))
    if not rows:
        raise FileNotFoundError(f"No deadband validation result.json files found under {raw_root}")
    df = pd.DataFrame(rows)
    numeric_cols = [
        "seed",
        "kappa",
        "delta",
        "eta_db",
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
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _format_delta_key(value: float) -> str:
    return f"{float(value):.2f}"


def _format_eta_key(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


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
            "yes"
            if float(row.kappa) > 0.0 and float(audit.delta_exec) > 0.0
            else ("no" if float(row.kappa) > 0.0 else "n/a")
        )
        out_rows.append(
            {
                "config_key": f"delta_{_format_delta_key(float(row.delta))}__eta_{_format_eta_key(float(row.eta_db))}",
                "delta": float(row.delta),
                "eta_db": float(row.eta_db),
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
                "selected_arm": str(row.selected_arm),
                "reference_arm": str(row.reference_arm),
            }
        )
    out_df = pd.DataFrame(out_rows)
    return out_df.sort_values(["delta", "eta_db", "kappa"], key=lambda s: s.map(kappa_sort_key) if s.name == "kappa" else s).reset_index(drop=True)


def _simplicity_sort_tuple(delta: float, eta_db: float) -> tuple[float, float]:
    # Simpler means wider deadband and smaller partial move.
    return (-float(delta), float(eta_db))


def _attach_config_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_rows: list[dict[str, object]] = []
    for (delta, eta_db), group in df.groupby(["delta", "eta_db"], sort=True):
        group = group.sort_values("kappa", key=lambda s: s.map(kappa_sort_key))
        zero_row = group[np.isclose(group["kappa"], 0.0, atol=1e-15)]
        pos_rows = group[group["kappa"].isin(POSITIVE_KAPPAS)].copy()

        near_flat_pass = bool((zero_row["zero_cost_near_flat_flag"] == "yes").all()) if not zero_row.empty else False
        positive_pass = bool((pos_rows["delta_sharpe_exec"] > 0.0).all()) if len(pos_rows) == len(POSITIVE_KAPPAS) else False
        disagreement_pass = bool(pos_rows["disagreement_type"].isin(["ranking_mismatch", "sign_flip"]).any())
        eligibility_flag = bool(near_flat_pass and positive_pass and disagreement_pass)

        mean_positive_cost_delta = float(pos_rows["delta_sharpe_exec"].mean()) if not pos_rows.empty else float("nan")
        sum_positive_disagreement = int(pos_rows["disagreement_strength"].sum()) if not pos_rows.empty else 0
        mean_positive_turnover_reduction = (
            float(pos_rows["turnover_reduction_pct"].mean()) if not pos_rows.empty else float("nan")
        )

        config_rows.append(
            {
                "config_key": f"delta_{_format_delta_key(float(delta))}__eta_{_format_eta_key(float(eta_db))}",
                "delta": float(delta),
                "eta_db": float(eta_db),
                "eligibility_flag": "yes" if eligibility_flag else "no",
                "zero_cost_near_flat_pass": "yes" if near_flat_pass else "no",
                "positive_cost_direction_pass": "yes" if positive_pass else "no",
                "positive_cost_disagreement_pass": "yes" if disagreement_pass else "no",
                "mean_positive_cost_delta_sharpe_exec": mean_positive_cost_delta,
                "sum_positive_cost_disagreement_strength": sum_positive_disagreement,
                "mean_positive_cost_turnover_reduction_pct": mean_positive_turnover_reduction,
                "simplicity_delta_preference": -float(delta),
                "simplicity_eta_preference": float(eta_db),
            }
        )

    config_df = pd.DataFrame(config_rows).sort_values(
        by=[
            "eligibility_flag",
            "mean_positive_cost_delta_sharpe_exec",
            "sum_positive_cost_disagreement_strength",
            "mean_positive_cost_turnover_reduction_pct",
            "simplicity_delta_preference",
            "simplicity_eta_preference",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)

    champion_key = None
    eligible_df = config_df[config_df["eligibility_flag"] == "yes"].copy()
    if not eligible_df.empty:
        champion_key = str(eligible_df.iloc[0]["config_key"])
    elif not config_df.empty:
        champion_key = str(config_df.iloc[0]["config_key"])

    out_df = df.merge(
        config_df.drop(columns=["simplicity_delta_preference", "simplicity_eta_preference"]),
        on=["config_key", "delta", "eta_db"],
        how="left",
    )
    out_df["champion_recommendation"] = out_df["config_key"].map(lambda key: "yes" if key == champion_key else "no")
    config_df["champion_recommendation"] = config_df["config_key"].map(lambda key: "yes" if key == champion_key else "no")
    return out_df, config_df


def _write_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _write_note(config_df: pd.DataFrame, output_note: Path) -> None:
    output_note.parent.mkdir(parents=True, exist_ok=True)
    eligible_df = config_df[config_df["eligibility_flag"] == "yes"].copy()
    champion = eligible_df.iloc[0] if not eligible_df.empty else None

    eligible_list = []
    for row in eligible_df.itertuples(index=False):
        eligible_list.append(
            f"- `{row.config_key}`: mean positive-cost `ΔSharpe_exec={format_float(row.mean_positive_cost_delta_sharpe_exec)}`, "
            f"sum disagreement strength `{int(row.sum_positive_cost_disagreement_strength)}`, "
            f"mean turnover reduction `{format_float(row.mean_positive_cost_turnover_reduction_pct, digits=1)}%`"
        )
    if not eligible_list:
        eligible_list.append("- No deadband configuration met the current validation eligibility rule. Treat this as a grid-adjustment candidate rather than as a hard Red.")

    if champion is None and not config_df.empty:
        best_overall = config_df.iloc[0]
        champion_text = (
            "No eligible test candidate is recommended yet. The best observed validation configuration under the requested tie-break "
            f"is provisionally `{best_overall.config_key}`, "
            "but it remains ineligible because it fails the zero-cost near-flat screen."
        )
    elif champion is None:
        champion_text = "No champion is recommended yet because no configuration satisfied the validation eligibility rule."
    else:
        champion_text = (
            "The recommended deadband configuration for later test evaluation is "
            f"`{champion.config_key}`. "
            "It is selected by largest mean positive-cost `ΔSharpe_exec`, then largest summed disagreement strength, "
            "then largest mean positive-cost turnover reduction, and finally by the simpler parameter preference "
            "(wider deadband, smaller partial step)."
        )

    text = f"""# Deadband Partial Validation Note

This note summarizes the validation-only grid search for `arch_deadband_partial`.

The validation eligibility rule uses the following documented near-flat threshold:

- `|ΔSharpe_exec(kappa=0)| <= 0.01`

The remaining eligibility checks are:

- both positive-cost rows (`kappa in {{5e-4, 1e-3}}`) must have `ΔSharpe_exec > 0`
- at least one positive-cost row must show `disagreement_type in {{ranking_mismatch, sign_flip}}`

This stage uses the same Step 8 pair-audit logic for ranking, sign, and disagreement labels, but it keeps the validation qualification threshold at `0.01` for the zero-cost near-flat screen. Test has not been run in this step.

Eligible configurations:
{chr(10).join(eligible_list)}

{champion_text}
"""
    output_note.write_text(text)


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_note = Path(args.output_note).resolve()

    raw_df = _load_rows(raw_root)
    row_df = _build_row_metrics(raw_df)
    out_df, config_df = _attach_config_summary(row_df)
    _write_csv(out_df, output_csv)
    _write_note(config_df, output_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
