#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_same_interface_rank_summary import build_domain_rank_summary  # noqa: E402


DEFAULT_LOCK_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "load_dispatch_support_locked"
DEFAULT_PAPER_RESULTS_DIR = REPO_ROOT / "paper" / "forecasting_workshop" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build locked summaries for the load-following support domain.")
    parser.add_argument("--q1-csv", default=str(DEFAULT_LOCK_DIR / "load_dispatch_seed_stability_q1.csv"))
    parser.add_argument("--q2-csv", default=str(DEFAULT_LOCK_DIR / "load_dispatch_seed_stability_q2.csv"))
    parser.add_argument("--diagnostics-csv", default=str(DEFAULT_LOCK_DIR / "load_dispatch_seed_stability_diagnostics.csv"))
    parser.add_argument("--freeze-csv", default=str(DEFAULT_LOCK_DIR / "load_dispatch_seed_stability_freeze_check.csv"))
    parser.add_argument("--window-schedule-csv", default=str(DEFAULT_LOCK_DIR / "load_dispatch_seed_stability_window_schedule.csv"))
    parser.add_argument("--run-summary-csv", default=str(DEFAULT_LOCK_DIR / "load_dispatch_seed_stability_summary.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_LOCK_DIR))
    parser.add_argument("--paper-results-dir", default=str(DEFAULT_PAPER_RESULTS_DIR))
    return parser.parse_args()


def build_q1_threshold_summary(q1_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = (
        q1_df.pivot_table(
            index=["seed", "friction_level"],
            columns="interface_id",
            values=["executed_metric", "target_executed_gap"],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = [
        column[0] if isinstance(column, tuple) and column[1] == "" else f"{column[0]}__{column[1]}"
        for column in pivot.columns
    ]
    pivot["window_abs_gap_mean"] = pivot[["target_executed_gap__responsive", "target_executed_gap__tempered"]].abs().mean(axis=1)
    pivot["executed_delta_tempered_minus_responsive"] = (
        pivot["executed_metric__tempered"] - pivot["executed_metric__responsive"]
    )
    pivot["tempered_win"] = pivot["executed_delta_tempered_minus_responsive"] > 0.0

    summary = (
        pivot.groupby("friction_level", as_index=False)
        .agg(
            windows=("seed", "count"),
            tempered_win_count=("tempered_win", "sum"),
            tempered_win_rate=("tempered_win", "mean"),
            mean_executed_delta_tempered_minus_responsive=("executed_delta_tempered_minus_responsive", "mean"),
            median_executed_delta_tempered_minus_responsive=("executed_delta_tempered_minus_responsive", "median"),
            mean_window_abs_gap=("window_abs_gap_mean", "mean"),
            median_window_abs_gap=("window_abs_gap_mean", "median"),
            mean_abs_target_executed_gap_tempered=("target_executed_gap__tempered", lambda s: float(np.mean(np.abs(s)))),
            mean_executed_metric_responsive=("executed_metric__responsive", "mean"),
            mean_executed_metric_tempered=("executed_metric__tempered", "mean"),
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )
    return summary, pivot


def build_diagnostics_share_summary(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        diagnostics_df.groupby(["question_id", "scenario_id", "interface_id", "friction_level"], as_index=False)
        .agg(
            windows=("seed", "count"),
            mean_shortage_cost=("mean_shortage_cost", "mean"),
            mean_surplus_cost=("mean_surplus_cost", "mean"),
            mean_ramp_cost=("mean_ramp_cost", "mean"),
            mean_dispatch_adjustment=("mean_dispatch_adjustment", "mean"),
            mean_dispatch_target_clip_rate=("dispatch_target_clip_rate", "mean"),
            mean_dispatch_exec_clip_rate=("dispatch_exec_clip_rate", "mean"),
            mean_dispatch=("mean_dispatch", "mean"),
            mean_load=("mean_load", "mean"),
        )
        .sort_values(["question_id", "interface_id", "friction_level"])
        .reset_index(drop=True)
    )
    total = (summary["mean_shortage_cost"] + summary["mean_surplus_cost"] + summary["mean_ramp_cost"]).replace(0.0, np.nan)
    summary["shortage_cost_share"] = summary["mean_shortage_cost"] / total
    summary["surplus_cost_share"] = summary["mean_surplus_cost"] / total
    summary["ramp_cost_share"] = summary["mean_ramp_cost"] / total
    return summary.fillna(0.0)


def build_q2_forecast_vs_deployed_summary(
    rank_corr: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    strongest = (
        pairwise.sort_values(
            ["friction_level", "flip_seed_share", "model_a", "model_b"],
            ascending=[True, False, True, True],
        )
        .groupby("friction_level", as_index=False)
        .first()
    )
    strongest["strongest_flip_pair"] = strongest["model_a"].fillna("") + "|" + strongest["model_b"].fillna("")
    strongest.loc[strongest["model_a"].isna() | strongest["model_b"].isna(), "strongest_flip_pair"] = ""
    summary = rank_corr.merge(
        strongest[["friction_level", "strongest_flip_pair", "flip_seed_share"]],
        on="friction_level",
        how="left",
    ).rename(columns={"flip_seed_share": "strongest_flip_share"})
    return summary


def _bool_all(frame: pd.DataFrame, column: str) -> bool:
    return bool(frame[column].astype(bool).all()) if not frame.empty else False


def assess_q1(q1_df: pd.DataFrame, freeze_df: pd.DataFrame) -> dict[str, Any]:
    summary, window_level = build_q1_threshold_summary(q1_df)
    zero_row = summary[np.isclose(summary["friction_level"], 0.0, atol=1e-15)].iloc[0]
    positive = summary[summary["friction_level"] > 0.0].copy()
    high_row = summary[np.isclose(summary["friction_level"], 1.0, atol=1e-15)].iloc[0]

    freeze_ok = bool(
        _bool_all(freeze_df, "forecast_hash_identical_flag")
        and _bool_all(freeze_df, "target_hash_identical_flag")
        and _bool_all(freeze_df, "initial_prev_target_match_flag")
        and _bool_all(freeze_df, "initial_prev_dispatch_match_flag")
        and int(freeze_df["pairing_failure_count"].sum()) == 0
    )
    zero_gap_ok = float(zero_row["mean_window_abs_gap"]) <= 1e-12
    positive_gap_exists = bool((positive["mean_window_abs_gap"] > 1e-12).any())
    high_win_rate = float(high_row["tempered_win_rate"])
    threshold_ok = high_win_rate >= 0.6

    if freeze_ok and zero_gap_ok and positive_gap_exists and threshold_ok:
        status = "support_pass"
    elif freeze_ok and zero_gap_ok and positive_gap_exists:
        status = "support_mixed"
    else:
        status = "support_fail"

    return {
        "status": status,
        "freeze_ok": freeze_ok,
        "zero_gap_ok": zero_gap_ok,
        "positive_gap_exists": positive_gap_exists,
        "high_friction_tempered_win_rate": high_win_rate,
        "zero_friction_mean_window_abs_gap": float(zero_row["mean_window_abs_gap"]),
        "summary_df": summary,
        "window_level_df": window_level,
    }


def assess_q2(q2_df: pd.DataFrame) -> dict[str, Any]:
    outputs, meta = build_domain_rank_summary(
        q2_df,
        domain="load_dispatch",
        expected_interface_id="responsive",
    )
    rank_corr = outputs["rank_correlation_by_friction"].copy()
    pairwise = outputs["pairwise_flips_by_friction"].copy()
    zero = rank_corr[np.isclose(rank_corr["friction_level"], 0.0, atol=1e-15)]
    if zero.empty:
        raise RuntimeError("Q2 summary is missing the zero-friction row.")
    zero_row = zero.iloc[0]
    positive = rank_corr[rank_corr["friction_level"] > 0.0].copy()

    zero_flip_ok = float(zero_row["mean_flip_rate"]) <= 0.10
    positive_flip_exists = bool((positive["mean_flip_rate"] > float(zero_row["mean_flip_rate"])).any())
    positive_corr_drop_exists = bool((positive["mean_spearman_rho"] < float(zero_row["mean_spearman_rho"])).any())
    pair_share_ok = bool((pairwise[pairwise["friction_level"] > 0.0]["flip_seed_share"] >= 0.50).any())
    min_forecasters_ok = int(meta.min_n_forecasters_per_seed_friction) >= 4
    paper_facing_valid = bool(min_forecasters_ok)
    support_pass = bool(zero_flip_ok and positive_flip_exists and positive_corr_drop_exists and pair_share_ok and paper_facing_valid)
    support_mixed = bool(zero_flip_ok and positive_flip_exists and positive_corr_drop_exists and pair_share_ok and not paper_facing_valid)
    status = "support_pass" if support_pass else ("invalid" if not paper_facing_valid else "support_mixed")

    strongest_pair = ""
    strongest_share = 0.0
    if not pairwise.empty:
        strongest = (
            pairwise[pairwise["friction_level"] > 0.0]
            .sort_values(["flip_seed_share", "friction_level", "model_a", "model_b"], ascending=[False, True, True, True])
            .reset_index(drop=True)
        )
        if not strongest.empty:
            row = strongest.iloc[0]
            strongest_pair = f"{row['model_a']}|{row['model_b']}"
            strongest_share = float(row["flip_seed_share"])

    return {
        "status": status,
        "paper_facing_valid": paper_facing_valid,
        "min_forecasters_per_seed_friction": int(meta.min_n_forecasters_per_seed_friction),
        "zero_friction_mean_flip_rate": float(zero_row["mean_flip_rate"]),
        "zero_friction_mean_spearman_rho": float(zero_row["mean_spearman_rho"]),
        "positive_flip_exists": positive_flip_exists,
        "positive_corr_drop_exists": positive_corr_drop_exists,
        "pair_share_ok": pair_share_ok,
        "strongest_flip_pair": strongest_pair,
        "strongest_flip_share": strongest_share,
        "outputs": outputs,
        "meta": meta,
    }


def _safe_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _latex_escape(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\textbackslash ").replace("_", "\\_")


def _support_note(q1_assessment: dict[str, Any], q2_assessment: dict[str, Any]) -> str:
    if not q2_assessment["paper_facing_valid"]:
        return "archived raw outputs only"
    if q2_assessment["status"] == "support_pass" and q1_assessment["status"] == "support_pass":
        return "overlapping rolling windows; appendix-only support"
    if q2_assessment["status"] == "support_pass":
        return "Q2 compatible; Q1 mixed threshold support"
    if q1_assessment["status"] == "support_pass":
        return "Q1 support only; Q2 mixed"
    return "mixed support; appendix-only"


def write_paper_table(
    paper_results_dir: Path,
    *,
    q1_assessment: dict[str, Any],
    q2_assessment: dict[str, Any],
) -> pd.DataFrame:
    table_df = pd.DataFrame(
        [
            {
                "Domain": "Load-following proxy",
                "Role": "support",
                "Windows": 10,
                "Q1": q1_assessment["status"].replace("_", " "),
                "Q1 Zero Gap": q1_assessment["zero_friction_mean_window_abs_gap"],
                "Q1 Win@1.0": q1_assessment["high_friction_tempered_win_rate"],
                "Q2": q2_assessment["status"].replace("_", " "),
                "Zero Flip Rate": q2_assessment["zero_friction_mean_flip_rate"],
                "Zero Spearman": q2_assessment["zero_friction_mean_spearman_rho"],
                "Strongest Flip Pair": q2_assessment["strongest_flip_pair"] or "--",
                "Strongest Flip Share": q2_assessment["strongest_flip_share"],
                "Note": _support_note(q1_assessment, q2_assessment),
            }
        ]
    )
    csv_path = paper_results_dir / "table_load_following_support_summary.csv"
    tex_path = paper_results_dir / "table_load_following_support_summary.tex"
    table_df.to_csv(csv_path, index=False)

    row = table_df.iloc[0]
    tex_lines = [
        r"\begin{tabular}{llrllllllrl}",
        r"\toprule",
        r"Domain & Role & Windows & Q1 & Q1 Zero Gap & Q1 Win@1.0 & Q2 & Zero Flip Rate & Zero Spearman & Strongest Flip Pair & Note \\",
        r"\midrule",
        (
            f"{_latex_escape(row['Domain'])} & "
            f"{_latex_escape(row['Role'])} & "
            f"{int(row['Windows'])} & "
            f"{_latex_escape(row['Q1'])} & "
            f"{_safe_float(row['Q1 Zero Gap'])} & "
            f"{_safe_float(row['Q1 Win@1.0'])} & "
            f"{_latex_escape(row['Q2'])} & "
            f"{_safe_float(row['Zero Flip Rate'])} & "
            f"{_safe_float(row['Zero Spearman'])} & "
            f"{_latex_escape(row['Strongest Flip Pair'])} & "
            f"{_latex_escape(row['Note'])} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex_path.write_text("\n".join(tex_lines) + "\n")
    return table_df


def write_verdict_note(
    output_dir: Path,
    paper_results_dir: Path,
    *,
    q1_assessment: dict[str, Any],
    q2_assessment: dict[str, Any],
    run_summary: pd.Series,
) -> None:
    verdict_path = output_dir / "load_dispatch_support_verdict.md"
    paper_note_path = paper_results_dir / "load_following_support_note.md"

    main_text_eligible = bool(q2_assessment["status"] == "support_pass" and q2_assessment["paper_facing_valid"])
    lines = [
        "# Load-Following Support Domain Verdict",
        "",
        "- Domain role: appendix-only operational support domain.",
        "- Interpretation: the 10 window seeds are overlapping rolling stability windows rather than independent replicates.",
        (
            f"- Window schedule: train_hours={int(run_summary['train_hours'])}, "
            f"eval_hours={int(run_summary['eval_hours'])}, step_hours={int(run_summary['window_step_hours'])}."
        ),
        "- The public hourly series does not support a gap-free 2-year training window under the fixed 10-window schedule, so the locked support run uses a 365-day training slice and a 180-day evaluation slice.",
        "- Forecast-path hashes are retained for auditability, but target-path identity is the primary Q1 exact-control condition.",
        "",
        "## Q1",
        f"- Status: {q1_assessment['status']}.",
        f"- Zero-friction mean window abs gap: {_safe_float(q1_assessment['zero_friction_mean_window_abs_gap'])}.",
        f"- High-friction tempered win-rate: {_safe_float(q1_assessment['high_friction_tempered_win_rate'])}.",
        "",
        "## Q2",
        f"- Status: {q2_assessment['status']}.",
        f"- Zero-friction mean flip rate: {_safe_float(q2_assessment['zero_friction_mean_flip_rate'])}.",
        f"- Zero-friction mean Spearman rho: {_safe_float(q2_assessment['zero_friction_mean_spearman_rho'])}.",
        f"- Strongest positive-friction flip pair: {q2_assessment['strongest_flip_pair'] or '--'} ({_safe_float(q2_assessment['strongest_flip_share'])}).",
        "",
        "## Paper-facing use",
        (
            f"- Paper-facing use valid: {'yes' if q2_assessment['paper_facing_valid'] else 'no'}."
        ),
        (
            "- Invalid paper-facing use still allows archival of raw outputs, but it blocks both the optional "
            "main-text sentence and any positive appendix-support wording."
        ),
        (
            f"- Optional main-text sentence eligible: {'yes' if main_text_eligible else 'no'}."
        ),
        "- This support domain is not promoted into the main evidence hierarchy in the current submission round.",
        "",
    ]
    verdict_path.write_text("\n".join(lines))

    paper_note_lines = [
        "Load-following support-domain reading",
        "",
        f"- Q1 status: {q1_assessment['status']}",
        f"- Q2 status: {q2_assessment['status']}",
        "- The 10 window seeds are overlapping rolling stability windows rather than independent replicates.",
        "- This appendix-only domain is not part of the main evidence hierarchy in the current submission round.",
    ]
    paper_note_path.write_text("\n".join(paper_note_lines) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paper_results_dir = Path(args.paper_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_results_dir.mkdir(parents=True, exist_ok=True)

    q1_df = pd.read_csv(args.q1_csv)
    q2_df = pd.read_csv(args.q2_csv)
    diagnostics_df = pd.read_csv(args.diagnostics_csv)
    freeze_df = pd.read_csv(args.freeze_csv)
    window_schedule_df = pd.read_csv(args.window_schedule_csv)
    run_summary_df = pd.read_csv(args.run_summary_csv)
    run_summary = run_summary_df.iloc[0]

    q1_assessment = assess_q1(q1_df, freeze_df)
    q2_assessment = assess_q2(q2_df)

    q1_summary_path = output_dir / "load_dispatch_q1_friction_threshold_summary.csv"
    q2_summary_path = output_dir / "load_dispatch_q2_forecast_vs_deployed_summary.csv"
    rank_corr_path = output_dir / "load_dispatch_q2_rank_correlation_by_friction.csv"
    pairwise_path = output_dir / "load_dispatch_q2_pairwise_flips_by_friction.csv"
    diagnostics_path = output_dir / "load_dispatch_diagnostics_share_by_friction.csv"

    q1_assessment["summary_df"].to_csv(q1_summary_path, index=False)
    q2_forecast_vs_deployed = build_q2_forecast_vs_deployed_summary(
        q2_assessment["outputs"]["rank_correlation_by_friction"],
        q2_assessment["outputs"]["pairwise_flips_by_friction"],
    )
    q2_forecast_vs_deployed.to_csv(q2_summary_path, index=False)
    q2_assessment["outputs"]["rank_correlation_by_friction"].to_csv(rank_corr_path, index=False)
    q2_assessment["outputs"]["pairwise_flips_by_friction"].to_csv(pairwise_path, index=False)
    build_diagnostics_share_summary(diagnostics_df).to_csv(diagnostics_path, index=False)
    window_schedule_df.to_csv(output_dir / "load_dispatch_window_schedule.csv", index=False)

    write_paper_table(
        paper_results_dir,
        q1_assessment=q1_assessment,
        q2_assessment=q2_assessment,
    )
    write_verdict_note(
        output_dir,
        paper_results_dir,
        q1_assessment=q1_assessment,
        q2_assessment=q2_assessment,
        run_summary=run_summary,
    )

    print(f"[load-dispatch-summary] wrote {q1_summary_path}")
    print(f"[load-dispatch-summary] wrote {q2_summary_path}")
    print(f"[load-dispatch-summary] wrote {rank_corr_path}")
    print(f"[load-dispatch-summary] wrote {pairwise_path}")
    print(f"[load-dispatch-summary] wrote {diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
