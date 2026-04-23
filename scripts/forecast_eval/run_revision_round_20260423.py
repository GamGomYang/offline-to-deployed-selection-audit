#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for candidate in (str(SCRIPT_DIR), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from build_same_interface_rank_summary import build_domain_rank_summary, write_summary_outputs  # noqa: E402
from revision_round_20260423 import (  # noqa: E402
    DEFAULT_TIE_ABS_FLOOR,
    DEFAULT_TIE_REL_SCALE,
    EVENT_MICRO_CANONICAL_SEED40_CONFIG,
    EVENT_MICRO_CONFIG_DIR,
    EVENT_MICRO_DIR,
    EVENT_MICRO_PAPER_LABELS,
    EVENT_MICRO_REGIME_CONFIGS,
    EVENT_MICRO_REGIME_DIR,
    EVENT_MICRO_THRESHOLD_CONFIGS,
    EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS,
    EXPECTED_LOAD_FOLLOWING_FRICTIONS,
    EXTENSION_ROOT,
    LOAD_FOLLOWING_DIR,
    LOAD_FOLLOWING_PAPER_LABELS,
    PAPER_STAGING_DIR,
    REVISION_ROUND_ID,
    RUN_TO_PAPER_REGIME_LABELS,
    GateResult,
    build_q2_from_seed_metrics,
    compact_selection_summary,
    determine_track,
    draft_proposition_lines,
    ensure_dir,
    evaluate_c_seed_gate,
    evaluate_workstream_a,
    evaluate_workstream_c,
    evaluate_workstream_d,
    model_label,
    paper_selection_table,
    validate_load_following_raw_candidate,
    write_json,
    write_markdown,
    write_table_bundle,
)


RUN_EVENT_MICRO_SCRIPT = SCRIPT_DIR / "run_event_micro.py"
SETUP_SCRIPT = SCRIPT_DIR / "setup_revision_round_20260423.py"
LOAD_FOLLOWING_PROMOTION_RAW = (
    ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_promotion_locked" / "q2_diff_forecasts_same_interface.csv"
)
LOAD_FOLLOWING_BALANCE_RAW = (
    ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_balance_repair_locked" / "q2_diff_forecasts_same_interface.csv"
)
LOAD_FOLLOWING_RERUN_SCRIPT = SCRIPT_DIR / "run_load_following_elecdiag.py"

THRESHOLD_DISPLAY = {
    "tau045": "tau=0.45",
    "tau050": "tau=0.50",
    "tau055": "tau=0.55",
    "tau060": "tau=0.60",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the selective experiment-first workshop revision round.")
    parser.add_argument("--skip-setup", action="store_true", help="Assume the isolated workspace already exists.")
    parser.add_argument(
        "--run-tie-robustness",
        action="store_true",
        help="Run the optional canonical tie-tolerance robustness summary.",
    )
    parser.add_argument(
        "--allow-load-following-rerun",
        action="store_true",
        help="If all existing load-following candidates fail, authorize a fresh rerun in the isolated extension root.",
    )
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def _materialize_workspace(skip_setup: bool) -> None:
    if skip_setup:
        for path in [EXTENSION_ROOT, EVENT_MICRO_DIR, EVENT_MICRO_REGIME_DIR, LOAD_FOLLOWING_DIR, PAPER_STAGING_DIR]:
            ensure_dir(path)
        return
    _run_command([sys.executable, str(SETUP_SCRIPT)])


def _run_event_micro(config_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(output_dir)
    _run_command(
        [
            sys.executable,
            str(RUN_EVENT_MICRO_SCRIPT),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--skip-summary-refresh",
        ]
    )
    raw_df = pd.read_csv(output_dir / "q2_diff_forecasts_same_interface.csv")
    seed_metrics_df = pd.read_csv(output_dir / "seed_level_metrics.csv")
    outputs, _meta = build_domain_rank_summary(
        raw_df,
        domain="event_micro",
        expected_interface_id="fixed_threshold",
        tie_abs_floor=DEFAULT_TIE_ABS_FLOOR,
        tie_rel_scale=DEFAULT_TIE_REL_SCALE,
    )
    derived_dir = output_dir / "derived"
    write_summary_outputs(outputs, derived_dir)
    return raw_df, seed_metrics_df, outputs["selection_summary_by_friction"].copy()


def _write_event_micro_run_table(selection_summary: pd.DataFrame, output_stem: Path) -> pd.DataFrame:
    compact = compact_selection_summary(selection_summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS)
    paper_table = paper_selection_table(compact, label_map=EVENT_MICRO_PAPER_LABELS)
    write_table_bundle(paper_table, output_stem)
    return paper_table


def _build_logloss_summary(seed_metrics_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    logloss_q2_df = build_q2_from_seed_metrics(
        seed_metrics_df,
        forecast_metric_column="logloss",
        scenario_id="event_micro_revision_round_logloss",
    )
    outputs, _meta = build_domain_rank_summary(
        logloss_q2_df,
        domain="event_micro",
        expected_interface_id="fixed_threshold",
        tie_abs_floor=DEFAULT_TIE_ABS_FLOOR,
        tie_rel_scale=DEFAULT_TIE_REL_SCALE,
    )
    derived_dir = output_dir / "derived_logloss"
    write_summary_outputs(outputs, derived_dir)
    return outputs["selection_summary_by_friction"].copy()


def _build_tie_robustness_summary(raw_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    outputs, _meta = build_domain_rank_summary(
        raw_df,
        domain="event_micro",
        expected_interface_id="fixed_threshold",
        tie_abs_floor=1e-8,
        tie_rel_scale=1e-6,
    )
    derived_dir = output_dir / "derived_tie_robustness"
    write_summary_outputs(outputs, derived_dir)
    return outputs["selection_summary_by_friction"].copy()


def _threshold_family_table(selection_by_threshold: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for threshold_name, summary in selection_by_threshold.items():
        paper_table = paper_selection_table(
            compact_selection_summary(summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS),
            label_map=EVENT_MICRO_PAPER_LABELS,
        )
        for row in paper_table.to_dict(orient="records"):
            rows.append({"Threshold": THRESHOLD_DISPLAY[threshold_name], **row})
    return pd.DataFrame(rows)


def _metric_robustness_table(
    canonical_brier_summary: pd.DataFrame,
    canonical_logloss_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for metric_name, summary in [("Brier", canonical_brier_summary), ("Log loss", canonical_logloss_summary)]:
        paper_table = paper_selection_table(
            compact_selection_summary(summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS),
            label_map=EVENT_MICRO_PAPER_LABELS,
        )
        for row in paper_table.to_dict(orient="records"):
            rows.append({"Metric": metric_name, **row})
    return pd.DataFrame(rows)


def _regime_comparison_table(canonical_seed40: pd.DataFrame, regime_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    all_summaries = {"canonical_seed40": canonical_seed40, **regime_summaries}
    for regime_name, summary in all_summaries.items():
        paper_table = paper_selection_table(
            compact_selection_summary(summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS),
            label_map=EVENT_MICRO_PAPER_LABELS,
        )
        for row in paper_table.to_dict(orient="records"):
            rows.append({"Regime": RUN_TO_PAPER_REGIME_LABELS[regime_name], **row})
    return pd.DataFrame(rows)


def _stage_proposition() -> GateResult:
    proposition_lines = draft_proposition_lines()
    write_markdown(PAPER_STAGING_DIR / "appendix_proposition_draft.md", proposition_lines)
    write_markdown(
        PAPER_STAGING_DIR / "main_text_pointer_sentence.md",
        ["A short appendix proposition formalizes why forecast-side ordering need not be preserved after frictional execution."],
    )
    return GateResult(
        status="B GO",
        passed=True,
        details={
            "appendix_nonempty_line_count": int(len([line for line in proposition_lines if line.strip()])),
            "main_text_sentence_count": 1,
        },
    )


def _load_following_selection_summary(raw_path: Path, output_dir: Path) -> tuple[pd.DataFrame, GateResult, dict[str, object]]:
    raw_df = pd.read_csv(raw_path)
    validation = validate_load_following_raw_candidate(raw_df)
    if validation["raw_valid"]:
        outputs = validation["outputs"]
        selection_summary = outputs["selection_summary_by_friction"].copy()
        compact = compact_selection_summary(selection_summary, frictions=EXPECTED_LOAD_FOLLOWING_FRICTIONS)
        paper_table = paper_selection_table(compact, label_map=LOAD_FOLLOWING_PAPER_LABELS)
        write_table_bundle(paper_table, output_dir / "table_load_following_selection_candidate")
        write_summary_outputs(outputs, output_dir / "derived")
        d_gate = evaluate_workstream_d(selection_summary)
    else:
        selection_summary = pd.DataFrame()
        d_gate = GateResult(status="D NO-GO", passed=False, details={"invalid_reasons": validation["invalid_reasons"]})
    return selection_summary, d_gate, validation


def _choose_load_following_candidate(allow_rerun: bool) -> tuple[str, pd.DataFrame | None, GateResult | None, dict[str, object] | None]:
    candidates = [
        ("promotion_locked", LOAD_FOLLOWING_PROMOTION_RAW),
        ("balance_repair_locked", LOAD_FOLLOWING_BALANCE_RAW),
    ]
    candidate_summaries: list[dict[str, object]] = []
    for candidate_name, raw_path in candidates:
        candidate_dir = LOAD_FOLLOWING_DIR / candidate_name
        ensure_dir(candidate_dir)
        selection_summary, d_gate, validation = _load_following_selection_summary(raw_path, candidate_dir)
        candidate_summaries.append(
            {
                "candidate_name": candidate_name,
                "raw_path": str(raw_path),
                "raw_valid": bool(validation["raw_valid"]),
                "invalid_reasons": validation["invalid_reasons"],
                "d_status": d_gate.status,
                "d_passed": d_gate.passed,
            }
        )
        if validation["raw_valid"] and d_gate.passed:
            return candidate_name, selection_summary, d_gate, {"candidates": candidate_summaries, "selected_raw_path": str(raw_path)}

    if allow_rerun:
        rerun_dir = LOAD_FOLLOWING_DIR / "rerun_candidate"
        ensure_dir(rerun_dir)
        _run_command([sys.executable, str(LOAD_FOLLOWING_RERUN_SCRIPT), "--output-dir", str(rerun_dir)])
        rerun_raw = rerun_dir / "q2_diff_forecasts_same_interface.csv"
        selection_summary, d_gate, validation = _load_following_selection_summary(rerun_raw, rerun_dir)
        candidate_summaries.append(
            {
                "candidate_name": "rerun_candidate",
                "raw_path": str(rerun_raw),
                "raw_valid": bool(validation["raw_valid"]),
                "invalid_reasons": validation["invalid_reasons"],
                "d_status": d_gate.status,
                "d_passed": d_gate.passed,
            }
        )
        if validation["raw_valid"] and d_gate.passed:
            return "rerun_candidate", selection_summary, d_gate, {"candidates": candidate_summaries, "selected_raw_path": str(rerun_raw)}

    return "none", None, None, {"candidates": candidate_summaries}


def _write_status_markdown(
    *,
    a_gate: GateResult,
    b_gate: GateResult | None,
    c_seed_gate: GateResult | None,
    c_gate: GateResult | None,
    d_gate: GateResult | None,
    track_summary: dict[str, object],
) -> None:
    lines = [
        f"# {REVISION_ROUND_ID} Status",
        "",
        f"- Workstream A: {a_gate.status}",
        f"- Workstream B: {b_gate.status if b_gate is not None else 'not_run'}",
        f"- Workstream C seed gate: {c_seed_gate.status if c_seed_gate is not None else 'not_run'}",
        f"- Workstream C: {c_gate.status if c_gate is not None else 'not_run'}",
        f"- Workstream D: {d_gate.status if d_gate is not None else 'not_run'}",
        f"- Evidence track: {track_summary['evidence_track']}",
        f"- Final track: {track_summary['final_track']}",
    ]
    if track_summary["notes"]:
        lines.append("")
        lines.append("## Notes")
        lines.extend([f"- {note}" for note in track_summary["notes"]])
    write_markdown(EXTENSION_ROOT / "revision_round_status.md", lines)


def main() -> int:
    args = parse_args()
    _materialize_workspace(skip_setup=args.skip_setup)

    threshold_summaries: dict[str, pd.DataFrame] = {}
    canonical_raw_df: pd.DataFrame | None = None
    canonical_seed_metrics_df: pd.DataFrame | None = None
    for threshold_name, config_path in EVENT_MICRO_THRESHOLD_CONFIGS.items():
        run_dir = EVENT_MICRO_DIR / threshold_name
        raw_df, seed_metrics_df, selection_summary = _run_event_micro(config_path, run_dir)
        _write_event_micro_run_table(selection_summary, run_dir / "table_selection_compact")
        threshold_summaries[threshold_name] = selection_summary
        if threshold_name == "tau055":
            canonical_raw_df = raw_df
            canonical_seed_metrics_df = seed_metrics_df

    threshold_family_df = _threshold_family_table(threshold_summaries)
    write_table_bundle(threshold_family_df, PAPER_STAGING_DIR / "table_event_micro_threshold_family")

    if canonical_seed_metrics_df is None or canonical_raw_df is None:
        raise RuntimeError("Canonical tau055 event-micro run did not complete.")

    canonical_logloss_summary = _build_logloss_summary(canonical_seed_metrics_df, EVENT_MICRO_DIR / "tau055")
    metric_robustness_df = _metric_robustness_table(
        canonical_brier_summary=threshold_summaries["tau055"],
        canonical_logloss_summary=canonical_logloss_summary,
    )
    write_table_bundle(metric_robustness_df, PAPER_STAGING_DIR / "table_event_micro_metric_robustness")

    if args.run_tie_robustness:
        tie_summary = _build_tie_robustness_summary(canonical_raw_df, EVENT_MICRO_DIR / "tau055")
        tie_paper_table = paper_selection_table(
            compact_selection_summary(tie_summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS),
            label_map=EVENT_MICRO_PAPER_LABELS,
        )
        write_table_bundle(tie_paper_table, PAPER_STAGING_DIR / "table_event_micro_tie_robustness")

    a_gate = evaluate_workstream_a(threshold_summaries)
    write_json(EVENT_MICRO_DIR / "workstream_a_gate.json", {"status": a_gate.status, "passed": a_gate.passed, "details": a_gate.details})

    b_gate: GateResult | None = None
    c_seed_gate: GateResult | None = None
    c_gate: GateResult | None = None
    d_gate: GateResult | None = None
    d_candidate_info: dict[str, object] | None = None

    if not a_gate.passed:
        b_gate = GateResult(status="not_run_due_to_stop_loss", passed=False, details={})
        track_summary = determine_track(
            a_gate=a_gate,
            b_gate=b_gate,
            c_gate=None,
            d_gate=None,
            layout_preflight_passed=False,
        )
        write_json(
            EXTENSION_ROOT / "revision_round_decision_ledger.json",
            {
                "revision_round_id": REVISION_ROUND_ID,
                "stop_loss_triggered": True,
                "a_gate": {"status": a_gate.status, "details": a_gate.details},
                "track_summary": track_summary,
            },
        )
        _write_status_markdown(
            a_gate=a_gate,
            b_gate=b_gate,
            c_seed_gate=None,
            c_gate=None,
            d_gate=None,
            track_summary=track_summary,
        )
        return 0

    b_gate = _stage_proposition()
    write_json(PAPER_STAGING_DIR / "proposition_gate.json", {"status": b_gate.status, "passed": b_gate.passed, "details": b_gate.details})

    canonical_seed40_dir = EVENT_MICRO_REGIME_DIR / "canonical_seed40"
    _, _, canonical_seed40_summary = _run_event_micro(EVENT_MICRO_CANONICAL_SEED40_CONFIG, canonical_seed40_dir)
    _write_event_micro_run_table(canonical_seed40_summary, canonical_seed40_dir / "table_selection_compact")
    c_seed_gate = evaluate_c_seed_gate(canonical_seed40_summary)
    write_json(
        EVENT_MICRO_REGIME_DIR / "workstream_c_seed_gate.json",
        {"status": c_seed_gate.status, "passed": c_seed_gate.passed, "details": c_seed_gate.details},
    )

    regime_summaries: dict[str, pd.DataFrame] = {}
    if c_seed_gate.passed:
        for regime_name, config_path in EVENT_MICRO_REGIME_CONFIGS.items():
            run_dir = EVENT_MICRO_REGIME_DIR / regime_name
            _, _, regime_summary = _run_event_micro(config_path, run_dir)
            _write_event_micro_run_table(regime_summary, run_dir / "table_selection_compact")
            regime_summaries[regime_name] = regime_summary
        regime_comparison_df = _regime_comparison_table(canonical_seed40_summary, regime_summaries)
        write_table_bundle(regime_comparison_df, PAPER_STAGING_DIR / "table_event_micro_regime_comparison")
        c_gate = evaluate_workstream_c(canonical_seed40_summary, regime_summaries)
    else:
        c_gate = GateResult(status="C NO-GO", passed=False, details={"seed_gate": c_seed_gate.details, "regimes": {}})
    write_json(EVENT_MICRO_REGIME_DIR / "workstream_c_gate.json", {"status": c_gate.status, "passed": c_gate.passed, "details": c_gate.details})

    selected_candidate_name, d_selection_summary, d_gate_selected, d_candidate_info = _choose_load_following_candidate(
        allow_rerun=args.allow_load_following_rerun
    )
    if d_gate_selected is None:
        d_gate = GateResult(status="D NO-GO", passed=False, details={"reason": "no_valid_candidate"})
    else:
        d_gate = d_gate_selected
        if d_selection_summary is not None:
            staged_load_following_table = paper_selection_table(
                compact_selection_summary(d_selection_summary, frictions=EXPECTED_LOAD_FOLLOWING_FRICTIONS),
                label_map=LOAD_FOLLOWING_PAPER_LABELS,
            )
            write_table_bundle(staged_load_following_table, PAPER_STAGING_DIR / "table_load_following_second_domain")
    write_json(
        LOAD_FOLLOWING_DIR / "workstream_d_gate.json",
        {
            "status": d_gate.status,
            "passed": d_gate.passed,
            "selected_candidate": selected_candidate_name,
            "details": d_gate.details,
            "candidate_info": d_candidate_info,
        },
    )

    track_summary = determine_track(
        a_gate=a_gate,
        b_gate=b_gate,
        c_gate=c_gate,
        d_gate=d_gate,
        layout_preflight_passed=False,
    )
    ledger = {
        "revision_round_id": REVISION_ROUND_ID,
        "stop_loss_triggered": False,
        "a_gate": {"status": a_gate.status, "passed": a_gate.passed, "details": a_gate.details},
        "b_gate": {"status": b_gate.status, "passed": b_gate.passed, "details": b_gate.details},
        "c_seed_gate": {"status": c_seed_gate.status, "passed": c_seed_gate.passed, "details": c_seed_gate.details},
        "c_gate": {"status": c_gate.status, "passed": c_gate.passed, "details": c_gate.details},
        "d_gate": {"status": d_gate.status, "passed": d_gate.passed, "details": d_gate.details},
        "d_candidate_info": d_candidate_info,
        "track_summary": track_summary,
    }
    write_json(EXTENSION_ROOT / "revision_round_decision_ledger.json", ledger)
    _write_status_markdown(
        a_gate=a_gate,
        b_gate=b_gate,
        c_seed_gate=c_seed_gate,
        c_gate=c_gate,
        d_gate=d_gate,
        track_summary=track_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
