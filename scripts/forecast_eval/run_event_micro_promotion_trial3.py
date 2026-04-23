#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    EVENT_MICRO_PAPER_LABELS,
    EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS,
    EXTENSION_ROOT,
    GateResult,
    compact_selection_summary,
    ensure_dir,
    friction_row,
    paper_selection_table,
    write_json,
    write_markdown,
    write_table_bundle,
)


RUN_EVENT_MICRO_SCRIPT = SCRIPT_DIR / "run_event_micro.py"
TRIAL3_CONFIG_DIR = ROOT / "configs" / "event_micro_revision_round_20260423" / "trial3"
TRIAL_HISTORY_DIR = EXTENSION_ROOT / "trial_history"
TRIAL3_ROOT = TRIAL_HISTORY_DIR / "trial3"
PAPER_STAGING_DIR = EXTENSION_ROOT / "paper_staging"

TRIAL3_CONFIGS = {
    "canonical_seed80": TRIAL3_CONFIG_DIR / "event_micro_tau055_seed80.yaml",
    "rare_event_softened": TRIAL3_CONFIG_DIR / "event_micro_rare_event_softened_seed40.yaml",
    "bursty_moderate": TRIAL3_CONFIG_DIR / "event_micro_bursty_moderate_seed40.yaml",
}

RUN_LABELS = {
    "canonical_seed80": "Canonical regime (80 seeds)",
    "rare_event_softened": "Rare-event / softened regime",
    "bursty_moderate": "Common-event / moderate-bursty regime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the event-micro promotion-focused trial 3 package.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing trial3 outputs when present.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def run_event_micro(config_path: Path, output_dir: Path, *, skip_existing: bool) -> pd.DataFrame:
    ensure_dir(output_dir)
    raw_path = output_dir / "q2_diff_forecasts_same_interface.csv"
    seed_path = output_dir / "seed_level_metrics.csv"
    if not (skip_existing and raw_path.exists() and seed_path.exists()):
        run_command(
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
    raw_df = pd.read_csv(raw_path)
    outputs, _meta = build_domain_rank_summary(
        raw_df,
        domain="event_micro",
        expected_interface_id="fixed_threshold",
        tie_abs_floor=DEFAULT_TIE_ABS_FLOOR,
        tie_rel_scale=DEFAULT_TIE_REL_SCALE,
    )
    write_summary_outputs(outputs, output_dir / "derived")
    return outputs["selection_summary_by_friction"].copy()


def compact_paper_table(selection_summary: pd.DataFrame, output_stem: Path) -> pd.DataFrame:
    compact = compact_selection_summary(selection_summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS)
    paper_table = paper_selection_table(compact, label_map=EVENT_MICRO_PAPER_LABELS)
    write_table_bundle(paper_table, output_stem)
    return paper_table


def evaluate_canonical(selection_summary: pd.DataFrame) -> GateResult:
    compact = compact_selection_summary(selection_summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS)
    zero_row = friction_row(compact, 0.0)
    mid_row = friction_row(compact, 0.5)
    high_row = friction_row(compact, 1.0)
    zero_agreement = float(zero_row["agreement_rate"])
    mid_share = float(mid_row["deployed_suboptimal_seed_fraction"])
    high_share = float(high_row["deployed_suboptimal_seed_fraction"])
    mid_forecast = str(mid_row["most_frequent_forecast_best"])
    mid_deployed = str(mid_row["most_frequent_deployed_best"])
    high_forecast = str(high_row["most_frequent_forecast_best"])
    high_deployed = str(high_row["most_frequent_deployed_best"])
    passed = bool(
        zero_agreement >= 0.65
        and mid_share >= 0.65
        and high_share >= 0.90
        and mid_forecast == "reactive_sharp"
        and high_forecast == "reactive_sharp"
        and mid_deployed in {"calibrated_baseline", "lagged_smoother"}
        and mid_deployed != "reactive_sharp"
        and high_deployed == "lagged_smoother"
    )
    return GateResult(
        status="Trial3 canonical gate GO" if passed else "Trial3 canonical gate NO-GO",
        passed=passed,
        details={
            "zero_agreement": zero_agreement,
            "mid_suboptimal_share": mid_share,
            "high_suboptimal_share": high_share,
            "mid_forecast_winner": mid_forecast,
            "mid_deployed_winner": mid_deployed,
            "high_forecast_winner": high_forecast,
            "high_deployed_winner": high_deployed,
        },
    )


def evaluate_regime(selection_summary: pd.DataFrame) -> dict[str, object]:
    compact = compact_selection_summary(selection_summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS)
    zero_row = friction_row(compact, 0.0)
    mid_row = friction_row(compact, 0.5)
    high_row = friction_row(compact, 1.0)
    mid_share = float(mid_row["deployed_suboptimal_seed_fraction"])
    high_share = float(high_row["deployed_suboptimal_seed_fraction"])
    mid_forecast = str(mid_row["most_frequent_forecast_best"])
    mid_deployed = str(mid_row["most_frequent_deployed_best"])
    high_forecast = str(high_row["most_frequent_forecast_best"])
    high_deployed = str(high_row["most_frequent_deployed_best"])
    passed = bool(
        float(zero_row["agreement_rate"]) >= 0.45
        and mid_share >= 0.60
        and high_share >= 0.85
        and mid_forecast == "reactive_sharp"
        and high_forecast == "reactive_sharp"
        and mid_deployed in {"calibrated_baseline", "lagged_smoother"}
        and mid_deployed != "reactive_sharp"
        and high_deployed == "lagged_smoother"
    )
    return {
        "passed": passed,
        "zero_agreement": float(zero_row["agreement_rate"]),
        "mid_suboptimal_share": mid_share,
        "high_suboptimal_share": high_share,
        "mid_forecast_winner": mid_forecast,
        "mid_deployed_winner": mid_deployed,
        "high_forecast_winner": high_forecast,
        "high_deployed_winner": high_deployed,
    }


def combined_regime_table(selection_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for run_name, summary in selection_summaries.items():
        paper_table = paper_selection_table(
            compact_selection_summary(summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS),
            label_map=EVENT_MICRO_PAPER_LABELS,
        )
        for row in paper_table.to_dict(orient="records"):
            rows.append({"Regime": RUN_LABELS[run_name], **row})
    return pd.DataFrame(rows)


def stage_note() -> None:
    lines = [
        "# Event Micro Promotion: Trial 3 Notes",
        "",
        "- Trial 3 keeps the promotion gate unchanged and only changes the candidate configs.",
        "- Canonical depth is widened from 60 to 80 seeds.",
        "- The bursty regime is softened so forecast-side identity is more likely to remain Reactive sharp.",
        "- If Trial 3 still fails, the main-benchmark promotion path should be treated as exhausted for this round.",
    ]
    write_markdown(PAPER_STAGING_DIR / "event_micro_promotion_trial3_notes.md", lines)


def main() -> None:
    args = parse_args()
    ensure_dir(TRIAL3_ROOT)
    ensure_dir(PAPER_STAGING_DIR)

    selection_summaries: dict[str, pd.DataFrame] = {}
    paper_tables: dict[str, str] = {}
    for run_name, config_path in TRIAL3_CONFIGS.items():
        output_dir = TRIAL3_ROOT / run_name
        selection_summary = run_event_micro(config_path, output_dir, skip_existing=args.skip_existing)
        selection_summaries[run_name] = selection_summary
        compact_paper_table(selection_summary, output_dir / "table_selection_compact")
        paper_tables[run_name] = str((output_dir / "table_selection_compact.csv").relative_to(EXTENSION_ROOT))

    canonical_gate = evaluate_canonical(selection_summaries["canonical_seed80"])
    regime_results = {
        run_name: evaluate_regime(selection_summaries[run_name])
        for run_name in ("rare_event_softened", "bursty_moderate")
    }
    added_regime_passes = sum(1 for result in regime_results.values() if bool(result["passed"]))
    promotion_ready = bool(canonical_gate.passed and added_regime_passes == 2)

    combined_table = combined_regime_table(selection_summaries)
    write_table_bundle(combined_table, PAPER_STAGING_DIR / "table_event_micro_promotion_trial3")
    stage_note()

    ledger = {
        "trial_id": "trial3",
        "configs": {name: str(path.relative_to(ROOT)) for name, path in TRIAL3_CONFIGS.items()},
        "canonical_seed80_gate": {
            "status": canonical_gate.status,
            "passed": canonical_gate.passed,
            "details": canonical_gate.details,
        },
        "added_regimes": regime_results,
        "added_regime_pass_count": added_regime_passes,
        "promotion_ready": promotion_ready,
        "paper_tables": paper_tables,
        "staged_regime_table": str((PAPER_STAGING_DIR / "table_event_micro_promotion_trial3.csv").relative_to(EXTENSION_ROOT)),
    }
    write_json(TRIAL3_ROOT / "promotion_trial3_ledger.json", ledger)
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
