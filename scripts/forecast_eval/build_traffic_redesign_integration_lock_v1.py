#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for candidate in (str(SCRIPT_DIR), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from revision_round_20260423 import PHYSICAL_STORAGE_ROOT  # noqa: E402


DEFAULT_TRAFFIC_ROOT = (
    ROOT
    / "outputs"
    / "extensions"
    / "q2_pivot_revision_20260423"
    / "new_reruns"
    / "traffic_redesign"
)
DEFAULT_TOPK_DIR = DEFAULT_TRAFFIC_ROOT / "traffic_topk_alert_q2_v1"
DEFAULT_RELATIVE_DIR = DEFAULT_TRAFFIC_ROOT / "traffic_relative_rank_q2_v1"
DEFAULT_SURGE_DIR = DEFAULT_TRAFFIC_ROOT / "traffic_surge_onset_q2_v1"
DEFAULT_LOCK_DIR = PHYSICAL_STORAGE_ROOT / "traffic_redesign_integration_lock_v1"
DEFAULT_PAPER_DIR = ROOT / "paper" / "forecasting_workshop"

PAPER_LABELS = {
    "reactive_short": "Reactive short",
    "lagged_smoother": "Lagged smoother",
    "calibrated_baseline": "Calibrated baseline",
    "linear_ar_head": "Linear AR head",
}
EXPECTED_FRICTIONS = (0.0, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Traffic redesign integration lock and paper-facing assets.")
    parser.add_argument("--topk-dir", default=str(DEFAULT_TOPK_DIR))
    parser.add_argument("--relative-rank-dir", default=str(DEFAULT_RELATIVE_DIR))
    parser.add_argument("--surge-dir", default=str(DEFAULT_SURGE_DIR))
    parser.add_argument("--lock-dir", default=str(DEFAULT_LOCK_DIR))
    parser.add_argument("--paper-dir", default=str(DEFAULT_PAPER_DIR))
    parser.add_argument("--paper-results-dir", default=None)
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_label(value: str) -> str:
    return PAPER_LABELS.get(str(value), str(value))


def format_decimal(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_report(report_path: Path, *, expected_verdict: str, expected_prefix: str) -> dict[str, Any]:
    report = load_json(report_path)
    verdict = str(report.get("verdict"))
    prefix = str(report.get("prefix"))
    if verdict != expected_verdict:
        raise RuntimeError(f"{report_path} verdict {verdict!r} did not match expected {expected_verdict!r}")
    if prefix != expected_prefix:
        raise RuntimeError(f"{report_path} prefix {prefix!r} did not match expected {expected_prefix!r}")
    return report


def select_summary_rows(summary_path: Path, *, variant_id: str, k: int) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    work = frame.loc[(frame["variant_id"] == variant_id) & (frame["k"] == k)].copy()
    if work.empty:
        raise RuntimeError(f"No rows found in {summary_path} for variant={variant_id!r}, k={k}")
    work["friction"] = work["friction"].astype(float)
    work = work.loc[work["friction"].isin(EXPECTED_FRICTIONS)].sort_values("friction").reset_index(drop=True)
    if tuple(work["friction"].tolist()) != EXPECTED_FRICTIONS:
        raise RuntimeError(f"{summary_path} did not contain exactly the expected friction rows for {variant_id!r}, k={k}")
    return work


def build_paper_table(summary_rows: pd.DataFrame, *, total_replicates: int) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in summary_rows.itertuples(index=False):
        suboptimal_count = int(round(float(row.deployed_suboptimal_share) * total_replicates))
        rows.append(
            {
                "Friction": format_decimal(float(row.friction), 2),
                "Forecast-side winner": paper_label(str(row.forecast_winner)),
                "Deployed winner": paper_label(str(row.deployed_winner)),
                "Agreement rate": format_decimal(float(row.agreement), 2),
                "Mean deployed gap": format_decimal(float(row.mean_gap), 3),
                "Median deployed gap": format_decimal(float(row.median_gap), 3),
                "Deployed-suboptimal seeds / total": f"{suboptimal_count}/{total_replicates}",
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_tex(frame: pd.DataFrame) -> str:
    return frame.to_latex(index=False, escape=True)


def write_table_bundle(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(dataframe_to_tex(frame))


def copytree_clean(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def freeze_baseline(*, paper_dir: Path, paper_results_dir: Path, freeze_dir: Path) -> dict[str, Any]:
    freeze_dir.mkdir(parents=True, exist_ok=True)
    tex_path = paper_dir / "paper_forecasting_workshop_v2.tex"
    pdf_path = paper_dir / "paper_forecasting_workshop_v2.pdf"
    frozen_tex = freeze_dir / tex_path.name
    frozen_pdf = freeze_dir / pdf_path.name
    shutil.copy2(tex_path, frozen_tex)
    shutil.copy2(pdf_path, frozen_pdf)
    copytree_clean(paper_results_dir, freeze_dir / "results")
    payload = {
        "paper_tex": {"source": repo_relative(tex_path), "sha256": sha256_file(tex_path)},
        "paper_pdf": {"source": repo_relative(pdf_path), "sha256": sha256_file(pdf_path)},
        "paper_results_dir": {"source": repo_relative(paper_results_dir)},
    }
    write_json(freeze_dir / "baseline_freeze_manifest.json", payload)
    return payload


def build_source_manifest(
    *,
    lock_dir: Path,
    freeze_manifest: dict[str, Any],
    topk_dir: Path,
    relative_dir: Path,
    surge_dir: Path,
    topk_report: dict[str, Any],
    relative_report: dict[str, Any],
    surge_report: dict[str, Any],
) -> None:
    artifacts = [
        {
            "candidate_id": "C1",
            "scenario_id": "traffic_topk_alert_q2_v1",
            "role": "main_text_family_2",
            "report_path": repo_relative(topk_dir / "full_report.json"),
            "report_sha256": sha256_file(topk_dir / "full_report.json"),
            "selection_summary_path": repo_relative(topk_dir / "full_selection_summary_by_friction.csv"),
            "selection_summary_sha256": sha256_file(topk_dir / "full_selection_summary_by_friction.csv"),
            "verdict": topk_report["verdict"],
            "best_variant_id": topk_report["best_variant_id"],
            "best_k": int(topk_report["best_k"]),
        },
        {
            "candidate_id": "C2",
            "scenario_id": "traffic_relative_rank_q2_v1",
            "role": "appendix_forecasting_native_support",
            "report_path": repo_relative(relative_dir / "full_report.json"),
            "report_sha256": sha256_file(relative_dir / "full_report.json"),
            "selection_summary_path": repo_relative(relative_dir / "full_selection_summary_by_friction.csv"),
            "selection_summary_sha256": sha256_file(relative_dir / "full_selection_summary_by_friction.csv"),
            "verdict": relative_report["verdict"],
            "best_variant_id": relative_report["best_variant_id"],
            "best_k": int(relative_report["best_k"]),
        },
        {
            "candidate_id": "C3",
            "scenario_id": "traffic_surge_onset_q2_v1",
            "role": "internal_only_drop",
            "report_path": repo_relative(surge_dir / "pilot_report.json"),
            "report_sha256": sha256_file(surge_dir / "pilot_report.json"),
            "selection_summary_path": repo_relative(surge_dir / "pilot_selection_summary_by_friction.csv"),
            "selection_summary_sha256": sha256_file(surge_dir / "pilot_selection_summary_by_friction.csv"),
            "verdict": surge_report["verdict"],
            "best_variant_id": surge_report["best_variant_id"],
            "best_k": int(surge_report["best_k"]),
        },
    ]
    payload = {
        "integration_id": "traffic_redesign_integration_lock_v1",
        "baseline_freeze": freeze_manifest,
        "artifacts": artifacts,
    }
    write_json(lock_dir / "source_of_truth_manifest.json", payload)
    lines = [
        "# Traffic Redesign Source of Truth Manifest",
        "",
        f"- Integration id: `traffic_redesign_integration_lock_v1`",
        f"- Frozen paper tex: `{freeze_manifest['paper_tex']['source']}`",
        f"- Frozen paper pdf: `{freeze_manifest['paper_pdf']['source']}`",
        "",
        "| Candidate | Scenario | Role | Report | Selection summary | Verdict | Selected variant | Selected k |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in artifacts:
        lines.append(
            f"| {artifact['candidate_id']} | {artifact['scenario_id']} | {artifact['role']} | "
            f"`{artifact['report_path']}` | `{artifact['selection_summary_path']}` | {artifact['verdict']} | "
            f"`{artifact['best_variant_id']}` | {artifact['best_k']} |"
        )
    write_text(lock_dir / "source_of_truth_manifest.md", "\n".join(lines))


def build_claim_addendum(lock_dir: Path) -> None:
    payload = {
        "forecasting_native_q2_evidence": {
            "event_micro": "main forecasting-native evidence",
            "traffic_topk_alert": "complementary second forecasting-native family",
            "traffic_relative_rank": "appendix forecasting-native support",
        },
        "traffic_redesign_addendum": [
            {"candidate_id": "C1", "status": "strong pass", "paper_role": "main-text family 2"},
            {"candidate_id": "C2", "status": "strong pass", "paper_role": "appendix-only forecasting-native support"},
            {"candidate_id": "C3", "status": "fail", "paper_role": "internal-only, manuscript hidden"},
        ],
        "limitation_update": "Forecasting-native Q2 evidence now spans more than one task family, though it is still not a broad benchmark suite.",
        "drop_record": "Traffic Surge-Onset is internal-only due to zero-row mismatch under the current gate narrative.",
    }
    write_json(lock_dir / "claim_to_evidence_addendum.json", payload)
    lines = [
        "# Traffic Redesign Claim-to-Evidence Addendum",
        "",
        "- `C1`: strong pass, main-text family 2",
        "- `C2`: strong pass, appendix-only forecasting-native support",
        "- `C3`: fail, internal-only, manuscript hidden",
        "",
        "## Hierarchy update",
        "- `Event-micro`: main forecasting-native evidence",
        "- `Traffic Top-k Alert`: complementary second forecasting-native family",
        "- `Traffic Relative-Rank`: appendix forecasting-native support",
        "",
        "## Limitation update",
        "- Old: `one narrow task family`",
        "- New: `more than one task family, though still not a broad benchmark suite`",
        "",
        "## Internal-only drop record",
        "- `Traffic Surge-Onset`: internal-only fail due to zero-row mismatch under the current gate narrative",
    ]
    write_text(lock_dir / "claim_to_evidence_addendum.md", "\n".join(lines))


def build_evidence_map_v3(paper_results_dir: Path) -> None:
    lines = [
        "\\begin{tabularx}{\\columnwidth}{@{}l c l l >{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X@{}}",
        "\\toprule",
        "Domain & Question & Fixed & Varying & Zero-fric. & Positive-fric. & Role \\\\",
        "\\midrule",
        "Synthetic & Q2 & interface & forecaster & aligned at zero & ranking flips & zero-fric. anchor \\\\",
        "Event micro & Q2 & interface & forecaster & mixed at zero & selection failure recurs & main fcst.-native Q2 evid. \\\\",
        "Traffic Top-k Alert & Q2 & interface & forecaster & aligned at zero & switching-cost divergence recurs & comp. second fcst.-native family \\\\",
        "Traffic Relative-Rank & Q2 & interface & forecaster & aligned at zero & qualitative separation recurs & appx.-only fcst.-native support \\\\",
        "Inventory & Q2 & interface & forecaster & mixed at zero & stronger recurrence & main oper. corrob. \\\\",
        "Load-following & Q2 & interface & forecaster & mixed at zero & directional recurrence & appx.-only sec. corrob. \\\\",
        "Synthetic & Q1 & proposal & interface & exact agreement & mismatch emerges & mech. support \\\\",
        "Inventory & Q1 & proposal & interface & coincide at zero & threshold story & mech. support \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
    ]
    write_text(paper_results_dir / "table_evidence_map_v3.tex", "\n".join(lines))


def main() -> int:
    args = parse_args()
    topk_dir = Path(args.topk_dir).resolve()
    relative_dir = Path(args.relative_rank_dir).resolve()
    surge_dir = Path(args.surge_dir).resolve()
    lock_dir = Path(args.lock_dir).resolve()
    paper_dir = Path(args.paper_dir).resolve()
    paper_results_dir = Path(args.paper_results_dir).resolve() if args.paper_results_dir else (paper_dir / "results").resolve()

    topk_report = require_report(topk_dir / "full_report.json", expected_verdict="strong", expected_prefix="full")
    relative_report = require_report(relative_dir / "full_report.json", expected_verdict="strong", expected_prefix="full")
    surge_report = require_report(surge_dir / "pilot_report.json", expected_verdict="fail", expected_prefix="pilot")

    freeze_manifest = freeze_baseline(
        paper_dir=paper_dir,
        paper_results_dir=paper_results_dir,
        freeze_dir=lock_dir / "baseline_freeze",
    )

    topk_rows = select_summary_rows(
        topk_dir / "full_selection_summary_by_friction.csv",
        variant_id=str(topk_report["best_variant_id"]),
        k=int(topk_report["best_k"]),
    )
    relative_rows = select_summary_rows(
        relative_dir / "full_selection_summary_by_friction.csv",
        variant_id=str(relative_report["best_variant_id"]),
        k=int(relative_report["best_k"]),
    )

    topk_table = build_paper_table(topk_rows, total_replicates=100)
    relative_table = build_paper_table(relative_rows, total_replicates=100)
    write_table_bundle(topk_table, paper_results_dir / "table_q2_selection_drift_traffic_topk_main")
    write_table_bundle(relative_table, paper_results_dir / "table_q2_selection_drift_traffic_relative_rank_support")
    build_evidence_map_v3(paper_results_dir)

    build_source_manifest(
        lock_dir=lock_dir,
        freeze_manifest=freeze_manifest,
        topk_dir=topk_dir,
        relative_dir=relative_dir,
        surge_dir=surge_dir,
        topk_report=topk_report,
        relative_report=relative_report,
        surge_report=surge_report,
    )
    build_claim_addendum(lock_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
