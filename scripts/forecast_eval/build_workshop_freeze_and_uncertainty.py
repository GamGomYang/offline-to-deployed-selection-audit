#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["ps.useafm"] = True
matplotlib.rcParams["font.family"] = "serif"

from build_same_interface_rank_summary import build_domain_rank_summary
from revision_round_20260423 import (
    BASELINE_LOCK_DIR,
    CORE_IDENTITY_SENTENCE,
    LOGICAL_CANONICAL_ROOT,
    LOGICAL_ROOT_MAP_PATH,
    PHYSICAL_STORAGE_ROOT,
    STORY_REVISION_DIR,
    ANALYSIS_ADDITIONS_DIR,
    NEW_RERUNS_DIR,
    ensure_logical_alias,
    logical_root_map_payload,
    logical_root_relative,
    physical_root_relative,
    repo_relative,
    sha256_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

FREEZE_DIR = BASELINE_LOCK_DIR
STORY_DIR = STORY_REVISION_DIR
ANALYSIS_DIR = ANALYSIS_ADDITIONS_DIR
PAPER_DIR = ROOT / "paper" / "forecasting_workshop"
PAPER_RESULTS_DIR = PAPER_DIR / "results"
PAPER_FIGURES_DIR = PAPER_DIR / "assets" / "figures"
SNAPSHOT_DIR = FREEZE_DIR / "source_snapshot"
APPENDIX_TRIGGER = "Evidence Hierarchy"

WORKSHOP_PDF = PAPER_DIR / "paper_forecasting_workshop_v2.pdf"
WORKSHOP_TEX = PAPER_DIR / "paper_forecasting_workshop_v2.tex"

SYNTHETIC_Q2_RAW = ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q2_diff_forecasts_same_interface.csv"
SYNTHETIC_Q1_SUMMARY = ROOT / "outputs" / "forecast_eval" / "synthetic_step2_candidate_lock" / "q1_gap_by_friction.csv"
INVENTORY_Q1_SUMMARY = (
    ROOT
    / "outputs"
    / "forecast_eval"
    / "inventory_step4_seed_stability_locked"
    / "inventory_v2_seed_stability_q1_friction_threshold_summary.csv"
)
EVENT_MICRO_Q2_RAW = (
    ROOT
    / "outputs"
    / "extensions"
    / "revision_round_20260423"
    / "new_reruns"
    / "event_micro_hardening"
    / "fixed_threshold_tau055_seed100"
    / "q2_diff_forecasts_same_interface.csv"
)
INVENTORY_Q2_SEED = (
    ROOT
    / "outputs"
    / "forecast_eval"
    / "inventory_step4_seed_stability_locked"
    / "inventory_v2_seed_stability_q2_selection_seed_level.csv"
)
INVENTORY_Q2_SUMMARY = (
    ROOT
    / "outputs"
    / "forecast_eval"
    / "inventory_step4_seed_stability_locked"
    / "inventory_v2_seed_stability_q2_selection_summary_by_friction.csv"
)
LOAD_FOLLOWING_Q2_RAW = (
    ROOT / "outputs" / "forecast_eval" / "load_following_elecdiag_promotion_locked" / "q2_diff_forecasts_same_interface.csv"
)
PORTFOLIO_Q1_RAW = ROOT / "outputs" / "forecast_eval" / "portfolio_exact_control" / "q1_same_forecast_diff_interface.csv"

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260423


class AppendixDetectionError(RuntimeError):
    """Raised when appendix split detection cannot identify the canonical title trigger."""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def copy_repo_file_to_snapshot(relative_path: Path) -> Path:
    source_path = ROOT / relative_path
    destination = SNAPSHOT_DIR / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


def logical_display_path(path: Path) -> str:
    return str(logical_root_relative(path))


def physical_display_path(path: Path) -> str:
    return str(physical_root_relative(path))


def dataframe_to_tex(frame: pd.DataFrame) -> str:
    return frame.to_latex(index=False, escape=True)


def write_table_bundle(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".tex").write_text(dataframe_to_tex(frame))


def human_pvalue(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def human_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def bootstrap_interval(values: np.ndarray, *, statistic: str) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(values, dtype=float)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    n = values.size
    for idx in range(BOOTSTRAP_SAMPLES):
        sample = values[rng.integers(0, n, size=n)]
        if statistic == "mean":
            draws[idx] = float(np.mean(sample))
        elif statistic == "median":
            draws[idx] = float(np.median(sample))
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def majority_test_row(
    *,
    domain_label: str,
    friction: float,
    seed_df: pd.DataFrame,
) -> dict[str, str]:
    row_df = seed_df.loc[np.isclose(seed_df["friction_level"], friction, atol=1e-12)].copy()
    k = int(row_df["selection_disagreement_flag"].sum())
    n = int(row_df["seed"].nunique())
    result = binomtest(k, n, p=0.5, alternative="greater")
    ci = result.proportion_ci(confidence_level=0.95, method="exact")
    return {
        "Domain": domain_label,
        "Friction": f"{friction:.2f}",
        "Deployed-suboptimal seeds / total": f"{k}/{n}",
        "Share": f"{k / n:.2f}",
        "95% exact CI": human_ci(float(ci.low), float(ci.high)),
        "One-sided binomial p": human_pvalue(float(result.pvalue)),
    }


def gap_ci_row(
    *,
    domain_label: str,
    friction: float,
    seed_df: pd.DataFrame,
) -> dict[str, str]:
    row_df = seed_df.loc[np.isclose(seed_df["friction_level"], friction, atol=1e-12)].copy()
    gaps = row_df["deployed_gap_of_forecast_selected"].to_numpy(dtype=float)
    mean_gap = float(np.mean(gaps))
    median_gap = float(np.median(gaps))
    mean_lo, mean_hi = bootstrap_interval(gaps, statistic="mean")
    median_lo, median_hi = bootstrap_interval(gaps, statistic="median")
    return {
        "Domain": domain_label,
        "Friction": f"{friction:.2f}",
        "Mean deployed gap": f"{mean_gap:.3f}",
        "Mean gap 95% bootstrap CI": human_ci(mean_lo, mean_hi),
        "Median deployed gap": f"{median_gap:.3f}",
        "Median gap 95% bootstrap CI": human_ci(median_lo, median_hi),
    }


def paper_load_following_table(selection_summary: pd.DataFrame) -> pd.DataFrame:
    label_map = {
        "linear_ar_ridge": "Linear AR",
        "mlp_small": "Small MLP",
        "moving_average_24h": "Moving average (24h)",
        "naive_last": "Naive last",
    }
    rows = []
    for row in selection_summary.itertuples(index=False):
        rows.append(
            {
                "Friction": f"{float(row.friction_level):.2f}",
                "Forecast-side winner": label_map.get(str(row.most_frequent_forecast_best), str(row.most_frequent_forecast_best)),
                "Deployed winner": label_map.get(str(row.most_frequent_deployed_best), str(row.most_frequent_deployed_best)),
                "Agreement rate": f"{float(row.agreement_rate):.2f}",
                "Mean deployed gap": f"{float(row.mean_deployed_gap_of_forecast_selected):.3f}",
                "Median deployed gap": f"{float(row.median_deployed_gap_of_forecast_selected):.3f}",
                "Deployed-suboptimal seeds / total": str(row.deployed_suboptimal_seeds_over_total),
            }
        )
    return pd.DataFrame(rows)


def detect_appendix_start_page(pdf_path: Path) -> dict[str, object]:
    pdfinfo_text = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    pages = 0
    for line in pdfinfo_text.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
            break
    if pages <= 0:
        raise RuntimeError("Could not determine page count for frozen PDF.")

    page_debug: list[dict[str, object]] = []
    for page in range(1, pages + 1):
        page_text = subprocess.check_output(
            ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
            text=True,
        )
        page_debug.append(
            {
                "page_number": page,
                "contains_trigger": APPENDIX_TRIGGER in page_text,
                "excerpt": "\n".join(page_text.splitlines()[:24]),
            }
        )
        if APPENDIX_TRIGGER in page_text:
            return {
                "trigger": APPENDIX_TRIGGER,
                "detected_page": page,
                "page_count": pages,
                "pdfinfo_excerpt": "\n".join(pdfinfo_text.splitlines()[:20]),
                "page_debug": page_debug,
            }

    debug_payload = {
        "trigger": APPENDIX_TRIGGER,
        "status": "not_found",
        "page_count": pages,
        "pdf_path": str(repo_relative(pdf_path)),
        "pdfinfo_excerpt": "\n".join(pdfinfo_text.splitlines()[:20]),
        "page_debug": page_debug,
    }
    write_json(FREEZE_DIR / "appendix_split_debug.json", debug_payload)
    raise AppendixDetectionError(
        f"Could not detect appendix start page via title trigger `{APPENDIX_TRIGGER}`."
    )


def extract_appendix_pdf(pdf_path: Path, appendix_start_page: int, output_path: Path) -> None:
    info = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    total_pages = 0
    for line in info.splitlines():
        if line.startswith("Pages:"):
            total_pages = int(line.split(":", 1)[1].strip())
            break
    tmp_dir = ensure_dir(ANALYSIS_DIR / "_tmp_pdf_split")
    for file_path in tmp_dir.glob("*"):
        file_path.unlink()
    subprocess.run(
        [
            "pdfseparate",
            "-f",
            str(appendix_start_page),
            "-l",
            str(total_pages),
            str(pdf_path),
            str(tmp_dir / "page-%d.pdf"),
        ],
        check=True,
    )
    page_files = sorted(tmp_dir.glob("page-*.pdf"))
    subprocess.run(["pdfunite", *(str(path) for path in page_files), str(output_path)], check=True)
    shutil.rmtree(tmp_dir)


def build_manifest_and_claim_map(appendix_detection: dict[str, object]) -> None:
    manuscript_artifacts = [
        {
            "artifact": "Q2 main figure",
            "role": "main_text_figure",
            "paper_facing": True,
            "relative_path": Path("paper/forecasting_workshop/assets/figures/fig_q2_results_v2.pdf"),
            "raw_source_of_truth": "outputs/forecast_eval/synthetic_step2_candidate_lock/q2_diff_forecasts_same_interface.csv and outputs/extensions/revision_round_20260423/new_reruns/event_micro_hardening/fixed_threshold_tau055_seed100/q2_diff_forecasts_same_interface.csv via paper/forecasting_workshop/results/build_v2_main_figures.py",
        },
        {
            "artifact": "Inventory Q2 main table",
            "role": "main_text_table",
            "paper_facing": True,
            "relative_path": Path("paper/forecasting_workshop/results/table_q2_selection_drift_inventory.csv"),
            "raw_source_of_truth": "outputs/forecast_eval/inventory_step4_seed_stability_locked/inventory_v2_seed_stability_q2_selection_seed_level.csv",
        },
        {
            "artifact": "Event-micro Q2 main table",
            "role": "main_text_table",
            "paper_facing": True,
            "relative_path": Path("paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_main.csv"),
            "raw_source_of_truth": "outputs/extensions/revision_round_20260423/new_reruns/event_micro_hardening/fixed_threshold_tau055_seed100/q2_diff_forecasts_same_interface.csv",
        },
        {
            "artifact": "Load-following Q2 appendix table",
            "role": "appendix_table",
            "paper_facing": True,
            "relative_path": Path("paper/forecasting_workshop/results/table_load_following_support_summary.csv"),
            "raw_source_of_truth": "outputs/forecast_eval/load_following_elecdiag_promotion_locked/q2_diff_forecasts_same_interface.csv",
        },
        {
            "artifact": "Event-micro threshold robustness",
            "role": "appendix_table",
            "paper_facing": True,
            "relative_path": Path("paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_threshold_robustness.csv"),
            "raw_source_of_truth": "outputs/extensions/revision_round_20260423/new_reruns/event_micro_hardening/fixed_threshold_tau050_seed100/q2_diff_forecasts_same_interface.csv",
        },
        {
            "artifact": "Event-micro log-loss robustness",
            "role": "appendix_table",
            "paper_facing": True,
            "relative_path": Path("paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_logloss.csv"),
            "raw_source_of_truth": "outputs/extensions/revision_round_20260423/new_reruns/event_micro_hardening/fixed_threshold_tau055_seed100/derived_logloss/selection_summary_by_friction.csv",
        },
        {
            "artifact": "Q1 support figure",
            "role": "appendix_figure",
            "paper_facing": True,
            "relative_path": Path("paper/forecasting_workshop/assets/figures/fig_q1_results_v2.pdf"),
            "raw_source_of_truth": "outputs/forecast_eval/synthetic_step2_candidate_lock/q1_gap_by_friction.csv and outputs/forecast_eval/inventory_step4_seed_stability_locked/inventory_v2_seed_stability_q1_friction_threshold_summary.csv",
        },
    ]

    artifact_rows: list[dict[str, object]] = []
    frozen_main_pdf = FREEZE_DIR / "frozen_workshop_main.pdf"
    frozen_appendix_pdf = FREEZE_DIR / "frozen_workshop_appendix.pdf"
    artifact_rows.extend(
        [
            {
                "artifact": "Main workshop PDF",
                "role": "frozen_pdf",
                "paper_facing": False,
                "logical_frozen_path": logical_display_path(frozen_main_pdf),
                "physical_frozen_path": physical_display_path(frozen_main_pdf),
                "manuscript_source": str(repo_relative(WORKSHOP_PDF)),
                "raw_source_of_truth": "Compiled from the current workshop TeX and paper-facing assets.",
                "sha256": sha256_file(frozen_main_pdf),
            },
            {
                "artifact": "Appendix-only PDF",
                "role": "frozen_appendix_pdf",
                "paper_facing": False,
                "logical_frozen_path": logical_display_path(frozen_appendix_pdf),
                "physical_frozen_path": physical_display_path(frozen_appendix_pdf),
                "manuscript_source": str(repo_relative(WORKSHOP_PDF)),
                "raw_source_of_truth": f"Pages extracted by title-trigger detection on `{APPENDIX_TRIGGER}`.",
                "sha256": sha256_file(frozen_appendix_pdf),
            },
        ]
    )

    for artifact in manuscript_artifacts:
        relative_path = Path(artifact["relative_path"])
        snapshot_path = copy_repo_file_to_snapshot(relative_path)
        artifact_rows.append(
            {
                "artifact": str(artifact["artifact"]),
                "role": str(artifact["role"]),
                "paper_facing": bool(artifact["paper_facing"]),
                "logical_frozen_path": logical_display_path(snapshot_path),
                "physical_frozen_path": physical_display_path(snapshot_path),
                "manuscript_source": str(relative_path),
                "raw_source_of_truth": str(artifact["raw_source_of_truth"]),
                "sha256": sha256_file(snapshot_path),
            }
        )

    manifest_md = [
        "# Result Manifest",
        "",
        f"- Core identity sentence: `{CORE_IDENTITY_SENTENCE}`",
        f"- Canonical logical root: `{repo_relative(LOGICAL_CANONICAL_ROOT)}`",
        f"- Physical storage root: `{repo_relative(PHYSICAL_STORAGE_ROOT)}`",
        f"- Appendix split trigger: `{APPENDIX_TRIGGER}`",
        "- Page numbers are debug metadata only; title detection is the canonical split rule.",
        "",
        "| Artifact | Role | Canonical logical path | Manuscript source | Raw/source of truth | Physical provenance path |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in artifact_rows:
        manifest_md.append(
            f"| {row['artifact']} | {row['role']} | `{row['logical_frozen_path']}` | `{row['manuscript_source']}` | {row['raw_source_of_truth']} | `{row['physical_frozen_path']}` |"
        )
    write_text(FREEZE_DIR / "result_manifest.md", "\n".join(manifest_md))
    write_json(
        FREEZE_DIR / "result_manifest.json",
        {
            "logical_canonical_root": str(repo_relative(LOGICAL_CANONICAL_ROOT)),
            "physical_storage_root": str(repo_relative(PHYSICAL_STORAGE_ROOT)),
            "core_identity_sentence": CORE_IDENTITY_SENTENCE,
            "appendix_detection": appendix_detection,
            "artifacts": artifact_rows,
        },
    )

    claim_map_md = [
        "# Claim-to-Evidence Map",
        "",
        f"- Core sentence: `{CORE_IDENTITY_SENTENCE}`",
        f"- Canonical logical root: `{repo_relative(LOGICAL_CANONICAL_ROOT)}`",
        f"- Physical storage root: `{repo_relative(PHYSICAL_STORAGE_ROOT)}`",
        "",
        "## Q2-first hierarchy",
        "- `Synthetic Q2`: zero-friction sanity anchor for deployed-selection robustness.",
        "- `Event-micro Q2`: main forecasting-native evidence under one fixed thresholding interface.",
        "- `Inventory Q2`: main operational corroboration under one fixed replenishment interface.",
        "- `Load-following Q2`: short second corroboration with appendix detail.",
        "- `Q1 support`: mechanism-only explanation of why ranking can fail after deployment friction.",
        "",
        "## Claim links",
        "- `Moderate-to-high friction can produce recurrent deployed misselection under a fixed interface.`",
        "  - Event-micro main evidence: `paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_main.csv`",
        "  - Inventory main corroboration: `paper/forecasting_workshop/results/table_q2_selection_drift_inventory.csv`",
        "  - Load-following second corroboration: `paper/forecasting_workshop/results/table_load_following_support_summary.csv`",
        "- `Zero-friction alignment is a sanity anchor, not the deployed-selection endpoint.`",
        "  - Synthetic Q2 in `paper/forecasting_workshop/assets/figures/fig_q2_results_v2.pdf`",
        "  - Event-micro zero-friction row in `paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_main.csv`",
        "- `Q1 is mechanism support only.`",
        "  - Synthetic and inventory Q1 support figure: `paper/forecasting_workshop/assets/figures/fig_q1_results_v2.pdf`",
        "  - Inventory Q1 raw summary: `outputs/forecast_eval/inventory_step4_seed_stability_locked/inventory_v2_seed_stability_q1_friction_threshold_summary.csv`",
        "- `Broader-family inventory sweep and event-micro robustness stay appendix-only.`",
        "  - `paper/forecasting_workshop/results/table_q2_stronger_baseline_selection.csv`",
        "  - `paper/forecasting_workshop/results/table_q2_stronger_baseline_robustness.csv`",
        "  - `paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_threshold_robustness.csv`",
        "  - `paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_logloss.csv`",
    ]
    write_text(FREEZE_DIR / "claim_to_evidence_map.md", "\n".join(claim_map_md))
    write_json(
        FREEZE_DIR / "claim_to_evidence_map.json",
        {
            "logical_canonical_root": str(repo_relative(LOGICAL_CANONICAL_ROOT)),
            "physical_storage_root": str(repo_relative(PHYSICAL_STORAGE_ROOT)),
            "core_identity_sentence": CORE_IDENTITY_SENTENCE,
            "hierarchy": [
                "Synthetic Q2 zero-friction anchor",
                "Event-micro Q2 main forecasting-native evidence",
                "Inventory Q2 main operational corroboration",
                "Load-following Q2 short second corroboration",
                "Q1 short mechanism support",
            ],
            "claims": [
                {
                    "claim": "Moderate-to-high friction can produce recurrent deployed misselection under a fixed interface.",
                    "evidence": [
                        "paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_main.csv",
                        "paper/forecasting_workshop/results/table_q2_selection_drift_inventory.csv",
                        "paper/forecasting_workshop/results/table_load_following_support_summary.csv",
                    ],
                },
                {
                    "claim": "Zero-friction alignment is a sanity anchor, not the deployed-selection endpoint.",
                    "evidence": [
                        "paper/forecasting_workshop/assets/figures/fig_q2_results_v2.pdf",
                        "paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_main.csv",
                    ],
                },
                {
                    "claim": "Q1 is mechanism support only.",
                    "evidence": [
                        "paper/forecasting_workshop/assets/figures/fig_q1_results_v2.pdf",
                        "outputs/forecast_eval/inventory_step4_seed_stability_locked/inventory_v2_seed_stability_q1_friction_threshold_summary.csv",
                    ],
                },
            ],
        },
    )


def build_outputs() -> None:
    ensure_logical_alias()
    write_json(LOGICAL_ROOT_MAP_PATH, logical_root_map_payload())
    ensure_dir(FREEZE_DIR)
    ensure_dir(STORY_DIR)
    ensure_dir(ANALYSIS_DIR)
    ensure_dir(NEW_RERUNS_DIR)
    appendix_detection = detect_appendix_start_page(WORKSHOP_PDF)
    write_json(FREEZE_DIR / "appendix_split_debug.json", appendix_detection)
    shutil.copy2(WORKSHOP_PDF, FREEZE_DIR / "frozen_workshop_main.pdf")
    extract_appendix_pdf(
        WORKSHOP_PDF,
        int(appendix_detection["detected_page"]),
        FREEZE_DIR / "frozen_workshop_appendix.pdf",
    )
    build_manifest_and_claim_map(appendix_detection)

    story_note = [
        "# Story Revision Workspace",
        "",
        f"- Canonical logical root: `{repo_relative(LOGICAL_CANONICAL_ROOT)}`.",
        f"- Physical storage root: `{repo_relative(PHYSICAL_STORAGE_ROOT)}`.",
        f"- Core identity sentence: `{CORE_IDENTITY_SENTENCE}`",
        "- This folder is reserved for manuscript-facing narrative changes.",
        "- The current identity lock for this round is: synthetic Q2 anchor, event-micro main Q2 evidence, inventory main operational corroboration, load-following short second corroboration, and Q1 short mechanism support.",
        "- Portfolio is excluded from main text and paper-facing appendix in the revised manuscript.",
    ]
    write_text(STORY_DIR / "identity_lock_note.md", "\n".join(story_note))

    synthetic_raw = pd.read_csv(SYNTHETIC_Q2_RAW)
    synthetic_outputs, _ = build_domain_rank_summary(
        synthetic_raw,
        domain="synthetic",
        expected_interface_id="tempered",
    )
    event_micro_raw = pd.read_csv(EVENT_MICRO_Q2_RAW)
    event_micro_outputs, _ = build_domain_rank_summary(
        event_micro_raw,
        domain="event_micro",
        expected_interface_id="fixed_threshold",
    )
    load_following_raw = pd.read_csv(LOAD_FOLLOWING_Q2_RAW)
    load_following_outputs, _ = build_domain_rank_summary(
        load_following_raw,
        domain="load_following_elecdiag",
        expected_interface_id="responsive",
    )

    inventory_seed = pd.read_csv(INVENTORY_Q2_SEED)
    inventory_summary = pd.read_csv(INVENTORY_Q2_SUMMARY)
    event_micro_seed = event_micro_outputs["seed_level_selection_stats"].copy()
    event_micro_rank = event_micro_outputs["rank_correlation_by_friction"].copy()
    event_micro_seed_rank = event_micro_outputs["seed_level_rank_stats"].copy()
    load_following_seed = load_following_outputs["seed_level_selection_stats"].copy()
    load_following_summary = load_following_outputs["selection_summary_by_friction"].copy()
    synthetic_summary = synthetic_outputs["selection_summary_by_friction"].copy()
    synthetic_rank = synthetic_outputs["rank_correlation_by_friction"].copy()

    recurrence_rows = [
        majority_test_row(domain_label="Event micro", friction=0.50, seed_df=event_micro_seed),
        majority_test_row(domain_label="Event micro", friction=1.00, seed_df=event_micro_seed),
        majority_test_row(domain_label="Inventory", friction=0.50, seed_df=inventory_seed),
        majority_test_row(domain_label="Inventory", friction=1.00, seed_df=inventory_seed),
        majority_test_row(domain_label="Load-following", friction=0.25, seed_df=load_following_seed),
        majority_test_row(domain_label="Load-following", friction=0.50, seed_df=load_following_seed),
        majority_test_row(domain_label="Load-following", friction=1.00, seed_df=load_following_seed),
    ]
    recurrence_table = pd.DataFrame(recurrence_rows)
    write_table_bundle(recurrence_table, ANALYSIS_DIR / "table_q2_recurrence_tests")
    write_table_bundle(recurrence_table, PAPER_RESULTS_DIR / "table_q2_recurrence_tests")

    gap_rows = [
        gap_ci_row(domain_label="Event micro", friction=0.50, seed_df=event_micro_seed),
        gap_ci_row(domain_label="Event micro", friction=1.00, seed_df=event_micro_seed),
        gap_ci_row(domain_label="Inventory", friction=0.50, seed_df=inventory_seed),
        gap_ci_row(domain_label="Inventory", friction=1.00, seed_df=inventory_seed),
        gap_ci_row(domain_label="Load-following", friction=0.25, seed_df=load_following_seed),
        gap_ci_row(domain_label="Load-following", friction=0.50, seed_df=load_following_seed),
        gap_ci_row(domain_label="Load-following", friction=1.00, seed_df=load_following_seed),
    ]
    gap_table = pd.DataFrame(gap_rows)
    write_table_bundle(gap_table, ANALYSIS_DIR / "table_q2_gap_bootstrap_cis")
    write_table_bundle(gap_table, PAPER_RESULTS_DIR / "table_q2_gap_bootstrap_cis")

    write_table_bundle(paper_load_following_table(load_following_summary), ANALYSIS_DIR / "table_load_following_second_corroboration")
    write_table_bundle(paper_load_following_table(load_following_summary), PAPER_RESULTS_DIR / "table_load_following_support_summary")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
    for label, color, rank_df, summary_df in [
        ("Synthetic", "#1f77b4", synthetic_rank, synthetic_summary),
        ("Event micro", "#d62728", event_micro_rank, event_micro_outputs["selection_summary_by_friction"].copy()),
    ]:
        x = rank_df["friction_level"].to_numpy(dtype=float)
        mean_flip = rank_df["mean_flip_rate"].to_numpy(dtype=float)
        err = 1.96 * rank_df["stderr_flip_rate"].to_numpy(dtype=float)
        axes[0].plot(x, mean_flip, color=color, marker="o", linewidth=2.0, label=label)
        axes[0].fill_between(x, np.clip(mean_flip - err, 0.0, 1.0), np.clip(mean_flip + err, 0.0, 1.0), color=color, alpha=0.15)

        x2 = summary_df["friction_level"].to_numpy(dtype=float)
        y2 = summary_df["disagreement_rate"].to_numpy(dtype=float)
        lows = []
        highs = []
        for row in summary_df.itertuples(index=False):
            ci = binomtest(
                int(row.deployed_suboptimal_seed_count),
                int(row.n_seeds),
            ).proportion_ci(confidence_level=0.95, method="exact")
            lows.append(float(ci.low))
            highs.append(float(ci.high))
        axes[1].plot(x2, y2, color=color, marker="o", linewidth=2.0, label=label)
        axes[1].fill_between(x2, lows, highs, color=color, alpha=0.15)

    axes[0].set_title("Ranking disagreement with 95% bands")
    axes[0].set_xlabel("Friction")
    axes[0].set_ylabel("Mean flip rate")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_title("Deployed-suboptimal share with 95% bands")
    axes[1].set_xlabel("Friction")
    axes[1].set_ylabel("Forecast-selected not deployed-best")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    uncertainty_fig = ANALYSIS_DIR / "fig_q2_uncertainty_appendix.pdf"
    fig.savefig(uncertainty_fig, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(uncertainty_fig, PAPER_FIGURES_DIR / "fig_q2_uncertainty_appendix.pdf")

    strip_domains = [
        ("Event micro", event_micro_seed, [0.0, 0.5, 1.0]),
        ("Inventory", inventory_seed, [0.0, 0.25, 0.5, 1.0]),
        ("Load-following", load_following_seed, [0.0, 0.25, 0.5, 1.0]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, (title, seed_df, frictions) in zip(axes, strip_domains):
        work = seed_df.loc[seed_df["friction_level"].isin(frictions)].copy()
        work["seed_order"] = work["seed"].rank(method="dense").astype(int)
        for x_idx, friction in enumerate(frictions):
            fr_df = work.loc[np.isclose(work["friction_level"], friction, atol=1e-12)].copy()
            y = fr_df["seed_order"].to_numpy(dtype=float)
            disagree = fr_df["selection_disagreement_flag"].to_numpy(dtype=bool)
            ax.scatter(np.full_like(y, x_idx), y, s=18, facecolors=np.where(disagree, "#111111", "white"), edgecolors="#111111", linewidths=0.6)
        ax.set_title(title)
        ax.set_xticks(range(len(frictions)))
        ax.set_xticklabels([f"{value:.2f}" for value in frictions], rotation=0)
        ax.set_xlabel("Friction")
        ax.set_ylabel("Seed")
        ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    strip_fig = ANALYSIS_DIR / "fig_q2_seed_recurrence_appendix.pdf"
    fig.savefig(strip_fig, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(strip_fig, PAPER_FIGURES_DIR / "fig_q2_seed_recurrence_appendix.pdf")

    uncertainty_note = [
        "# Analysis-Only Additions",
        "",
        "- Recurrence tests use one-sided exact binomial tests against a 0.5 majority null on seed-level deployed-suboptimal indicators.",
        f"- Gap intervals use {BOOTSTRAP_SAMPLES} bootstrap resamples with seed {BOOTSTRAP_SEED}.",
        "- Synthetic and event-micro uncertainty bands are appendix-only additions; they do not change the main-text object budget.",
        "- Seed-level strip plots record whether the forecast-selected model is deployed-best in each seed/friction slice.",
    ]
    write_text(ANALYSIS_DIR / "analysis_note.md", "\n".join(uncertainty_note))

    summary_payload = {
        "logical_canonical_root": str(repo_relative(LOGICAL_CANONICAL_ROOT)),
        "physical_storage_root": str(repo_relative(PHYSICAL_STORAGE_ROOT)),
        "freeze_dir": logical_display_path(FREEZE_DIR),
        "physical_freeze_dir": physical_display_path(FREEZE_DIR),
        "story_revision_dir": logical_display_path(STORY_DIR),
        "analysis_additions_dir": logical_display_path(ANALYSIS_DIR),
        "new_reruns_dir": logical_display_path(NEW_RERUNS_DIR),
        "generated_tables": [
            "paper/forecasting_workshop/results/table_q2_recurrence_tests.tex",
            "paper/forecasting_workshop/results/table_q2_gap_bootstrap_cis.tex",
            "paper/forecasting_workshop/results/table_load_following_support_summary.tex",
        ],
        "generated_figures": [
            "paper/forecasting_workshop/assets/figures/fig_q2_uncertainty_appendix.pdf",
            "paper/forecasting_workshop/assets/figures/fig_q2_seed_recurrence_appendix.pdf",
        ],
    }
    write_json(ANALYSIS_DIR / "analysis_summary.json", summary_payload)


def main() -> int:
    build_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
