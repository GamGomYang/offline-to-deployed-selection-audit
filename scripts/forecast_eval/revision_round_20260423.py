from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_same_interface_rank_summary import build_domain_rank_summary, validate_q2_source


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

REVISION_ROUND_ID = "revision_round_20260423"
CANONICAL_LOGICAL_ROOT_ID = "q2_pivot_revision_20260423"
EXTENSIONS_ROOT = ROOT / "outputs" / "extensions"
PHYSICAL_STORAGE_ROOT = EXTENSIONS_ROOT / REVISION_ROUND_ID
LOGICAL_CANONICAL_ROOT = EXTENSIONS_ROOT / CANONICAL_LOGICAL_ROOT_ID

# Backward-compatible aliases used by existing scripts.
EXTENSION_ROOT = PHYSICAL_STORAGE_ROOT
BASELINE_LOCK_DIR = PHYSICAL_STORAGE_ROOT / "freeze_q2_pivot_base"
BASELINE_SNAPSHOT_DIR = BASELINE_LOCK_DIR / "source_snapshot"
STORY_REVISION_DIR = PHYSICAL_STORAGE_ROOT / "story_revision"
ANALYSIS_ADDITIONS_DIR = PHYSICAL_STORAGE_ROOT / "analysis_additions"
NEW_RERUNS_DIR = PHYSICAL_STORAGE_ROOT / "new_reruns"
EVENT_MICRO_DIR = NEW_RERUNS_DIR / "event_micro"
EVENT_MICRO_REGIME_DIR = NEW_RERUNS_DIR / "event_micro_regimes"
LOAD_FOLLOWING_DIR = NEW_RERUNS_DIR / "load_following"
PAPER_STAGING_DIR = STORY_REVISION_DIR
LOGICAL_ROOT_MAP_PATH = PHYSICAL_STORAGE_ROOT / "logical_root_map.json"
CORE_IDENTITY_SENTENCE = (
    "Forecasting systems under frictional deployment interfaces should report deployed-selection robustness, "
    "not forecast-side ranking alone."
)

WORKSHOP_TEX = Path("paper/forecasting_workshop/paper_forecasting_workshop_v2.tex")
WORKSHOP_PDF = Path("paper/forecasting_workshop/paper_forecasting_workshop_v2.pdf")
Q2_MAIN_FIGURE = Path("paper/forecasting_workshop/assets/figures/fig_q2_results_v2.pdf")
Q2_INVENTORY_CSV = Path("paper/forecasting_workshop/results/table_q2_selection_drift_inventory.csv")
Q2_EVENT_MICRO_MAIN_CSV = Path("paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_main.csv")
Q2_EVENT_MICRO_FULL_CSV = Path("paper/forecasting_workshop/results/table_q2_selection_drift_event_micro.csv")
Q2_EVENT_MICRO_THRESHOLD_CSV = Path(
    "paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_threshold_robustness.csv"
)
Q2_EVENT_MICRO_LOGLOSS_CSV = Path("paper/forecasting_workshop/results/table_q2_selection_drift_event_micro_logloss.csv")
LOAD_FOLLOWING_SUPPORT_CSV = Path("paper/forecasting_workshop/results/table_load_following_support_summary.csv")
V2_NUMERIC_LOCK = Path("paper/forecasting_workshop/results/v2_numeric_lock.md")

BASELINE_KEY_ARTIFACTS = [
    {"role": "workshop_tex", "path": WORKSHOP_TEX},
    {"role": "workshop_pdf", "path": WORKSHOP_PDF},
    {"role": "q2_main_figure_asset", "path": Q2_MAIN_FIGURE},
    {"role": "q2_inventory_table_csv", "path": Q2_INVENTORY_CSV},
    {"role": "q2_event_micro_main_csv", "path": Q2_EVENT_MICRO_MAIN_CSV},
    {"role": "q2_event_micro_full_csv", "path": Q2_EVENT_MICRO_FULL_CSV},
    {"role": "q2_event_micro_threshold_csv", "path": Q2_EVENT_MICRO_THRESHOLD_CSV},
    {"role": "q2_event_micro_logloss_csv", "path": Q2_EVENT_MICRO_LOGLOSS_CSV},
    {"role": "load_following_support_csv", "path": LOAD_FOLLOWING_SUPPORT_CSV},
    {"role": "v2_numeric_lock_note", "path": V2_NUMERIC_LOCK},
]

BASELINE_SNAPSHOT_DIRECTORIES = [
    Path("paper/forecasting_workshop/assets"),
    Path("paper/forecasting_workshop/results"),
]

EVENT_MICRO_CONFIG_DIR = ROOT / "configs" / "event_micro_revision_round_20260423"
EVENT_MICRO_THRESHOLD_CONFIGS = {
    "tau045": EVENT_MICRO_CONFIG_DIR / "event_micro_tau045.yaml",
    "tau050": EVENT_MICRO_CONFIG_DIR / "event_micro_tau050.yaml",
    "tau055": EVENT_MICRO_CONFIG_DIR / "event_micro_tau055.yaml",
    "tau060": EVENT_MICRO_CONFIG_DIR / "event_micro_tau060.yaml",
}
EVENT_MICRO_CANONICAL_SEED40_CONFIG = EVENT_MICRO_CONFIG_DIR / "event_micro_tau055_seed40.yaml"
EVENT_MICRO_REGIME_CONFIGS = {
    "rare_event_bursty": EVENT_MICRO_CONFIG_DIR / "event_micro_rare_event_bursty_seed40.yaml",
    "persistent_low_snr": EVENT_MICRO_CONFIG_DIR / "event_micro_persistent_low_snr_seed40.yaml",
}

EVENT_MICRO_PAPER_LABELS = {
    "calibrated_baseline": "Calibrated baseline",
    "reactive_sharp": "Reactive sharp",
    "lagged_smoother": "Lagged smoother",
    "noisy_heuristic": "Noisy heuristic",
}

LOAD_FOLLOWING_PAPER_LABELS = {
    "linear_ar_ridge": "Linear AR",
    "mlp_small": "Small MLP",
    "moving_average_24h": "Moving average (24h)",
    "naive_last": "Naive last",
}

RUN_TO_PAPER_REGIME_LABELS = {
    "canonical_seed40": "Canonical regime",
    "rare_event_bursty": "Rare-event / bursty regime",
    "persistent_low_snr": "Persistent / low-SNR regime",
}

BASELINE_FACTS = {
    "event_micro_main_role": "current main forecasting-native Q2 evidence",
    "inventory_role": "current operational corroboration",
    "event_micro_limitation": "deliberately small benchmark",
    "inventory_limitation": "one operational domain",
    "existing_robustness": "threshold and log-loss robustness already exist at one level",
}

EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS = (0.0, 0.5, 1.0)
EXPECTED_LOAD_FOLLOWING_FRICTIONS = (0.0, 0.25, 0.5, 1.0)
DEFAULT_TIE_ABS_FLOOR = 1e-10
DEFAULT_TIE_REL_SCALE = 1e-8
CORE_A_THRESHOLDS = {"tau050", "tau055"}
TRACK4_OBJECT_BUDGET = [
    "Keep the existing Q2 figure as the main-text Q2 figure.",
    "Keep the event-micro main table as the main-text forecasting-native Q2 table.",
    "Keep the inventory selection table as the main-text operational Q2 table.",
    "Allow the second domain into main text only as a short corroboration paragraph with an appendix pointer.",
]
REWRITE_ORDER = [
    "Appendix tables and robustness summaries",
    "Discussion limitations",
    "Evidence Design",
    "Results 4.1 / 4.2",
    "Abstract / Introduction last",
]


@dataclass(frozen=True)
class GateResult:
    status: str
    passed: bool
    details: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_logical_alias() -> Path:
    ensure_dir(PHYSICAL_STORAGE_ROOT.parent)
    if LOGICAL_CANONICAL_ROOT.is_symlink():
        current_target = LOGICAL_CANONICAL_ROOT.resolve()
        if current_target != PHYSICAL_STORAGE_ROOT.resolve():
            LOGICAL_CANONICAL_ROOT.unlink()
    elif LOGICAL_CANONICAL_ROOT.exists():
        if LOGICAL_CANONICAL_ROOT.resolve() != PHYSICAL_STORAGE_ROOT.resolve():
            raise RuntimeError(
                f"Logical canonical root already exists and is not the expected alias: {LOGICAL_CANONICAL_ROOT}"
            )
        return LOGICAL_CANONICAL_ROOT

    if not LOGICAL_CANONICAL_ROOT.exists():
        os.symlink(PHYSICAL_STORAGE_ROOT.name, LOGICAL_CANONICAL_ROOT)
    return LOGICAL_CANONICAL_ROOT


def repo_relative(path: Path) -> Path:
    return path.relative_to(ROOT)


def logical_root_relative(path: Path) -> Path:
    return repo_relative(LOGICAL_CANONICAL_ROOT) / path.relative_to(PHYSICAL_STORAGE_ROOT)


def physical_root_relative(path: Path) -> Path:
    return repo_relative(PHYSICAL_STORAGE_ROOT) / path.relative_to(PHYSICAL_STORAGE_ROOT)


def logical_root_map_payload() -> dict[str, Any]:
    return {
        "revision_round_id": REVISION_ROUND_ID,
        "canonical_identity_sentence": CORE_IDENTITY_SENTENCE,
        "logical_canonical_root": str(repo_relative(LOGICAL_CANONICAL_ROOT)),
        "physical_storage_root": str(repo_relative(PHYSICAL_STORAGE_ROOT)),
        "legacy_name_status": "backward-compatible alias only",
        "logical_to_physical_equivalence": {
            str(repo_relative(LOGICAL_CANONICAL_ROOT)): str(repo_relative(PHYSICAL_STORAGE_ROOT))
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_rel_path(relative_path: Path, *, snapshot_root: Path = BASELINE_SNAPSHOT_DIR) -> Path:
    source = ROOT / relative_path
    destination = snapshot_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def snapshot_rel_directory(relative_dir: Path, *, snapshot_root: Path = BASELINE_SNAPSHOT_DIR) -> Path:
    source = ROOT / relative_dir
    destination = snapshot_root / relative_dir
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def load_csv_records(relative_path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(ROOT / relative_path)
    return frame.to_dict(orient="records")


def build_baseline_manifest() -> dict[str, Any]:
    ensure_logical_alias()
    ensure_dir(BASELINE_SNAPSHOT_DIR)
    for rel_dir in BASELINE_SNAPSHOT_DIRECTORIES:
        snapshot_rel_directory(rel_dir)

    artifact_entries: list[dict[str, Any]] = []
    for artifact in BASELINE_KEY_ARTIFACTS:
        relative_path = Path(artifact["path"])
        snapshot_path = snapshot_rel_path(relative_path)
        artifact_entries.append(
            {
                "role": str(artifact["role"]),
                "source_path": str(relative_path),
                "snapshot_path": str(logical_root_relative(snapshot_path)),
                "physical_snapshot_path": str(physical_root_relative(snapshot_path)),
                "sha256": sha256_file(ROOT / relative_path),
            }
        )

    manifest = {
        "revision_round_id": REVISION_ROUND_ID,
        "created_utc": utc_now_iso(),
        "workspace_root": str(ROOT),
        "canonical_identity_sentence": CORE_IDENTITY_SENTENCE,
        "logical_canonical_root": str(repo_relative(LOGICAL_CANONICAL_ROOT)),
        "physical_storage_root": str(repo_relative(PHYSICAL_STORAGE_ROOT)),
        "baseline_root": str(logical_root_relative(BASELINE_LOCK_DIR)),
        "physical_baseline_root": str(physical_root_relative(BASELINE_LOCK_DIR)),
        "logical_root_map_path": str(physical_root_relative(LOGICAL_ROOT_MAP_PATH)),
        "baseline_claims": BASELINE_FACTS,
        "artifacts": artifact_entries,
        "figure_asset_filename": Q2_MAIN_FIGURE.name,
        "anchor_tables": {
            "event_micro_main": load_csv_records(Q2_EVENT_MICRO_MAIN_CSV),
            "inventory_main": load_csv_records(Q2_INVENTORY_CSV),
            "event_micro_threshold_robustness": load_csv_records(Q2_EVENT_MICRO_THRESHOLD_CSV),
            "event_micro_logloss_robustness": load_csv_records(Q2_EVENT_MICRO_LOGLOSS_CSV),
            "load_following_support": load_csv_records(LOAD_FOLLOWING_SUPPORT_CSV),
        },
    }
    return manifest


def build_q2_from_seed_metrics(
    seed_metrics_df: pd.DataFrame,
    *,
    forecast_metric_column: str,
    scenario_id: str,
    domain: str = "event_micro",
    interface_id: str = "fixed_threshold",
) -> pd.DataFrame:
    metric_values = -pd.to_numeric(seed_metrics_df[forecast_metric_column], errors="raise")
    q2_df = pd.DataFrame(
        {
            "question_id": "Q2",
            "scenario_id": str(scenario_id),
            "domain": str(domain),
            "seed": seed_metrics_df["seed"].astype(int),
            "forecaster_id": seed_metrics_df["model"].astype(str),
            "interface_id": str(interface_id),
            "friction_level": seed_metrics_df["friction"].astype(float),
            "forecast_metric": metric_values.astype(float),
            "target_metric": seed_metrics_df["deployed_utility"].astype(float),
            "executed_metric": seed_metrics_df["deployed_utility"].astype(float),
            "target_executed_gap": 0.0,
            "realized_cost": 0.0,
            "realized_turnover_or_adjustment": seed_metrics_df["switch_rate"].astype(float),
            "rank_within_forecast_metric": 0,
            "rank_within_executed_metric": 0,
        }
    )
    return q2_df


def friction_row(summary_df: pd.DataFrame, friction: float) -> pd.Series:
    mask = np.isclose(summary_df["friction_level"], float(friction), atol=1e-12)
    matched = summary_df.loc[mask]
    if matched.empty:
        raise KeyError(f"Missing friction row {friction}.")
    return matched.iloc[0]


def compact_selection_summary(
    selection_summary: pd.DataFrame,
    *,
    frictions: tuple[float, ...],
) -> pd.DataFrame:
    ordered_rows = [friction_row(selection_summary, friction) for friction in frictions]
    return pd.DataFrame(ordered_rows).reset_index(drop=True)


def model_label(model_id: str, labels: dict[str, str]) -> str:
    return labels.get(str(model_id), str(model_id))


def membership_winner(members: dict[str, int], *, mean_scores: dict[str, float]) -> str:
    if not members:
        return ""
    ordered = sorted(
        members,
        key=lambda model_id: (-int(members[str(model_id)]), -float(mean_scores[str(model_id)]), str(model_id)),
    )
    return str(ordered[0])


def paper_selection_table(
    selection_summary: pd.DataFrame,
    *,
    label_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for row in selection_summary.itertuples(index=False):
        rows.append(
            {
                "Friction": f"{float(row.friction_level):.2f}",
                "Forecast-side winner": model_label(str(row.most_frequent_forecast_best), label_map),
                "Deployed winner": model_label(str(row.most_frequent_deployed_best), label_map),
                "Agreement rate": f"{float(row.agreement_rate):.2f}",
                "Mean deployed gap": f"{float(row.mean_deployed_gap_of_forecast_selected):.3f}",
                "Median deployed gap": f"{float(row.median_deployed_gap_of_forecast_selected):.3f}",
                "Deployed-suboptimal seeds / total": str(row.deployed_suboptimal_seeds_over_total),
            }
        )
    return pd.DataFrame(rows)


def write_table_bundle(frame: pd.DataFrame, output_stem: Path) -> tuple[Path, Path]:
    csv_path = output_stem.with_suffix(".csv")
    tex_path = output_stem.with_suffix(".tex")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, escape=True))
    return csv_path, tex_path


def selection_pattern_holds(selection_summary: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    compact = compact_selection_summary(selection_summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS)
    zero_row = friction_row(compact, 0.0)
    mid_row = friction_row(compact, 0.5)
    high_row = friction_row(compact, 1.0)

    mid_share = float(mid_row["deployed_suboptimal_seed_fraction"])
    high_share = float(high_row["deployed_suboptimal_seed_fraction"])
    passed = bool(
        float(zero_row["agreement_rate"]) >= 0.50
        and str(mid_row["most_frequent_forecast_best"]) == "reactive_sharp"
        and str(high_row["most_frequent_forecast_best"]) == "reactive_sharp"
        and str(mid_row["most_frequent_deployed_best"]) in {"calibrated_baseline", "lagged_smoother"}
        and str(mid_row["most_frequent_deployed_best"]) != str(mid_row["most_frequent_forecast_best"])
        and str(high_row["most_frequent_deployed_best"]) == "lagged_smoother"
        and mid_share >= 0.50
        and high_share >= 0.50
    )
    return passed, {
        "zero_agreement": float(zero_row["agreement_rate"]),
        "mid_suboptimal_share": mid_share,
        "high_suboptimal_share": high_share,
        "mid_forecast_winner": str(mid_row["most_frequent_forecast_best"]),
        "mid_deployed_winner": str(mid_row["most_frequent_deployed_best"]),
        "high_forecast_winner": str(high_row["most_frequent_forecast_best"]),
        "high_deployed_winner": str(high_row["most_frequent_deployed_best"]),
    }


def evaluate_workstream_a(threshold_summaries: dict[str, pd.DataFrame]) -> GateResult:
    per_threshold: dict[str, dict[str, Any]] = {}
    qualifying: list[str] = []
    for threshold_name, summary in threshold_summaries.items():
        passed, details = selection_pattern_holds(summary)
        per_threshold[threshold_name] = {"pattern_holds": passed, **details}
        if passed:
            qualifying.append(threshold_name)

    status = "NO-GO"
    passed = False
    if len(qualifying) >= 3:
        status = "A Strong GO"
        passed = True
    elif len(qualifying) >= 2 and CORE_A_THRESHOLDS.issubset(set(qualifying)):
        status = "A Weak GO"
        passed = True

    return GateResult(
        status=status,
        passed=passed,
        details={
            "qualifying_thresholds": qualifying,
            "core_thresholds_required_for_weak_go": sorted(CORE_A_THRESHOLDS),
            "per_threshold": per_threshold,
        },
    )


def evaluate_c_seed_gate(selection_summary: pd.DataFrame) -> GateResult:
    compact = compact_selection_summary(selection_summary, frictions=EXPECTED_EVENT_MICRO_COMPACT_FRICTIONS)
    mid_row = friction_row(compact, 0.5)
    high_row = friction_row(compact, 1.0)

    mid_share = float(mid_row["deployed_suboptimal_seed_fraction"])
    high_share = float(high_row["deployed_suboptimal_seed_fraction"])
    passed = bool(
        mid_share >= 0.60
        and high_share >= 0.85
        and str(mid_row["most_frequent_forecast_best"]) == "reactive_sharp"
        and str(high_row["most_frequent_forecast_best"]) == "reactive_sharp"
        and str(mid_row["most_frequent_deployed_best"]) in {"calibrated_baseline", "lagged_smoother"}
        and str(mid_row["most_frequent_deployed_best"]) != "reactive_sharp"
        and str(high_row["most_frequent_deployed_best"]) == "lagged_smoother"
    )
    return GateResult(
        status="C seed-entry GO" if passed else "C seed-entry NO-GO",
        passed=passed,
        details={
            "mid_suboptimal_share": mid_share,
            "high_suboptimal_share": high_share,
            "mid_forecast_winner": str(mid_row["most_frequent_forecast_best"]),
            "mid_deployed_winner": str(mid_row["most_frequent_deployed_best"]),
            "high_forecast_winner": str(high_row["most_frequent_forecast_best"]),
            "high_deployed_winner": str(high_row["most_frequent_deployed_best"]),
        },
    )


def evaluate_workstream_c(
    canonical_seed40_summary: pd.DataFrame,
    regime_summaries: dict[str, pd.DataFrame],
) -> GateResult:
    seed_gate = evaluate_c_seed_gate(canonical_seed40_summary)
    if not seed_gate.passed:
        return GateResult(status="C NO-GO", passed=False, details={"seed_gate": seed_gate.details, "regimes": {}})

    regime_outcomes: dict[str, dict[str, Any]] = {}
    added_regime_passes = 0
    for regime_name, summary in regime_summaries.items():
        passed, details = selection_pattern_holds(summary)
        regime_outcomes[regime_name] = {"pattern_holds": passed, **details}
        if passed:
            added_regime_passes += 1

    if added_regime_passes == len(regime_summaries):
        status = "C Strong GO"
        passed = True
    elif added_regime_passes >= 1:
        status = "C Medium GO"
        passed = True
    else:
        status = "C NO-GO"
        passed = False

    return GateResult(
        status=status,
        passed=passed,
        details={
            "seed_gate": seed_gate.details,
            "added_regime_pass_count": int(added_regime_passes),
            "regimes": regime_outcomes,
        },
    )


def validate_load_following_raw_candidate(
    raw_df: pd.DataFrame,
    *,
    expected_interface_id: str = "responsive",
    tie_abs_floor: float = DEFAULT_TIE_ABS_FLOOR,
    tie_rel_scale: float = DEFAULT_TIE_REL_SCALE,
) -> dict[str, Any]:
    invalid_reasons: list[str] = []
    failures = validate_q2_source(
        raw_df,
        expected_interface_id=expected_interface_id,
        min_forecasters_per_seed_friction=4,
    )
    invalid_reasons.extend(failures)

    required_numeric_columns = ["forecast_metric", "executed_metric"]
    for column in required_numeric_columns:
        if column not in raw_df.columns or raw_df[column].isna().any():
            invalid_reasons.append(f"{column}_unavailable_or_null")

    observed_frictions = sorted(float(value) for value in raw_df["friction_level"].dropna().unique().tolist())
    seeds = sorted(int(value) for value in raw_df["seed"].dropna().unique().tolist())
    model_sets = raw_df.groupby(["seed", "friction_level"], dropna=False)["forecaster_id"].agg(lambda s: tuple(sorted(str(v) for v in s)))
    reference_models = tuple(sorted(raw_df["forecaster_id"].dropna().astype(str).unique().tolist()))

    for seed in seeds:
        for friction in EXPECTED_LOAD_FOLLOWING_FRICTIONS:
            if float(friction) not in observed_frictions:
                invalid_reasons.append(f"missing_friction_{friction:.2f}")
                continue
            if (seed, float(friction)) not in model_sets.index:
                invalid_reasons.append(f"missing_seed_friction_slice_seed{seed}_friction{friction:.2f}")
                continue
            if tuple(model_sets.loc[(seed, float(friction))]) != reference_models:
                invalid_reasons.append(f"missing_model_coverage_seed{seed}_friction{friction:.2f}")

    outputs, meta = build_domain_rank_summary(
        raw_df,
        domain="load_following_elecdiag",
        expected_interface_id=expected_interface_id,
        tie_abs_floor=tie_abs_floor,
        tie_rel_scale=tie_rel_scale,
    )
    selection_summary = outputs["selection_summary_by_friction"].copy()
    seed_selection = outputs["seed_level_selection_stats"].copy()
    reproduced = (
        seed_selection.groupby("friction_level", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            agreement_rate=("agreement_flag", "mean"),
            disagreement_rate=("selection_disagreement_flag", "mean"),
            deployed_suboptimal_seed_count=("selection_disagreement_flag", "sum"),
            mean_deployed_gap_of_forecast_selected=("deployed_gap_of_forecast_selected", "mean"),
            median_deployed_gap_of_forecast_selected=("deployed_gap_of_forecast_selected", "median"),
        )
        .sort_values("friction_level")
        .reset_index(drop=True)
    )
    winner_mode_rows: list[dict[str, Any]] = []
    for friction in sorted(seed_selection["friction_level"].unique().tolist()):
        friction_seed = seed_selection.loc[np.isclose(seed_selection["friction_level"], float(friction), atol=1e-12)].copy()
        friction_raw = raw_df.loc[np.isclose(raw_df["friction_level"], float(friction), atol=1e-12)].copy()
        forecast_membership: dict[str, int] = {}
        deployed_membership: dict[str, int] = {}
        for row in friction_seed.itertuples(index=False):
            for model_id in str(row.forecast_best_set).split("|"):
                if model_id:
                    forecast_membership[model_id] = forecast_membership.get(model_id, 0) + 1
            for model_id in str(row.deployed_best_set).split("|"):
                if model_id:
                    deployed_membership[model_id] = deployed_membership.get(model_id, 0) + 1
        mean_forecast_scores = (
            friction_raw.groupby("forecaster_id", as_index=False)["forecast_metric"].mean().set_index("forecaster_id")["forecast_metric"].to_dict()
        )
        mean_executed_scores = (
            friction_raw.groupby("forecaster_id", as_index=False)["executed_metric"].mean().set_index("forecaster_id")["executed_metric"].to_dict()
        )
        winner_mode_rows.append(
            {
                "friction_level": float(friction),
                "most_frequent_forecast_best": membership_winner(
                    forecast_membership,
                    mean_scores=mean_forecast_scores,
                ),
                "most_frequent_deployed_best": membership_winner(
                    deployed_membership,
                    mean_scores=mean_executed_scores,
                ),
            }
        )
    winner_modes = pd.DataFrame(winner_mode_rows).sort_values("friction_level").reset_index(drop=True)
    merged = selection_summary.merge(
        reproduced.merge(winner_modes, on="friction_level", how="left"),
        on=["friction_level", "n_seeds", "most_frequent_forecast_best", "most_frequent_deployed_best"],
        suffixes=("_summary", "_reproduced"),
        how="outer",
        indicator=True,
    )
    if not bool((merged["_merge"] == "both").all()):
        invalid_reasons.append("inconsistent_winner_derivation")
    else:
        compare_columns = [
            ("agreement_rate_summary", "agreement_rate_reproduced"),
            ("disagreement_rate_summary", "disagreement_rate_reproduced"),
            ("deployed_suboptimal_seed_count_summary", "deployed_suboptimal_seed_count_reproduced"),
            ("mean_deployed_gap_of_forecast_selected_summary", "mean_deployed_gap_of_forecast_selected_reproduced"),
            ("median_deployed_gap_of_forecast_selected_summary", "median_deployed_gap_of_forecast_selected_reproduced"),
        ]
        for left, right in compare_columns:
            if not np.allclose(
                merged[left].astype(float),
                merged[right].astype(float),
                atol=1e-12,
                rtol=0.0,
            ):
                invalid_reasons.append("gap_fields_unavailable_or_non_reproducible")
                break

    invalid_reasons = sorted(set(invalid_reasons))
    return {
        "raw_valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "outputs": outputs,
        "meta": meta,
    }


def evaluate_workstream_d(selection_summary: pd.DataFrame) -> GateResult:
    zero_row = friction_row(selection_summary, 0.0)
    mid_row = friction_row(selection_summary, 0.5)
    high_row = friction_row(selection_summary, 1.0)

    winner_mode_mismatch = bool(
        str(mid_row["most_frequent_forecast_best"]) != str(mid_row["most_frequent_deployed_best"])
        or str(high_row["most_frequent_forecast_best"]) != str(high_row["most_frequent_deployed_best"])
    )
    passed = bool(
        float(zero_row["agreement_rate"]) >= 0.60
        and int(mid_row["deployed_suboptimal_seed_count"]) >= 7
        and int(high_row["deployed_suboptimal_seed_count"]) >= 7
        and winner_mode_mismatch
        and float(mid_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
        and float(mid_row["median_deployed_gap_of_forecast_selected"]) > 0.0
        and float(high_row["mean_deployed_gap_of_forecast_selected"]) > 0.0
        and float(high_row["median_deployed_gap_of_forecast_selected"]) > 0.0
    )
    return GateResult(
        status="D GO" if passed else "D NO-GO",
        passed=passed,
        details={
            "zero_agreement": float(zero_row["agreement_rate"]),
            "mid_suboptimal": int(mid_row["deployed_suboptimal_seed_count"]),
            "high_suboptimal": int(high_row["deployed_suboptimal_seed_count"]),
            "mid_forecast_winner": str(mid_row["most_frequent_forecast_best"]),
            "mid_deployed_winner": str(mid_row["most_frequent_deployed_best"]),
            "high_forecast_winner": str(high_row["most_frequent_forecast_best"]),
            "high_deployed_winner": str(high_row["most_frequent_deployed_best"]),
            "winner_mode_mismatch": winner_mode_mismatch,
            "mid_mean_gap": float(mid_row["mean_deployed_gap_of_forecast_selected"]),
            "mid_median_gap": float(mid_row["median_deployed_gap_of_forecast_selected"]),
            "high_mean_gap": float(high_row["mean_deployed_gap_of_forecast_selected"]),
            "high_median_gap": float(high_row["median_deployed_gap_of_forecast_selected"]),
        },
    )


def draft_proposition_lines() -> list[str]:
    return [
        "### Appendix Proposition Draft",
        "",
        r"**Proposition.** Under a fixed frictional deployment interface, forecast-side ordering need not preserve deployed ordering even when the forecast metric remains valid for prediction.",
        "",
        r"**Proof sketch.** Let two forecasters induce different probability paths under one fixed thresholding rule. If one forecaster is slightly sharper on the forecast metric but also induces more switching, then its realized utility can fall below that of a smoother forecaster once switching friction is positive. The forecast metric still correctly scores predictive quality, but the model-selection object changes because realized utility depends on realized action under friction, not forecast quality alone.",
        "",
        "### Main-text Pointer Sentence",
        "",
        "A short appendix proposition formalizes why forecast-side ordering need not be preserved after frictional execution.",
    ]


def determine_track(
    *,
    a_gate: GateResult,
    b_gate: GateResult,
    c_gate: GateResult | None,
    d_gate: GateResult | None,
    layout_preflight_passed: bool,
) -> dict[str, Any]:
    c_status = c_gate.status if c_gate is not None else "not_run"
    d_status = d_gate.status if d_gate is not None else "not_run"

    evidence_track = "Track 5"
    final_track = "Track 5"
    notes: list[str] = []

    if a_gate.status in {"A Strong GO", "A Weak GO"}:
        evidence_track = "Track 1"
        final_track = "Track 1"

    if a_gate.status == "A Strong GO" and c_status == "C Strong GO" and d_status != "D GO":
        evidence_track = "Track 2"
        final_track = "Track 2"
    if a_gate.status == "A Strong GO" and d_status == "D GO" and c_status != "C Strong GO":
        evidence_track = "Track 3"
        final_track = "Track 3"
    if a_gate.status == "A Strong GO" and b_gate.passed and c_status == "C Strong GO" and d_status == "D GO":
        evidence_track = "Track 4"
        if layout_preflight_passed:
            final_track = "Track 4"
        else:
            final_track = "Track 4 candidate pending layout preflight"
            notes.append("Track 4 requires staged PDF layout preflight before it becomes the final track.")

    if final_track == "Track 1":
        notes.append("Track 1 cannot alter evidence hierarchy, Figure 2, Table 2, or limitation wording.")
    if c_status == "C Medium GO":
        notes.append("C Medium GO is appendix-strengthening only and does not relax the deliberately-small limitation.")
    if a_gate.status == "A Weak GO":
        notes.append("A Weak GO is appendix-strengthening only and cannot strengthen main-text claims.")

    return {
        "evidence_track": evidence_track,
        "final_track": final_track,
        "layout_preflight_passed": bool(layout_preflight_passed),
        "a_status": a_gate.status,
        "b_status": b_gate.status,
        "c_status": c_status,
        "d_status": d_status,
        "notes": notes,
    }
