import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FORECAST_EVAL_DIR = REPO_ROOT / "scripts" / "forecast_eval"
if str(FORECAST_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(FORECAST_EVAL_DIR))

import run_traffic_redesign_pilot as traffic  # noqa: E402


def _write_tiny_traffic_zip(path: Path, *, n_series: int = 8, n_time: int = 520) -> Path:
    lines = [
        "@attribute series_name string",
        "@frequency hourly",
        "@horizon 1",
        "@missing false",
        "@equallength true",
        "@data",
    ]
    for series_idx in range(n_series):
        values = []
        for t in range(n_time):
            daily = 10.0 * np.sin(2 * np.pi * (t % 24) / 24.0)
            weekly = 4.0 * np.cos(2 * np.pi * (t % 168) / 168.0)
            trend = 0.01 * t
            offset = series_idx * 2.0
            pulse = 20.0 if (t + series_idx) % 31 == 0 else 0.0
            values.append(f"{100.0 + offset + trend + daily + weekly + pulse:.4f}")
        lines.append(f"s{series_idx}:{','.join(values)}")
    tsf = "\n".join(lines) + "\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("traffic_hourly_dataset.tsf", tsf)
    return path


def test_parse_tsf_and_source_lock_fixture(tmp_path):
    source_zip = _write_tiny_traffic_zip(tmp_path / "traffic.zip")
    panel = traffic.load_traffic_panel(
        cache_dir=tmp_path / "cache",
        source_zip=source_zip,
        skip_download=True,
        allow_noncanonical_panel=True,
    )

    assert panel.values.shape == (8, 520)
    assert panel.source_lock["hard_lock_policy"] == "downloaded_zip_sha256_plus_parsed_panel_assertions"
    assert panel.source_lock["content_length_policy"] == "warning_only"
    assert panel.source_lock["parsed_assertions"]["no_missing"] is True


def test_split_labels_k_and_topk_utility():
    panel = np.vstack(
        [
            np.arange(520, dtype=float),
            np.arange(520, dtype=float) + 10,
            np.sin(np.arange(520) / 5.0) + 30,
            np.cos(np.arange(520) / 7.0) + 40,
        ]
    )
    split = traffic.build_split(panel.shape[1])
    labels, provenance = traffic.c1_labels(panel, split)
    task = traffic.materialize_tasks(panel, split, "traffic_topk_alert_q2_v1")[0]

    assert split.validation_start == panel.shape[1] - 336
    assert split.test_end == panel.shape[1]
    assert labels.shape == panel.shape
    assert "validation_event_rate_at_q70" in provenance
    assert task.k_ref >= 1

    probabilities = np.array([[0.9, 0.1], [0.8, 0.7], [0.2, 0.6]])
    outcomes = np.array([[1, 0], [0, 1], [0, 1]])
    metrics = traffic.evaluate_top_k(probabilities, outcomes, k=2, friction=0.5)
    assert metrics["alert_or_set_size_rate"] == 2 / 3
    assert metrics["switch_rate"] >= 0.0
    assert isinstance(metrics["deployed_utility"], float)


def test_tie_top_set_uses_existing_tolerance():
    winners = traffic.top_set({"a": 1.0, "b": 1.0 + 1e-11, "c": 0.5}, higher_is_better=True)
    assert winners == ("a", "b")


def test_cli_smoke_writes_required_pilot_outputs(tmp_path):
    source_zip = _write_tiny_traffic_zip(tmp_path / "traffic.zip")
    output_root = tmp_path / "outputs"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "forecast_eval" / "run_traffic_redesign_pilot.py"),
        "--scenario-id",
        "traffic_topk_alert_q2_v1",
        "--replicates",
        "2",
        "--frictions",
        "0,1",
        "--skip-summary-refresh",
        "--skip-download",
        "--allow-noncanonical-panel",
        "--source-zip",
        str(source_zip),
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    scenario_dir = output_root / "traffic_topk_alert_q2_v1"
    expected_files = [
        "pilot_selection_summary_by_friction.csv",
        "pilot_seed_level_metrics.csv",
        "pilot_zero_row_diagnostics.csv",
        "pilot_candidate_ledger.json",
        "pilot_table.csv",
        "pilot_report.json",
    ]
    for filename in expected_files:
        assert (scenario_dir / filename).exists()

    seed_metrics = pd.read_csv(scenario_dir / "pilot_seed_level_metrics.csv")
    required_seed_columns = {
        "scenario_id",
        "replicate_id",
        "friction",
        "family_id",
        "forecast_metric_brier",
        "forecast_metric_logloss",
        "deployed_utility",
        "switch_rate",
        "alert_or_set_size_rate",
        "forecast_winner_flag",
        "deployed_winner_flag",
        "tie_involved",
    }
    assert required_seed_columns.issubset(seed_metrics.columns)

    zero_diag = pd.read_csv(scenario_dir / "pilot_zero_row_diagnostics.csv")
    required_zero_columns = {
        "scenario_id",
        "replicate_id",
        "forecast_winner",
        "deployed_winner",
        "agreement_zero",
        "mean_gap_zero",
        "median_gap_zero",
        "tie_involved_zero",
        "set_switch_rate_zero",
        "alert_or_set_size_zero",
        "verdict_zero_explainable",
    }
    assert required_zero_columns.issubset(zero_diag.columns)
