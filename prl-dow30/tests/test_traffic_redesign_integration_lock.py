import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "forecast_eval" / "build_traffic_redesign_integration_lock_v1.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_fixture_dirs(tmp_path: Path) -> dict[str, Path]:
    paper_dir = tmp_path / "paper"
    results_dir = paper_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "paper_forecasting_workshop_v2.tex").write_text("placeholder tex\n")
    (paper_dir / "paper_forecasting_workshop_v2.pdf").write_bytes(b"%PDF-1.4\n% fixture\n")
    (results_dir / "table_evidence_map_v2.tex").write_text("old\n")

    topk_dir = tmp_path / "traffic_topk"
    relative_dir = tmp_path / "traffic_relative"
    surge_dir = tmp_path / "traffic_surge"
    lock_dir = tmp_path / "lock"
    return {
        "paper_dir": paper_dir,
        "results_dir": results_dir,
        "topk_dir": topk_dir,
        "relative_dir": relative_dir,
        "surge_dir": surge_dir,
        "lock_dir": lock_dir,
    }


def _seed_rows(variant_id: str, k: int, deployed_winners: tuple[str, str, str]) -> list[dict]:
    frictions = [0.0, 0.5, 1.0]
    rows = []
    for friction, deployed, mean_gap, median_gap, suboptimal in zip(
        frictions,
        deployed_winners,
        [0.0, 7.0155505952, 24.6512648809],
        [0.0, 6.9769345238, 24.4732142857],
        [0.0, 1.0, 1.0],
    ):
        rows.append(
            {
                "variant_id": variant_id,
                "k": k,
                "friction": friction,
                "forecast_winner": "reactive_short",
                "deployed_winner": deployed,
                "agreement": 1.0 if friction == 0.0 else 0.0,
                "mean_gap": mean_gap,
                "median_gap": median_gap,
                "deployed_suboptimal_share": suboptimal,
                "modal_winner_divergence": bool(friction > 0.0),
                "mean_set_switch_rate": 10.0,
                "tie_involved_fraction": 0.0,
            }
        )
    return rows


def test_integration_lock_builder_writes_expected_assets(tmp_path):
    paths = _make_fixture_dirs(tmp_path)
    _write_json(
        paths["topk_dir"] / "full_report.json",
        {
            "best_k": 249,
            "best_variant_id": "q0.70_kgrid",
            "prefix": "full",
            "scenario_id": "traffic_topk_alert_q2_v1",
            "verdict": "strong",
        },
    )
    _write_json(
        paths["relative_dir"] / "full_report.json",
        {
            "best_k": 87,
            "best_variant_id": "m0.10",
            "prefix": "full",
            "scenario_id": "traffic_relative_rank_q2_v1",
            "verdict": "strong",
        },
    )
    _write_json(
        paths["surge_dir"] / "pilot_report.json",
        {
            "best_k": 107,
            "best_variant_id": "qdelta0.70",
            "prefix": "pilot",
            "scenario_id": "traffic_surge_onset_q2_v1",
            "verdict": "fail",
        },
    )
    _write_csv(
        paths["topk_dir"] / "full_selection_summary_by_friction.csv",
        _seed_rows("q0.70_kgrid", 249, ("reactive_short", "lagged_smoother", "lagged_smoother")),
    )
    _write_csv(
        paths["relative_dir"] / "full_selection_summary_by_friction.csv",
        [
            {
                "variant_id": "m0.10",
                "k": 87,
                "friction": 0.0,
                "forecast_winner": "reactive_short",
                "deployed_winner": "reactive_short",
                "agreement": 1.0,
                "mean_gap": 0.0,
                "median_gap": 0.0,
                "deployed_suboptimal_share": 0.0,
                "modal_winner_divergence": False,
                "mean_set_switch_rate": 10.0,
                "tie_involved_fraction": 0.0,
            },
            {
                "variant_id": "m0.10",
                "k": 87,
                "friction": 0.5,
                "forecast_winner": "reactive_short",
                "deployed_winner": "calibrated_baseline",
                "agreement": 0.0,
                "mean_gap": 10.1030505952,
                "median_gap": 10.1220238095,
                "deployed_suboptimal_share": 1.0,
                "modal_winner_divergence": True,
                "mean_set_switch_rate": 10.0,
                "tie_involved_fraction": 0.0,
            },
            {
                "variant_id": "m0.10",
                "k": 87,
                "friction": 1.0,
                "forecast_winner": "reactive_short",
                "deployed_winner": "calibrated_baseline",
                "agreement": 0.0,
                "mean_gap": 25.2051339286,
                "median_gap": 25.2313988095,
                "deployed_suboptimal_share": 1.0,
                "modal_winner_divergence": True,
                "mean_set_switch_rate": 10.0,
                "tie_involved_fraction": 0.0,
            },
        ],
    )
    _write_csv(
        paths["surge_dir"] / "pilot_selection_summary_by_friction.csv",
        [
            {
                "variant_id": "qdelta0.70",
                "k": 107,
                "friction": 0.0,
                "forecast_winner": "calibrated_baseline",
                "deployed_winner": "reactive_short",
                "agreement": 0.0,
                "mean_gap": 2.0,
                "median_gap": 2.1,
                "deployed_suboptimal_share": 1.0,
                "modal_winner_divergence": True,
                "mean_set_switch_rate": 10.0,
                "tie_involved_fraction": 0.0,
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--topk-dir",
            str(paths["topk_dir"]),
            "--relative-rank-dir",
            str(paths["relative_dir"]),
            "--surge-dir",
            str(paths["surge_dir"]),
            "--lock-dir",
            str(paths["lock_dir"]),
            "--paper-dir",
            str(paths["paper_dir"]),
            "--paper-results-dir",
            str(paths["results_dir"]),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    topk_table = list(csv.DictReader((paths["results_dir"] / "table_q2_selection_drift_traffic_topk_main.csv").open()))
    assert topk_table[1]["Forecast-side winner"] == "Reactive short"
    assert topk_table[1]["Deployed winner"] == "Lagged smoother"
    assert topk_table[1]["Deployed-suboptimal seeds / total"] == "100/100"
    assert topk_table[1]["Mean deployed gap"] == "7.016"

    relative_table = list(csv.DictReader((paths["results_dir"] / "table_q2_selection_drift_traffic_relative_rank_support.csv").open()))
    assert relative_table[2]["Deployed winner"] == "Calibrated baseline"

    evidence_map = (paths["results_dir"] / "table_evidence_map_v3.tex").read_text()
    assert "Traffic Top-k Alert" in evidence_map
    assert "switching-cost divergence recurs" in evidence_map

    addendum = json.loads((paths["lock_dir"] / "claim_to_evidence_addendum.json").read_text())
    assert addendum["traffic_redesign_addendum"][0]["status"] == "strong pass"
    assert addendum["traffic_redesign_addendum"][2]["status"] == "fail"


def test_integration_lock_builder_fails_on_wrong_verdict(tmp_path):
    paths = _make_fixture_dirs(tmp_path)
    _write_json(
        paths["topk_dir"] / "full_report.json",
        {
            "best_k": 249,
            "best_variant_id": "q0.70_kgrid",
            "prefix": "full",
            "scenario_id": "traffic_topk_alert_q2_v1",
            "verdict": "fail",
        },
    )
    _write_json(
        paths["relative_dir"] / "full_report.json",
        {
            "best_k": 87,
            "best_variant_id": "m0.10",
            "prefix": "full",
            "scenario_id": "traffic_relative_rank_q2_v1",
            "verdict": "strong",
        },
    )
    _write_json(
        paths["surge_dir"] / "pilot_report.json",
        {
            "best_k": 107,
            "best_variant_id": "qdelta0.70",
            "prefix": "pilot",
            "scenario_id": "traffic_surge_onset_q2_v1",
            "verdict": "fail",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--topk-dir",
            str(paths["topk_dir"]),
            "--relative-rank-dir",
            str(paths["relative_dir"]),
            "--surge-dir",
            str(paths["surge_dir"]),
            "--lock-dir",
            str(paths["lock_dir"]),
            "--paper-dir",
            str(paths["paper_dir"]),
            "--paper-results-dir",
            str(paths["results_dir"]),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "did not match expected" in completed.stderr or "did not match expected" in completed.stdout
