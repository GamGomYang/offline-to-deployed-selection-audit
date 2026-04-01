#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[2]
PAPER_ARTIFACT_MANIFEST = ROOT / "paper_artifact_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a reproduced baseline run matches the frozen canonical run.")
    parser.add_argument("--manifest", type=str, required=True, help="Baseline manifest JSON.")
    parser.add_argument("--run-root", type=str, required=True, help="Reproduced baseline run root.")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_csv(path: Path, *, drop_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_csv(path)
    existing = [column for column in drop_columns if column in df.columns]
    if existing:
        df = df.drop(columns=existing)
    if not df.empty:
        df = df.sort_values(list(df.columns), kind="mergesort").reset_index(drop=True)
    return df


def _compare_csv(source: Path, target: Path, *, drop_columns: tuple[str, ...] = ()) -> None:
    source_df = _normalize_csv(source, drop_columns=drop_columns)
    target_df = _normalize_csv(target, drop_columns=drop_columns)
    assert_frame_equal(
        source_df,
        target_df,
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def _check(condition: bool, description: str, payload: dict) -> dict:
    return {
        "description": description,
        "status": "PASS" if condition else "FAIL",
        "details": payload,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(_resolve(args.manifest).read_text())
    canonical_root = _resolve(manifest["canonical_run_root"])
    run_root = _resolve(args.run_root)

    results: list[dict] = []

    csv_pairs = [
        ("validation_eta/aggregate.csv", ()),
        ("validation_eta/selection/validation_eta_selection.csv", ()),
        ("final_eta/aggregate.csv", ()),
        ("external_baselines/aggregate.csv", ("trace_path",)),
        ("paper_pack/tables/validation_selection.csv", ()),
        ("paper_pack/tables/test_selected_vs_eta1.csv", ()),
        ("paper_pack/tables/test_selected_vs_external_baselines.csv", ()),
        ("paper_pack/tables/diagnostic_selected_eta.csv", ()),
        ("paper_pack/stats/selected_eta_vs_eta1_stats.csv", ()),
        ("paper_pack/diagnostics/diagnostic_selected_eta_v2.csv", ()),
    ]

    for rel_path, drop_columns in csv_pairs:
        source = canonical_root / rel_path
        target = run_root / rel_path
        payload = {"source": str(source), "target": str(target)}
        try:
            _compare_csv(source, target, drop_columns=drop_columns)
            results.append(_check(True, f"CSV match: {rel_path}", payload))
        except Exception as exc:  # pragma: no cover - surfaced in CLI output
            payload["error"] = str(exc)
            results.append(_check(False, f"CSV match: {rel_path}", payload))

    selection_json = json.loads((run_root / "validation_eta/selection/validation_eta_selection.json").read_text())
    selected_eta_expected = float(manifest["selection_rule"]["selected_eta"])
    results.append(
        _check(
            float(selection_json["selected_eta"]) == selected_eta_expected,
            "Selected eta matches frozen manifest",
            {
                "expected": selected_eta_expected,
                "actual": float(selection_json["selected_eta"]),
            },
        )
    )

    stats_df = pd.read_csv(run_root / "paper_pack/stats/selected_eta_vs_eta1_stats.csv")
    snapshot = manifest["paper_result_snapshot"]["heldout_selected_vs_eta1"]
    for kappa_str, expected in snapshot.items():
        row = stats_df.loc[(stats_df["kappa"] - float(kappa_str)).abs() < 1e-12]
        if row.empty:
            results.append(
                _check(
                    False,
                    f"Paper snapshot row exists for kappa={kappa_str}",
                    {"expected_kappa": kappa_str},
                )
            )
            continue
        record = row.iloc[0]
        delta_ok = abs(float(record["median_delta_sharpe_net_lin"]) - float(expected["median_delta_sharpe_net_lin"])) < 1e-12
        win_ok = abs(float(record["win_rate_sharpe"]) - float(expected["win_rate_sharpe"])) < 1e-12
        results.append(
            _check(
                delta_ok and win_ok,
                f"Paper snapshot values match for kappa={kappa_str}",
                {
                    "expected_delta_sharpe": expected["median_delta_sharpe_net_lin"],
                    "actual_delta_sharpe": float(record["median_delta_sharpe_net_lin"]),
                    "expected_win_rate": expected["win_rate_sharpe"],
                    "actual_win_rate": float(record["win_rate_sharpe"]),
                },
            )
        )

    figure_pairs = [
        ("paper_pack/figures/fig_validation_frontier.png", "paper_pack/figures/fig_validation_frontier.png"),
        ("paper_pack/figures/fig_selected_trace.png", "paper_pack/figures/fig_selected_trace.png"),
        ("paper_pack/figures/fig_seed_scatter.png", "paper_pack/figures/fig_seed_scatter.png"),
    ]
    for source_rel, target_rel in figure_pairs:
        source = canonical_root / source_rel
        target = run_root / target_rel
        source_hash = _sha256(source)
        target_hash = _sha256(target)
        results.append(
            _check(
                source_hash == target_hash,
                f"Figure hash match: {target_rel}",
                {"source_hash": source_hash, "target_hash": target_hash},
            )
        )

    paper_figure_pairs = [
        ("fig_frontier.png", canonical_root / "paper_pack/figures/fig_validation_frontier.png"),
        ("fig_validation_frontier.png", canonical_root / "paper_pack/figures/fig_validation_frontier.png"),
        ("fig_misalignment.png", canonical_root / "paper_pack/figures/fig_selected_trace.png"),
        ("fig_selected_trace.png", canonical_root / "paper_pack/figures/fig_selected_trace.png"),
        ("fig_seed_scatter.png", canonical_root / "paper_pack/figures/fig_seed_scatter.png"),
    ]
    for repo_rel, canonical_path in paper_figure_pairs:
        repo_path = ROOT / repo_rel
        repo_hash = _sha256(repo_path)
        canonical_hash = _sha256(canonical_path)
        results.append(
            _check(
                repo_hash == canonical_hash,
                f"Paper-facing figure matches canonical baseline: {repo_rel}",
                {"repo_hash": repo_hash, "canonical_hash": canonical_hash},
            )
        )

    figure_manifest = json.loads((ROOT / "figure_manifest.json").read_text())
    results.append(
        _check(
            float(figure_manifest["selected_eta"]) == selected_eta_expected,
            "Root figure manifest selected eta matches frozen manifest",
            {
                "expected": selected_eta_expected,
                "actual": float(figure_manifest["selected_eta"]),
            },
        )
    )

    if PAPER_ARTIFACT_MANIFEST.exists():
        artifact_manifest = json.loads(PAPER_ARTIFACT_MANIFEST.read_text())
        exported_manifest_path = run_root / "paper_artifacts/manifest.json"
        results.append(
            _check(
                exported_manifest_path.exists(),
                "Exported paper artifact manifest exists",
                {"path": str(exported_manifest_path)},
            )
        )
        if exported_manifest_path.exists():
            exported_manifest = json.loads(exported_manifest_path.read_text())
            for key, payload in artifact_manifest["tables"].items():
                canonical_csv = _resolve(payload["canonical_csv"])
                exported_csv = Path(exported_manifest["tables"][key]["export_csv"])
                table_ok = exported_csv.exists()
                payload_out = {
                    "canonical_csv": str(canonical_csv),
                    "exported_csv": str(exported_csv),
                }
                if table_ok:
                    try:
                        _compare_csv(canonical_csv, exported_csv)
                    except Exception as exc:  # pragma: no cover
                        table_ok = False
                        payload_out["error"] = str(exc)
                results.append(_check(table_ok, f"Exported paper table matches canonical: {key}", payload_out))

            for key, payload in artifact_manifest["figures"].items():
                canonical_png = _resolve(payload["canonical_png"])
                exported_png = Path(exported_manifest["figures"][key]["export_png"])
                figure_ok = exported_png.exists()
                payload_out = {
                    "canonical_png": str(canonical_png),
                    "exported_png": str(exported_png),
                }
                if figure_ok:
                    payload_out["canonical_hash"] = _sha256(canonical_png)
                    payload_out["exported_hash"] = _sha256(exported_png)
                    figure_ok = payload_out["canonical_hash"] == payload_out["exported_hash"]
                results.append(_check(figure_ok, f"Exported paper figure matches canonical: {key}", payload_out))

    passed = all(item["status"] == "PASS" for item in results)
    summary = {
        "baseline_id": manifest["baseline_id"],
        "canonical_run_root": str(canonical_root),
        "reproduced_run_root": str(run_root),
        "passed": passed,
        "checks": results,
    }
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
