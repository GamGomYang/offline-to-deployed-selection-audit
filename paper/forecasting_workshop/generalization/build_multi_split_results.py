#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOCKED_KAPPAS = [0.0, 5e-4, 1e-3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the compact multi-split robustness summary CSV.")
    parser.add_argument("--manifest", required=True, help="Prepared multi-split manifest JSON.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV.")
    return parser.parse_args()


def _kappa_label(kappa: float) -> str:
    if np.isclose(kappa, 0.0):
        return "k0"
    if np.isclose(kappa, 5e-4):
        return "k5e4"
    if np.isclose(kappa, 1e-3):
        return "k1e3"
    return f"k{str(kappa).replace('.', 'p').replace('-', 'm')}"


def _compute_sharpe(returns: pd.Series) -> float:
    arr = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return 0.0
    std = float(arr.std(ddof=0))
    if std <= 1e-8:
        return 0.0
    return float((arr.mean() / std) * np.sqrt(252.0))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _load_selection(run_root: Path) -> tuple[float, dict]:
    selection_path = run_root / "validation_eta" / "selection" / "validation_eta_selection.json"
    if not selection_path.exists():
        selection_path = run_root / "paper_pack" / "validation" / "validation_eta_selection.json"
    payload = json.loads(selection_path.read_text())
    return float(payload["selected_eta"]), payload


def _load_pair_stats(run_root: Path) -> pd.DataFrame:
    return pd.read_csv(run_root / "paper_pack" / "stats" / "selected_eta_vs_eta1_stats.csv")


def _load_diagnostics(run_root: Path) -> pd.DataFrame:
    return pd.read_csv(run_root / "paper_pack" / "diagnostics" / "diagnostic_selected_eta_v2.csv")


def _final_metrics_map(run_root: Path) -> dict[tuple[float, float, int], Path]:
    out: dict[tuple[float, float, int], Path] = {}
    for metrics_path in sorted((run_root / "final_eta").glob("kappa_*/*/seed_*/metrics.csv")):
        seed_dir = metrics_path.parent
        eta_dir = seed_dir.parent
        kappa_dir = eta_dir.parent
        kappa = float(kappa_dir.name.split("kappa_", 1)[1])
        eta = float(eta_dir.name.split("eta_", 1)[1])
        seed = int(seed_dir.name.split("seed_", 1)[1])
        out[(kappa, eta, seed)] = metrics_path
    return out


def _target_delta_pair_median(run_root: Path, *, selected_eta: float, baseline_eta: float = 1.0) -> dict[float, float]:
    metrics_map = _final_metrics_map(run_root)
    out: dict[float, float] = {}
    for kappa in LOCKED_KAPPAS:
        deltas: list[float] = []
        seed_candidates = sorted(
            {
                int(seed)
                for (seen_kappa, seen_eta, seed) in metrics_map.keys()
                if np.isclose(seen_kappa, kappa, atol=1e-15)
                and (np.isclose(seen_eta, selected_eta, atol=1e-15) or np.isclose(seen_eta, baseline_eta, atol=1e-15))
            }
        )
        for seed in seed_candidates:
            sel_metrics = metrics_map.get((kappa, selected_eta, seed))
            base_metrics = metrics_map.get((kappa, baseline_eta, seed))
            if sel_metrics is None or base_metrics is None:
                continue
            sel_trace = pd.read_parquet(sel_metrics.parent / "trace.parquet")
            base_trace = pd.read_parquet(base_metrics.parent / "trace.parquet")
            sel_target = _compute_sharpe(sel_trace["net_return_lin_target"])
            base_target = _compute_sharpe(base_trace["net_return_lin_target"])
            deltas.append(sel_target - base_target)
        out[kappa] = _median(deltas)
    return out


def _zero_cost_flag(delta0: float) -> str:
    return "yes" if abs(delta0) <= 0.005 else "no"


def _positive_cost_direction_flag(deltas: list[float]) -> str:
    signs = [float(x) > 0.0 for x in deltas if np.isfinite(x)]
    if not signs:
        return "missing"
    if all(signs):
        return "yes"
    if any(signs):
        return "mixed"
    return "no"


def _disagreement_flag(exec_deltas: list[float], target_deltas: list[float]) -> str:
    rows = []
    for exec_delta, target_delta in zip(exec_deltas, target_deltas):
        if not np.isfinite(exec_delta) or not np.isfinite(target_delta):
            continue
        rows.append(exec_delta > 0.0 and target_delta <= 0.0)
    if not rows:
        return "missing"
    if all(rows):
        return "yes"
    if any(rows):
        return "mixed"
    return "no"


def _resolve_run_root(split_payload: dict, *, manifest_path: Path) -> Path:
    if split_payload["status"] == "canonical_reference":
        return Path(split_payload["canonical_run_root"]).resolve()
    if "run_root" in split_payload:
        return Path(split_payload["run_root"]).resolve()
    split_id = str(split_payload["split_id"])
    inferred = manifest_path.parent / "multi_split_compact_v1" / "splits" / split_id
    return inferred.resolve()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, object]] = []

    for split_payload in manifest["splits"]:
        split_id = str(split_payload["split_id"])
        run_root = _resolve_run_root(split_payload, manifest_path=manifest_path)
        selected_eta, _selection_payload = _load_selection(run_root)
        pair_stats = _load_pair_stats(run_root)
        diagnostics = _load_diagnostics(run_root)
        target_deltas = _target_delta_pair_median(run_root, selected_eta=selected_eta, baseline_eta=1.0)

        row: dict[str, object] = {
            "split_id": split_id,
            "label": split_payload["label"],
            "status": split_payload["status"],
            "run_root": str(run_root),
            "selected_eta": selected_eta,
        }

        exec_positive: list[float] = []
        target_positive: list[float] = []
        for kappa in LOCKED_KAPPAS:
            label = _kappa_label(kappa)
            pair_row = pair_stats[np.isclose(pair_stats["kappa"], kappa, atol=1e-15)].iloc[0]
            diag_row = diagnostics[np.isclose(diagnostics["kappa"], kappa, atol=1e-15)].iloc[0]
            exec_delta = float(pair_row["median_delta_sharpe_net_lin"])
            target_delta = float(target_deltas.get(kappa, float("nan")))
            row[f"delta_sharpe_exec_{label}"] = exec_delta
            row[f"selected_toexec_{label}"] = float(pair_row["selected_median_turnover_exec"])
            row[f"baseline_toexec_{label}"] = float(pair_row["baseline_median_turnover_exec"])
            row[f"totgt_selected_{label}"] = float(diag_row["median_turnover_target"])
            row[f"tracking_selected_{label}"] = float(diag_row["median_tracking_error_l2"])
            row[f"target_delta_sharpe_{label}"] = target_delta
            row[f"n_pairs_{label}"] = int(pair_row["n_pairs"])
            if kappa > 0:
                exec_positive.append(exec_delta)
                target_positive.append(target_delta)

        row["zero_cost_near_flat_flag"] = _zero_cost_flag(float(row["delta_sharpe_exec_k0"]))
        row["positive_cost_direction_flag"] = _positive_cost_direction_flag(exec_positive)
        row["target_vs_executed_disagreement_flag"] = _disagreement_flag(exec_positive, target_positive)
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"WROTE={out_path}")


if __name__ == "__main__":
    main()
