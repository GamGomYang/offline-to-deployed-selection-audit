#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rolling-window held-out frontier robustness for the frozen-policy study.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_selection_eta(run_root: Path) -> float:
    payload = json.loads((run_root / "validation_eta" / "selection" / "validation_eta_selection.json").read_text())
    return float(payload["selected_eta"])


def _resolve_run_root(split: dict) -> Path:
    if split["status"] == "canonical_reference":
        return Path(split["canonical_run_root"])
    return Path(split["run_root"])


def _resolve_frontier_root(experiment_root: Path, split: dict, run_root: Path) -> Path:
    candidates = []
    if split["status"] == "canonical_reference":
        candidates.append(experiment_root / "splits" / f"{split['split_id']}_reference" / "final_eta_full_grid")
        candidates.append(run_root / "final_eta_full_grid")
        candidates.append(run_root / "final_eta")
    else:
        candidates.append(run_root / "final_eta_full_grid")
        candidates.append(run_root / "final_eta")
    for candidate in candidates:
        if (candidate / "aggregate.csv").exists():
            return candidate
    raise FileNotFoundError(f"No frontier aggregate found for split {split['split_id']}")


def _build_split_rows(experiment_root: Path, split: dict) -> list[dict]:
    run_root = _resolve_run_root(split)
    selected_eta = _load_selection_eta(run_root)
    frontier_root = _resolve_frontier_root(experiment_root, split, run_root)
    agg = pd.read_csv(frontier_root / "aggregate.csv")

    rows: list[dict] = []
    for kappa in sorted(agg["kappa"].unique().tolist()):
        frame = agg[agg["kappa"] == kappa].copy()
        baseline = frame.loc[frame["eta"] == 1.0].iloc[0]
        interior = frame.loc[frame["eta"] < 1.0].copy().sort_values(["median_sharpe", "eta"], ascending=[False, False])
        best_interior = interior.iloc[0] if not interior.empty else None
        best_any = frame.sort_values(["median_sharpe", "eta"], ascending=[False, False]).iloc[0]

        out = {
            "split_id": split["split_id"],
            "label": split["label"],
            "selected_eta": selected_eta,
            "kappa": float(kappa),
            "frontier_root": str(frontier_root),
            "baseline_eta": 1.0,
            "baseline_median_sharpe": float(baseline["median_sharpe"]),
            "baseline_median_turnover_exec": float(baseline["median_turnover_exec"]),
            "best_any_eta": float(best_any["eta"]),
            "best_any_median_sharpe": float(best_any["median_sharpe"]),
            "best_any_median_turnover_exec": float(best_any["median_turnover_exec"]),
        }
        if best_interior is not None:
            out.update(
                {
                    "best_interior_eta": float(best_interior["eta"]),
                    "best_interior_median_sharpe": float(best_interior["median_sharpe"]),
                    "best_interior_median_turnover_exec": float(best_interior["median_turnover_exec"]),
                    "delta_sharpe_best_interior_vs_eta1": float(best_interior["median_sharpe"] - baseline["median_sharpe"]),
                    "delta_turnover_exec_best_interior_vs_eta1": float(
                        best_interior["median_turnover_exec"] - baseline["median_turnover_exec"]
                    ),
                }
            )
        else:
            out.update(
                {
                    "best_interior_eta": None,
                    "best_interior_median_sharpe": None,
                    "best_interior_median_turnover_exec": None,
                    "delta_sharpe_best_interior_vs_eta1": None,
                    "delta_turnover_exec_best_interior_vs_eta1": None,
                }
            )

        delta_sharpe = out["delta_sharpe_best_interior_vs_eta1"]
        delta_toexec = out["delta_turnover_exec_best_interior_vs_eta1"]
        out["frontier_signal"] = bool(
            delta_sharpe is not None and delta_toexec is not None and delta_sharpe > 0.0 and delta_toexec < 0.0
        )
        out["selected_eta_is_baseline"] = bool(selected_eta == 1.0)
        out["selection_missed_frontier"] = bool(out["selected_eta_is_baseline"] and out["frontier_signal"] and kappa > 0.0)
        rows.append(out)
    return rows


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((experiment_root / "prepared" / "manifest.json").read_text())
    rows: list[dict] = []
    for split in manifest["splits"].values():
        rows.extend(_build_split_rows(experiment_root, split))

    df = pd.DataFrame(rows).sort_values(["split_id", "kappa"]).reset_index(drop=True)
    positive = df[df["kappa"] > 0].copy()
    split_summary = positive.groupby(["split_id", "label"], as_index=False).agg(
        selected_eta=("selected_eta", "first"),
        mean_delta_sharpe_best_interior_vs_eta1=("delta_sharpe_best_interior_vs_eta1", "mean"),
        min_delta_sharpe_best_interior_vs_eta1=("delta_sharpe_best_interior_vs_eta1", "min"),
        mean_delta_turnover_exec_best_interior_vs_eta1=("delta_turnover_exec_best_interior_vs_eta1", "mean"),
        all_positive_kappa_frontier_signal=("frontier_signal", "all"),
        any_positive_kappa_frontier_signal=("frontier_signal", "any"),
        any_selection_missed_frontier=("selection_missed_frontier", "any"),
    )

    verdict = {
        "n_splits_total": int(split_summary["split_id"].nunique()),
        "n_splits_all_positive_kappa_frontier_signal": int(split_summary["all_positive_kappa_frontier_signal"].sum()),
        "n_splits_any_positive_kappa_frontier_signal": int(split_summary["any_positive_kappa_frontier_signal"].sum()),
        "n_splits_selection_missed_frontier": int(split_summary["any_selection_missed_frontier"].sum()),
        "passes_two_of_three_frontier_rule": bool(split_summary["all_positive_kappa_frontier_signal"].sum() >= 2),
    }

    df.to_csv(output_dir / "frontier_split_kappa_summary.csv", index=False)
    split_summary.to_csv(output_dir / "frontier_split_summary.csv", index=False)
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

    lines = [
        "# Rolling Frontier Robustness",
        "",
        f"- total splits: `{verdict['n_splits_total']}`",
        f"- splits with frontier signal on all positive kappas: `{verdict['n_splits_all_positive_kappa_frontier_signal']}`",
        f"- splits with frontier signal on any positive kappa: `{verdict['n_splits_any_positive_kappa_frontier_signal']}`",
        f"- splits where selection stayed at eta=1 despite a positive-cost frontier signal: `{verdict['n_splits_selection_missed_frontier']}`",
        f"- passes two-of-three frontier rule: `{verdict['passes_two_of_three_frontier_rule']}`",
        "",
    ]
    for _, row in split_summary.iterrows():
        lines.append(
            f"- {row['label']} ({row['split_id']}): selected eta `{row['selected_eta']}`, "
            f"mean delta Sharpe(best interior vs eta=1) `{row['mean_delta_sharpe_best_interior_vs_eta1']:.6f}`, "
            f"mean delta TOexec `{row['mean_delta_turnover_exec_best_interior_vs_eta1']:.6f}`, "
            f"all positive-kappa frontier signal `{bool(row['all_positive_kappa_frontier_signal'])}`, "
            f"selection missed frontier `{bool(row['any_selection_missed_frontier'])}`"
        )
    (output_dir / "frontier_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
