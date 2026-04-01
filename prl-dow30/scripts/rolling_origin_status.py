#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report per-split status for the v1 rolling-origin runs.")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def _step_state(master_log: Path) -> str:
    if not master_log.exists():
        return "not_started"
    text = master_log.read_text()
    if "[DONE] phase=control_eta_validation_first complete" in text or "[STEP-END] check_validation_first_outputs rc=0" in text:
        return "completed"
    if "[STEP-END] external_baselines rc=0" in text:
        return "completed"
    if "[STEP-START] final_selected_eta_eval" in text:
        return "final"
    if "[STEP-START] select_validation_eta" in text:
        return "selection"
    if "[STEP-START] validation_eta_frontier" in text:
        return "validation"
    if "[STEP-START] train_control_seed0" in text:
        return "training"
    return "started"


def _completed_seeds(run_root: Path) -> int:
    models_dir = run_root / "train_control" / "models"
    if not models_dir.exists():
        return 0
    seeds = set()
    for path in models_dir.glob("*_final.zip"):
        name = path.name
        if "seed" not in name:
            continue
        try:
            token = name.split("seed", 1)[1]
            digits = []
            for ch in token:
                if ch.isdigit():
                    digits.append(ch)
                else:
                    break
            if digits:
                seeds.add(int("".join(digits)))
        except Exception:
            continue
    return len(seeds)


def _has_train_activity(run_root: Path) -> bool:
    logs_dir = run_root / "train_control" / "logs"
    if not logs_dir.exists():
        return False
    return any(logs_dir.glob("train_*.csv"))


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    manifest = json.loads((experiment_root / "prepared" / "manifest.json").read_text())
    rows = []
    for split_id, split in manifest["splits"].items():
        if split["status"] == "canonical_reference":
            rows.append(
                {
                    "split_id": split_id,
                    "status": "canonical_reference",
                    "run_root": split["canonical_run_root"],
                    "completed_seeds": 10,
                }
            )
            continue
        run_root = Path(split["run_root"])
        checklist_json = run_root / "checklists" / "control_eta_validation_first.json"
        if checklist_json.exists():
            rows.append(
                {
                    "split_id": split_id,
                    "status": "completed",
                    "run_root": str(run_root),
                    "completed_seeds": _completed_seeds(run_root),
                }
            )
            continue
        master_log = run_root / "logs" / "master.log"
        state = _step_state(master_log)
        completed_seeds = _completed_seeds(run_root)
        if state == "not_started" and (_has_train_activity(run_root) or completed_seeds > 0):
            state = "training"
        rows.append(
            {
                "split_id": split_id,
                "status": state,
                "run_root": str(run_root),
                "completed_seeds": completed_seeds,
            }
        )
    print(json.dumps({"experiment_root": str(experiment_root), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
