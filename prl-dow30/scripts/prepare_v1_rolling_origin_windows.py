#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare split-specific current configs for the v1 rolling-origin study.")
    parser.add_argument(
        "--base-current-config",
        default=str(ROOT / "frozen_protocol/paper_v3/current_config.yaml"),
        help="Frozen baseline current config used as the source template.",
    )
    parser.add_argument(
        "--splits-json",
        default=str(ROOT / "frozen_protocol/rolling_windows_v1/split_definitions.json"),
        help="Rolling split definition JSON.",
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Experiment root under outputs/extensions/v1_rolling_origin_windows/<stamp>.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def main() -> None:
    args = parse_args()
    base_current_config = Path(args.base_current_config).resolve()
    splits_json = Path(args.splits_json).resolve()
    experiment_root = Path(args.experiment_root).resolve()
    prepared_root = experiment_root / "prepared"
    configs_root = prepared_root / "configs"
    prepared_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _read_yaml(base_current_config)
    split_payload = json.loads(splits_json.read_text())

    manifest: dict[str, object] = {
        "experiment_root": str(experiment_root),
        "prepared_root": str(prepared_root),
        "base_current_config": str(base_current_config),
        "splits_json": str(splits_json),
        "splits": {},
    }

    for split in split_payload["splits"]:
        split_id = str(split["split_id"])
        split_root = experiment_root / "splits" / split_id
        split_cfg = dict(base_cfg)
        split_cfg["dates"] = dict(base_cfg["dates"])
        split_cfg["dates"]["train_start"] = split["train"]["start"]
        split_cfg["dates"]["train_end"] = split["train"]["end"]
        split_cfg["dates"]["test_start"] = split["validation"]["start"]
        split_cfg["dates"]["test_end"] = split["test"]["end"]
        split_cfg.setdefault("output", {})
        split_cfg["output"]["root"] = str(split_root / "train_control")

        current_config_out = configs_root / split_id / "current_config.yaml"
        split_definition_out = configs_root / split_id / "split_definition.json"
        _write_yaml(current_config_out, split_cfg)
        split_definition_out.parent.mkdir(parents=True, exist_ok=True)
        split_definition_out.write_text(json.dumps(split, indent=2) + "\n")

        split_entry = {
            "split_id": split_id,
            "label": split["label"],
            "status": split["status"],
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
            "current_config": str(current_config_out),
            "split_definition": str(split_definition_out),
            "run_root": str(split_root),
        }
        if split["status"] == "canonical_reference":
            split_entry["canonical_run_root"] = split["canonical_run_root"]
        manifest["splits"][split_id] = split_entry

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
