#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "paper_artifact_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-facing tables and figures for a rebuilt run.")
    parser.add_argument("--run-root", required=True, help="Run root containing paper_pack outputs.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional destination directory. Defaults to <run-root>/paper_artifacts.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def export_artifacts(run_root: Path, output_dir: Path) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    export_manifest: dict[str, object] = {
        "baseline_id": manifest["baseline_id"],
        "source_run_root": str(run_root),
        "tables": {},
        "figures": {},
    }

    for key, payload in manifest["tables"].items():
        canonical_csv = resolve_repo_path(payload["canonical_csv"])
        rel_csv = canonical_csv.relative_to(resolve_repo_path(manifest["canonical_run_root"]))
        source_csv = run_root / rel_csv
        dest_csv = tables_dir / f"{key.lower().replace(' ', '_')}.csv"
        shutil.copy2(source_csv, dest_csv)
        table_record = {
            "label": payload["label"],
            "title": payload["title"],
            "source_csv": str(source_csv),
            "export_csv": str(dest_csv),
        }
        canonical_md = payload.get("canonical_md")
        if canonical_md:
            source_md = run_root / resolve_repo_path(canonical_md).relative_to(resolve_repo_path(manifest["canonical_run_root"]))
            dest_md = tables_dir / f"{key.lower().replace(' ', '_')}.md"
            shutil.copy2(source_md, dest_md)
            table_record["source_md"] = str(source_md)
            table_record["export_md"] = str(dest_md)
        canonical_extended_csv = payload.get("canonical_extended_csv")
        if canonical_extended_csv:
            source_ext = run_root / resolve_repo_path(canonical_extended_csv).relative_to(resolve_repo_path(manifest["canonical_run_root"]))
            dest_ext = tables_dir / f"{key.lower().replace(' ', '_')}_extended.csv"
            shutil.copy2(source_ext, dest_ext)
            table_record["source_extended_csv"] = str(source_ext)
            table_record["export_extended_csv"] = str(dest_ext)
        export_manifest["tables"][key] = table_record

    for key, payload in manifest["figures"].items():
        canonical_png = resolve_repo_path(payload["canonical_png"])
        rel_png = canonical_png.relative_to(resolve_repo_path(manifest["canonical_run_root"]))
        source_png = run_root / rel_png
        dest_png = figures_dir / f"{key.lower().replace(' ', '_')}.png"
        shutil.copy2(source_png, dest_png)
        export_manifest["figures"][key] = {
            "label": payload["label"],
            "title": payload["title"],
            "source_png": str(source_png),
            "export_png": str(dest_png),
        }

    (output_dir / "manifest.json").write_text(json.dumps(export_manifest, indent=2) + "\n")
    return export_manifest


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root / "paper_artifacts"
    export_manifest = export_artifacts(run_root, output_dir)
    print(json.dumps(export_manifest, indent=2))


if __name__ == "__main__":
    main()
