#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import build_pairing_report, discover_result_files, merge_result_files  # noqa: E402


DEFAULT_RESULTS_ROOT = ROOT / "outputs" / "forecast_eval"
DEFAULT_MASTER_CSV = DEFAULT_RESULTS_ROOT / "summary" / "master_results.csv"
DEFAULT_PAIRING_CSV = DEFAULT_RESULTS_ROOT / "summary" / "pairing_report.csv"
DEFAULT_SCHEMA_JSON = DEFAULT_RESULTS_ROOT / "summary" / "schema_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-domain forecast-eval CSVs into one master summary.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root directory containing per-domain result CSVs.")
    parser.add_argument("--output-csv", default=str(DEFAULT_MASTER_CSV), help="Destination for the merged master CSV.")
    parser.add_argument("--pairing-csv", default=str(DEFAULT_PAIRING_CSV), help="Destination for the pairing validation CSV.")
    parser.add_argument("--schema-json", default=str(DEFAULT_SCHEMA_JSON), help="Destination for a compact schema report JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    csv_paths = discover_result_files(results_root)
    master_df = merge_result_files(csv_paths)
    pairing_df = build_pairing_report(master_df)

    output_csv = Path(args.output_csv)
    pairing_csv = Path(args.pairing_csv)
    schema_json = Path(args.schema_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairing_csv.parent.mkdir(parents=True, exist_ok=True)
    schema_json.parent.mkdir(parents=True, exist_ok=True)

    master_df.to_csv(output_csv, index=False)
    pairing_df.to_csv(pairing_csv, index=False)

    schema_report = {
        "result_file_count": len(csv_paths),
        "domains": sorted(master_df["domain"].unique().tolist()),
        "questions": sorted(master_df["question_id"].unique().tolist()),
        "row_count": int(len(master_df)),
        "columns": master_df.columns.tolist(),
        "paired_group_count": int(len(pairing_df)),
        "paired_group_failures": int((~pairing_df["paired_ok"]).sum()) if not pairing_df.empty else 0,
    }
    schema_json.write_text(json.dumps(schema_report, indent=2))

    print(f"[summary] merged {len(csv_paths)} files into {output_csv}")
    print(f"[summary] domains={schema_report['domains']} questions={schema_report['questions']} rows={schema_report['row_count']}")
    print(f"[summary] pairing failures={schema_report['paired_group_failures']} report={pairing_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
