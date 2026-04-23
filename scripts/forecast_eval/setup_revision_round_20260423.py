#!/usr/bin/env python3
from __future__ import annotations

import argparse

from revision_round_20260423 import (
    ANALYSIS_ADDITIONS_DIR,
    BASELINE_LOCK_DIR,
    CORE_IDENTITY_SENTENCE,
    EVENT_MICRO_DIR,
    EVENT_MICRO_REGIME_DIR,
    EXTENSION_ROOT,
    LOAD_FOLLOWING_DIR,
    LOGICAL_CANONICAL_ROOT,
    LOGICAL_ROOT_MAP_PATH,
    PAPER_STAGING_DIR,
    PHYSICAL_STORAGE_ROOT,
    REWRITE_ORDER,
    REVISION_ROUND_ID,
    STORY_REVISION_DIR,
    TRACK4_OBJECT_BUDGET,
    build_baseline_manifest,
    ensure_logical_alias,
    ensure_dir,
    logical_root_map_payload,
    repo_relative,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the isolated workspace for revision round 20260423.")
    parser.add_argument(
        "--skip-baseline-manifest",
        action="store_true",
        help="Create the directory layout only without copying the current baseline snapshot.",
    )
    return parser.parse_args()


def write_protocol_docs() -> None:
    write_markdown(
        EXTENSION_ROOT / "README.md",
        [
            f"# {LOGICAL_CANONICAL_ROOT.name}",
            "",
            f"- Canonical logical root: `{repo_relative(LOGICAL_CANONICAL_ROOT)}`.",
            f"- Physical storage root: `{repo_relative(PHYSICAL_STORAGE_ROOT)}`.",
            f"- Legacy round id `{REVISION_ROUND_ID}` remains a backward-compatible storage alias only.",
            f"- Core identity sentence: `{CORE_IDENTITY_SENTENCE}`",
            "- Live manuscript sources under `paper/forecasting_workshop/` should be frozen before narrative rewrites land.",
            "- New work is separated into `freeze_q2_pivot_base/`, `story_revision/`, `analysis_additions/`, and `new_reruns/`.",
        ],
    )
    write_markdown(
        BASELINE_LOCK_DIR / "BASELINE_FREEZE_NOTE.md",
        [
            "# Baseline Freeze Note",
            "",
            "- This snapshot freezes the current workshop manuscript and paper-facing artifacts before any new revision-round experiment is interpreted.",
            "- If the round fails the stop-loss rule, revert to this frozen baseline and do not promote any staged result.",
        ],
    )
    write_markdown(
        STORY_REVISION_DIR / "rewrite_order_lock.md",
        ["# Rewrite Order Lock", ""] + [f"- {item}" for item in REWRITE_ORDER],
    )
    write_markdown(
        STORY_REVISION_DIR / "layout_preflight_checklist.md",
        ["# Track 4 Layout Preflight", ""] + [f"- {item}" for item in TRACK4_OBJECT_BUDGET],
    )
    write_markdown(
        STORY_REVISION_DIR / "identity_lock_note.md",
        [
            "# Q2 Pivot Identity Lock",
            "",
            f"- Core sentence: `{CORE_IDENTITY_SENTENCE}`",
            "- Main story order: Synthetic Q2 anchor -> Event-micro Q2 main evidence -> Inventory Q2 main corroboration -> Load-following short corroboration -> Q1 short mechanism support.",
            "- Portfolio is excluded from main text and paper-facing appendix in the revised manuscript.",
        ],
    )


def main() -> int:
    args = parse_args()

    ensure_logical_alias()
    for path in [
        EXTENSION_ROOT,
        BASELINE_LOCK_DIR,
        STORY_REVISION_DIR,
        ANALYSIS_ADDITIONS_DIR,
        EVENT_MICRO_DIR,
        EVENT_MICRO_REGIME_DIR,
        LOAD_FOLLOWING_DIR,
        PAPER_STAGING_DIR,
    ]:
        ensure_dir(path)

    write_json(LOGICAL_ROOT_MAP_PATH, logical_root_map_payload())
    write_protocol_docs()

    if not args.skip_baseline_manifest:
        manifest = build_baseline_manifest()
        write_json(BASELINE_LOCK_DIR / "baseline_manifest.json", manifest)
        print(f"[revision-setup] wrote {BASELINE_LOCK_DIR / 'baseline_manifest.json'}")

    print(f"[revision-setup] logical root alias: {LOGICAL_CANONICAL_ROOT}")
    print(f"[revision-setup] physical workspace: {EXTENSION_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
