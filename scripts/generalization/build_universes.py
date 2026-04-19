#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TICKER_COUNT = 27
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")
DEFAULT_SPEC_DIR = (
    Path(__file__).resolve().parents[2]
    / "paper"
    / "forecasting_workshop"
    / "generalization"
    / "universe_specs"
)


class UniverseSpecError(Exception):
    """Base exception for universe spec loading and validation errors."""


class UniverseValidationError(UniverseSpecError):
    """Raised when one or more universe specs fail validation."""


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    description: str
    construction_rule: str
    seed: int | None
    tickers: tuple[str, ...]
    evaluation_role: str
    notes: tuple[str, ...]
    source_path: str
    ticker_count: int
    expected_ticker_count: int | None
    allow_non_27_count: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tickers"] = list(self.tickers)
        payload["notes"] = list(self.notes)
        return payload


def _normalize_ticker(raw: Any, *, source: Path, name: str, index: int) -> str:
    if not isinstance(raw, str):
        raise UniverseValidationError(
            f"{source}: universe '{name}' ticker at index {index} must be a string, got {type(raw).__name__}."
        )
    ticker = raw.strip().upper()
    if not ticker:
        raise UniverseValidationError(f"{source}: universe '{name}' contains an empty ticker at index {index}.")
    if not TICKER_PATTERN.match(ticker):
        raise UniverseValidationError(
            f"{source}: universe '{name}' ticker '{raw}' normalizes to invalid symbol '{ticker}'."
        )
    return ticker


def _require_string(payload: dict[str, Any], key: str, *, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UniverseValidationError(f"{source}: field '{key}' must be a non-empty string.")
    return value.strip()


def _parse_seed(payload: dict[str, Any], *, source: Path) -> int | None:
    seed = payload.get("seed")
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise UniverseValidationError(f"{source}: field 'seed' must be an integer or null.")
    return int(seed)


def _parse_notes(payload: dict[str, Any], *, source: Path) -> tuple[str, ...]:
    notes = payload.get("notes", [])
    if notes is None:
        return ()
    if isinstance(notes, str):
        note = notes.strip()
        return (note,) if note else ()
    if not isinstance(notes, list):
        raise UniverseValidationError(f"{source}: field 'notes' must be a list of strings, a string, or null.")
    normalized: list[str] = []
    for idx, item in enumerate(notes):
        if not isinstance(item, str) or not item.strip():
            raise UniverseValidationError(f"{source}: notes[{idx}] must be a non-empty string.")
        normalized.append(item.strip())
    return tuple(normalized)


def _parse_ticker_count_rules(payload: dict[str, Any], *, source: Path) -> tuple[int | None, bool]:
    allow_non_27 = bool(payload.get("allow_non_27_count", False))
    expected = payload.get("expected_ticker_count")
    if expected is None:
        return (None if allow_non_27 else DEFAULT_TICKER_COUNT), allow_non_27
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise UniverseValidationError(
            f"{source}: field 'expected_ticker_count' must be a positive integer when provided."
        )
    return int(expected), allow_non_27


def load_universe_spec(spec_path: str | Path) -> UniverseSpec:
    path = Path(spec_path).resolve()
    if not path.exists():
        raise UniverseValidationError(f"Universe spec not found: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise UniverseValidationError(f"Universe spec must be a YAML file: {path}")

    try:
        payload = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise UniverseValidationError(f"{path}: failed to parse YAML: {exc}") from exc

    if not isinstance(payload, dict):
        raise UniverseValidationError(f"{path}: top-level YAML must be a mapping/object.")

    name = _require_string(payload, "name", source=path)
    description = _require_string(payload, "description", source=path)
    construction_rule = _require_string(payload, "construction_rule", source=path)
    evaluation_role = _require_string(payload, "evaluation_role", source=path)
    seed = _parse_seed(payload, source=path)
    notes = _parse_notes(payload, source=path)
    expected_ticker_count, allow_non_27_count = _parse_ticker_count_rules(payload, source=path)

    tickers_raw = payload.get("tickers")
    if not isinstance(tickers_raw, list):
        raise UniverseValidationError(f"{path}: field 'tickers' must be a non-empty list of ticker strings.")
    if not tickers_raw:
        raise UniverseValidationError(f"{path}: universe '{name}' must define a non-empty ticker list.")

    tickers = tuple(_normalize_ticker(item, source=path, name=name, index=idx) for idx, item in enumerate(tickers_raw))
    duplicates = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
    if duplicates:
        raise UniverseValidationError(f"{path}: universe '{name}' contains duplicate tickers: {duplicates}.")

    ticker_count = len(tickers)
    if expected_ticker_count is not None and ticker_count != expected_ticker_count:
        raise UniverseValidationError(
            f"{path}: universe '{name}' has {ticker_count} tickers; expected {expected_ticker_count}."
        )
    if expected_ticker_count is None and not allow_non_27_count and ticker_count != DEFAULT_TICKER_COUNT:
        raise UniverseValidationError(
            f"{path}: universe '{name}' has {ticker_count} tickers; expected exactly {DEFAULT_TICKER_COUNT} "
            "unless explicitly marked otherwise."
        )

    return UniverseSpec(
        name=name,
        description=description,
        construction_rule=construction_rule,
        seed=seed,
        tickers=tickers,
        evaluation_role=evaluation_role,
        notes=notes,
        source_path=str(path),
        ticker_count=ticker_count,
        expected_ticker_count=expected_ticker_count,
        allow_non_27_count=allow_non_27_count,
    )


def load_universe_specs(spec_dir: str | Path = DEFAULT_SPEC_DIR) -> dict[str, UniverseSpec]:
    root = Path(spec_dir).resolve()
    if not root.exists():
        raise UniverseValidationError(f"Universe spec directory not found: {root}")
    if not root.is_dir():
        raise UniverseValidationError(f"Universe spec path is not a directory: {root}")

    spec_paths = sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}]
    )
    if not spec_paths:
        raise UniverseValidationError(f"No YAML universe specs found in: {root}")

    specs: dict[str, UniverseSpec] = {}
    duplicates: dict[str, list[str]] = {}
    for path in spec_paths:
        spec = load_universe_spec(path)
        if spec.name in specs:
            duplicates.setdefault(spec.name, []).append(str(path))
            duplicates[spec.name].append(specs[spec.name].source_path)
            continue
        specs[spec.name] = spec

    if duplicates:
        parts = []
        for name, paths in sorted(duplicates.items()):
            unique_paths = sorted(set(paths))
            parts.append(f"name '{name}' appears in multiple files: {unique_paths}")
        raise UniverseValidationError("; ".join(parts))

    return dict(sorted(specs.items()))


def get_universe_spec(universe_name: str, spec_dir: str | Path = DEFAULT_SPEC_DIR) -> UniverseSpec:
    name = universe_name.strip()
    if not name:
        raise UniverseValidationError("Universe name must be a non-empty string.")
    specs = load_universe_specs(spec_dir)
    try:
        return specs[name]
    except KeyError as exc:
        available = ", ".join(sorted(specs))
        raise UniverseValidationError(f"Unknown universe '{name}'. Available universes: {available}") from exc


def get_universe_tickers(universe_name: str, spec_dir: str | Path = DEFAULT_SPEC_DIR) -> list[str]:
    return list(get_universe_spec(universe_name, spec_dir).tickers)


def _build_list_payload(specs: dict[str, UniverseSpec], *, spec_dir: Path) -> dict[str, Any]:
    return {
        "spec_dir": str(spec_dir),
        "count": len(specs),
        "universes": [
            {
                "name": spec.name,
                "ticker_count": spec.ticker_count,
                "evaluation_role": spec.evaluation_role,
                "seed": spec.seed,
                "source_path": spec.source_path,
            }
            for spec in specs.values()
        ],
    }


def _build_validate_payload(specs: dict[str, UniverseSpec], *, spec_dir: Path) -> dict[str, Any]:
    return {
        "status": "ok",
        "spec_dir": str(spec_dir),
        "count": len(specs),
        "names": list(specs.keys()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, validate, and normalize generalization universe specs.",
    )
    parser.add_argument(
        "--spec-dir",
        type=str,
        default=str(DEFAULT_SPEC_DIR),
        help="Directory containing YAML universe specs.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List available universe specs.")
    group.add_argument("--validate", action="store_true", help="Validate all universe specs.")
    group.add_argument("--show", type=str, metavar="UNIVERSE_NAME", help="Show one validated universe spec.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_dir = Path(args.spec_dir).resolve()

    try:
        if args.list:
            specs = load_universe_specs(spec_dir)
            payload = _build_list_payload(specs, spec_dir=spec_dir)
        elif args.validate:
            specs = load_universe_specs(spec_dir)
            payload = _build_validate_payload(specs, spec_dir=spec_dir)
        else:
            spec = get_universe_spec(args.show, spec_dir)
            payload = spec.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    except UniverseSpecError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
