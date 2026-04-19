# Universe Validation Note

`scripts/generalization/build_universes.py` is the reusable loader and validator for the generalization package's universe specs.

## Purpose

The script provides one stable place to:

- read YAML universe specs from `paper/forecasting_workshop/generalization/universe_specs/`
- normalize ticker symbols into a consistent uppercase representation
- validate the basic structural rules before downstream runners consume a universe
- emit structured JSON output that later experiment wrappers can reuse

This keeps the universe layer standalone and reduces the risk of silently inconsistent support baskets.

## Default Validation Rules

The validator checks:

- unique universe name across all loaded spec files
- non-empty ticker list
- no duplicate tickers within a universe after normalization
- exactly `27` tickers by default

The 27-name rule is the default because the first multi-universe support package is centered on fixed 27-name snapshots that can be compared against the current U27.

## Explicit Count Overrides

If a future universe intentionally uses a non-27 ticker count, it must say so explicitly in the YAML. The validator supports two conservative escape hatches:

- `expected_ticker_count: <positive integer>`
- `allow_non_27_count: true`

If neither field is present, the validator enforces the default `27`-ticker rule.

## CLI Shape

Current CLI entrypoints:

- `--list`
- `--validate`
- `--show <universe_name>`

The CLI prints structured JSON on success and clear stderr errors on failure so that downstream wrappers can call it directly without inventing their own parsing logic.

## Scope Boundary

This utility validates the universe spec layer only. It does not run experiments, and it does not change the main training pipeline.
