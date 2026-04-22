#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from build_same_interface_rank_summary import build_domain_rank_summary, validate_q2_source, write_summary_outputs  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "forecast_eval" / "inventory_q2_stronger_baselines"
DEFAULT_PAPER_RESULTS_DIR = REPO_ROOT / "paper" / "forecasting_workshop" / "results"

PAPER_FORECASTER_LABELS = {
    "naive_last": "Naive persistence",
    "moving_average_7": "Moving average (7)",
    "linear_ar_ridge": "Linear AR",
    "mlp_small": "Small MLP",
    "gru_small": "Small GRU",
    "reg_linear_lag_search": "Regularized linear + lag search",
    "gbrt_lagged": "Gradient-boosted trees",
    "mlp_large": "Large MLP",
    "gru_variant": "GRU variant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build appendix artifacts for inventory Q2 stronger baselines.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--paper-results-dir", default=str(DEFAULT_PAPER_RESULTS_DIR))
    parser.add_argument("--paper-stem", default="table_q2_stronger_baseline")
    parser.add_argument("--note-name", default="stronger_baseline_note_v1.md")
    parser.add_argument("--robustness-only", action="store_true")
    return parser.parse_args()


def _paper_label(model_id: str) -> str:
    return PAPER_FORECASTER_LABELS.get(str(model_id), str(model_id))


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _write_tex(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _protocol_table(protocol_df: pd.DataFrame) -> pd.DataFrame:
    table = protocol_df.copy()
    table["Family"] = table["display_name"]
    table["Input"] = table["input_form"]
    table["Stage 1"] = table["stage1_candidate_count"].map(lambda value: "--" if int(value) == 0 else str(int(value)))
    table["Stage 2"] = table["stage2_top_k"].map(lambda value: "--" if int(value) == 0 else f"top {int(value)}")
    table["Seeds"] = table.apply(
        lambda row: (
            "final 10"
            if not bool(row["tunable"])
            else f"s1 {row['stage1_seeds'].replace('|', ',')} / s2 {row['stage2_seeds'].replace('|', ',')} / final 0-9"
        ),
        axis=1,
    )
    table["Representative rule"] = table["selection_rule"].map(
        lambda value: "fixed ex ante"
        if value == "fixed_ex_ante"
        else "best mean val. metric, then median, then pre-registered simpler config"
    )
    return table[["Family", "Input", "Stage 1", "Stage 2", "Seeds", "Representative rule"]]


def _write_protocol_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabularx}{\\textwidth}{@{}l l c c >{\\raggedright\\arraybackslash}X >{\\raggedright\\arraybackslash}X@{}}",
        "\\toprule",
        "Family & Input & Stage 1 & Stage 2 & Seeds & Representative rule \\\\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabularx}"])
    _write_tex(path, lines)


def _selection_table(representatives_df: pd.DataFrame) -> pd.DataFrame:
    table = representatives_df.copy()
    family_order = {family: idx for idx, family in enumerate(PAPER_FORECASTER_LABELS)}
    table["Family"] = table["display_name"]
    table["Representative"] = table.apply(_selection_repr_string, axis=1)
    table["Validation mean"] = table["validation_mean_metric"].map(lambda value: _format_float(value, 3))
    table["Validation median"] = table["validation_median_metric"].map(lambda value: _format_float(value, 3))
    table["_family_order"] = table["family"].map(family_order).fillna(999).astype(int)
    table = table.sort_values(["_family_order", "Family"]).reset_index(drop=True)
    return table[["Family", "Representative", "Validation mean", "Validation median"]]


def _selection_repr_string(row: pd.Series) -> str:
    if str(row["selected_config_id"]) == "fixed_ex_ante":
        return "fixed ex ante"
    params = {}
    raw = row.get("selected_params_json", "")
    if isinstance(raw, str) and raw:
        params = dict(json.loads(raw))
    family = str(row["family"])
    if family == "linear_ar_ridge":
        return f"alpha={params['alpha']}"
    if family == "mlp_small":
        return f"lr={params['lr']}, wd={params['weight_decay']}, batch={params['batch_size']}"
    if family == "gru_small":
        return f"lr={params['lr']}, batch={params['batch_size']}"
    if family == "reg_linear_lag_search":
        return f"lag={params['lag']}, {params['penalty']}, alpha={params['alpha']}"
    if family == "gbrt_lagged":
        return (
            f"lag={params['lag']}, trees={params['n_estimators']}, depth={params['max_depth']}, "
            f"lr={params['learning_rate']}, leaf={params['min_samples_leaf']}"
        )
    if family == "mlp_large":
        return (
            f"w={params['hidden_width']}, d={params['depth']}, drop={params['dropout']}, "
            f"lr={params['lr']}, wd={params['weight_decay']}"
        )
    if family == "gru_variant":
        return (
            f"L={params['sequence_length']}, h={params['hidden_size']}, layers={params['num_layers']}, "
            f"drop={params['dropout']}, lr={params['lr']}"
        )
    return str(row["selected_config_id"])


def _write_selection_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabularx}{\\textwidth}{@{}l >{\\raggedright\\arraybackslash}X c c@{}}",
        "\\toprule",
        "Family & Representative & Validation mean & Validation median \\\\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}"])
    _write_tex(path, lines)


def _robustness_table(selection_summary: pd.DataFrame, selection_seed: pd.DataFrame) -> pd.DataFrame:
    suboptimal_counts = (
        selection_seed.assign(
            deployed_suboptimal_flag=selection_seed["deployed_gap_of_forecast_selected"].gt(1e-12)
        )
        .groupby("friction_level", as_index=False)
        .agg(
            deployed_suboptimal_seeds=("deployed_suboptimal_flag", "sum"),
            total_seeds=("seed", "count"),
        )
    )
    suboptimal_counts["Deployed-suboptimal seeds / total"] = suboptimal_counts.apply(
        lambda row: f"{int(row['deployed_suboptimal_seeds'])}/{int(row['total_seeds'])}",
        axis=1,
    )
    table = selection_summary.merge(
        suboptimal_counts[["friction_level", "Deployed-suboptimal seeds / total"]],
        on="friction_level",
        how="left",
    )
    table = table.rename(
        columns={
            "friction_level": "Friction",
            "most_frequent_forecast_best": "Forecast-side winner",
            "most_frequent_deployed_best": "Deployed winner",
            "agreement_rate": "Agreement rate",
            "mean_deployed_gap_of_forecast_selected": "Mean deployed gap",
        }
    )
    table["Forecast-side winner"] = table["Forecast-side winner"].map(_paper_label)
    table["Deployed winner"] = table["Deployed winner"].map(_paper_label)
    return table[
        [
            "Friction",
            "Forecast-side winner",
            "Deployed winner",
            "Agreement rate",
            "Mean deployed gap",
            "Deployed-suboptimal seeds / total",
        ]
    ]


def _write_robustness_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Friction & Forecast-side winner & Deployed winner & Agreement rate & Mean deployed gap & Deployed-suboptimal seeds / total \\\\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    _write_tex(path, lines)


def _qualitative_pass(robustness_table: pd.DataFrame) -> bool:
    zero = robustness_table.loc[robustness_table["Friction"].eq(0.0)].iloc[0]
    positive = robustness_table.loc[robustness_table["Friction"] > 0.0].copy()
    if positive.empty:
        return False
    high = positive.loc[positive["Friction"].ge(0.5)].copy()
    if high.empty:
        return False
    return bool(
        float(zero["Agreement rate"]) >= 0.7
        and float(high["Agreement rate"].min()) <= 0.4
        and float(high["Mean deployed gap"].max()) >= 2.0 * float(zero["Mean deployed gap"])
    )


def _write_note(path: Path, protocol_df: pd.DataFrame, robustness_table: pd.DataFrame) -> None:
    zero = robustness_table.loc[robustness_table["Friction"].eq(0.0)].iloc[0]
    high = robustness_table.loc[robustness_table["Friction"].ge(0.5)].sort_values(
        ["Mean deployed gap", "Friction"], ascending=[False, True]
    ).iloc[0]
    lines = [
        "# Inventory Q2 stronger-baseline note",
        "",
        f"- Verdict: {'main_text_ready' if _qualitative_pass(robustness_table) else 'appendix_only'}",
        "- Scope: defense-oriented robustness layer, not new headline evidence.",
        "- Selection metric for family representatives: validation negative MAE only.",
        "- Held-out forecast-side ranking metric: the same held-out raw forecast metric used by the locked inventory Q2 schema.",
        "- Standardized procedure: same split, held-out protocol, validation criterion, and two-stage tuning structure across tunable families.",
        f"- Tunable family count: {int(protocol_df['tunable'].sum())}",
        f"- Zero-friction agreement: {_format_float(float(zero['Agreement rate']), 2)}",
        f"- Strongest moderate/high-friction mean deployed gap: {_format_float(float(high['Mean deployed gap']), 3)} at friction {_format_float(float(high['Friction']), 2)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    paper_results_dir = Path(args.paper_results_dir).resolve()
    paper_results_dir.mkdir(parents=True, exist_ok=True)
    paper_stem = str(args.paper_stem)

    q2_path = output_dir / "q2_diff_forecasts_same_interface.csv"
    protocol_path = output_dir / "protocol_summary.csv"
    representatives_path = output_dir / "family_representatives.csv"

    q2_df = pd.read_csv(q2_path)
    protocol_df = pd.read_csv(protocol_path)
    representatives_df = None if args.robustness_only else pd.read_csv(representatives_path)

    failures = validate_q2_source(q2_df, expected_interface_id="responsive", min_forecasters_per_seed_friction=9)
    if failures:
        raise SystemExit(f"[inventory-q2-stronger-baseline-appendix] invalid input: {failures}")

    rank_outputs, _meta = build_domain_rank_summary(
        q2_df,
        domain="inventory",
        expected_interface_id="responsive",
    )
    write_summary_outputs(rank_outputs, output_dir)

    robustness_table = _robustness_table(
        rank_outputs["selection_summary_by_friction"].copy(),
        rank_outputs["seed_level_selection_stats"].copy(),
    )

    robustness_csv = paper_results_dir / f"{paper_stem}_robustness.csv"
    robustness_tex = paper_results_dir / f"{paper_stem}_robustness.tex"
    note_path = paper_results_dir / str(args.note_name)

    if not args.robustness_only:
        protocol_table = _protocol_table(protocol_df)
        selection_table = _selection_table(representatives_df)
        protocol_csv = paper_results_dir / f"{paper_stem}_protocol.csv"
        protocol_tex = paper_results_dir / f"{paper_stem}_protocol.tex"
        selection_csv = paper_results_dir / f"{paper_stem}_selection.csv"
        selection_tex = paper_results_dir / f"{paper_stem}_selection.tex"
        protocol_table.to_csv(protocol_csv, index=False)
        selection_table.to_csv(selection_csv, index=False)
        _write_protocol_tex(protocol_table, protocol_tex)
        _write_selection_tex(selection_table, selection_tex)

    robustness_csv_frame = robustness_table.copy()
    robustness_csv_frame["Friction"] = robustness_csv_frame["Friction"].map(lambda value: _format_float(value, 2))
    robustness_csv_frame["Agreement rate"] = robustness_csv_frame["Agreement rate"].map(lambda value: _format_float(value, 2))
    robustness_csv_frame["Mean deployed gap"] = robustness_csv_frame["Mean deployed gap"].map(lambda value: _format_float(value, 3))
    robustness_csv_frame.to_csv(robustness_csv, index=False)

    robustness_tex_frame = robustness_table.copy()
    robustness_tex_frame["Friction"] = robustness_tex_frame["Friction"].map(lambda value: _format_float(value, 2))
    robustness_tex_frame["Agreement rate"] = robustness_tex_frame["Agreement rate"].map(lambda value: _format_float(value, 2))
    robustness_tex_frame["Mean deployed gap"] = robustness_tex_frame["Mean deployed gap"].map(lambda value: _format_float(value, 3))
    _write_robustness_tex(robustness_tex_frame, robustness_tex)
    _write_note(note_path, protocol_df, robustness_table)

    if not args.robustness_only:
        print(f"[inventory-q2-stronger-baseline-appendix] wrote {protocol_csv}")
        print(f"[inventory-q2-stronger-baseline-appendix] wrote {selection_csv}")
    print(f"[inventory-q2-stronger-baseline-appendix] wrote {robustness_csv}")
    print(f"[inventory-q2-stronger-baseline-appendix] wrote {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
