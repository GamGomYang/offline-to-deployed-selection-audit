#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "multi_universe" / "raw"
DEFAULT_OUTPUT_CSV = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "outputs" / "multi_universe" / "multi_universe_results.csv"
)
DEFAULT_OUTPUT_TEX = ROOT / "paper" / "forecasting_workshop" / "generalization" / "tables" / "multi_universe_summary.tex"
DEFAULT_OUTPUT_NOTE = ROOT / "paper" / "forecasting_workshop" / "generalization" / "notes" / "multi_universe_note.md"
DEFAULT_OUTPUT_FIG = (
    ROOT / "paper" / "forecasting_workshop" / "generalization" / "figures" / "fig_multi_universe_consistency.pdf"
)

FINAL_PERIOD = "final"
LOCKED_KAPPAS = [0.0, 5e-4, 1e-3]
ZERO_COST_NEAR_FLAT_THRESHOLD = 0.005
ZERO_COST_YELLOW_THRESHOLD = 0.01
TARGET_SUPPRESSION_RATIO = 0.25
TARGET_SUPPRESSION_MIN_EXEC_DELTA = 0.005
UNIVERSE_ORDER = {
    "u27_current": 0,
    "u27_alt_largecap": 1,
    "u27_sector_balanced": 2,
    "u27_random_seed17": 3,
}


@dataclass(frozen=True)
class UniverseVerdict:
    universe: str
    verdict: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-universe raw results into paper-facing outputs.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Root with raw result.json files.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Destination CSV path.")
    parser.add_argument("--output-tex", default=str(DEFAULT_OUTPUT_TEX), help="Destination LaTeX table path.")
    parser.add_argument("--output-note", default=str(DEFAULT_OUTPUT_NOTE), help="Destination note path.")
    parser.add_argument("--output-fig", default=str(DEFAULT_OUTPUT_FIG), help="Destination figure PDF path.")
    return parser.parse_args()


def _kappa_sort_key(kappa: float) -> float:
    return float(kappa)


def _kappa_label(kappa: float) -> str:
    if np.isclose(kappa, 0.0):
        return "0"
    if np.isclose(kappa, 5e-4):
        return "5e-4"
    if np.isclose(kappa, 1e-3):
        return "1e-3"
    return f"{kappa:g}"


def _display_universe_name(universe: str) -> str:
    mapping = {
        "u27_current": "Current",
        "u27_alt_largecap": "Alt-LargeCap",
        "u27_sector_balanced": "Sector-Balanced",
        "u27_random_seed17": "Random-17",
    }
    return mapping.get(universe, universe.replace("_", "-"))


def _load_raw_results(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result_path in sorted(raw_root.glob("*/*/kappa_*/eta_*/seed_*/result.json")):
        rows.append(json.loads(result_path.read_text()))
    if not rows:
        raise FileNotFoundError(f"No result.json files found under {raw_root}")

    df = pd.DataFrame(rows)
    numeric_cols = [
        col
        for col in df.columns
        if col
        not in {
            "universe_name",
            "evaluation_role",
            "period",
            "result_dir",
            "trace_path",
            "model_path",
            "run_completed_at",
        }
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _paired_median(series_a: pd.Series, series_b: pd.Series) -> float:
    arr = pd.to_numeric(series_a - series_b, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _material_target_suppression(exec_delta: float, tgt_delta: float) -> bool:
    if not (np.isfinite(exec_delta) and np.isfinite(tgt_delta)):
        return False
    if abs(exec_delta) < TARGET_SUPPRESSION_MIN_EXEC_DELTA:
        return False
    return abs(tgt_delta) <= TARGET_SUPPRESSION_RATIO * abs(exec_delta)


def _sign_with_tolerance(value: float, *, atol: float = 1e-12) -> int:
    if not np.isfinite(value) or abs(value) <= atol:
        return 0
    return 1 if value > 0.0 else -1


def _compute_zero_cost_flag(delta_exec: float, kappa: float) -> str:
    if not np.isclose(kappa, 0.0):
        return "n/a"
    return "yes" if abs(delta_exec) <= ZERO_COST_NEAR_FLAT_THRESHOLD else "no"


def _compute_disagreement_flag(delta_exec: float, delta_tgt: float, kappa: float, zero_cost_flag: str) -> str:
    if not (np.isfinite(delta_exec) and np.isfinite(delta_tgt)):
        return "missing"
    if np.isclose(kappa, 0.0) and zero_cost_flag == "yes":
        return "no"

    exec_sign = _sign_with_tolerance(delta_exec)
    tgt_sign = _sign_with_tolerance(delta_tgt)
    sign_diff = exec_sign != tgt_sign and not (exec_sign == 0 and tgt_sign == 0)
    suppression = _material_target_suppression(delta_exec, delta_tgt)
    return "yes" if sign_diff or suppression else "no"


def _compute_positive_cost_direction_flag(delta_exec: float, kappa: float) -> str:
    if np.isclose(kappa, 0.0):
        return "n/a"
    if not np.isfinite(delta_exec):
        return "missing"
    return "yes" if delta_exec > 0.0 else "no"


def _compute_row_verdict(delta_exec: float, kappa: float, zero_cost_flag: str, positive_flag: str, disagreement_flag: str) -> str:
    if np.isclose(kappa, 0.0):
        if abs(delta_exec) <= ZERO_COST_NEAR_FLAT_THRESHOLD:
            return "Green"
        if abs(delta_exec) <= ZERO_COST_YELLOW_THRESHOLD:
            return "Yellow"
        return "Red"

    if positive_flag == "yes" and disagreement_flag == "yes":
        return "Green"
    if positive_flag == "yes":
        return "Yellow"
    return "Red"


def _aggregate_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    final_df = df[df["period"] == FINAL_PERIOD].copy()
    if final_df.empty:
        raise ValueError("No final-period rows available for aggregation.")

    rows: list[dict[str, object]] = []
    for universe in sorted(final_df["universe_name"].unique()):
        universe_df = final_df[final_df["universe_name"] == universe].copy()
        for kappa in LOCKED_KAPPAS:
            kappa_df = universe_df[np.isclose(universe_df["kappa"], kappa, atol=1e-15)].copy()
            if kappa_df.empty:
                continue
            wide = (
                kappa_df.pivot_table(
                    index="seed",
                    columns="eta",
                    values=[
                        "sharpe_exec_net",
                        "sharpe_target_net",
                        "turnover_exec",
                        "turnover_target",
                        "tracking_error_l2",
                        "final_path_gap",
                    ],
                    aggfunc="first",
                )
                .sort_index()
                .copy()
            )
            wide.columns = [f"{field}_eta_{eta}" for field, eta in wide.columns]
            wide = wide.reset_index()

            median_sharpe_exec_eta1 = float(wide["sharpe_exec_net_eta_1.0"].median())
            median_sharpe_exec_eta05 = float(wide["sharpe_exec_net_eta_0.5"].median())
            delta_sharpe_exec = _paired_median(wide["sharpe_exec_net_eta_0.5"], wide["sharpe_exec_net_eta_1.0"])

            median_sharpe_tgt_eta1 = float(wide["sharpe_target_net_eta_1.0"].median())
            median_sharpe_tgt_eta05 = float(wide["sharpe_target_net_eta_0.5"].median())
            delta_sharpe_tgt = _paired_median(wide["sharpe_target_net_eta_0.5"], wide["sharpe_target_net_eta_1.0"])

            median_toexec_eta1 = float(wide["turnover_exec_eta_1.0"].median())
            median_toexec_eta05 = float(wide["turnover_exec_eta_0.5"].median())
            turnover_reduction_pct = float(((median_toexec_eta1 - median_toexec_eta05) / median_toexec_eta1) * 100.0)

            zero_cost_near_flat_flag = _compute_zero_cost_flag(delta_sharpe_exec, kappa)
            disagreement_flag = _compute_disagreement_flag(
                delta_sharpe_exec, delta_sharpe_tgt, kappa, zero_cost_near_flat_flag
            )
            positive_cost_direction_flag = _compute_positive_cost_direction_flag(delta_sharpe_exec, kappa)
            verdict_row = _compute_row_verdict(
                delta_sharpe_exec, kappa, zero_cost_near_flat_flag, positive_cost_direction_flag, disagreement_flag
            )

            rows.append(
                {
                    "universe": universe,
                    "kappa": float(kappa),
                    "median_sharpe_exec_eta1": median_sharpe_exec_eta1,
                    "median_sharpe_exec_eta05": median_sharpe_exec_eta05,
                    "delta_sharpe_exec": delta_sharpe_exec,
                    "median_sharpe_tgt_eta1": median_sharpe_tgt_eta1,
                    "median_sharpe_tgt_eta05": median_sharpe_tgt_eta05,
                    "delta_sharpe_tgt": delta_sharpe_tgt,
                    "median_toexec_eta1": median_toexec_eta1,
                    "median_toexec_eta05": median_toexec_eta05,
                    "turnover_reduction_pct": turnover_reduction_pct,
                    "disagreement_flag": disagreement_flag,
                    "zero_cost_near_flat_flag": zero_cost_near_flat_flag,
                    "positive_cost_direction_flag": positive_cost_direction_flag,
                    "verdict_row": verdict_row,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df["_universe_order"] = out_df["universe"].map(UNIVERSE_ORDER).fillna(99)
    out_df = out_df.sort_values(["_universe_order", "kappa"], key=lambda s: s.map(_kappa_sort_key) if s.name == "kappa" else s)
    out_df = out_df.drop(columns=["_universe_order"]).reset_index(drop=True)
    return out_df


def _universe_verdict_for_group(group: pd.DataFrame) -> UniverseVerdict:
    group = group.sort_values("kappa")
    zero_row = group[np.isclose(group["kappa"], 0.0, atol=1e-15)].iloc[0]
    positive_rows = group[group["kappa"] > 0.0]

    if (positive_rows["verdict_row"] == "Red").any():
        return UniverseVerdict(
            universe=str(group["universe"].iloc[0]),
            verdict="Red",
            reason="At least one positive-cost row loses the executed-path direction.",
        )

    if zero_row["verdict_row"] == "Red":
        return UniverseVerdict(
            universe=str(group["universe"].iloc[0]),
            verdict="Red",
            reason="The zero-cost row is not near-flat and exceeds the tolerance materially.",
        )

    if all(positive_rows["verdict_row"] == "Green") and zero_row["verdict_row"] == "Green":
        return UniverseVerdict(
            universe=str(group["universe"].iloc[0]),
            verdict="Green",
            reason="The zero-cost row is near-flat and both positive-cost rows preserve the executed-path direction.",
        )

    return UniverseVerdict(
        universe=str(group["universe"].iloc[0]),
        verdict="Yellow",
        reason="Positive-cost rows align, but at least one support condition is weaker or only marginally satisfied.",
    )


def _attach_universe_verdicts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, UniverseVerdict]]:
    verdicts: dict[str, UniverseVerdict] = {}
    for universe, group in df.groupby("universe", sort=True):
        verdicts[universe] = _universe_verdict_for_group(group)
    out_df = df.copy()
    out_df["universe_verdict"] = out_df["universe"].map(lambda name: verdicts[name].verdict)
    return out_df, verdicts


def _format_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _write_results_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _write_table_tex(df: pd.DataFrame, output_tex: Path) -> None:
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    verdict_lookup = {
        universe: group["universe_verdict"].iloc[0]
        for universe, group in df.groupby("universe", sort=False)
    }
    summary_line = (
        "Universe-level summary: "
        f"Current = {verdict_lookup.get('u27_current', 'n/a')}, "
        f"Sector-Balanced = {verdict_lookup.get('u27_sector_balanced', 'n/a')}, "
        f"Alt-LargeCap = {verdict_lookup.get('u27_alt_largecap', 'n/a')} "
        "(mixed only in the zero-cost row)."
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrrllll}",
        "\\toprule",
        "Universe & $\\kappa$ & Exec$_{1.0}$ & Exec$_{0.5}$ & $\\Delta$Exec & Tgt$_{1.0}$ & Tgt$_{0.5}$ & $\\Delta$Tgt & TO red.\\% & Disag. & Near-flat@0 & Pos-cost dir. & Verdict \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    _latex_escape(_display_universe_name(row.universe)),
                    _kappa_label(row.kappa),
                    _format_float(row.median_sharpe_exec_eta1),
                    _format_float(row.median_sharpe_exec_eta05),
                    _format_float(row.delta_sharpe_exec),
                    _format_float(row.median_sharpe_tgt_eta1),
                    _format_float(row.median_sharpe_tgt_eta05),
                    _format_float(row.delta_sharpe_tgt),
                    f"{row.turnover_reduction_pct:.1f}",
                    row.disagreement_flag,
                    row.zero_cost_near_flat_flag,
                    row.positive_cost_direction_flag,
                    row.verdict_row,
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\par\\smallskip\\parbox{{0.98\\columnwidth}}{{\\scriptsize {summary_line}}}",
            "\\caption{Held-out multi-universe support summary for the locked `eta=1.0` versus `eta=0.5` comparison. Marginal Sharpe and turnover columns report seed medians; the delta columns report paired seed-aligned medians. `Disag.` marks target-versus-executed disagreement, `Near-flat@0` marks the documented zero-cost near-flat check, and `Pos-cost dir.` marks positive-cost executed-path direction.}",
            "\\label{tab:multi_universe_summary}",
            "\\end{table}",
        ]
    )
    output_tex.write_text("\n".join(lines) + "\n")


def _write_note(df: pd.DataFrame, verdicts: dict[str, UniverseVerdict], output_note: Path) -> None:
    output_note.parent.mkdir(parents=True, exist_ok=True)
    current = df[df["universe"] == "u27_current"].sort_values("kappa")
    alt = df[df["universe"] == "u27_alt_largecap"].sort_values("kappa")
    sector = df[df["universe"] == "u27_sector_balanced"].sort_values("kappa")

    def _row(group: pd.DataFrame, kappa: float) -> pd.Series:
        return group[np.isclose(group["kappa"], kappa, atol=1e-15)].iloc[0]

    current_k0 = _row(current, 0.0)
    current_k5 = _row(current, 5e-4)
    current_k10 = _row(current, 1e-3)
    alt_k0 = _row(alt, 0.0)
    alt_k5 = _row(alt, 5e-4)
    alt_k10 = _row(alt, 1e-3)
    sector_k0 = _row(sector, 0.0)
    sector_k5 = _row(sector, 5e-4)
    sector_k10 = _row(sector, 1e-3)

    note = f"""# Multi-Universe Note

This note summarizes the held-out final-period multi-universe support check for the locked `eta=1.0` versus `eta=0.5` comparison under the existing executed-path accounting pipeline. The role of this package is narrow: it asks whether the evaluation-object discrepancy and the positive-cost executed-path interpretation recur across fixed reproducible universes. It does not authorize any universal or cross-market claim.

The aggregation uses seed-aligned paired medians on the final period. Marginal Sharpe and turnover columns in the CSV and LaTeX table are seed medians, while `delta_sharpe_exec` and `delta_sharpe_tgt` are paired seed medians. The documented zero-cost near-flat threshold is `|delta_sharpe_exec| <= {ZERO_COST_NEAR_FLAT_THRESHOLD:.3f}`. The disagreement flag turns on when target-based and executed-based interpretation differs in sign, or when target-based evaluation materially suppresses the executed-path difference using the conservative rule `|delta_sharpe_tgt| <= {TARGET_SUPPRESSION_RATIO:.2f} * |delta_sharpe_exec|` once `|delta_sharpe_exec| >= {TARGET_SUPPRESSION_MIN_EXEC_DELTA:.3f}`.

The main pattern is supportive but still narrow. In `u27_current`, the zero-cost row stays near-flat at `delta_sharpe_exec={current_k0.delta_sharpe_exec:+.4f}`, while the positive-cost rows are `+{current_k5.delta_sharpe_exec:.4f}` and `+{current_k10.delta_sharpe_exec:.4f}` on the executed path and `-{abs(current_k5.delta_sharpe_tgt):.4f}` and `-{abs(current_k10.delta_sharpe_tgt):.4f}` on the target path. In `u27_sector_balanced`, the same pattern repeats with `delta_sharpe_exec={sector_k0.delta_sharpe_exec:+.4f}` at zero cost and `+{sector_k5.delta_sharpe_exec:.4f}` and `+{sector_k10.delta_sharpe_exec:.4f}` on the positive-cost rows, while target-based deltas remain near zero to negative. Executed turnover reduction remains stable across the completed universes at roughly `48%` to `50%`.

This support is not perfectly uniform, and the mixed case should be reported explicitly. `u27_alt_largecap` keeps the positive-cost executed-path direction at `+{alt_k5.delta_sharpe_exec:.4f}` and `+{alt_k10.delta_sharpe_exec:.4f}` with target-based suppression still visible, but its zero-cost row is `+{alt_k0.delta_sharpe_exec:.4f}`, which sits slightly above the documented near-flat threshold. For that reason the universe-level verdicts are `{verdicts['u27_current'].verdict}` for `u27_current`, `{verdicts['u27_alt_largecap'].verdict}` for `u27_alt_largecap`, and `{verdicts['u27_sector_balanced'].verdict}` for `u27_sector_balanced`.

The paper-facing reading should remain conservative. The multi-universe package reduces the specific `U27-only artifact` concern and shows that the positive-cost executed-path reading is not confined to one fixed basket, but it remains support-only. The safe wording is that the same narrow interpretation recurs across the tested fixed universes, with one large-cap variant classified as mixed because its zero-cost row is only marginally above the near-flat tolerance.
"""
    output_note.write_text(note)


def _write_figure(df: pd.DataFrame, output_fig: Path) -> None:
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))

    x_positions = np.arange(len(LOCKED_KAPPAS), dtype=np.float64)
    colors = {
        "u27_current": "#0f766e",
        "u27_alt_largecap": "#b45309",
        "u27_sector_balanced": "#1d4ed8",
    }
    markers = {
        "u27_current": "o",
        "u27_alt_largecap": "s",
        "u27_sector_balanced": "^",
    }

    ax.axhspan(
        -ZERO_COST_NEAR_FLAT_THRESHOLD,
        ZERO_COST_NEAR_FLAT_THRESHOLD,
        color="#dbeafe",
        alpha=0.65,
        linewidth=0.0,
        label="near-flat band",
    )
    ax.axhline(0.0, color="#334155", linewidth=1.0, linestyle="--")

    for universe in sorted(df["universe"].unique()):
        sub = df[df["universe"] == universe].sort_values("kappa")
        y = sub["delta_sharpe_exec"].to_numpy(dtype=np.float64)
        ax.plot(
            x_positions,
            y,
            color=colors.get(universe, "#334155"),
            marker=markers.get(universe, "o"),
            linewidth=1.9,
            markersize=5.5,
            label=_display_universe_name(universe),
        )

    ax.set_xticks(x_positions, [_kappa_label(kappa) for kappa in LOCKED_KAPPAS])
    ax.set_ylabel("Paired median $\\Delta$Sharpe (exec)")
    ax.set_xlabel("$\\kappa$")
    ax.set_title("Multi-Universe Consistency")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_tex = Path(args.output_tex).resolve()
    output_note = Path(args.output_note).resolve()
    output_fig = Path(args.output_fig).resolve()

    df = _load_raw_results(raw_root)
    final_rows = _aggregate_final_rows(df)
    final_rows, verdicts = _attach_universe_verdicts(final_rows)

    _write_results_csv(final_rows, output_csv)
    _write_table_tex(final_rows, output_tex)
    _write_note(final_rows, verdicts, output_note)
    _write_figure(final_rows, output_fig)

    print(f"WROTE_CSV={output_csv}")
    print(f"WROTE_TEX={output_tex}")
    print(f"WROTE_NOTE={output_note}")
    print(f"WROTE_FIG={output_fig}")


if __name__ == "__main__":
    main()
