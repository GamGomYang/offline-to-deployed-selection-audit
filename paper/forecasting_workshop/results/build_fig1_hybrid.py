#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


WORKSHOP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSHOP_DIR / "results"
FIGURES_DIR = WORKSHOP_DIR / "assets" / "figures"
OUTPUT_PATH = FIGURES_DIR / "fig1_fixed_interface_inversion.pdf"


@dataclass(frozen=True)
class FigureCell:
    domain: str
    friction: str
    forecast_winner: str
    deployed_winner: str
    suboptimal_seeds: str
    color: str
    source_file: str


def _clean_latex_cell(value: str) -> str:
    value = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", value)
    value = value.replace(r"\\", "")
    value = value.replace("$", "")
    return " ".join(value.strip().split())


def _parse_latex_table(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "&" not in line or line.startswith("\\"):
            continue
        line = re.sub(r"\\\\.*$", "", line)
        parts = [_clean_latex_cell(part) for part in line.split("&")]
        if parts and parts[0].lower() not in {"friction", "domain"}:
            rows.append(parts)
    return rows


def _lookup(rows: list[list[str]], key_index: int, key: str) -> list[str]:
    for row in rows:
        if len(row) > key_index and row[key_index] == key:
            return row
    raise KeyError(key)


def _suboptimal_rate(value: str) -> float:
    match = re.search(r"(\d+)\s*/\s*(\d+)", value)
    if not match:
        return 0.0
    numerator, denominator = int(match.group(1)), int(match.group(2))
    return numerator / denominator if denominator else 0.0


def _cell_color(forecast_winner: str, deployed_winner: str, friction: str, suboptimal_seeds: str) -> str:
    if friction == "0.00":
        return "neutralbg"
    rate = _suboptimal_rate(suboptimal_seeds)
    if forecast_winner == deployed_winner or rate < 0.5:
        return "partialbg"
    if rate >= 0.8:
        return "strongbg"
    return "shiftbg"


def build_figure_data() -> list[FigureCell]:
    event_path = RESULTS_DIR / "table_q2_selection_drift_event_micro_main.tex"
    corroboration_path = RESULTS_DIR / "table_q2_corroboration_compact.tex"
    inventory_path = RESULTS_DIR / "table_q2_selection_drift_inventory.tex"

    event_rows = _parse_latex_table(event_path)
    corroboration_rows = _parse_latex_table(corroboration_path)
    inventory_rows = _parse_latex_table(inventory_path)

    cells: list[FigureCell] = []

    for friction in ["0.00", "0.50", "1.00"]:
        row = _lookup(event_rows, 0, friction)
        cells.append(
            FigureCell(
                domain="Event-micro",
                friction=friction,
                forecast_winner=row[1],
                deployed_winner=row[2],
                suboptimal_seeds=row[4],
                color=_cell_color(row[1], row[2], friction, row[4]),
                source_file=str(event_path.relative_to(WORKSHOP_DIR)).replace("\\", "/"),
            )
        )

    for friction in ["0.00", "0.50", "1.00"]:
        row = next(
            r
            for r in corroboration_rows
            if len(r) >= 5 and r[0] == "Traffic-Hourly Top-k" and r[1] == friction
        )
        cells.append(
            FigureCell(
                domain="Traffic-Hourly Top-k",
                friction=friction,
                forecast_winner=row[2],
                deployed_winner=row[3],
                suboptimal_seeds=row[4],
                color=_cell_color(row[2], row[3], friction, row[4]),
                source_file=str(corroboration_path.relative_to(WORKSHOP_DIR)).replace("\\", "/"),
            )
        )

    for friction in ["0.00", "0.50", "1.00"]:
        row = _lookup(inventory_rows, 0, friction)
        cells.append(
            FigureCell(
                domain="Inventory",
                friction=friction,
                forecast_winner=row[1],
                deployed_winner=row[2],
                suboptimal_seeds=row[5],
                color=_cell_color(row[1], row[2], friction, row[5]),
                source_file=str(inventory_path.relative_to(WORKSHOP_DIR)).replace("\\", "/"),
            )
        )

    return cells


def _latex_escape(value: str) -> str:
    return (
        value.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _short_name(value: str) -> str:
    return {
        "Reactive sharp": "Sharp",
        "Calibrated baseline": "Calib.",
        "Lagged smoother": "Smooth",
        "Reactive short": "Short",
        "Small MLP": "MLP",
        "Moving average (7)": "MA(7)",
    }.get(value, value)


def _make_cell_node(cell: FigureCell, x: float, y: float) -> str:
    transition = _short_name(cell.forecast_winner) + r" $\rightarrow$ " + _short_name(cell.deployed_winner)
    body = r"{\bfseries " + _latex_escape(transition) + r"}\\[-1pt]{\scriptsize subopt. " + _latex_escape(cell.suboptimal_seeds) + "}"
    return rf"\node[cell,fill={cell.color}] at ({x:.2f},{y:.2f}) {{{body}}};"


def build_latex(cells: list[FigureCell]) -> str:
    domains = ["Event-micro", "Traffic-Hourly Top-k", "Inventory"]
    frictions = ["0.00", "0.50", "1.00"]
    domain_labels = {
        "Event-micro": r"Event-micro",
        "Traffic-Hourly Top-k": r"Traffic-Hourly\\Top-k",
        "Inventory": r"Inventory",
    }

    x_positions = {"0.00": 5.10, "0.50": 8.42, "1.00": 11.74}
    y_positions = {"Event-micro": -4.74, "Traffic-Hourly Top-k": -5.97, "Inventory": -7.20}
    cell_map = {(cell.domain, cell.friction): cell for cell in cells}

    cell_nodes = []
    for domain in domains:
        for friction in frictions:
            cell_nodes.append(_make_cell_node(cell_map[(domain, friction)], x_positions[friction], y_positions[domain]))

    row_labels = [
        rf"\node[rowlabel] at (2.90,{y_positions[domain]:.2f}) {{{domain_labels[domain]}}};"
        for domain in domains
    ]
    column_labels = []
    for friction in frictions:
        label = rf"\kappa = {friction}^{{*}}" if friction == "0.00" else rf"\kappa = {friction}"
        column_labels.append(rf"\node[colheader] at ({x_positions[friction]:.2f},-4.16) {{${label}$}};")

    return (
        r"""
\documentclass[tikz,border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{xcolor}
\usetikzlibrary{arrows.meta,positioning}
\definecolor{panelbg}{HTML}{F7F8FA}
\definecolor{panelrule}{HTML}{D6DADE}
\definecolor{titlegray}{HTML}{222222}
\definecolor{muted}{HTML}{555A60}
\definecolor{forecastbg}{HTML}{DCEBFF}
\definecolor{forecastline}{HTML}{4E79A7}
\definecolor{interfacebg}{HTML}{E9DDF7}
\definecolor{interfaceline}{HTML}{8E6BBE}
\definecolor{actionbg}{HTML}{DDF1EE}
\definecolor{actionline}{HTML}{59A14F}
\definecolor{metricbg}{HTML}{FDE5C8}
\definecolor{metricline}{HTML}{F28E2B}
\definecolor{neutralbg}{HTML}{E7E7E7}
\definecolor{partialbg}{HTML}{FFF0C2}
\definecolor{shiftbg}{HTML}{FBC878}
\definecolor{strongbg}{HTML}{E76F51}
\definecolor{framegray}{HTML}{8C9298}
\definecolor{footmuted}{HTML}{6F747A}
\begin{document}
\sffamily
\begin{tikzpicture}[
  stage/.style 2 args={draw=#1, rounded corners=3pt, fill=#2, align=center, font=\scriptsize, minimum width=2.23cm, minimum height=0.76cm, inner sep=2.5pt},
  arr/.style={-{Latex[length=2.25mm]}, line width=0.7pt, draw=framegray},
  cell/.style={draw=white, line width=1.0pt, minimum width=2.98cm, minimum height=0.88cm, align=center, font=\scriptsize, text width=2.70cm, inner sep=2pt},
  rowlabel/.style={anchor=east, align=right, font=\bfseries\scriptsize, text width=2.12cm},
  colheader/.style={align=center, font=\bfseries\scriptsize},
  legend/.style={font=\scriptsize, anchor=west}
]
\node[anchor=west,font=\bfseries\large,text=titlegray] at (0.00,0.00) {Offline-selected candidates can lose after fixed-interface deployment.};

\filldraw[fill=panelbg,draw=panelrule,rounded corners=4pt] (0.00,-0.42) rectangle (13.80,-2.36);
\node[anchor=west,font=\bfseries\footnotesize,text=titlegray] at (0.25,-0.68) {A. Fixed-interface selection-transfer audit};
\node[stage={forecastline}{forecastbg}] (forecast) at (2.02,-1.42) {Offline score\\[-1pt]{\fontsize{6.4}{6.8}\selectfont selects candidate}};
\node[stage={interfaceline}{interfacebg}] (interface) at (5.24,-1.42) {Fixed interface\\[-1pt]{\fontsize{6.4}{6.8}\selectfont maps outputs to actions}};
\node[stage={actionline}{actionbg}] (executed) at (8.46,-1.42) {Deployment friction\\[-1pt]{\fontsize{6.4}{6.8}\selectfont applies cost}};
\node[stage={metricline}{metricbg}] (metric) at (11.68,-1.42) {Deployed utility\\[-1pt]{\fontsize{6.4}{6.8}\selectfont ranks candidates}};
\draw[arr] (forecast) -- (interface);
\draw[arr] (interface) -- (executed);
\draw[arr] (executed) -- (metric);
\node[anchor=west,align=left,font=\fontsize{6.4}{6.8}\selectfont,text=muted] at (0.25,-2.13) {candidate rule: $\pi_m = I_\theta \circ f_m$};

\filldraw[fill=panelbg,draw=panelrule,rounded corners=4pt] (0.00,-2.61) rectangle (13.80,-8.83);
\node[anchor=west,font=\bfseries\footnotesize,text=titlegray] at (0.25,-2.91) {B. Winner inversion under fixed-interface deployment};
\node[anchor=north west,align=left,font=\scriptsize,text=muted] at (0.55,-3.14) {Arrows show offline-selected $\rightarrow$ deployed-utility winner\\[1pt]Counts show deployed-suboptimal units};
"""
        + "\n".join(column_labels)
        + "\n"
        + "\n".join(row_labels)
        + "\n"
        + "\n".join(cell_nodes)
        + r"""
\node[legend] at (1.10,-8.18) {\tikz{\fill[neutralbg,draw=framegray] (0,0) rectangle (0.18,0.12);} zero-friction reference \quad
\tikz{\fill[partialbg,draw=framegray] (0,0) rectangle (0.18,0.12);} partial transfer / mixed \quad
\tikz{\fill[shiftbg,draw=framegray] (0,0) rectangle (0.18,0.12);} winner changes \quad
\tikz{\fill[strongbg,draw=framegray] (0,0) rectangle (0.18,0.12);} recurrent suboptimality};
\node[anchor=west,font=\fontsize{6.6}{7.0}\selectfont,text=footmuted] at (1.10,-8.52) {*Gray cells are zero-friction references, not friction-causal evidence.};
\end{tikzpicture}
\end{document}
"""
    ).strip()


def write_data_table(cells: list[FigureCell]) -> None:
    print("Domain | Friction | Offline-selected model | Deployed-utility winner | Suboptimal seeds | Cell color | Source file")
    print("--- | --- | --- | --- | --- | --- | ---")
    color_name = {"neutralbg": "Gray", "partialbg": "Light", "shiftbg": "Amber", "strongbg": "Orange"}
    for cell in cells:
        print(
            " | ".join(
                [
                    cell.domain,
                    cell.friction,
                    cell.forecast_winner,
                    cell.deployed_winner,
                    cell.suboptimal_seeds,
                    color_name[cell.color],
                    cell.source_file,
                ]
            )
        )


def main() -> int:
    cells = build_figure_data()
    latex_source = build_latex(cells)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fig1_hybrid_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tex_path = tmp_dir / "fig1_fixed_interface_inversion.tex"
        pdf_path = tmp_dir / "fig1_fixed_interface_inversion.pdf"
        tex_path.write_text(latex_source, encoding="utf-8")
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(tmp_dir),
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(
                "Failed to build fig1_fixed_interface_inversion.pdf.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        shutil.copyfile(pdf_path, OUTPUT_PATH)

    write_data_table(cells)
    print(f"\nWrote {OUTPUT_PATH.relative_to(WORKSHOP_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
