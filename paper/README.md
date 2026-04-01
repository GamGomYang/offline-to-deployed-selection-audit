# Overleaf Paper Package

This directory is a self-contained package for compiling the current manuscript on Overleaf.

Contents:
- `paper.tex`: manuscript source
- `fig_frontier.png`
- `fig_misalignment.png`
- `fig_seed_scatter.png`
- `fig_rolling_frontier_robustness.png`
- `fig_kappa_benefit_curve.png`

Notes:
- The bibliography is embedded directly inside `paper.tex`, so no `.bib` file is required.
- The manuscript uses standard packages (`geometry`, `amsmath`, `graphicx`, `booktabs`, `subcaption`, `hyperref`, `authblk`, etc.) that Overleaf should provide.
- The document is written for `pdflatex`-style compilation.

Recommended Overleaf settings:
- Compiler: `pdfLaTeX`
- Main file: `paper.tex`
