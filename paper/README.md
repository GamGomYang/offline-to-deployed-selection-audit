# Paper Package

This directory is the canonical manuscript package for the current paper revision.

Contents:
- `paper.tex`: manuscript source
- `paper.pdf`: compiled submission PDF
- `submission_figures/`: print-ready PDF figures used by the manuscript
- `fig_misalignment.png`: local raster figure used in the appendix
- `legacy_figures/`: older exported PNG aliases kept for reproducibility and reference

Notes:
- The bibliography is embedded directly inside `paper.tex`; no separate `.bib` file is required.
- The manuscript is self-contained and designed for `pdflatex`-style compilation.

Recommended Overleaf settings:
- Compiler: `pdfLaTeX`
- Main file: `paper.tex`
