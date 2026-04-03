#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/workspace/execution-aware-portfolio-rl')
OUT = ROOT / 'submission_figures'
PAPER_OUT = ROOT / 'paper' / 'submission_figures'

PALETTE = {
    'blue': '#1f4e79',
    'orange': '#d17c24',
    'green': '#4c8c4a',
    'purple': '#6f5aa5',
    'red': '#b24c3d',
    'gray': '#666666',
    'lightgray': '#d9dde3',
}
KAPPA_STYLE = {
    0.0: (PALETTE['blue'], r'$\kappa=0$'),
    0.0005: (PALETTE['orange'], r'$\kappa=5\times10^{-4}$'),
    0.001: (PALETTE['green'], r'$\kappa=10^{-3}$'),
}


def set_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 180,
        'savefig.dpi': 180,
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'stix',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'axes.facecolor': 'white',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.2,
        'grid.color': '#cfd5dd',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.55,
        'lines.linewidth': 1.9,
        'lines.markersize': 5.5,
    })


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)


def finish(fig: plt.Figure, name: str) -> None:
    out_path = OUT / name
    paper_path = PAPER_OUT / name
    fig.savefig(out_path, bbox_inches='tight')
    shutil.copy2(out_path, paper_path)
    plt.close(fig)


def build_validation_frontier() -> None:
    df = pd.read_csv(ROOT / 'paper_rebuild_20260324T065755Z' / 'validation_eta' / 'aggregate.csv')
    df['kappa'] = pd.to_numeric(df['kappa'])
    df['eta'] = pd.to_numeric(df['eta'])
    df['median_sharpe'] = pd.to_numeric(df['median_sharpe'])
    df['iqr_sharpe'] = pd.to_numeric(df['iqr_sharpe'])
    df['median_turnover_exec'] = pd.to_numeric(df['median_turnover_exec'])

    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    for kappa, grp in df.groupby('kappa'):
        color, label = KAPPA_STYLE[float(kappa)]
        grp = grp.sort_values('median_turnover_exec', ascending=False)
        ax.errorbar(
            grp['median_turnover_exec'],
            grp['median_sharpe'],
            yerr=grp['iqr_sharpe'] / 2.0,
            color=color,
            marker='o',
            linewidth=1.8,
            capsize=2.4,
            label=label,
            zorder=2,
        )
        sel = grp[np.isclose(grp['eta'], 0.5)]
        ax.scatter(sel['median_turnover_exec'], sel['median_sharpe'], s=95, facecolor='white', edgecolor='#111111', linewidth=1.5, zorder=5)
        ax.scatter(sel['median_turnover_exec'], sel['median_sharpe'], s=45, facecolor=color, edgecolor='none', zorder=6)

    eta1 = df[(np.isclose(df['kappa'], 0.0005)) & (np.isclose(df['eta'], 1.0))].iloc[0]
    eta05 = df[(np.isclose(df['kappa'], 0.0005)) & (np.isclose(df['eta'], 0.5))].iloc[0]
    eta002 = df[(np.isclose(df['kappa'], 0.0005)) & (np.isclose(df['eta'], 0.02))].iloc[0]
    ax.annotate(r'immediate $\eta=1.0$', (eta1['median_turnover_exec'], eta1['median_sharpe']), xytext=(8, -10), textcoords='offset points', fontsize=8.4, color=PALETTE['gray'])
    ax.annotate(r'selected $\eta=0.5$', (eta05['median_turnover_exec'], eta05['median_sharpe']), xytext=(8, -2), textcoords='offset points', fontsize=8.7, color='#111111', fontweight='bold')
    ax.annotate(r'raw best $\eta=0.02$', (eta002['median_turnover_exec'], eta002['median_sharpe']), xytext=(8, 8), textcoords='offset points', fontsize=8.4, color=PALETTE['gray'])

    ax.set_xlabel('Seed-median executed turnover')
    ax.set_ylabel('Validation net Sharpe')
    ax.set_title('Validation Frontier')
    ax.grid(axis='y')
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.14))
    finish(fig, 'fig_frontier_submission.pdf')


def build_rolling_summary() -> None:
    split_data = pd.DataFrame([
        {'split': 'A', 'selected_eta': 1.0, 'delta_5e4': 0.0082, 'delta_1e3': 0.0165, 'turnover_reduction': 0.01936},
        {'split': 'B', 'selected_eta': 1.0, 'delta_5e4': 0.0163, 'delta_1e3': 0.0308, 'turnover_reduction': 0.01879},
        {'split': 'C', 'selected_eta': 0.5, 'delta_5e4': 0.0231, 'delta_1e3': 0.0439, 'turnover_reduction': 0.02141},
    ])
    x = np.arange(len(split_data))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True, gridspec_kw={'width_ratios': [1.0, 1.45]})
    ax = axes[0]
    ax.hlines(x, xmin=0, xmax=split_data['selected_eta'], color=PALETTE['lightgray'], linewidth=2.6)
    ax.scatter(split_data['selected_eta'], x, s=55, color=PALETTE['purple'], zorder=3)
    for idx, row in split_data.iterrows():
        ax.text(row['selected_eta'] + 0.04, idx, rf'$\eta={row["selected_eta"]:g}$', va='center', fontsize=8.5)
    ax.set_yticks(x, split_data['split'])
    ax.set_xlim(-0.02, 1.16)
    ax.set_xlabel('Locked selected $\eta$')
    ax.set_title('Selected-Point Transfer')
    ax.grid(axis='x')

    ax = axes[1]
    ax.plot(x, split_data['delta_5e4'], color=PALETTE['orange'], marker='o', label=r'$\kappa=5\times10^{-4}$')
    ax.plot(x, split_data['delta_1e3'], color=PALETTE['green'], marker='o', label=r'$\kappa=10^{-3}$')
    for idx, row in split_data.iterrows():
        ax.text(idx, min(row['delta_5e4'], row['delta_1e3']) - 0.0042, rf'$\Delta\overline{{TO}}={row["turnover_reduction"]:.5f}$', ha='center', fontsize=7.8, color=PALETTE['gray'])
    ax.set_xticks(x, split_data['split'])
    ax.set_ylabel('Best-interior held-out $\Delta$ Sharpe')
    ax.set_title('Frontier Robustness')
    ax.set_ylim(0.0, 0.05)
    ax.grid(axis='y')
    ax.legend(frameon=False, loc='upper left')
    finish(fig, 'fig_rolling_summary.pdf')


def build_kappa_curve() -> None:
    kappas = np.array([2e-4, 5e-4, 1e-3, 2e-3])
    labels = [r'$2\times10^{-4}$', r'$5\times10^{-4}$', r'$10^{-3}$', r'$2\times10^{-3}$']
    selected = np.array([0.0041, 0.0105, 0.0213, 0.0409])
    best = np.array([0.0113, 0.0230, 0.0424, 0.0761])
    selector_eta = np.array([1.0, 1.0, 0.5, 0.2])
    xpos = np.arange(len(kappas))

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.0), constrained_layout=True, sharex=True)
    ax = axes[0]
    ax.plot(xpos, selected, color=PALETTE['blue'], marker='o', label=r'global selected $\eta=0.5$')
    ax.plot(xpos, best, color=PALETTE['orange'], marker='o', label=r'best interior $\eta<1$')
    ax.set_ylabel('Held-out paired $\Delta$ Sharpe')
    ax.set_title('Dense Friction Grid')
    ax.grid(axis='y')
    ax.legend(frameon=False, loc='upper left')

    ax = axes[1]
    ax.step(xpos, selector_eta, where='mid', color=PALETTE['purple'])
    ax.scatter(xpos, selector_eta, s=45, color=PALETTE['purple'])
    for idx, val in enumerate(selector_eta):
        ax.text(idx, val + 0.06, rf'$\eta={val:g}$', ha='center', fontsize=8.2)
    ax.set_ylabel('Per-$\kappa$ selected $\eta$')
    ax.set_xticks(xpos, labels)
    ax.set_xlabel(r'Friction level $\kappa$')
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y')
    finish(fig, 'fig_kappa_curve_submission.pdf')


def build_aux_frontiers() -> None:
    retrain = pd.read_csv(ROOT / 'prl-dow30' / 'outputs' / 'v2_u27_eta05_retrain_pilot' / 'final_main_vs_baseline' / 'aggregate.csv')
    u36 = pd.read_csv(ROOT / 'prl-dow30' / 'outputs' / 'v2_u36_sector_frozen_pilot' / 'final_eta' / 'aggregate.csv')
    for df in (retrain, u36):
        for col in df.columns:
            if col in {'kappa', 'eta', 'median_sharpe', 'median_turnover_exec'}:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.9), constrained_layout=True)

    ax = axes[0]
    for kappa, grp in retrain.groupby('kappa'):
        color, label = KAPPA_STYLE[float(kappa)]
        grp = grp.sort_values('median_turnover_exec', ascending=False)
        ax.plot(grp['median_turnover_exec'], grp['median_sharpe'], color=color, marker='o', label=label)
    main = retrain[retrain['arm'] == 'main']
    base = retrain[retrain['arm'] == 'baseline']
    ax.scatter(main['median_turnover_exec'], main['median_sharpe'], s=85, facecolor='white', edgecolor='#111111', linewidth=1.4, zorder=4)
    ax.scatter(main['median_turnover_exec'], main['median_sharpe'], s=40, facecolor=PALETTE['blue'], edgecolor='none', zorder=5)
    ax.set_title('Eta-Aligned Retraining')
    ax.set_xlabel('Median executed turnover')
    ax.set_ylabel('Median net Sharpe')
    ax.grid(axis='y')
    ax.legend(frameon=False, loc='lower right')
    first = main[np.isclose(main['kappa'], 0.0005)].iloc[0]
    ax.annotate(r'retrained $\eta=0.5$', (first['median_turnover_exec'], first['median_sharpe']), xytext=(8, 4), textcoords='offset points', fontsize=8.3, fontweight='bold')

    ax = axes[1]
    for kappa, grp in u36.groupby('kappa'):
        color, label = KAPPA_STYLE[float(kappa)]
        grp = grp.sort_values('median_turnover_exec', ascending=False)
        ax.plot(grp['median_turnover_exec'], grp['median_sharpe'], color=color, marker='o', label=label)
        sel = grp[np.isclose(grp['eta'], 0.5)]
        best = grp[np.isclose(grp['eta'], 0.02)]
        ax.scatter(sel['median_turnover_exec'], sel['median_sharpe'], s=85, facecolor='white', edgecolor='#111111', linewidth=1.4, zorder=4)
        ax.scatter(sel['median_turnover_exec'], sel['median_sharpe'], s=40, facecolor=color, edgecolor='none', zorder=5)
        ax.scatter(best['median_turnover_exec'], best['median_sharpe'], s=36, facecolor='none', edgecolor=color, linewidth=1.2, zorder=5)
    sel = u36[(np.isclose(u36['kappa'], 0.0005)) & (np.isclose(u36['eta'], 0.5))].iloc[0]
    best = u36[(np.isclose(u36['kappa'], 0.0005)) & (np.isclose(u36['eta'], 0.02))].iloc[0]
    ax.annotate(r'selected $\eta=0.5$', (sel['median_turnover_exec'], sel['median_sharpe']), xytext=(8, -2), textcoords='offset points', fontsize=8.3, fontweight='bold')
    ax.annotate(r'raw best $\eta=0.02$', (best['median_turnover_exec'], best['median_sharpe']), xytext=(8, 8), textcoords='offset points', fontsize=8.1, color=PALETTE['gray'])
    ax.set_title('U36 Replication')
    ax.set_xlabel('Median executed turnover')
    ax.grid(axis='y')
    finish(fig, 'fig_aux_frontiers_submission.pdf')


def build_seed_dotplot() -> None:
    df = pd.read_csv(ROOT / 'paper_rebuild_20260324T065755Z' / 'paper_pack' / 'stats' / 'selected_eta_seedwise_deltas.csv')
    df['kappa'] = pd.to_numeric(df['kappa'])
    df['seed'] = pd.to_numeric(df['seed'])
    df['delta_sharpe_net_lin'] = pd.to_numeric(df['delta_sharpe_net_lin'])
    kappas = [0.0, 0.0005, 0.001]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 5.0), constrained_layout=True, sharex=True, sharey=True)
    for ax, kappa in zip(axes, kappas):
        color, label = KAPPA_STYLE[float(kappa)]
        sub = df[np.isclose(df['kappa'], kappa)].sort_values('seed').reset_index(drop=True)
        y = sub['seed'].to_numpy()
        x = sub['delta_sharpe_net_lin'].to_numpy()
        ax.axvline(0.0, color=PALETTE['gray'], linestyle=':', linewidth=1.1)
        ax.scatter(x, y, color=color, s=34, zorder=3)
        ax.hlines(np.median(y), xmin=np.median(x), xmax=np.median(x), alpha=0)
        ax.plot([np.median(x), np.median(x)], [y.min()-0.4, y.max()+0.4], color=color, linewidth=1.1, alpha=0.55)
        ax.set_title(label)
        ax.grid(axis='x')
        ax.set_xlabel(r'$\Delta$ Sharpe')
    axes[0].set_ylabel('Seed')
    finish(fig, 'fig_seed_scatter_submission.pdf')


def main() -> None:
    set_style()
    ensure_dirs()
    build_validation_frontier()
    build_rolling_summary()
    build_kappa_curve()
    build_aux_frontiers()
    build_seed_dotplot()


if __name__ == '__main__':
    main()
