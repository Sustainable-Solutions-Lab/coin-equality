"""
Figure 5.9: Policy Tradeoff Scatter Frontier.

Single panel: X = OCP@2030, Y = delta_T@2100.
Each point = one simulation run. Color = tax policy.
Columns of sub-panels by sweep type. Confidence ellipses overlay.

Usage:
    python scripts/plot_fig5_9_frontier.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR_061 = PROJECT_ROOT / 'data' / 'output' / 'config_061'
OUTPUT_DIR_058 = PROJECT_ROOT / 'data' / 'output' / 'config_058'
GRID_CSV = PROJECT_ROOT / 'json' / 'config_061_grid.csv'
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'figures-new'

TAX_EQUITY_ORDER = [0.0, 0.5, 0.998]
TAX_EQUITY_LABELS = {
    0.0: 'Flat tax',
    0.5: 'Progressive tax',
    0.998: 'Highly progressive tax',
}
TAX_EQUITY_COLORS = {
    0.0: '#5a7d9a',
    0.5: '#c0392b',
    0.998: '#2980b9',
}

SWEEP_ORDER = ['all', 'rho', 'beta', 'eta']
SWEEP_TITLES = {
    'all': 'All parameters\n(combined)',
    'rho': 'Effective rate of pure\ntime preference (\u03c1)',
    'beta': 'Damage distribution\nexponent (\u03b2)',
    'eta': 'Elasticity of\nmarginal utility (\u03b7)',
}


def load_results_061():
    results = {}
    for entry in sorted(OUTPUT_DIR_061.iterdir()):
        status = entry / 'run_status.csv'
        if not status.exists():
            continue
        with open(status, 'r') as f:
            for row in csv.DictReader(f):
                if row['success'] == 'True':
                    results[int(row['grid_id'])] = row
    return results


def load_results_058():
    results = {}
    for entry in sorted(OUTPUT_DIR_058.iterdir()):
        status = entry / 'run_status.csv'
        if not status.exists():
            continue
        with open(status, 'r') as f:
            for row in csv.DictReader(f):
                if row['success'] == 'True':
                    results[int(row['grid_id'])] = row
    return results


def load_grid():
    grid = {}
    with open(GRID_CSV, 'r') as f:
        for row in csv.DictReader(f):
            grid[int(row['grid_id'])] = row
    return grid


def confidence_ellipse(x, y, ax, n_std, color, linestyle='-'):
    """Draw a confidence ellipse at n_std standard deviations."""
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                       width=width, height=height, angle=angle,
                       facecolor='none', edgecolor=color,
                       linewidth=1.0, linestyle=linestyle, alpha=0.7)
    ax.add_patch(ellipse)


def main():
    results_061 = load_results_061()
    results_058 = load_results_058()
    grid = load_grid()
    print(f'Loaded {len(results_061)} config_061 runs, {len(results_058)} config_058 runs')

    by_sweep_tax = defaultdict(list)
    for gid, r in results_061.items():
        g = grid[gid]
        sweep = g['sweep']
        tax = float(g['tax_equity'])
        by_sweep_tax[(sweep, tax)].append(r)

    for r in results_058.values():
        tax = float(r['tax_equity'])
        by_sweep_tax[('all', tax)].append(r)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for col, sweep in enumerate(SWEEP_ORDER):
        ax = axes[col]

        for te_val in TAX_EQUITY_ORDER:
            runs = by_sweep_tax.get((sweep, te_val), [])
            mac = np.array([float(r['mac_2030']) for r in runs])
            dt = np.array([float(r['delta_t_2100']) for r in runs])
            color = TAX_EQUITY_COLORS[te_val]

            ax.scatter(mac, dt, c=color, s=4, alpha=0.15,
                       edgecolors='none', rasterized=True, zorder=1)

            # 90% confidence ellipse
            confidence_ellipse(mac, dt, ax, n_std=1.645, color=color,
                               linestyle='-')
            # 50% confidence ellipse
            confidence_ellipse(mac, dt, ax, n_std=0.674, color=color,
                               linestyle='--')

            # Median cross
            ax.scatter([np.median(mac)], [np.median(dt)],
                       c='white', edgecolors=color, s=40,
                       linewidths=1.2, zorder=5, marker='o')

        ax.set_title(SWEEP_TITLES[sweep], fontsize=9, weight='bold', pad=8)
        ax.set_xlabel('Optimal Carbon Price\nin 2030 ($/tCO\u2082)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Temperature Change in 2100 (\u00b0C)', fontsize=9)

        ax.set_xlim(0, 500)
        ax.set_ylim(1, 4)

        ax.xaxis.grid(True, alpha=0.12, linewidth=0.3, color='#999999')
        ax.yaxis.grid(True, alpha=0.12, linewidth=0.3, color='#999999')
        ax.set_axisbelow(True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(width=0.4, labelsize=8, colors='#333333')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=TAX_EQUITY_COLORS[te], markersize=7,
               label=TAX_EQUITY_LABELS[te])
        for te in TAX_EQUITY_ORDER
    ]
    legend_elements.extend([
        Line2D([0], [0], color='#666666', linewidth=1.0, linestyle='--',
               label='50% confidence'),
        Line2D([0], [0], color='#666666', linewidth=1.0, linestyle='-',
               label='90% confidence'),
    ])
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=2.0, columnspacing=1.8)

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / 'fig5_9_frontier.png'
    pdf_path = FIGURES_DIR / 'fig5_9_frontier.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {png_path}')


if __name__ == '__main__':
    main()
