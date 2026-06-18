"""
Production Figure 4.6: Policy Tradeoff Scatter Frontier.

4 sub-panels by sweep type. X = OCP@2030, Y = delta_T@2100.
Each point = one simulation run. Color = tax policy.
Confidence ellipses overlay.

Usage:
    python scripts/plot_prod_fig4_6_frontier.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prod_fig4_data import load_prod_fig4_data
from fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS, SWEEP_TITLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'

PANEL_ORDER = ['all', 'rho', 'beta', 'eta']

# Muted palette for frontier (matches old-tcre version)
TAX_EQUITY_COLORS = {
    0.0: '#5a7d9a',
    0.5: '#c0392b',
    0.998: '#2980b9',
}


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
    by_sweep_tax = load_prod_fig4_data()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for col, sweep in enumerate(PANEL_ORDER):
        ax = axes[col]

        for te_val in TAX_EQUITY_ORDER:
            runs = by_sweep_tax.get((sweep, te_val), [])
            mac = np.array([float(r['mac_2030']) for r in runs])
            dt = np.array([float(r['delta_t_2100']) for r in runs])
            color = TAX_EQUITY_COLORS[te_val]

            ax.scatter(mac, dt, c=color, s=4, alpha=0.15,
                       edgecolors='none', rasterized=True, zorder=1)

            # Highlight unvalidated outliers in pink
            bad = [r for r in runs if r.get('validated', 'True') != 'True']
            if bad:
                mac_bad = np.array([float(r['mac_2030']) for r in bad])
                dt_bad = np.array([float(r['delta_t_2100']) for r in bad])
                ax.scatter(mac_bad, dt_bad,
                           c='#FF69B4', s=20, alpha=0.9,
                           edgecolors='black', linewidths=0.5,
                           zorder=10)

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
    pdf_path = FIGURES_DIR / 'prod_fig4_6_frontier.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
