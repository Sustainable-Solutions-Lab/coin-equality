"""
Production Figure 4.4: Violin plots.

4x2 grid:
  Columns: all, rho, beta, eta
  Row 1: Optimal Carbon Price in 2030 ($/tCO2)
  Row 2: Temperature Change in 2100 (deg C)

Each panel has 3 violins (Flat tax, Progressive tax, Highly progressive tax).

Usage:
    python scripts/plot_prod_fig4_4_violins.py
    python scripts/plot_prod_fig4_4_violins.py --interactive
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prod_fig4_data import load_prod_fig4_data
from fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS, TAX_EQUITY_COLORS, SWEEP_TITLES
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'

PANEL_ORDER = ['all', 'rho', 'beta', 'eta']

METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030 ($/tCO\u2082)'),
    ('delta_t_2100', 'Temperature Change in 2100 (\u00b0C)'),
]

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_all_delta', 'flat'): {'x': 0.0, 'y': 3.9, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_all_delta', 'prog'): {'x': 1.0, 'y': 3.7, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_all_delta', 'hprog'): {'x': 1.9, 'y': 2.8, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
}
LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
}
LABEL_COLORS = {
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
}
MARKERS = {

}
ARROWS = {

}


def main():
    by_sweep_tax = load_prod_fig4_data()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, sweep in enumerate(PANEL_ORDER):
        for row, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[row, col]

            violin_data = []
            positions = []
            colors = []

            for i, te_val in enumerate(TAX_EQUITY_ORDER):
                runs = by_sweep_tax.get((sweep, te_val), [])
                vals = np.array([float(r[metric_key]) for r in runs])
                violin_data.append(vals)
                positions.append(i)
                colors.append(TAX_EQUITY_COLORS[te_val])

            parts = ax.violinplot(violin_data, positions=positions,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)
                pc.set_alpha(0.7)

            bp = ax.boxplot(violin_data, positions=positions, widths=0.15,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color='white', linewidth=1.5),
                            boxprops=dict(linewidth=0.8),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.9)

            ax.set_xticks([])
            ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(metric_label, fontsize=10)

            if row == 0:
                ax.set_title(SWEEP_TITLES[sweep], fontsize=10, weight='bold')

            ax.grid(axis='y', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)

            if row == 0:
                ax.set_ylim(0, 500)
            else:
                ax.set_ylim(1, 4)

    # Place labels
    panel_labels = {}
    panel_markers = {}
    panel_arrows = {}
    for col, sweep in enumerate(PANEL_ORDER):
        for row, (metric_key, _) in enumerate(METRICS):
            pname = f'panel_{sweep}_{metric_key.split("_")[0]}'
            ax = axes[row, col]
            lbl = place_labels(ax, pname, LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
            mkr = place_markers(ax, pname, MARKERS)
            arr = place_arrows(ax, pname, ARROWS)
            for k, v in lbl.items():
                panel_labels[(pname, k)] = v
            for k, v in mkr.items():
                panel_markers[(pname, k)] = v
            for k, v in arr.items():
                panel_arrows[(pname, k)] = v

    plt.tight_layout()

    if '--interactive' in sys.argv:
        panel_axes = {}
        for col, sweep in enumerate(PANEL_ORDER):
            for row, (metric_key, _) in enumerate(METRICS):
                pname = f'panel_{sweep}_{metric_key.split("_")[0]}'
                panel_axes[pname] = axes[row, col]
        enable_interactive(fig, panel_axes, panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / 'prod_fig4_4_violins.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
