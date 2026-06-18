"""
Production Figure 4.3: Box Plots with Jittered Points.

2-panel layout grouped by tax scenario.
Within each tax group, 4 boxes side-by-side for the parameter sweeps.
Individual data points jittered behind each box.

Usage:
    python scripts/plot_prod_fig4_3_boxjitter.py
    python scripts/plot_prod_fig4_3_boxjitter.py --interactive
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prod_fig4_data import load_prod_fig4_data
from fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS, TAX_EQUITY_COLORS
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'

PANEL_ORDER = ['all', 'rho', 'beta', 'eta']
SWEEP_LABELS = {
    'all': 'Combined',
    'rho': r'$\rho$',
    'beta': r'$\beta$',
    'eta': r'$\eta$',
}
SWEEP_HATCHES = {
    'all': '',
    'rho': '///',
    'beta': '\\\\\\',
    'eta': '---',
}

METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030 ($/tCO\u2082)'),
    ('delta_t_2100', 'Temperature Change in 2100 (\u00b0C)'),
]

GROUP_SPACING = 1.6
BOX_SPACING = 0.28
BOX_WIDTH = 0.22

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_l', 'grp_flat'): {'x': 0.0, 'y': -30.0, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'grp_prog'): {'x': 1.6, 'y': -30.0, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'grp_hprog'): {'x': 3.2, 'y': -30.0, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'grp_flat'): {'x': 0.0, 'y': 0.8, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'grp_prog'): {'x': 1.6, 'y': 0.8, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'grp_hprog'): {'x': 3.2, 'y': 0.8, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
}
LABEL_TEXTS = {
    'grp_flat': 'Flat tax',
    'grp_prog': 'Progressive tax',
    'grp_hprog': 'Highly progressive tax',
}
LABEL_COLORS = {
    'grp_flat': '#2C3E50',
    'grp_prog': '#d62728',
    'grp_hprog': '#1f77b4',
}
MARKERS = {

}
ARROWS = {

}


def main():
    by_sweep_tax = load_prod_fig4_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.default_rng(42)
    n_sweeps = len(PANEL_ORDER)

    for panel_idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[panel_idx]

        group_centers = []
        for gi, te_val in enumerate(TAX_EQUITY_ORDER):
            group_center = gi * GROUP_SPACING
            group_centers.append(group_center)

            all_vals = []
            all_positions = []

            for si, sweep in enumerate(PANEL_ORDER):
                runs = by_sweep_tax.get((sweep, te_val), [])
                vals = np.array([float(r[metric_key]) for r in runs])

                if len(vals) < 3:
                    continue

                pos = group_center + (si - (n_sweeps - 1) / 2) * BOX_SPACING
                all_vals.append(vals)
                all_positions.append(pos)

                # Jittered points behind boxes
                jitter = rng.uniform(-BOX_WIDTH * 0.35, BOX_WIDTH * 0.35,
                                     size=len(vals))
                ax.scatter(pos + jitter, vals,
                           c=TAX_EQUITY_COLORS[te_val],
                           s=3, alpha=0.12, edgecolors='none',
                           zorder=1, rasterized=True)

                # Highlight unvalidated outliers in pink
                bad_mask = np.array([r.get('validated', 'True') != 'True' for r in runs])
                if bad_mask.any():
                    bad_jitter = jitter[bad_mask]
                    bad_vals = vals[bad_mask]
                    ax.scatter(pos + bad_jitter, bad_vals,
                               c='#FF69B4', s=20, alpha=0.9,
                               edgecolors='black', linewidths=0.5,
                               zorder=10)

            # Boxplots
            bp = ax.boxplot(all_vals, positions=all_positions, widths=BOX_WIDTH,
                            showfliers=False, patch_artist=True, zorder=2,
                            medianprops=dict(color='white', linewidth=1.5),
                            boxprops=dict(linewidth=0.8),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8))

            for si, patch in enumerate(bp['boxes']):
                patch.set_facecolor(TAX_EQUITY_COLORS[te_val])
                patch.set_alpha(0.7)
                patch.set_hatch(SWEEP_HATCHES[PANEL_ORDER[si]])

        ax.set_title(metric_label, fontsize=10, weight='bold')
        if panel_idx == 0:
            ax.set_ylabel(metric_label, fontsize=10)

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        if panel_idx == 0:
            ax.set_ylim(0, 500)
        else:
            ax.set_ylim(1, 4)

        x_pad = 0.7
        ax.set_xlim(min(group_centers) - x_pad, max(group_centers) + x_pad)

    # Place labels, markers, and arrows
    labels_l = place_labels(axes[0], 'panel_l', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    labels_r = place_labels(axes[1], 'panel_r', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    markers_l = place_markers(axes[0], 'panel_l', MARKERS)
    markers_r = place_markers(axes[1], 'panel_r', MARKERS)
    arrows_l = place_arrows(axes[0], 'panel_l', ARROWS)
    arrows_r = place_arrows(axes[1], 'panel_r', ARROWS)

    plt.tight_layout()

    if '--interactive' in sys.argv:
        panel_labels = {}
        for k, v in labels_l.items():
            panel_labels[('panel_l', k)] = v
        for k, v in labels_r.items():
            panel_labels[('panel_r', k)] = v
        panel_markers = {}
        for k, v in markers_l.items():
            panel_markers[('panel_l', k)] = v
        for k, v in markers_r.items():
            panel_markers[('panel_r', k)] = v
        panel_arrows = {}
        for k, v in arrows_l.items():
            panel_arrows[('panel_l', k)] = v
        for k, v in arrows_r.items():
            panel_arrows[('panel_r', k)] = v
        enable_interactive(fig, {'panel_l': axes[0], 'panel_r': axes[1]},
                           panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / 'prod_fig4_3_boxjitter.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
