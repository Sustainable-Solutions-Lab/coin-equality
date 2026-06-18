"""
Production Figure 4.5: Left-oriented half-violin (raincloud style) uncertainty decomposition.

Two panels: OCP@2030 (left), delta_T@2100 (right).
Grouped by tax policy. Within each group: Combined, rho, beta, eta.
Left-side half-violins with small white median dots, thin IQR stems.
Combined violin is wider for visual prominence.

Usage:
    python scripts/plot_prod_fig4_5_raincloud.py
    python scripts/plot_prod_fig4_5_raincloud.py --interactive
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prod_fig4_data import load_prod_fig4_data
from fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'

PANEL_ORDER = ['all', 'rho', 'beta', 'eta']

# Muted palette for raincloud (matches old-tcre version)
TAX_EQUITY_COLORS = {
    0.0: '#5a7d9a',
    0.5: '#c0392b',
    0.998: '#2980b9',
}

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
SWEEP_WIDTH = {
    'all': 0.26,
    'rho': 0.18,
    'beta': 0.18,
    'eta': 0.18,
}
SWEEP_ALPHA_FILL = {
    'all': 0.50,
    'rho': 0.40,
    'beta': 0.40,
    'eta': 0.40,
}
SWEEP_EDGE_LW = {
    'all': 0.6,
    'rho': 0.4,
    'beta': 0.4,
    'eta': 0.4,
}

METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030 ($/tCO\u2082)'),
    ('delta_t_2100', 'Temperature Change in 2100 (\u00b0C)'),
]

GROUP_SPACING = 1.8
VIOLIN_SPACING = 0.27

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_l', 'all_flat'): {'x': -0.5, 'y': 39.6, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'rho_flat'): {'x': -0.2, 'y': 69.8, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'beta_flat'): {'x': 0.0, 'y': 70.5, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'eta_flat'): {'x': 0.3, 'y': 71.8, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'all_prog'): {'x': 1.3, 'y': 96.3, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'rho_prog'): {'x': 1.6, 'y': 117.1, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'beta_prog'): {'x': 1.8, 'y': 117.8, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'eta_prog'): {'x': 2.1, 'y': 117.1, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'all_hprog'): {'x': 3.1, 'y': 204.3, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'rho_hprog'): {'x': 3.4, 'y': 234.4, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'beta_hprog'): {'x': 3.6, 'y': 235.1, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'eta_hprog'): {'x': 3.9, 'y': 235.8, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'all_flat'): {'x': -0.5, 'y': 2.8, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'rho_flat'): {'x': -0.2, 'y': 3.1, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'beta_flat'): {'x': 0.1, 'y': 3.0, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'eta_flat'): {'x': 0.3, 'y': 3.1, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'all_prog'): {'x': 1.3, 'y': 2.4, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'rho_prog'): {'x': 1.6, 'y': 2.7, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'beta_prog'): {'x': 1.9, 'y': 2.7, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'eta_prog'): {'x': 2.1, 'y': 2.7, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'all_hprog'): {'x': 3.1, 'y': 1.7, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'rho_hprog'): {'x': 3.4, 'y': 1.9, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'beta_hprog'): {'x': 3.6, 'y': 1.9, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'eta_hprog'): {'x': 3.9, 'y': 1.9, 'fontsize': 7, 'rotation': 90, 'ha': 'center', 'va': 'bottom'},
}
LABEL_TEXTS = {
    'all_flat': 'Combined',
    'rho_flat': '\u03c1',
    'beta_flat': '\u03b2',
    'eta_flat': '\u03b7',
    'all_prog': 'Combined',
    'rho_prog': '\u03c1',
    'beta_prog': '\u03b2',
    'eta_prog': '\u03b7',
    'all_hprog': 'Combined',
    'rho_hprog': '\u03c1',
    'beta_hprog': '\u03b2',
    'eta_hprog': '\u03b7',
}
LABEL_COLORS = {
    'all_flat': '#000000',
    'rho_flat': '#000000',
    'beta_flat': '#000000',
    'eta_flat': '#000000',
    'all_prog': '#000000',
    'rho_prog': '#000000',
    'beta_prog': '#000000',
    'eta_prog': '#000000',
    'all_hprog': '#000000',
    'rho_hprog': '#000000',
    'beta_hprog': '#000000',
    'eta_hprog': '#000000',
}
MARKERS = {

}
ARROWS = {

}


def draw_left_half_violin(ax, vals, center, max_width, color, hatch,
                           alpha_fill, edge_lw):
    """Draw a left-side half-violin with KDE, white median dot, thin IQR stem."""
    kde = gaussian_kde(vals, bw_method=0.2)
    y_grid = np.linspace(vals.min(), vals.max(), 300)
    density = kde(y_grid)
    density = density / density.max() * max_width

    # Left-side: density extends to the left of center
    x_coords = np.concatenate([[center], center - density, [center]])
    y_coords = np.concatenate([[y_grid[0]], y_grid, [y_grid[-1]]])

    verts = np.column_stack([x_coords, y_coords])

    # Fill polygon
    poly = plt.Polygon(verts, closed=True,
                        facecolor=color, edgecolor='#444444',
                        linewidth=edge_lw, alpha=alpha_fill, zorder=2)
    ax.add_patch(poly)

    # Subtle hatch overlay
    if hatch:
        poly_h = plt.Polygon(verts, closed=True,
                              facecolor='none', edgecolor='#666666',
                              linewidth=0.0, hatch=hatch, alpha=0.35, zorder=3)
        ax.add_patch(poly_h)

    # Statistics
    median = np.median(vals)
    q25, q75 = np.percentile(vals, [25, 75])

    # Thin IQR stem — right of center (clean side)
    stem_offset = max_width * 0.08
    ax.plot([center + stem_offset, center + stem_offset], [q25, q75],
            color='#333333', linewidth=0.8, solid_capstyle='butt', zorder=5)

    # Small white median dot
    ax.scatter([center + stem_offset], [median],
               c='white', edgecolors='#222222', s=18, linewidths=0.7,
               zorder=6)


def main():
    by_sweep_tax = load_prod_fig4_data()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    n_sweeps = len(PANEL_ORDER)

    for panel_idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[panel_idx]

        group_centers = []
        for gi, te_val in enumerate(TAX_EQUITY_ORDER):
            group_center = gi * GROUP_SPACING

            for si, sweep in enumerate(PANEL_ORDER):
                runs = by_sweep_tax.get((sweep, te_val), [])
                vals = np.array([float(r[metric_key]) for r in runs])

                if len(vals) < 3:
                    continue

                pos = group_center + (si - (n_sweeps - 1) / 2) * VIOLIN_SPACING
                draw_left_half_violin(ax, vals, pos,
                                      SWEEP_WIDTH[sweep],
                                      TAX_EQUITY_COLORS[te_val],
                                      SWEEP_HATCHES[sweep],
                                      SWEEP_ALPHA_FILL[sweep],
                                      SWEEP_EDGE_LW[sweep])

            group_centers.append(group_center)

        ax.set_title(metric_label, fontsize=10, weight='bold')
        if panel_idx == 0:
            ax.set_ylabel(metric_label, fontsize=10)

        ax.set_xticks([])
        ax.set_xticklabels([])

        # Subtle gridlines
        ax.yaxis.grid(True, alpha=0.15, linewidth=0.4, color='#999999', zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(width=0.4, labelsize=9, colors='#333333')

        if panel_idx == 0:
            ax.set_ylim(0, 500)
        else:
            ax.set_ylim(1, 4)

        x_pad = 0.8
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
    pdf_path = FIGURES_DIR / 'prod_fig4_5_raincloud.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {pdf_path}')


if __name__ == '__main__':
    main()
