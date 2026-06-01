"""
Figure 5.7: Left-oriented half-violin (raincloud style) uncertainty decomposition.

Two panels: OCP@2030 (left), delta_T@2100 (right).
Grouped by tax policy. Within each group: Combined, rho, beta, eta.
Left-side half-violins with small white median dots, thin IQR stems.
Combined violin is wider for visual prominence.

Usage:
    python scripts/plot_fig5_7_raincloud.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from label_utils import place_labels, place_markers, place_arrows, enable_interactive

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
    'rho_flat': 'ρ',
    'beta_flat': 'β',
    'eta_flat': 'η',
    'all_prog': 'Combined',
    'rho_prog': 'ρ',
    'beta_prog': 'β',
    'eta_prog': 'η',
    'all_hprog': 'Combined',
    'rho_hprog': 'ρ',
    'beta_hprog': 'β',
    'eta_hprog': 'η',
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    n_sweeps = len(SWEEP_ORDER)

    for panel_idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[panel_idx]

        group_centers = []
        for gi, te_val in enumerate(TAX_EQUITY_ORDER):
            group_center = gi * GROUP_SPACING

            for si, sweep in enumerate(SWEEP_ORDER):
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
    png_path = FIGURES_DIR / 'fig5_7_raincloud.png'
    pdf_path = FIGURES_DIR / 'fig5_7_raincloud.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {png_path}')


if __name__ == '__main__':
    main()
