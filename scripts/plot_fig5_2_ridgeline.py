"""
Figure 5.2: Ridgeline Density Plots.

Overlapping KDE density curves arranged vertically by sweep type.
Two panels: OCP@2030 (left), delta_T@2100 (right).
Each sweep row has 3 overlapping densities (one per tax_equity).

Usage:
    python scripts/plot_fig5_2_ridgeline.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    0.0: '#2C3E50',
    0.5: '#d62728',
    0.998: '#1f77b4',
}

SWEEP_ORDER = ['all', 'rho', 'beta', 'eta']
SWEEP_TITLES = {
    'eta': 'Elasticity of marginal utility (η)',
    'rho': 'Effective rate of pure time preference (ρ)',
    'beta': 'Damage distribution exponent (β)',
    'all': 'All parameters (combined)',
}

METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030 ($/tCO₂)', 0, 500),
    ('delta_t_2100', 'Temperature Change in 2100 (°C)', 1, 4),
]

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_l', 'txt_2'): {'x': 278.8, 'y': 3.7, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_l', 'txt_3'): {'x': 91.2, 'y': 4.0, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_l', 'txt_6'): {'x': 8.4, 'y': 4.0, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_l', 'med_l_flat'): {'x': 75.0, 'y': 3.9, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'med_l_prog'): {'x': 132.2, 'y': 3.6, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_l', 'med_l_hprog'): {'x': 239.0, 'y': 3.9, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'flat'): {'x': 3.1, 'y': 4.0, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_r', 'prog'): {'x': 2.2, 'y': 4.0, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_r', 'hprog'): {'x': 1.0, 'y': 3.7, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_r', 'med_r_flat'): {'x': 3.1, 'y': 3.8, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'med_r_prog'): {'x': 2.7, 'y': 3.7, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
    ('panel_r', 'med_r_hprog'): {'x': 1.9, 'y': 3.8, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'bottom'},
}
LABEL_TEXTS = {
    'txt_2': 'Highly progressive tax',
    'txt_3': 'Progressive tax',
    'txt_6': 'Flat tax',
    'med_l_flat': '75',
    'med_l_prog': '121',
    'med_l_hprog': '239',
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'med_r_flat': '3.1',
    'med_r_prog': '2.7',
    'med_r_hprog': '1.9',
}
LABEL_COLORS = {
    'txt_2': '#1f77b4',
    'txt_3': '#d62728',
    'txt_6': '#2C3E50',
    'med_l_flat': '#000000',
    'med_l_prog': '#000000',
    'med_l_hprog': '#000000',
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'med_r_flat': '#000000',
    'med_r_prog': '#000000',
    'med_r_hprog': '#000000',
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    row_spacing = 1.0
    kde_height = 0.8

    for panel_idx, (metric_key, metric_label, xmin, xmax) in enumerate(METRICS):
        ax = axes[panel_idx]
        x_grid = np.linspace(xmin, xmax, 500)

        for row_idx, sweep in enumerate(reversed(SWEEP_ORDER)):
            y_base = row_idx * row_spacing

            for te_val in TAX_EQUITY_ORDER:
                runs = by_sweep_tax.get((sweep, te_val), [])
                vals = np.array([float(r[metric_key]) for r in runs])

                if len(vals) < 3:
                    continue

                kde = gaussian_kde(vals, bw_method=0.15)
                density = kde(x_grid)
                density_scaled = density / density.max() * kde_height

                ax.fill_between(x_grid, y_base, y_base + density_scaled,
                                color=TAX_EQUITY_COLORS[te_val],
                                alpha=0.4, linewidth=0)
                ax.plot(x_grid, y_base + density_scaled,
                        color=TAX_EQUITY_COLORS[te_val],
                        linewidth=1.2)

                median_val = np.median(vals)
                median_density = kde(np.array([median_val]))[0]
                median_height = median_density / density.max() * kde_height
                ax.vlines(median_val, y_base, y_base + median_height,
                          color=TAX_EQUITY_COLORS[te_val],
                          linewidth=1.5, linestyle='--', alpha=0.8)

            ax.axhline(y_base, color='grey', linewidth=0.3, alpha=0.5)

        sweep_labels = list(reversed(SWEEP_ORDER))
        ax.set_yticks([i * row_spacing for i in range(len(SWEEP_ORDER))])
        ax.set_yticklabels([SWEEP_TITLES[s] for s in sweep_labels], fontsize=9)
        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.2, len(SWEEP_ORDER) * row_spacing + 0.1)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Place labels, markers, and arrows (replace legend)
    labels_l = place_labels(axes[0], 'panel_l', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    labels_r = place_labels(axes[1], 'panel_r', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    markers_l = place_markers(axes[0], 'panel_l', MARKERS)
    markers_r = place_markers(axes[1], 'panel_r', MARKERS)
    arrows_l = place_arrows(axes[0], 'panel_l', ARROWS)
    arrows_r = place_arrows(axes[1], 'panel_r', ARROWS)

    plt.tight_layout()

    # Interactive mode
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
    png_path = FIGURES_DIR / 'fig5_2_ridgeline.png'
    pdf_path = FIGURES_DIR / 'fig5_2_ridgeline.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {png_path}')


if __name__ == '__main__':
    main()
