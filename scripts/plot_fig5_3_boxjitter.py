"""
Figure 5.3: Box Plots with Jittered Points.

2-panel layout grouped by tax scenario.
Within each tax group, 4 boxes side-by-side for the parameter sweeps.
Individual data points jittered behind each box.

Usage:
    python scripts/plot_fig5_3_boxjitter.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

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
    ('panel_l', 'all_flat'): {'x': -0.4000, 'y': 15.3000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'rho_flat'): {'x': -0.1419, 'y': 29.3000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'beta_flat'): {'x': 0.1327, 'y': 37.8205, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'eta_flat'): {'x': 0.4145, 'y': 47.5897, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'all_prog'): {'x': 1.2000, 'y': 28.7000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'rho_prog'): {'x': 1.5000, 'y': 64.1000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'beta_prog'): {'x': 1.7000, 'y': 69.6000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'eta_prog'): {'x': 2.0000, 'y': 83.0000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'all_hprog'): {'x': 2.8000, 'y': 93.4000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'rho_hprog'): {'x': 3.0436, 'y': 153.2000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'beta_hprog'): {'x': 3.3436, 'y': 145.3000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'eta_hprog'): {'x': 3.6254, 'y': 210.5000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'grp_flat'): {'x': 0.0000, 'y': -30.0000, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'grp_prog'): {'x': 1.6000, 'y': -30.0000, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_l', 'grp_hprog'): {'x': 3.2000, 'y': -30.0000, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'all_flat'): {'x': -0.4218, 'y': 1.9000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'rho_flat'): {'x': -0.1436, 'y': 2.1963, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'beta_flat'): {'x': 0.1363, 'y': 2.7707, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'eta_flat'): {'x': 0.4254, 'y': 2.8293, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'all_prog'): {'x': 1.1891, 'y': 1.6670, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'rho_prog'): {'x': 1.4528, 'y': 1.8073, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'beta_prog'): {'x': 1.7399, 'y': 2.3744, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'eta_prog'): {'x': 2.0145, 'y': 2.3220, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'all_hprog'): {'x': 2.7746, 'y': 1.1744, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'rho_hprog'): {'x': 3.0601, 'y': 1.3000, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'beta_hprog'): {'x': 3.3436, 'y': 1.5890, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'eta_hprog'): {'x': 3.6254, 'y': 1.8561, 'fontsize': 7, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'grp_flat'): {'x': 0.0000, 'y': 0.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'grp_prog'): {'x': 1.6000, 'y': 0.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
    ('panel_r', 'grp_hprog'): {'x': 3.2000, 'y': 0.8000, 'fontsize': 9, 'rotation': 0, 'ha': 'center', 'va': 'top'},
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
    'grp_flat': 'Flat tax',
    'grp_prog': 'Progressive tax',
    'grp_hprog': 'Highly progressive tax',
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
    'grp_flat': '#2C3E50',
    'grp_prog': '#d62728',
    'grp_hprog': '#1f77b4',
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.default_rng(42)
    n_sweeps = len(SWEEP_ORDER)

    for panel_idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[panel_idx]

        group_centers = []
        for gi, te_val in enumerate(TAX_EQUITY_ORDER):
            group_center = gi * GROUP_SPACING
            group_centers.append(group_center)

            all_vals = []
            all_positions = []

            for si, sweep in enumerate(SWEEP_ORDER):
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
                patch.set_hatch(SWEEP_HATCHES[SWEEP_ORDER[si]])

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
    png_path = FIGURES_DIR / 'fig5_3_boxjitter.png'
    pdf_path = FIGURES_DIR / 'fig5_3_boxjitter.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {png_path}')


if __name__ == '__main__':
    main()
