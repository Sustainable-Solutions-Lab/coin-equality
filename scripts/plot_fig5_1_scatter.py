"""
Figure 5.1: Scatter Summary — Carbon Price vs Temperature.

Each point = one model run. Color by tax_equity. Panels by sweep type (columns).
Shows the joint distribution of OCP@2030 and delta_T@2100.

Usage:
    python scripts/plot_fig5_1_scatter.py
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
SWEEP_TITLES = {
    'eta': 'Elasticity of\nmarginal utility (η)',
    'rho': 'Effective rate of pure\ntime preference (ρ)',
    'beta': 'Damage distribution\nexponent (β)',
    'all': 'All parameters\n(combined)',
}

# ---------------------------------------------------------------------------
# Label system — positions are rewritten in-place by --interactive mode
# ---------------------------------------------------------------------------
LABEL_POSITIONS = {
    ('panel_eta', 'flat'): {'x': 97.4, 'y': 3.3, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_eta', 'prog'): {'x': 161.2, 'y': 2.8, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_eta', 'hprog'): {'x': 240.9, 'y': 2.3, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_all', 'txt_1'): {'x': 62.9, 'y': 3.6, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_rho', 'txt_3'): {'x': 83.8, 'y': 3.4, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_beta', 'txt_4'): {'x': 79.6, 'y': 3.3, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_rho', 'txt_6'): {'x': 167.8, 'y': 2.8, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_beta', 'txt_7'): {'x': 158.5, 'y': 2.8, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_all', 'txt_8'): {'x': 153.5, 'y': 2.9, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_rho', 'txt_10'): {'x': 107.3, 'y': 1.3, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_beta', 'txt_11'): {'x': 41.9, 'y': 1.6, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_all', 'txt_12'): {'x': 98.9, 'y': 1.3, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
}
LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'txt_1': 'Flat tax',
    'txt_3': 'Flat tax',
    'txt_4': 'Flat tax',
    'txt_6': 'Progressive tax',
    'txt_7': 'Progressive tax',
    'txt_8': 'Progressive tax',
    'txt_10': 'Highly progressive tax',
    'txt_11': 'Highly progressive tax',
    'txt_12': 'Highly progressive tax',
}
LABEL_COLORS = {
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'txt_1': '#2C3E50',
    'txt_3': '#2C3E50',
    'txt_4': '#2C3E50',
    'txt_6': '#d62728',
    'txt_7': '#d62728',
    'txt_8': '#d62728',
    'txt_10': '#1f77b4',
    'txt_11': '#1f77b4',
    'txt_12': '#1f77b4',
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

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for col, sweep in enumerate(SWEEP_ORDER):
        ax = axes[col]

        for te_val in TAX_EQUITY_ORDER:
            runs = by_sweep_tax.get((sweep, te_val), [])
            mac = np.array([float(r['mac_2030']) for r in runs])
            dt = np.array([float(r['delta_t_2100']) for r in runs])
            ax.scatter(mac, dt,
                       c=TAX_EQUITY_COLORS[te_val],
                       s=8, alpha=0.3, edgecolors='none',
                       label=TAX_EQUITY_LABELS[te_val],
                       rasterized=True)

        ax.set_title(SWEEP_TITLES[sweep], fontsize=10, weight='bold')
        ax.set_xlabel('Optimal Carbon Price in 2030 ($/tCO₂)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Temperature Change in 2100 (°C)', fontsize=9)
        ax.set_xlim(0, 500)
        ax.set_ylim(1, 4)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # Place labels (colored text instead of legend)
    panel_names = {s: f'panel_{s}' for s in SWEEP_ORDER}
    panel_labels = {}
    panel_markers = {}
    panel_arrows = {}
    for col, sweep in enumerate(SWEEP_ORDER):
        pname = panel_names[sweep]
        lbl = place_labels(axes[col], pname, LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
        mkr = place_markers(axes[col], pname, MARKERS)
        arr = place_arrows(axes[col], pname, ARROWS)
        for k, v in lbl.items():
            panel_labels[(pname, k)] = v
        for k, v in mkr.items():
            panel_markers[(pname, k)] = v
        for k, v in arr.items():
            panel_arrows[(pname, k)] = v

    plt.tight_layout()

    if '--interactive' in sys.argv:
        panel_axes = {panel_names[s]: axes[i] for i, s in enumerate(SWEEP_ORDER)}
        enable_interactive(fig, panel_axes, panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / 'fig5_1_scatter.png'
    pdf_path = FIGURES_DIR / 'fig5_1_scatter.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {png_path}')


if __name__ == '__main__':
    main()
