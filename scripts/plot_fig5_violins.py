"""
Figure 5: Single-parameter violin sweeps from config_061 + joint uncertainty from config_058.

4x2 grid:
  Columns: eta, rho, beta (from config_061), all (from config_058 LHS)
  Row 1: Optimal Carbon Price in 2030 ($/tCO2)
  Row 2: Temperature Change in 2100 (°C)

Each panel has 3 violins (Flat tax, Progressive tax, Highly progressive tax).

Usage:
    python scripts/plot_fig5_violins.py
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

METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030 ($/tCO₂)'),
    ('delta_t_2100', 'Temperature Change in 2100 (°C)'),
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


def load_results_061():
    """Load all successful config_061 run_status.csv entries, keyed by grid_id."""
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
    """Load all successful config_058 run_status.csv entries, keyed by grid_id."""
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
    """Load grid CSV and return dict keyed by grid_id."""
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

    # Group config_061 results by sweep and tax_equity
    by_sweep_tax = defaultdict(list)
    for gid, r in results_061.items():
        g = grid[gid]
        sweep = g['sweep']
        tax = float(g['tax_equity'])
        by_sweep_tax[(sweep, tax)].append(r)

    # Use config_058 (LHS, all parameters joint) for the "all" column
    for r in results_058.values():
        tax = float(r['tax_equity'])
        by_sweep_tax[('all', tax)].append(r)

    for key in sorted(by_sweep_tax):
        print(f'  {key}: {len(by_sweep_tax[key])} runs')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, sweep in enumerate(SWEEP_ORDER):
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

    # Place labels on first panel only (colors serve as legend for all panels)
    panel_labels = {}
    panel_markers = {}
    panel_arrows = {}
    for col, sweep in enumerate(SWEEP_ORDER):
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
        for col, sweep in enumerate(SWEEP_ORDER):
            for row, (metric_key, _) in enumerate(METRICS):
                pname = f'panel_{sweep}_{metric_key.split("_")[0]}'
                panel_axes[pname] = axes[row, col]
        enable_interactive(fig, panel_axes, panel_labels, os.path.abspath(__file__),
                           panel_markers, panel_arrows)
        plt.show()
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / 'fig5_violin.png'
    svg_path = FIGURES_DIR / 'fig5_violin.svg'
    pdf_path = FIGURES_DIR / 'fig5_violin.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {png_path}')


if __name__ == '__main__':
    main()
