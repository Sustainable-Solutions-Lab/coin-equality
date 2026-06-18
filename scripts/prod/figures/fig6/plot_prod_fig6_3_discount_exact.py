"""Discount-rate figure v3: OCP and ΔT vs r_consumption@2030 for target-rate runs.

Loads the 30 prod_discount_exact runs (targeting r_consumption = 3,4,5,6,7%)
and plots scatter points for each target rate, with original prod_discount
lines as reference background.

Two output modes:
  --lines-only   Show only the target-rate scatter points
  --overlay       Overlay target points on top of prod_discount lines

Usage:
    python scripts/plot_prod_fig6_3_discount_exact.py
    python scripts/plot_prod_fig6_3_discount_exact.py --overlay
    python scripts/plot_prod_fig6_3_discount_exact.py --interactive
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fig_style import TAX_EQUITY_COLORS, TAX_EQUITY_LABELS, TAX_EQUITY_ORDER
from label_utils import place_labels, enable_interactive

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXACT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod', 'prod_discount_exact')
EXACT_GRID_CSV = os.path.join(PROJECT_ROOT, 'json', 'prod_discount_exact_grid.csv')
ORIG_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod', 'prod_discount')
ORIG_GRID_CSV = os.path.join(PROJECT_ROOT, 'json', 'prod_discount_grid.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod', 'figures')

OCP_YEAR = 2030
DT_YEAR = 2100
R_YEAR = 2030

LINE_WIDTH = 2.0
MARKER_SIZE = 7
SCATTER_SIZE = 50

# DICE-2023 reference (still using 2025 for now)
BN_DISCOUNT_RATES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
BN_SCC_2025 = np.array([571, 207, 102, 58, 37])

# Target r values — use distinct colors from a perceptually uniform palette
TARGETS = [3.0, 4.0, 5.0, 6.0, 7.0]
TARGET_COLORS = {
    3.0: '#e41a1c',   # Red
    4.0: '#377eb8',   # Blue
    5.0: '#4daf4a',   # Green
    6.0: '#ff7f00',   # Orange
    7.0: '#984ea3',   # Purple
}

LABEL_POSITIONS = {
    ('panel_a', 'flat'): {'x': 6.0000, 'y': 22.0000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'prog'): {'x': 6.2000, 'y': 55.0000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'hprog'): {'x': 5.5000, 'y': 100.0000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'dice'): {'x': 1.2000, 'y': 700.0000, 'fontsize': 8, 'rotation': 0, 'ha': 'left', 'va': 'bottom'},
    ('panel_b', 'flat'): {'x': 1.5000, 'y': 3.2000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_b', 'prog'): {'x': 1.0000, 'y': 2.6000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_b', 'hprog'): {'x': 1.0000, 'y': 1.3000, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
}

LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
    'dice': 'DICE-2023 (B&N 2024)',
}

LABEL_COLORS = {
    'flat': '#2C3E50',
    'prog': '#d62728',
    'hprog': '#1f77b4',
    'dice': '#228B22',
}

MARKERS = {}
ARROWS = {}


def _find_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0] if matches else None


def load_prod_discount_data():
    """Load original prod_discount lines for reference."""
    grid = pd.read_csv(ORIG_GRID_CSV)
    discount_dir = Path(ORIG_DIR)

    dir_index = {}
    for entry in sorted(discount_dir.iterdir()):
        status_path = entry / 'run_status.csv'
        if not status_path.exists():
            continue
        with open(status_path, 'r') as f:
            for row in csv.DictReader(f):
                if row['success'] == 'True':
                    dir_index[int(row['grid_id'])] = entry.name

    results = {}
    for _, grow in grid.iterrows():
        grid_id = int(grow['grid_id'])
        tax_equity = float(grow['tax_equity'])
        ydmg = float(grow['y_damage_distribution_exponent'])
        rho = float(grow['rho'])

        dir_name = dir_index.get(grid_id)
        if dir_name is None:
            continue

        run_dir = discount_dir / dir_name
        results_files = sorted(run_dir.glob('*_results.csv'))
        df = pd.read_csv(results_files[-1])

        t_col = _find_col(df, 't,')
        years = df[t_col].values
        idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
        idx_dt = np.argmin(np.abs(years - DT_YEAR))
        idx_r = np.argmin(np.abs(years - R_YEAR))

        mac_col = _find_col(df, 'marginal_abatement_cost')
        delta_t_col = _find_col(df, 'delta_T,')

        entry = {
            'rho': rho,
            'ocp': df[mac_col].values[idx_ocp],
            'dt': df[delta_t_col].values[idx_dt],
            'r': df['r_consumption'].values[idx_r] * 100,
        }

        key = (tax_equity, ydmg)
        results.setdefault(key, []).append(entry)

    data = {}
    for key in sorted(results):
        entries = sorted(results[key], key=lambda e: e['rho'])
        data[key] = {
            'rho': np.array([e['rho'] for e in entries]),
            'ocp': np.array([e['ocp'] for e in entries]),
            'dt': np.array([e['dt'] for e in entries]),
            'r': np.array([e['r'] for e in entries]),
        }

    return data


def load_prod_discount_exact_data():
    """Load new prod_discount_exact target-rate data."""
    grid = pd.read_csv(EXACT_GRID_CSV)
    exact_dir = Path(EXACT_DIR)

    data = {}

    for _, grow in grid.iterrows():
        grid_id = int(grow['grid_id'])
        tax_equity = float(grow['tax_equity'])
        ydmg = float(grow['y_damage_distribution_exponent'])
        rho = float(grow['rho'])

        matching_dirs = sorted(exact_dir.glob(f'*_gid-{grid_id}_*'))
        if not matching_dirs:
            continue

        run_dir = matching_dirs[-1]
        status_file = run_dir / 'run_status.csv'
        if not status_file.exists():
            continue

        with open(status_file, 'r') as f:
            status_row = list(csv.DictReader(f))[0]

        if status_row['success'] != 'True':
            continue

        results_files = sorted(run_dir.glob('*_results.csv'))
        df = pd.read_csv(results_files[-1])

        t_col = _find_col(df, 't,')
        years = df[t_col].values
        idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
        idx_dt = np.argmin(np.abs(years - DT_YEAR))
        idx_r = np.argmin(np.abs(years - R_YEAR))

        mac_col = _find_col(df, 'marginal_abatement_cost')
        delta_t_col = _find_col(df, 'delta_T,')

        r_actual = df['r_consumption'].values[idx_r] * 100
        target_idx = (grid_id - 1) % 5
        target_r = TARGETS[target_idx]

        entry = {
            'rho': rho,
            'ocp': df[mac_col].values[idx_ocp],
            'dt': df[delta_t_col].values[idx_dt],
            'r': r_actual,
            'r_target': target_r,
        }

        key = (tax_equity, ydmg)
        data.setdefault(key, []).append(entry)

    for key in data:
        data[key] = sorted(data[key], key=lambda e: e['r_target'])

    return data


def plot_figure(overlay_orig):
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading prod_discount_exact data...')
    exact_data = load_prod_discount_exact_data()

    orig_data = None
    if overlay_orig:
        print('Loading prod_discount original data...')
        orig_data = load_prod_discount_data()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel A: OCP vs r_consumption
    if orig_data is not None:
        for key in sorted(orig_data.keys()):
            tax_equity = key[0]
            d = orig_data[key]
            ax_a.plot(d['r'], d['ocp'], linestyle='-', color=TAX_EQUITY_COLORS[tax_equity],
                      linewidth=LINE_WIDTH, marker='o', markersize=4, alpha=0.4, zorder=2)

    ax_a.plot(BN_DISCOUNT_RATES, BN_SCC_2025, 'o-',
              color='#228B22', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, zorder=4)

    for tax_equity in TAX_EQUITY_ORDER:
        if tax_equity not in exact_data:
            continue
        d_list = exact_data[tax_equity]
        for d in d_list:
            target_r = d['r_target']
            ax_a.scatter(d['r'], d['ocp'], s=SCATTER_SIZE,
                        color=TARGET_COLORS[target_r], edgecolors='black', linewidths=0.5,
                        zorder=5, alpha=0.8)

    ax_a.set_yscale('log')
    ax_a.set_ylim(20, 1000)
    ax_a.set_xlim(0, 8)
    ax_a.set_yticks([20, 30, 50, 100, 200, 300, 500, 1000])
    ax_a.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax_a.yaxis.set_tick_params(which='minor', size=0)

    ax_a.set_xlabel(f'Discount rate in {R_YEAR} (%/year)', fontsize=11)
    ax_a.set_ylabel(f'Optimal carbon price in {OCP_YEAR} ($/tCO\u2082)', fontsize=11)
    ax_a.set_title('(A)  Optimal Carbon Price', fontsize=12, fontweight='bold')
    ax_a.grid(axis='y', alpha=0.3, which='major')
    ax_a.set_axisbelow(True)

    # Add target r labels for Panel A
    for target_r in TARGETS:
        for tax_equity in TAX_EQUITY_ORDER:
            if tax_equity not in exact_data:
                continue
            for d in exact_data[tax_equity]:
                if abs(d['r_target'] - target_r) < 0.01:
                    ax_a.text(d['r'], d['ocp'], f'{target_r:.0f}%',
                             fontsize=7, ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=TARGET_COLORS[target_r],
                                      alpha=0.5, edgecolor='none'))

    # Panel B: ΔT vs r_consumption
    if orig_data is not None:
        for key in sorted(orig_data.keys()):
            tax_equity = key[0]
            ydmg = key[1]
            d = orig_data[key]
            linestyle = '-' if ydmg == 0.0 else '--'
            ax_b.plot(d['r'], d['dt'], linestyle=linestyle, color=TAX_EQUITY_COLORS[tax_equity],
                      linewidth=LINE_WIDTH-0.5, marker='o', markersize=4, alpha=0.3, zorder=2)

    for tax_equity in TAX_EQUITY_ORDER:
        if tax_equity not in exact_data:
            continue
        d_list = exact_data[tax_equity]
        for d in d_list:
            target_r = d['r_target']
            ax_b.scatter(d['r'], d['dt'], s=SCATTER_SIZE,
                        color=TARGET_COLORS[target_r], edgecolors='black', linewidths=0.5,
                        zorder=5, alpha=0.8)

    ax_b.set_xlim(0, 8)
    ax_b.set_ylim(1.2, 3.8 if overlay_orig else 3.5)
    ax_b.set_xlabel(f'Discount rate in {R_YEAR} (%/year)', fontsize=11)
    ax_b.set_ylabel(f'Temperature change in {DT_YEAR} (\u00b0C)', fontsize=11)
    ax_b.set_title('(B)  Temperature Change', fontsize=12, fontweight='bold')
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.set_axisbelow(True)

    # Add target r labels for Panel B (only highly progressive for clarity)
    for target_r in TARGETS:
        if 0.998 in exact_data:
            for d in exact_data[0.998]:
                if abs(d['r_target'] - target_r) < 0.01:
                    ax_b.text(d['r'], d['dt'], f'{target_r:.0f}%',
                             fontsize=7, ha='center', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=TARGET_COLORS[target_r],
                                      alpha=0.5, edgecolor='none'))

    # Legend: original lines + target points
    legend_handles = []
    if orig_data is not None:
        # Tax equity sweep
        for tax_equity in TAX_EQUITY_ORDER:
            h = mlines.Line2D([], [], color=TAX_EQUITY_COLORS[tax_equity],
                              linewidth=LINE_WIDTH, label=TAX_EQUITY_LABELS[tax_equity], alpha=0.4)
            legend_handles.append(h)
        # Beta line styles
        legend_handles.append(mlines.Line2D([], [], color='none', label=''))  # Spacer
        h_solid = mlines.Line2D([], [], color='black', linestyle='-', linewidth=LINE_WIDTH-0.5, alpha=0.4, label=r'Sweep: $\beta=0$')
        h_dashed = mlines.Line2D([], [], color='black', linestyle='--', linewidth=LINE_WIDTH-0.5, alpha=0.4, label=r'Sweep: $\beta=0.5$')
        legend_handles.extend([h_solid, h_dashed])
    # Target points
    legend_handles.append(mlines.Line2D([], [], color='none', label=''))  # Spacer
    legend_handles.append(mlines.Line2D([], [], color='none', label='Target-rate points:'))
    for target_r in TARGETS:
        h = mlines.Line2D([], [], color=TARGET_COLORS[target_r], marker='o', linewidth=0,
                          markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                          label=f'r={target_r:.0f}%')
        legend_handles.append(h)
    ax_b.legend(handles=legend_handles, fontsize=8, loc='lower right', framealpha=0.95, ncol=1)

    # Labels
    panel_map = {'panel_a': ax_a, 'panel_b': ax_b}
    panel_labels = {}
    for pname, pax in panel_map.items():
        artists = place_labels(pax, pname, LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
        for label_key, artist in artists.items():
            panel_labels[(pname, label_key)] = artist

    fig.tight_layout(w_pad=3)

    if '--interactive' in sys.argv:
        enable_interactive(fig, panel_map, panel_labels, os.path.abspath(__file__))
        plt.show()
        return

    fig_name = 'prod_fig6_3_discount_exact' if overlay_orig else 'prod_fig6_3_discount_exact_only'
    out_base = os.path.join(OUT_DIR, fig_name)
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out_base + ext, bbox_inches='tight',
                    dpi=300 if ext == '.png' else None)
        print(f'Saved: {out_base}{ext}')

    plt.close(fig)


def main():
    overlay = '--overlay' in sys.argv
    plot_figure(overlay)


if __name__ == '__main__':
    main()
