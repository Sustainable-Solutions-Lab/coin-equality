"""Update prod_fig6_3_verified_points with improved warmstart solutions.

Loads verified points and compares with warmstart results.
If warmstart provides a better (lower) OCP solution, uses that instead.
Regenerates the figure with the same axes but updated point values.

Usage:
    python scripts/update_fig6_3_verified_points.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import TAX_EQUITY_COLORS, TAX_EQUITY_LABELS, TAX_EQUITY_ORDER
from label_utils import place_labels, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFIED_POINTS_CSV = Path('/tmp/selected_verified_points_corrected.csv')
WARMSTART_RESULTS = PROJECT_ROOT / 'warmstart_19_results_local.json'
ORIGINAL_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'prod_discount_exact'
ORIGINAL_GRID_CSV = PROJECT_ROOT / 'json' / 'prod_discount_exact_grid.csv'
OUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'

os.makedirs(OUT_DIR, exist_ok=True)

OCP_YEAR, DT_YEAR, R_YEAR = 2030, 2100, 2030
LINE_WIDTH = 2.0
SCATTER_SIZE = 50
MARKER_SIZE = 7

# DICE-2023 reference
BN_DISCOUNT_RATES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
BN_SCC_2025 = np.array([571, 207, 102, 58, 37])

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
    'dice': '#666666',
}

MARKERS = {}
ARROWS = {}


def load_verified_points():
    """Load verified points from CSV."""
    data = {}
    with open(VERIFIED_POINTS_CSV, 'r') as f:
        for row in csv.DictReader(f):
            try:
                tax_equity = float(row['tax_equity'])
                beta = float(row['beta'])
                r = float(row['r_consumption'])
                ocp = float(row['ocp'])
                dt = float(row['dt'])
            except (ValueError, KeyError):
                continue

            key = (tax_equity, beta)
            if key not in data:
                data[key] = []

            data[key].append({
                'r': r,
                'ocp': ocp,
                'dt': dt,
                'is_warmstart': False,
            })

    # Sort by r within each key
    for key in data:
        data[key] = sorted(data[key], key=lambda x: x['r'])

    return data


def load_warmstart_results():
    """Load warmstart results and index by location."""
    warmstart_by_location = {}

    if not WARMSTART_RESULTS.exists():
        return warmstart_by_location

    with open(WARMSTART_RESULTS, 'r') as f:
        results = json.load(f)

    for result in results:
        gid = result['grid_id']
        output_dir = Path(result['output_dir'])

        # Load the original grid entry to get parameters
        grid_csv = ORIGINAL_GRID_CSV
        with open(grid_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['grid_id']) == gid:
                    tax_equity = float(row['tax_equity'])
                    beta = float(row['y_damage_distribution_exponent'])
                    break

        key = (tax_equity, beta)
        warmstart_by_location[key] = {
            'ocp': result['ocp_2030'],
            'dt': result['dt_2100'],
            'r': result['r_consumption_2030'],
            'gid': gid,
        }

    return warmstart_by_location


def find_best_point(verified_list, warmstart_result):
    """
    Find the best point to use.
    If warmstart OCP is lower, use warmstart. Otherwise use verified.
    """
    if warmstart_result is None:
        return verified_list[0] if verified_list else None

    verified_ocp = verified_list[0]['ocp'] if verified_list else float('inf')
    warmstart_ocp = warmstart_result['ocp']

    if warmstart_ocp < verified_ocp:
        return {
            'r': warmstart_result['r'],
            'ocp': warmstart_ocp,
            'dt': warmstart_result['dt'],
            'is_warmstart': True,
            'gid': warmstart_result['gid'],
        }
    else:
        return verified_list[0]


def create_figure():
    """Create the updated figure."""
    print("Loading data...")

    # Load verified points and warmstart results
    verified_data = load_verified_points()
    warmstart_data = load_warmstart_results()

    print(f"Loaded {sum(len(v) for v in verified_data.values())} verified points")
    print(f"Loaded {len(warmstart_data)} warmstart results")

    # Prepare plot data
    plot_data = {}
    improvements = 0

    for key in sorted(verified_data.keys()):
        tax_equity, beta = key
        verified_list = verified_data[key]
        warmstart_result = warmstart_data.get(key)

        # Use best available point
        best_point = find_best_point(verified_list, warmstart_result)

        if best_point:
            if best_point.get('is_warmstart'):
                improvements += 1

            plot_data[key] = {
                'points': verified_list,
                'best': best_point,
            }

    print(f"Using warmstart improvements for {improvements} location groups")

    # Create figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: OCP vs r_consumption
    for (tax_equity, beta), group_data in sorted(plot_data.items()):
        tax_label = {0.0: 'Flat tax', 0.5: 'Progressive tax', 0.998: 'Highly progressive tax'}.get(
            tax_equity, f'{tax_equity:.3f}'
        )
        color = TAX_EQUITY_COLORS.get(tax_equity, '#999999')

        points = group_data['points']
        r_vals = np.array([p['r'] for p in points])
        ocp_vals = np.array([p['ocp'] for p in points])

        # Plot verified points
        ax_a.scatter(r_vals, ocp_vals, s=SCATTER_SIZE, alpha=0.6, color=color, label=tax_label if beta == 0.0 else '')

        # Highlight best point if it's a warmstart
        best = group_data['best']
        if best.get('is_warmstart'):
            ax_a.scatter([best['r']], [best['ocp']], s=SCATTER_SIZE * 1.5, marker='*',
                        edgecolor=color, facecolor='yellow', linewidth=1.5, zorder=10)

    # Add DICE reference
    ax_a.plot(BN_DISCOUNT_RATES, BN_SCC_2025, 'o--', color='#666666', linewidth=1.5,
             markersize=6, label='DICE-2023 (B&N 2024)', alpha=0.7)

    ax_a.set_xlabel('Discount rate at 2030 (%)', fontsize=11)
    ax_a.set_ylabel('Carbon price at 2030 ($/tCO₂)', fontsize=11)
    ax_a.set_title('(A) Optimal Carbon Price', fontsize=12, weight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.set_axisbelow(True)
    ax_a.legend(fontsize=9, loc='upper right')
    ax_a.set_yscale('log')

    # Panel B: ΔT vs r_consumption
    for (tax_equity, beta), group_data in sorted(plot_data.items()):
        tax_label = {0.0: 'Flat tax', 0.5: 'Progressive tax', 0.998: 'Highly progressive tax'}.get(
            tax_equity, f'{tax_equity:.3f}'
        )
        color = TAX_EQUITY_COLORS.get(tax_equity, '#999999')

        points = group_data['points']
        r_vals = np.array([p['r'] for p in points])
        dt_vals = np.array([p['dt'] for p in points])

        # Plot verified points
        ax_b.scatter(r_vals, dt_vals, s=SCATTER_SIZE, alpha=0.6, color=color, label=tax_label if beta == 0.0 else '')

        # Highlight best point if it's a warmstart
        best = group_data['best']
        if best.get('is_warmstart'):
            ax_b.scatter([best['r']], [best['dt']], s=SCATTER_SIZE * 1.5, marker='*',
                        edgecolor=color, facecolor='yellow', linewidth=1.5, zorder=10)

    ax_b.set_xlabel('Discount rate at 2030 (%)', fontsize=11)
    ax_b.set_ylabel('Temperature increase at 2100 (°C)', fontsize=11)
    ax_b.set_title('(B) Temperature Change', fontsize=12, weight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=9, loc='upper left')

    plt.tight_layout()

    # Save figure
    base_name = 'prod_fig6_3_verified_points'
    for ext in ['png', 'pdf', 'svg']:
        out_path = OUT_DIR / f'{base_name}.{ext}'
        dpi = 300 if ext == 'png' else None
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == '__main__':
    create_figure()
