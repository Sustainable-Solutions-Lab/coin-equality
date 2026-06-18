"""Production figure 6.4: OCP and ΔT vs r_consumption for different eta values.

Creates a two-panel figure showing how optimal carbon price (MAC in 2030)
and temperature change in 2100 vary with the consumption growth rate (r_consumption),
for six eta (elasticity of marginal utility) values.

Usage:
    python3 scripts/plot_prod_fig6_4_eta_sweep.py
    python3 scripts/plot_prod_fig6_4_eta_sweep.py --interactive
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fig_style import TAX_EQUITY_COLORS, TAX_EQUITY_LABELS, TAX_EQUITY_ORDER
from label_utils import place_labels, enable_interactive

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod', 'prod_discount_exact_eta_sweep')
GRID_CSV = os.path.join(PROJECT_ROOT, 'json', 'prod_discount_exact_eta_sweep_grid.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod', 'figures')

# Style parameters
LINE_WIDTH = 2.0
MARKER_SIZE = 7

LABEL_POSITIONS = {
    ('panel_a', 'flat'): {'x': 0.5, 'y': 50.0, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'prog'): {'x': 0.5, 'y': 90.0, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_a', 'hprog'): {'x': 0.5, 'y': 140.0, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_b', 'flat'): {'x': 0.5, 'y': 3.2, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_b', 'prog'): {'x': 0.5, 'y': 2.8, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
    ('panel_b', 'hprog'): {'x': 0.5, 'y': 2.2, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'bottom', 'bbox': True},
}

LABEL_TEXTS = {
    'flat': 'Flat tax',
    'prog': 'Progressive tax',
    'hprog': 'Highly progressive tax',
}

LABEL_COLORS = {
    'flat': TAX_EQUITY_COLORS[0.0],
    'prog': TAX_EQUITY_COLORS[0.5],
    'hprog': TAX_EQUITY_COLORS[0.998],
}

MARKERS = {}
ARROWS = {}


def load_results():
    """Load eta-sweep results with r_consumption values."""
    results = {}
    sweep_path = Path(SWEEP_DIR)

    for result_dir in sorted(sweep_path.glob('prod_discount_exact_eta_sweep_gid-*')):
        status_file = result_dir / 'run_status.csv'
        if not status_file.exists():
            continue

        with open(status_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gid = int(row['grid_id'])
                eta = float(row.get('eta', 0))
                break

        # Get r_consumption from results CSV at 2030 (first data row)
        results_file = result_dir / f'prod_discount_exact_eta_sweep_gid-{gid}_results.csv'
        if results_file.exists():
            with open(results_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r_consumption = float(row.get('r_consumption', 0))
                    break

        # Get mac_2030 and delta_t_2100 from status
        with open(status_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[gid] = {
                    'eta': eta,
                    'r_consumption': r_consumption,
                    'mac_2030': float(row.get('mac_2030', 0)),
                    'delta_t_2100': float(row.get('delta_t_2100', 0)),
                    'tax_equity': float(row.get('tax_equity', 0)),
                }
                break

    return results


def load_and_organize(results):
    """Organize results by tax_equity scenario."""
    by_tax_equity = {tax_eq: {'r_consumption': [], 'mac_2030': [], 'delta_t_2100': []}
                     for tax_eq in TAX_EQUITY_ORDER}

    for gid, data in results.items():
        tax_equity = data['tax_equity']

        if tax_equity in by_tax_equity:
            by_tax_equity[tax_equity]['r_consumption'].append(data['r_consumption'])
            by_tax_equity[tax_equity]['mac_2030'].append(data['mac_2030'])
            by_tax_equity[tax_equity]['delta_t_2100'].append(data['delta_t_2100'])

    # Sort each tax_equity group by r_consumption
    for tax_equity in by_tax_equity:
        if by_tax_equity[tax_equity]['r_consumption']:
            sorted_idx = np.argsort(by_tax_equity[tax_equity]['r_consumption'])
            by_tax_equity[tax_equity]['r_consumption'] = np.array(by_tax_equity[tax_equity]['r_consumption'])[sorted_idx]
            by_tax_equity[tax_equity]['mac_2030'] = np.array(by_tax_equity[tax_equity]['mac_2030'])[sorted_idx]
            by_tax_equity[tax_equity]['delta_t_2100'] = np.array(by_tax_equity[tax_equity]['delta_t_2100'])[sorted_idx]

    return by_tax_equity


def main():
    results = load_results()
    by_tax_equity = load_and_organize(results)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: MAC 2030 vs r_consumption
    for tax_equity in TAX_EQUITY_ORDER:
        if len(by_tax_equity[tax_equity]['r_consumption']) == 0:
            continue

        r_consumption = by_tax_equity[tax_equity]['r_consumption']
        mac = by_tax_equity[tax_equity]['mac_2030']

        color = TAX_EQUITY_COLORS[tax_equity]
        label = TAX_EQUITY_LABELS[tax_equity]

        ax_a.plot(r_consumption, mac, color=color, linewidth=LINE_WIDTH, marker='o',
                  markersize=MARKER_SIZE, label=label, zorder=3)

    ax_a.set_xlabel('Consumption growth rate in 2030 (r_consumption)', fontsize=10)
    ax_a.set_ylabel('Optimal carbon price in 2030 ($/tCO₂)', fontsize=10)
    ax_a.set_title('(A) Optimal Carbon Price', fontsize=12, weight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.set_axisbelow(True)

    # Panel B: Temperature change vs r_consumption
    for tax_equity in TAX_EQUITY_ORDER:
        if len(by_tax_equity[tax_equity]['r_consumption']) == 0:
            continue

        r_consumption = by_tax_equity[tax_equity]['r_consumption']
        dt = by_tax_equity[tax_equity]['delta_t_2100']

        color = TAX_EQUITY_COLORS[tax_equity]
        label = TAX_EQUITY_LABELS[tax_equity]

        ax_b.plot(r_consumption, dt, color=color, linewidth=LINE_WIDTH, marker='o',
                  markersize=MARKER_SIZE, label=label, zorder=3)

    ax_b.set_xlabel('Consumption growth rate in 2030 (r_consumption)', fontsize=10)
    ax_b.set_ylabel('Temperature change in 2100 (°C)', fontsize=10)
    ax_b.set_title('(B) Temperature Change', fontsize=12, weight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_axisbelow(True)

    # Tight layout
    fig.tight_layout()

    # Place labels
    labels_a = place_labels(ax_a, 'panel_a', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    labels_b = place_labels(ax_b, 'panel_b', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)

    if '--interactive' in sys.argv:
        panel_labels = {**{('panel_a', k): v for k, v in labels_a.items()},
                       **{('panel_b', k): v for k, v in labels_b.items()}}
        panel_axes = {'panel_a': ax_a, 'panel_b': ax_b}
        enable_interactive(fig, panel_axes, panel_labels,
                          os.path.abspath(__file__), MARKERS, ARROWS)
        plt.show()
        return

    # Save figure
    os.makedirs(OUT_DIR, exist_ok=True)

    fig_path_png = os.path.join(OUT_DIR, 'prod_fig6_4_eta_sweep.png')
    fig_path_pdf = os.path.join(OUT_DIR, 'prod_fig6_4_eta_sweep.pdf')
    fig_path_svg = os.path.join(OUT_DIR, 'prod_fig6_4_eta_sweep.svg')

    fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_path_pdf, bbox_inches='tight')
    fig.savefig(fig_path_svg, bbox_inches='tight')

    print(f"✓ Saved: {fig_path_png}")
    print(f"✓ Saved: {fig_path_pdf}")
    print(f"✓ Saved: {fig_path_svg}")

    plt.close(fig)


if __name__ == '__main__':
    main()
