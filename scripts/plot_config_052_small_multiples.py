"""
Publication-quality small multiples visualization for config_052 results.

Creates a grid of 2D contour plots showing how the parameter space
(tax_equity vs y_damage_distribution_exponent) changes across different
discount rates (rho).

Layout: 2×3 grid (6 panels for 6 rho values)
Output: data/output/config_052_small_multiples.pdf
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output')
OUTPUT_PDF = os.path.join(OUTPUT_DIR, 'config_052_small_multiples.pdf')


def load_results_for_rho(rho):
    """Load all results for a specific rho value."""
    pattern = os.path.join(OUTPUT_DIR, 'config_052', f'config_052_rho-{rho:.3f}_tax-*_ydmg-*', 'run_status.csv')
    status_files = glob.glob(pattern)

    if len(status_files) == 0:
        print(f"Warning: No results found for rho={rho:.3f}")
        return []

    results_by_grid_id = {}
    for status_file in status_files:
        dir_mtime = os.path.getmtime(os.path.dirname(status_file))
        with open(status_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['success'] != 'True':
                    continue
                grid_id = int(row['grid_id'])
                prev = results_by_grid_id.get(grid_id)
                if prev is None or dir_mtime > prev['_mtime']:
                    row['_mtime'] = dir_mtime
                    results_by_grid_id[grid_id] = row

    results = []
    for row in results_by_grid_id.values():
        results.append({
            'rho': float(row['rho']),
            'tax_equity': float(row['tax_equity']),
            'y_damage_dist_exp': float(row['y_damage_distribution_exponent']),
            'mac_2030': float(row['mac_2030']),
            'mac_2100': float(row['mac_2100']),
            'dt_2100': float(row['delta_t_2100']),
        })

    print(f"Loaded {len(results)} results for rho={rho:.3f}")
    return results


def create_small_multiples(metric='dt_2100', metric_label='ΔT@2100 (°C)', cmap='RdYlBu_r'):
    """Create 2×3 small multiples grid for all 6 rho values."""

    # RHO values (including config_051 baseline)
    rho_values = [0.000, 0.005, 0.015, 0.026, 0.035, 0.045]

    # Load all data from config_052
    all_results = {}
    for rho in rho_values:
        all_results[rho] = load_results_for_rho(rho)

    # Compute global color scale
    all_values = []
    for results in all_results.values():
        all_values.extend([r[metric] for r in results])

    if len(all_values) == 0:
        print("ERROR: No data found!")
        return None

    vmin = min(all_values)
    vmax = max(all_values)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{metric_label} across Discount Rate and Policy Parameters',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, rho in enumerate(rho_values):
        ax = axes.flatten()[idx]
        results = all_results[rho]

        if len(results) == 0:
            ax.text(0.5, 0.5, f'ρ = {rho:.3f}\nNo data',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 0.45)
            continue

        # Extract data
        te = np.array([r['tax_equity'] for r in results])
        ydmg = np.array([r['y_damage_dist_exp'] for r in results])
        values = np.array([r[metric] for r in results])

        # Create regular grid for interpolation
        te_grid = np.linspace(0, 1, 60)
        ydmg_grid = np.linspace(0, 0.42, 60)
        TE, YDMG = np.meshgrid(te_grid, ydmg_grid)

        # Interpolate
        VALUES = griddata((te, ydmg), values, (TE, YDMG), method='cubic')

        # Contour plot
        levels = np.linspace(vmin, vmax, 20)
        cf = ax.contourf(TE, YDMG, VALUES, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        cs = ax.contour(TE, YDMG, VALUES, levels=levels, colors='black', linewidths=0.3, alpha=0.3)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f' if vmax < 10 else '%.0f')

        # Label
        label = f'ρ = {rho:.3f}'
        if abs(rho - 0.026) < 0.001:
            label += ' (baseline)'
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
               fontsize=11, fontweight='bold', va='top')

        # Axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.42)

        # Labels only on edges
        if idx >= 3:  # Bottom row
            ax.set_xlabel('Tax Equity Parameter', fontsize=11)
        else:
            ax.set_xticklabels([])

        if idx % 3 == 0:  # Left column
            ax.set_ylabel('Damage Distribution Exponent', fontsize=11)
        else:
            ax.set_yticklabels([])

        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.15, linestyle='--')

    # Colorbar
    fig.subplots_adjust(right=0.92, hspace=0.12, wspace=0.08, left=0.08)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(metric_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    return fig


def main():
    print("Generating config_052 small multiples visualization...\n")

    # Create figures for different metrics
    metrics = [
        ('dt_2100', 'ΔT@2100 (°C)', 'RdYlBu_r'),
        ('mac_2030', 'Optimal Carbon Price 2030 ($/tCO₂)', 'viridis'),
        ('mac_2100', 'Optimal Carbon Price 2100 ($/tCO₂)', 'viridis'),
    ]

    with PdfPages(OUTPUT_PDF) as pdf:
        for metric, label, cmap in metrics:
            print(f"Creating figure for {label}...")
            fig = create_small_multiples(metric, label, cmap)
            if fig:
                pdf.savefig(fig, dpi=200, bbox_inches='tight')
                plt.close(fig)

    print(f"\nSaved: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
