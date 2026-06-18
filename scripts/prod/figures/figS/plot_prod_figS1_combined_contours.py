"""
Production Figure S1: Combined contour overlay on τ–β parameter grid.

Uses the same data source as Fig. 3: master_results_contour.csv from the 2D sweep.
Grid: 41 tax_equity x 41 y_damage_distribution_exponent = 1681 runs.

Visualization:
- Background color: Optimal Carbon Price 2030 (OPC2030, mac_2030)
- Black contour lines: Temperature Change 2100 (ΔT2100, dt_2100)
- X-axis: Tax Equity Parameter (τ)
- Y-axis: Damage Distribution Exponent (β)

This combines the two Fig. 3 panels into a single supplementary figure.

Output: data/output/prod/figures/prod_figS1_combined_contours.pdf/png
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod')
MASTER_CSV = os.path.join(PROD_OUTPUT, 'master_results_contour.csv')
FIGURES_DIR = os.path.join(PROD_OUTPUT, 'figures')


def load_contour_data():
    """Load contour data from master results CSV.

    Only validated points are used.  Outliers and missing cells are left as
    NaN and filled by interpolation (cubic + nearest fallback).
    """
    rows = []
    with open(MASTER_CSV, 'r') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    n_total = len(rows)
    n_validated = sum(1 for r in rows if r['validated'] == 'True')

    source_counts = {}
    for r in rows:
        src = r['selected_source']
        if src:
            source_counts[src] = source_counts.get(src, 0) + 1

    print(f'Loaded {n_total} rows from {os.path.basename(MASTER_CSV)}')
    print(f'  Validated: {n_validated}')
    for src in sorted(source_counts):
        print(f'  {src}: {source_counts[src]}')

    results = {}
    for r in rows:
        if r['validated'] != 'True':
            continue
        gid = int(r['grid_id'])
        results[gid] = {
            'tax_equity': r['tax_equity'],
            'y_damage_distribution_exponent': r['y_damage_distribution_exponent'],
            'mac_2030': r['mac_selected'],
            'delta_t_2100': r['dt_selected'],
        }

    n_excluded = n_total - len(results)
    print(f'  Excluded (outliers + missing): {n_excluded}')

    return rows, results


def build_grids(master_rows, results):
    """Build 2D arrays for contour plotting."""
    te_set = sorted(set(float(r['tax_equity']) for r in master_rows))
    ydmg_set = sorted(set(float(r['y_damage_distribution_exponent']) for r in master_rows))

    tax_equity_vals = np.array(te_set)
    ydmg_vals = np.array(ydmg_set)
    n_te = len(tax_equity_vals)
    n_ydmg = len(ydmg_vals)

    mac_2030 = np.full((n_ydmg, n_te), np.nan)
    delta_t_2100 = np.full((n_ydmg, n_te), np.nan)

    for gid, row in results.items():
        te = float(row['tax_equity'])
        ydmg = float(row['y_damage_distribution_exponent'])
        i_te = np.argmin(np.abs(tax_equity_vals - te))
        j_ydmg = np.argmin(np.abs(ydmg_vals - ydmg))
        mac_2030[j_ydmg, i_te] = float(row['mac_2030'])
        delta_t_2100[j_ydmg, i_te] = float(row['delta_t_2100'])

    n_valid = np.sum(~np.isnan(mac_2030))
    n_total = n_te * n_ydmg
    print(f'Grid: {n_te} x {n_ydmg} = {n_total}')
    print(f'Grid coverage: {n_valid}/{n_total} ({100 * n_valid / n_total:.1f}%)')
    print(f'Tax equity range: {tax_equity_vals[0]:.4f} to {tax_equity_vals[-1]:.4f}')
    print(f'Beta range: {ydmg_vals[0]:.4f} to {ydmg_vals[-1]:.4f}')

    # Interpolate missing grid points (cubic, then nearest fallback)
    TE_grid, YDMG_grid = np.meshgrid(tax_equity_vals, ydmg_vals)
    for data in [mac_2030, delta_t_2100]:
        mask = ~np.isnan(data)
        if mask.all():
            continue
        points = np.column_stack([TE_grid[mask], YDMG_grid[mask]])
        values = data[mask]
        filled_cubic = griddata(points, values, (TE_grid, YDMG_grid), method='cubic')
        data[np.isnan(data)] = filled_cubic[np.isnan(data)]
        # Nearest-neighbor fallback for any remaining NaN (corners/edges)
        still_nan = np.isnan(data)
        if still_nan.any():
            filled_nn = griddata(points, values, (TE_grid, YDMG_grid), method='nearest')
            data[still_nan] = filled_nn[still_nan]

    n_filled = np.sum(~np.isnan(mac_2030))
    print(f'After interpolation: {n_filled}/{n_total} ({100 * n_filled / n_total:.1f}%)')

    return tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100


def plot_combined_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100):
    """Create combined contour figure with OPC2030 color and DT2100 black contours.

    Interpolates to a 400x400 fine grid for smooth contour lines.
    """
    # Upsample to fine grid for smooth contours
    n_fine = 400
    te_fine = np.linspace(tax_equity_vals[0], tax_equity_vals[-1], n_fine)
    ydmg_fine = np.linspace(ydmg_vals[0], ydmg_vals[-1], n_fine)
    TE_coarse, YDMG_coarse = np.meshgrid(tax_equity_vals, ydmg_vals)
    TE, YDMG = np.meshgrid(te_fine, ydmg_fine)

    coarse_points = np.column_stack([TE_coarse.ravel(), YDMG_coarse.ravel()])
    mac_fine = griddata(coarse_points, mac_2030.ravel(), (TE, YDMG), method='cubic')
    dt_fine = griddata(coarse_points, delta_t_2100.ravel(), (TE, YDMG), method='cubic')

    # Gaussian smoothing for clean contour lines
    mac_fine = gaussian_filter(mac_fine, sigma=12.0)
    dt_fine = gaussian_filter(dt_fine, sigma=12.0)

    # Create figure
    fig_w, fig_h = 9, 8
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)

    # OPC2030 as filled contours (background color)
    mac_vmin, mac_vmax = 50, 350
    mac_fill_step = 10
    mac_levels = np.arange(mac_vmin, mac_vmax + mac_fill_step / 2, mac_fill_step)
    mac_plot = np.where(np.isnan(mac_fine), np.nan, np.clip(mac_fine, mac_vmin, mac_vmax))

    cf = ax.contourf(TE, YDMG, mac_plot, levels=mac_levels, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Optimal Carbon Price 2030 ($/tCO₂)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # DT2100 as black contour lines
    dt_vmin, dt_vmax = 1.5, 3.5
    dt_contour_step = 0.2
    dt_levels = np.arange(dt_vmin, dt_vmax + dt_contour_step / 2, dt_contour_step)
    cs = ax.contour(TE, YDMG, dt_fine, levels=dt_levels, colors='black', linewidths=0.8)
    cl = ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f °C')
    if cl:
        for txt in cl:
            txt.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # Axis labels and limits
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_aspect('equal')
    ax.set_xlabel('Tax Equity Parameter (τ)', fontsize=12)
    ax.set_ylabel('Damage Distribution Exponent (β)', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Title
    ax.set_title('Joint Response of 2030 Optimal Carbon Price and 2100 Temperature\nAcross the τ–β Parameter Grid',
                 fontsize=12, pad=15)

    plt.tight_layout()

    return fig


def main():
    master_rows, results = load_contour_data()

    tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100 = build_grids(master_rows, results)

    fig = plot_combined_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    pdf_path = os.path.join(FIGURES_DIR, 'prod_figS1_combined_contours.pdf')
    png_path = os.path.join(FIGURES_DIR, 'prod_figS1_combined_contours.png')

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f'\nSaved {pdf_path}')
    print(f'Saved {png_path}')

    print(f'\nData source: {os.path.basename(MASTER_CSV)}')
    print(f'Column mapping:')
    print(f'  Background color (OPC2030): mac_selected')
    print(f'  Black contour lines (ΔT2100): dt_selected')
    print(f'  X-axis (τ): tax_equity')
    print(f'  Y-axis (β): y_damage_distribution_exponent')


if __name__ == '__main__':
    main()
