"""
Production Figure 3: Contour plots for prod_contour 2D sweep results.

Grid: 41 tax_equity x 41 y_damage_distribution_exponent = 1681 runs.
Reads from master_results_contour.csv (validated points from the
multi-source selection pipeline).

Output: data/output/prod/figures/prod_fig3_contours.pdf
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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


def plot_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100):
    """Create 1x2 filled contour figure.

    Interpolates to a 200x200 fine grid for smooth contour lines.
    """
    from scipy.ndimage import gaussian_filter

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

    panel_labels = ['A)', 'B)']
    panels = [
        (mac_fine, 'viridis', 'Optimal Carbon Price 2030 ($/tCO\u2082)', 50, 350, 50, 10, 20, '%.0f'),
        (dt_fine, 'RdYlBu_r', 'Temperature Change 2100 (\u00b0C)', 1.5, 3.5, 0.5, 0.1, 0.1, '%.1f'),
    ]

    fig_w, fig_h = 13, 6.5
    panel_inches = 4.5
    pw = panel_inches / fig_w
    ph = panel_inches / fig_h
    cbar_w = 0.015
    gap = 0.02
    bottom = 0.12
    left1 = 0.07
    left2 = left1 + pw + cbar_w + gap + 0.10

    fig = plt.figure(figsize=(fig_w, fig_h))
    plot_axes = []

    max_te_with_data = tax_equity_vals[-1]

    for i, (data, cmap, title, fixed_vmin, fixed_vmax, tick_step, fill_step, contour_step, label_fmt) in enumerate(panels):
        left = left1 if i == 0 else left2
        ax = fig.add_axes([left, bottom, pw, ph])
        plot_axes.append(ax)

        vmin = fixed_vmin
        vmax = fixed_vmax

        fill_levels = np.arange(vmin, vmax + fill_step / 2, fill_step)
        data_plot = np.where(np.isnan(data), np.nan, np.clip(data, vmin, vmax))

        cf = ax.contourf(TE, YDMG, data_plot, levels=fill_levels, cmap=cmap)
        contour_levels = np.arange(vmin, vmax + contour_step / 2, contour_step)
        cs = ax.contour(TE, YDMG, data, levels=contour_levels, colors='black', linewidths=0.3)
        cl = ax.clabel(cs, inline=True, fontsize=7, fmt=label_fmt)

        if i == 0 and cl:
            for txt in cl:
                txt.set_color('white')

        cbar_ax = fig.add_axes([left + pw + gap, bottom, cbar_w, ph])
        cb = fig.colorbar(cf, cax=cbar_ax)
        cb.set_ticks(np.arange(vmin, vmax + tick_step, tick_step))

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect('equal')
        ax.set_xlabel('Tax Equity Parameter (\u03c4)', fontsize=10)
        ax.set_ylabel('Damage Distribution Exponent (\u03b2)', fontsize=10)
        ax.set_title(f'{panel_labels[i]} {title}', fontsize=12)

    # Grey overlay for no-data region
    if max_te_with_data < 1.0:
        grey_img = np.ones((2, 2, 4)) * 0.92
        grey_img[:, :, 3] = 1.0
        for ax in plot_axes:
            ax.imshow(grey_img, extent=[max_te_with_data, 1.0, 0, 1.0],
                      aspect='auto', zorder=100, interpolation='nearest')
            ax.axvline(max_te_with_data, color='black', linewidth=0.5,
                       linestyle='-', zorder=101)

    return fig


def main():
    master_rows, results = load_contour_data()

    tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100 = build_grids(master_rows, results)

    fig = plot_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, 'prod_fig3_contours.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {pdf_path}')


if __name__ == '__main__':
    main()
