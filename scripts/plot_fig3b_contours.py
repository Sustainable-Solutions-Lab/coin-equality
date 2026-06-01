"""
Contour plots for config_055 2D sweep results.

Clean grid: 27 tax_equity points (step 0.04 from 0 to 0.96, plus 0.98 and 0.998)
x 26 y_damage_distribution_exponent points (step 0.04 from 0 to 1.0) = 702 runs.
rho=0.025 (round number per Ken's suggestion).

Output: data/output/config_055/config_055_contour_plots_2panel.pdf
"""

import os
import csv
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output')
OUTPUT_PDF = os.path.join(OUTPUT_DIR, 'config_055', 'config_055_contour_plots_2panel.pdf')


def aggregate_results():
    """Glob all config_055 run_status.csv files and aggregate successful runs."""
    pattern = os.path.join(OUTPUT_DIR, 'config_055', 'config_055_tax-*_ydmg-*', 'run_status.csv')
    status_files = glob.glob(pattern)

    print(f"Found {len(status_files)} run_status.csv files")

    results_by_key = {}
    failures = []
    for status_file in status_files:
        dir_mtime = os.path.getmtime(os.path.dirname(status_file))
        with open(status_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                te = float(row['tax_equity'])
                ydmg = float(row['y_damage_distribution_exponent'])
                key = (round(te, 5), round(ydmg, 5))
                if row['success'] != 'True':
                    failures.append(key)
                    continue
                prev = results_by_key.get(key)
                if prev is None or dir_mtime > prev['_mtime']:
                    row['_mtime'] = dir_mtime
                    results_by_key[key] = row

    n_success = len(results_by_key)
    n_fail = len(set(failures) - set(results_by_key.keys()))
    print(f"Successful unique grid points: {n_success}")
    print(f"Failed unique grid points: {n_fail}")

    if failures:
        failed_te = sorted(set(te for te, _ in set(failures) - set(results_by_key.keys())))
        print(f"Failed tax_equity values: {failed_te}")

    return results_by_key


def build_grids(results):
    """Build 2D arrays for contour plotting from successful results.

    Uses all successful tax_equity values from the data, sorted.
    """
    te_set = sorted(set(round(float(row['tax_equity']), 5) for row in results.values()))
    ydmg_set = sorted(set(round(float(row['y_damage_distribution_exponent']), 5) for row in results.values()))

    tax_equity_vals = np.array(te_set)
    ydmg_vals = np.array(ydmg_set)

    n_te = len(tax_equity_vals)
    n_ydmg = len(ydmg_vals)

    mac_2030 = np.full((n_ydmg, n_te), np.nan)
    delta_t_2100 = np.full((n_ydmg, n_te), np.nan)

    for row in results.values():
        te = float(row['tax_equity'])
        ydmg = float(row['y_damage_distribution_exponent'])

        i_te = np.argmin(np.abs(tax_equity_vals - te))
        j_ydmg = np.argmin(np.abs(ydmg_vals - ydmg))

        mac_2030[j_ydmg, i_te] = float(row['mac_2030'])
        delta_t_2100[j_ydmg, i_te] = float(row['delta_t_2100'])

    n_valid = np.sum(~np.isnan(mac_2030))
    n_total = n_te * n_ydmg
    print(f"Grid: {n_te} x {n_ydmg} = {n_total}")
    print(f"Grid coverage: {n_valid}/{n_total} ({100 * n_valid / n_total:.1f}%)")
    print(f"Tax equity range: {tax_equity_vals[0]:.4f} to {tax_equity_vals[-1]:.4f}")

    # Interpolate missing grid points from neighbours
    from scipy.interpolate import griddata
    TE_grid, YDMG_grid = np.meshgrid(tax_equity_vals, ydmg_vals)
    for grid in [mac_2030, delta_t_2100]:
        mask = ~np.isnan(grid)
        if mask.all():
            continue
        points = np.column_stack([TE_grid[mask], YDMG_grid[mask]])
        values = grid[mask]
        filled = griddata(points, values,
                          (TE_grid, YDMG_grid), method='cubic')
        grid[np.isnan(grid)] = filled[np.isnan(grid)]
    n_filled = np.sum(~np.isnan(mac_2030))
    print(f"After interpolation: {n_filled}/{n_total} ({100 * n_filled / n_total:.1f}%)")

    return tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100


def plot_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100):
    """Create 1x2 filled contour figure with x-axis extending to 1.0."""
    TE, YDMG = np.meshgrid(tax_equity_vals, ydmg_vals)

    panels = [
        (mac_2030, 'viridis', 'Optimal Carbon Price 2030 ($/tCO\u2082)', 50, 350, 50, 10, 20, '%.0f'),
        (delta_t_2100, 'RdYlBu_r', 'Temperature Change 2100 (\u00b0C)', 1.5, 3.5, 0.5, 0.1, 0.1, '%.1f'),
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

    # Find the highest successful tax_equity value for the grey stripe
    max_te_with_data = tax_equity_vals[-1]

    for i, (data, cmap, title, fixed_vmin, fixed_vmax, tick_step, fill_step, contour_step, label_fmt) in enumerate(panels):
        left = left1 if i == 0 else left2
        ax = fig.add_axes([left, bottom, pw, ph])
        plot_axes.append(ax)

        n_valid = np.sum(~np.isnan(data))
        if n_valid < 4:
            ax.set_title(title + f'\n({n_valid} points)')
            continue

        if fixed_vmin is not None:
            vmin = fixed_vmin
            vmax = fixed_vmax
        else:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        fill_levels = np.arange(vmin, vmax + fill_step / 2, fill_step)

        # Clip data to fixed range so colormap maps tightly to [vmin, vmax]
        data_plot = np.where(np.isnan(data), np.nan, np.clip(data, vmin, vmax)) if fixed_vmin is not None else data

        cf = ax.contourf(TE, YDMG, data_plot, levels=fill_levels, cmap=cmap)
        contour_levels = np.arange(vmin, vmax + contour_step / 2, contour_step)
        cs = ax.contour(TE, YDMG, data, levels=contour_levels, colors='black', linewidths=0.3)
        cl = ax.clabel(cs, inline=True, fontsize=7, fmt=label_fmt)

        # In left panel (OCP), make all labels white for visibility
        if i == 0 and cl:
            for txt in cl:
                txt.set_color('white')

        cbar_ax = fig.add_axes([left + pw + gap, bottom, cbar_w, ph])
        cb = fig.colorbar(cf, cax=cbar_ax)
        if tick_step is not None:
            cb.set_ticks(np.arange(vmin, vmax + tick_step, tick_step))

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect('equal')
        ax.set_xlabel('Tax Equity Parameter (\u03c4)', fontsize=10)
        ax.set_ylabel('Damage Distribution Exponent (β)', fontsize=10)
        ax.set_title(title, fontsize=12)

    # Light grey overlay for no-data region beyond highest successful tax_equity
    if max_te_with_data < 1.0:
        grey_img = np.ones((2, 2, 4)) * 0.92
        grey_img[:, :, 3] = 1.0
        for ax in plot_axes:
            ax.imshow(grey_img, extent=[max_te_with_data, 1.0, 0, 1.0], aspect='auto', zorder=100, interpolation='nearest')
            ax.axvline(max_te_with_data, color='black', linewidth=0.5, linestyle='-', zorder=101)

    return fig


def main():
    results = aggregate_results()
    if not results:
        print("No successful results found. Exiting.")
        return

    tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100 = build_grids(results)

    fig = plot_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100)

    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig, dpi=150)

    # Also save to figures directory as PDF and PNG
    figures_dir = os.path.join(PROJECT_ROOT, 'data', 'output', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    fig_pdf = os.path.join(figures_dir, 'fig3b_contour_plots_2panel.pdf')
    fig_png = os.path.join(figures_dir, 'fig3b_contour_plots_2panel.png')
    with PdfPages(fig_pdf) as pdf:
        pdf.savefig(fig, dpi=150)
    fig.savefig(fig_png, dpi=150, bbox_inches='tight')

    # Also save to figures-new directory
    figures_new_dir = os.path.join(PROJECT_ROOT, 'data', 'output', 'figures-new')
    os.makedirs(figures_new_dir, exist_ok=True)
    fig_new_png = os.path.join(figures_new_dir, 'fig3b_contour_plots_2panel.png')
    fig_new_pdf = os.path.join(figures_new_dir, 'fig3b_contour_plots_2panel.pdf')
    fig.savefig(fig_new_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_new_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved: {OUTPUT_PDF}")
    print(f"Saved: {fig_pdf}")
    print(f"Saved: {fig_png}")
    print(f"Saved: {fig_new_png}")


if __name__ == '__main__':
    main()
