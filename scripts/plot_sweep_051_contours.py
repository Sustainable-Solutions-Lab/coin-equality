"""
Contour plots for config_051 2D sensitivity sweep results.

Aggregates run_status.csv files from all config_051 output directories,
produces 1x3 filled contour plots of OCP@2030, OCP@2100, and delta_T@2100.

Output: data/output/config_051_contour_plots.pdf
"""

import os
import csv
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output')
OUTPUT_PDF = os.path.join(OUTPUT_DIR, 'config_051_contour_plots.pdf')

N_LEVELS = 20


def aggregate_results():
    """Glob all config_051 run_status.csv files and aggregate successful runs."""
    pattern = os.path.join(OUTPUT_DIR, 'config_051_tax-*_ydmg-*', 'run_status.csv')
    status_files = glob.glob(pattern)

    print(f"Found {len(status_files)} run_status.csv files")

    # Collect all rows, keyed by grid_id (latest timestamp wins for duplicates)
    results_by_grid_id = {}
    for status_file in status_files:
        dir_mtime = os.path.getmtime(os.path.dirname(status_file))
        with open(status_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                grid_id = int(row['grid_id'])
                if row['success'] != 'True':
                    continue
                prev = results_by_grid_id.get(grid_id)
                if prev is None or dir_mtime > prev['_mtime']:
                    row['_mtime'] = dir_mtime
                    results_by_grid_id[grid_id] = row

    n_success = len(results_by_grid_id)
    print(f"Successful unique grid points: {n_success}/441")

    if n_success < 441:
        all_ids = set(range(1, 442))
        missing = sorted(all_ids - set(results_by_grid_id.keys()))
        print(f"Missing grid IDs ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    return results_by_grid_id


def build_grids(results):
    """Build 2D arrays for contour plotting from aggregated results."""
    tax_equity_vals = np.linspace(0.00, 0.95, 21)
    ydmg_vals = np.linspace(0.00, 0.41, 21)

    mac_2030 = np.full((21, 21), np.nan)
    mac_2100 = np.full((21, 21), np.nan)
    delta_t_2100 = np.full((21, 21), np.nan)

    for row in results.values():
        te = float(row['tax_equity'])
        ydmg = float(row['y_damage_distribution_exponent'])

        # Find nearest grid indices
        i_te = np.argmin(np.abs(tax_equity_vals - te))
        j_ydmg = np.argmin(np.abs(ydmg_vals - ydmg))

        mac_2030[j_ydmg, i_te] = float(row['mac_2030'])
        mac_2100[j_ydmg, i_te] = float(row['mac_2100'])
        delta_t_2100[j_ydmg, i_te] = float(row['delta_t_2100'])

    n_valid = np.sum(~np.isnan(mac_2030))
    print(f"Grid coverage: {n_valid}/441 ({100 * n_valid / 441:.1f}%)")

    return tax_equity_vals, ydmg_vals, mac_2030, mac_2100, delta_t_2100


def plot_contours(tax_equity_vals, ydmg_vals, mac_2030, mac_2100, delta_t_2100):
    """Create 1x3 filled contour figure."""
    TE, YDMG = np.meshgrid(tax_equity_vals, ydmg_vals)

    panels = [
        (mac_2030, 'viridis', 'Optimal Carbon Price 2030 ($/tCO2)'),
        (mac_2100, 'viridis', 'Optimal Carbon Price 2100 ($/tCO2)'),
        (delta_t_2100, 'RdYlBu_r', 'Temperature Change 2100 (\u00b0C)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (data, cmap, title) in zip(axes, panels):
        n_valid = np.sum(~np.isnan(data))
        if n_valid < 4:
            ax.set_title(title + f'\n({n_valid} points — need >= 4)')
            continue

        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        levels = np.linspace(vmin, vmax, N_LEVELS + 1)

        cf = ax.contourf(TE, YDMG, data, levels=levels, cmap=cmap)
        cs = ax.contour(TE, YDMG, data, levels=levels, colors='black', linewidths=0.3)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f' if vmax > 10 else '%.2f')

        fig.colorbar(cf, ax=ax, shrink=0.85)

        ax.set_xlabel('Tax Equity Parameter')
        ax.set_ylabel('Damage Distribution Exponent')
        ax.set_title(title)

    fig.tight_layout()
    return fig


def main():
    results = aggregate_results()
    if not results:
        print("No successful results found. Exiting.")
        return

    tax_equity_vals, ydmg_vals, mac_2030, mac_2100, delta_t_2100 = build_grids(results)

    fig = plot_contours(tax_equity_vals, ydmg_vals, mac_2030, mac_2100, delta_t_2100)

    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)

    print(f"\nSaved: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
