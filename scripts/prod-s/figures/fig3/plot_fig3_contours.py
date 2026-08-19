"""
Production Figure 3 (Prod_s): Contour plots for prod_contour_s 2D sweep results.

Grid: 41 tax_equity x 41 y_damage_distribution_exponent = 1681 runs.
Reads directly from output directories in data/output/prod_s/prod_contour_s/.

Output: data/output/prod_s/figures/fig3_contours.pdf
"""

import csv
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 11  # base size: tick labels, legends, colorbar ticks
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
PROD_S_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod_s')
CONTOUR_DIR = os.path.join(PROD_S_OUTPUT, 'prod_contour_s')
FIGURES_DIR = os.path.join(PROD_S_OUTPUT, 'figures')


def load_contour_data():
    """Load contour data from individual run directories."""
    run_dirs = sorted(glob.glob(os.path.join(CONTOUR_DIR, 'prod_contour_s_gid-*')))
    print(f'Found {len(run_dirs)} run directories')

    results = {}
    n_missing = 0
    for d in run_dirs:
        csvs = glob.glob(os.path.join(d, '*_results.csv'))
        jsons = glob.glob(os.path.join(d, '*.json'))
        if not csvs or not jsons:
            n_missing += 1
            continue

        cfg = json.load(open(jsons[0]))
        tax = cfg['scalar_parameters']['tax_equity']
        beta = cfg['scalar_parameters']['y_damage_distribution_exponent']

        import pandas as pd
        df = pd.read_csv(csvs[0])
        t_col = [c for c in df.columns if c.startswith('t,')][0]
        dt_col = [c for c in df.columns if c.startswith('delta_T,')][0]

        r2030 = df[df[t_col] == 2030.0].iloc[0]
        r2100 = df[df[t_col] == 2100.0].iloc[0]

        gid_str = os.path.basename(d).split('_gid-')[1].split('_')[0]
        gid = int(gid_str)
        results[gid] = {
            'tax_equity': tax,
            'y_damage_distribution_exponent': beta,
            'mac_2030': r2030['marginal_abatement_cost'],
            'delta_t_2100': r2100[dt_col],
        }

    print(f'  Loaded: {len(results)}')
    print(f'  Missing results: {n_missing}')
    return results


def build_grids(results):
    """Build 2D arrays for contour plotting."""
    te_set = sorted(set(r['tax_equity'] for r in results.values()))
    ydmg_set = sorted(set(r['y_damage_distribution_exponent'] for r in results.values()))

    tax_equity_vals = np.array(te_set)
    ydmg_vals = np.array(ydmg_set)
    n_te = len(tax_equity_vals)
    n_ydmg = len(ydmg_vals)

    mac_2030 = np.full((n_ydmg, n_te), np.nan)
    delta_t_2100 = np.full((n_ydmg, n_te), np.nan)

    for gid, row in results.items():
        te = row['tax_equity']
        ydmg = row['y_damage_distribution_exponent']
        i_te = np.argmin(np.abs(tax_equity_vals - te))
        j_ydmg = np.argmin(np.abs(ydmg_vals - ydmg))
        mac_2030[j_ydmg, i_te] = row['mac_2030']
        delta_t_2100[j_ydmg, i_te] = row['delta_t_2100']

    n_valid = np.sum(~np.isnan(mac_2030))
    n_total = n_te * n_ydmg
    print(f'Grid: {n_te} x {n_ydmg} = {n_total}')
    print(f'Grid coverage: {n_valid}/{n_total} ({100 * n_valid / n_total:.1f}%)')
    print(f'Tax equity range: {tax_equity_vals[0]:.4f} to {tax_equity_vals[-1]:.4f}')
    print(f'Beta range: {ydmg_vals[0]:.4f} to {ydmg_vals[-1]:.4f}')

    TE_grid, YDMG_grid = np.meshgrid(tax_equity_vals, ydmg_vals)
    for data in [mac_2030, delta_t_2100]:
        mask = ~np.isnan(data)
        if mask.all():
            continue
        points = np.column_stack([TE_grid[mask], YDMG_grid[mask]])
        values = data[mask]
        filled_cubic = griddata(points, values, (TE_grid, YDMG_grid), method='cubic')
        data[np.isnan(data)] = filled_cubic[np.isnan(data)]
        still_nan = np.isnan(data)
        if still_nan.any():
            filled_nn = griddata(points, values, (TE_grid, YDMG_grid), method='nearest')
            data[still_nan] = filled_nn[still_nan]

    n_filled = np.sum(~np.isnan(mac_2030))
    print(f'After interpolation: {n_filled}/{n_total} ({100 * n_filled / n_total:.1f}%)')

    return tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100


def plot_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100, smooth_sigma=0):
    """Create 1x2 filled contour figure.

    Interpolates to a 400x400 fine grid for smooth contour lines.
    """
    n_fine = 400
    te_fine = np.linspace(tax_equity_vals[0], tax_equity_vals[-1], n_fine)
    ydmg_fine = np.linspace(ydmg_vals[0], ydmg_vals[-1], n_fine)
    TE_coarse, YDMG_coarse = np.meshgrid(tax_equity_vals, ydmg_vals)
    TE, YDMG = np.meshgrid(te_fine, ydmg_fine)

    coarse_points = np.column_stack([TE_coarse.ravel(), YDMG_coarse.ravel()])
    mac_fine = griddata(coarse_points, mac_2030.ravel(), (TE, YDMG), method='cubic')
    dt_fine = griddata(coarse_points, delta_t_2100.ravel(), (TE, YDMG), method='cubic')

    if smooth_sigma > 0:
        mac_fine = gaussian_filter(mac_fine, sigma=smooth_sigma)
        dt_fine = gaussian_filter(dt_fine, sigma=smooth_sigma)

    panel_labels = ['A)', 'B)']
    panels = [
        (mac_fine, 'viridis', 'Optimal Carbon Price in 2030 ($/tCO\u2082)', 50, 350, 50, 10, 20, '%.0f'),
        (dt_fine, 'RdYlBu_r', 'Temperature Change in 2100 (\u00b0C)', 1.5, 3.5, 0.5, 0.1, 0.1, '%.1f'),
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

        if i == 0:
            # Panel A: place labels centrally along each contour (approved)
            margin = 0.05
            manual_positions = []
            for level_idx in range(len(cs.levels)):
                segments = cs.allsegs[level_idx]
                best_pos = None
                best_dist = -1
                for seg in segments:
                    if len(seg) < 2:
                        continue
                    edge_dist = np.minimum(
                        np.minimum(seg[:, 0], max_te_with_data - seg[:, 0]),
                        np.minimum(seg[:, 1], 1.0 - seg[:, 1])
                    )
                    interior = edge_dist > margin
                    if interior.any():
                        idx = np.where(interior)[0]
                        mid = idx[len(idx) // 2]
                        if edge_dist[mid] > best_dist:
                            best_dist = edge_dist[mid]
                            best_pos = (seg[mid, 0], seg[mid, 1])
                if best_pos is not None:
                    manual_positions.append(best_pos)
            if manual_positions:
                cl = ax.clabel(cs, inline=True, fontsize=10, fmt=label_fmt, manual=manual_positions)
            else:
                cl = ax.clabel(cs, inline=True, fontsize=10, fmt=label_fmt)
        else:
            # Panel B: place labels centrally, skip contours too small for a label
            margin = 0.08
            min_extent = 0.05
            manual_positions = []
            for level_idx in range(len(cs.levels)):
                segments = cs.allsegs[level_idx]
                best_pos = None
                best_dist = -1
                fallback_pos = None
                fallback_dist = -1
                for seg in segments:
                    if len(seg) < 2:
                        continue
                    seg_extent = max(np.ptp(seg[:, 0]), np.ptp(seg[:, 1]))
                    if seg_extent < min_extent:
                        continue
                    edge_dist = np.minimum(
                        np.minimum(seg[:, 0], max_te_with_data - seg[:, 0]),
                        np.minimum(seg[:, 1], 1.0 - seg[:, 1])
                    )
                    # Track fallback: point with max edge distance on any segment
                    seg_best_idx = np.argmax(edge_dist)
                    if edge_dist[seg_best_idx] > fallback_dist:
                        fallback_dist = edge_dist[seg_best_idx]
                        fallback_pos = (seg[seg_best_idx, 0], seg[seg_best_idx, 1])
                    # Prefer points above margin threshold
                    interior = edge_dist > margin
                    if interior.any():
                        idx = np.where(interior)[0]
                        mid = idx[len(idx) // 2]
                        if edge_dist[mid] > best_dist:
                            best_dist = edge_dist[mid]
                            best_pos = (seg[mid, 0], seg[mid, 1])
                if best_pos is not None:
                    manual_positions.append(best_pos)
                elif fallback_pos is not None:
                    manual_positions.append(fallback_pos)
            if manual_positions:
                cl = ax.clabel(cs, inline=True, fontsize=10, fmt=label_fmt, manual=manual_positions)
            else:
                cl = ax.clabel(cs, inline=True, fontsize=10, fmt=label_fmt)

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
        ax.set_xlabel('Tax Progressivity Parameter (\u03c4)', fontsize=13)
        ax.set_ylabel('Climate Damage Vulnerability Exponent (\u03b2)', fontsize=13)
        ax.set_title(f'{panel_labels[i]} {title}', fontsize=14)

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
    results = load_contour_data()
    tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100 = build_grids(results)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # No smoothing (current version)
    fig = plot_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100, smooth_sigma=0)
    pdf_path = os.path.join(FIGURES_DIR, 'fig3_contours.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {pdf_path}')

    # Light smoothing for comparison
    fig = plot_contours(tax_equity_vals, ydmg_vals, mac_2030, delta_t_2100, smooth_sigma=3)
    pdf_path_s = os.path.join(FIGURES_DIR, 'fig3_contours_smooth.pdf')
    fig.savefig(pdf_path_s, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf_path_s}')


if __name__ == '__main__':
    main()
