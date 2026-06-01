"""
Warmstart comparison for all fig5 variants.

Single multi-page PDF. Each page = one fig5 variant with:
  Top:    original data (before warmstart)
  Bottom: merged data (after warmstart)

Usage:
    python scripts/plot_fig5_warmstart_comparison.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from PyPDF2 import PdfMerger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR_061 = PROJECT_ROOT / 'data' / 'output' / 'config_061'
OUTPUT_DIR_058 = PROJECT_ROOT / 'data' / 'output' / 'config_058'
OUTPUT_DIR_061_WS = PROJECT_ROOT / 'data' / 'output' / 'config_061_warmstart_loose'
GRID_CSV = PROJECT_ROOT / 'json' / 'config_061_grid.csv'
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'figures-new'

TAX_EQUITY_ORDER = [0.0, 0.5, 0.998]
TAX_EQUITY_LABELS = {0.0: 'Flat tax', 0.5: 'Progressive tax', 0.998: 'Highly progressive tax'}
TAX_EQUITY_COLORS = {0.0: '#2C3E50', 0.5: '#d62728', 0.998: '#1f77b4'}

SWEEP_ORDER = ['all', 'rho', 'beta', 'eta']
SWEEP_TITLES = {
    'eta': 'Elasticity of\nmarginal utility (\u03b7)',
    'rho': 'Effective rate of pure\ntime preference (\u03c1)',
    'beta': 'Damage distribution\nexponent (\u03b2)',
    'all': 'All parameters\n(combined)',
}
SWEEP_HATCHES = {'all': '', 'rho': '///', 'beta': '\\\\\\', 'eta': '---'}

METRICS = [
    ('mac_2030', 'Optimal Carbon Price in 2030 ($/tCO\u2082)', 0, 500),
    ('delta_t_2100', 'Temperature Change in 2100 (\u00b0C)', 1, 4),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results_061_original():
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


def load_results_058_original():
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
    grid = {}
    with open(GRID_CSV, 'r') as f:
        for row in csv.DictReader(f):
            grid[int(row['grid_id'])] = row
    return grid


def load_results_061_warmstart_only():
    """Load only warmstart_loose results, best objective per grid_id."""
    results = {}
    if not OUTPUT_DIR_061_WS.exists():
        return results
    for entry in sorted(OUTPUT_DIR_061_WS.iterdir()):
        status = entry / 'run_status.csv'
        if not status.exists():
            continue
        with open(status, 'r') as f:
            for row in csv.DictReader(f):
                if row['success'] == 'True':
                    gid = int(row['grid_id'])
                    obj = float(row['objective_value'])
                    if gid not in results or obj < float(results[gid]['objective_value']):
                        results[gid] = row
    return results


# Map sweep type -> the grid CSV column that varies
SWEEP_PARAM_COL = {
    'eta': 'eta',
    'rho': 'rho',
    'beta': 'y_damage_distribution_exponent',
}

NEIGHBOR_K = 15  # neighbors on each side for rolling median


def select_by_neighbors(orig_061, ws_061, grid):
    """Select warmstart results only if they bring the point closer to the
    local trend of its neighbors.

    For each (sweep, tax_equity) group:
      1. Sort the 1000 points by the sweep parameter.
      2. Compute a rolling-median reference curve from the original data
         (window = 2*NEIGHBOR_K+1, excluding the center point).
      3. For each warmstart candidate, compute the normalized distance of
         original vs warmstart to this reference.
      4. Accept warmstart only if its distance is strictly smaller.

    Returns: (selected_061 dict, selection_report list of dicts)
    """
    selected = dict(orig_061)  # start with all original
    report = []

    # Group grid_ids by (sweep, tax_equity)
    groups = defaultdict(list)
    for gid, g in grid.items():
        if gid in orig_061:
            groups[(g['sweep'], float(g['tax_equity']))].append(gid)

    for (sweep, te), gids in sorted(groups.items()):
        param_col = SWEEP_PARAM_COL[sweep]
        # Sort by parameter value
        gids_sorted = sorted(gids, key=lambda g: float(grid[g][param_col]))
        n = len(gids_sorted)

        # Extract original metric arrays (in sorted order)
        orig_mac = np.array([float(orig_061[g]['mac_2030']) for g in gids_sorted])
        orig_dt = np.array([float(orig_061[g]['delta_t_2100']) for g in gids_sorted])

        # Compute scale factors (IQR) for normalization
        mac_iqr = np.percentile(orig_mac, 75) - np.percentile(orig_mac, 25)
        dt_iqr = np.percentile(orig_dt, 75) - np.percentile(orig_dt, 25)
        mac_scale = mac_iqr if mac_iqr > 0 else 1.0
        dt_scale = dt_iqr if dt_iqr > 0 else 1.0

        # Compute rolling median reference (excluding center point)
        ref_mac = np.empty(n)
        ref_dt = np.empty(n)
        for i in range(n):
            lo = max(0, i - NEIGHBOR_K)
            hi = min(n, i + NEIGHBOR_K + 1)
            # Exclude the point itself
            neighbor_idx = [j for j in range(lo, hi) if j != i]
            ref_mac[i] = np.median(orig_mac[neighbor_idx])
            ref_dt[i] = np.median(orig_dt[neighbor_idx])

        # Build index map: grid_id -> position in sorted array
        gid_to_idx = {g: i for i, g in enumerate(gids_sorted)}

        # Check each warmstart candidate
        for gid in gids_sorted:
            if gid not in ws_061:
                continue
            idx = gid_to_idx[gid]
            ws_row = ws_061[gid]

            ws_mac = float(ws_row['mac_2030'])
            ws_dt = float(ws_row['delta_t_2100'])
            o_mac = orig_mac[idx]
            o_dt = orig_dt[idx]

            # Normalized distance to reference
            dist_orig = np.sqrt(((o_mac - ref_mac[idx]) / mac_scale) ** 2 +
                                ((o_dt - ref_dt[idx]) / dt_scale) ** 2)
            dist_ws = np.sqrt(((ws_mac - ref_mac[idx]) / mac_scale) ** 2 +
                              ((ws_dt - ref_dt[idx]) / dt_scale) ** 2)

            accepted = dist_ws < dist_orig
            if accepted:
                ws_row_copy = dict(ws_row)
                ws_row_copy['source'] = 'warmstart_neighbor_selected'
                selected[gid] = ws_row_copy

            report.append({
                'grid_id': gid,
                'sweep': sweep,
                'tax_equity': te,
                'param_val': float(grid[gid][param_col]),
                'orig_mac': o_mac,
                'orig_dt': o_dt,
                'ws_mac': ws_mac,
                'ws_dt': ws_dt,
                'ref_mac': ref_mac[idx],
                'ref_dt': ref_dt[idx],
                'dist_orig': dist_orig,
                'dist_ws': dist_ws,
                'accepted': accepted,
            })

    return selected, report


MAC_OUTLIER_THRESHOLD = 15.0   # absolute $/tCO2 deviation from rolling median
DT_OUTLIER_THRESHOLD = 0.08    # absolute °C deviation from rolling median


def exclude_outliers(selected_061, grid):
    """Remove points that are still far from the local trend after warmstart selection.

    Uses absolute thresholds on the residual from the rolling median:
      - |mac_2030 - ref| > MAC_OUTLIER_THRESHOLD, or
      - |delta_t_2100 - ref| > DT_OUTLIER_THRESHOLD

    These correspond to ~3% of each axis range, the level at which individual
    points become visually distinguishable from the curve in the scatter plot.

    Returns: (cleaned dict, list of excluded grid_ids with details)
    """
    cleaned = dict(selected_061)
    excluded = []

    groups = defaultdict(list)
    for gid, g in grid.items():
        if gid in cleaned:
            groups[(g['sweep'], float(g['tax_equity']))].append(gid)

    for (sweep, te), gids in sorted(groups.items()):
        param_col = SWEEP_PARAM_COL[sweep]
        gids_sorted = sorted(gids, key=lambda g: float(grid[g][param_col]))
        n = len(gids_sorted)

        mac = np.array([float(cleaned[g]['mac_2030']) for g in gids_sorted])
        dt = np.array([float(cleaned[g]['delta_t_2100']) for g in gids_sorted])

        ref_mac = np.empty(n)
        ref_dt = np.empty(n)
        for i in range(n):
            lo = max(0, i - NEIGHBOR_K)
            hi = min(n, i + NEIGHBOR_K + 1)
            nbr = [j for j in range(lo, hi) if j != i]
            ref_mac[i] = np.median(mac[nbr])
            ref_dt[i] = np.median(dt[nbr])

        for i in range(n):
            resid_mac = abs(mac[i] - ref_mac[i])
            resid_dt = abs(dt[i] - ref_dt[i])
            if resid_mac > MAC_OUTLIER_THRESHOLD or resid_dt > DT_OUTLIER_THRESHOLD:
                gid = gids_sorted[i]
                excluded.append({
                    'grid_id': gid,
                    'sweep': sweep,
                    'tax_equity': te,
                    'param_val': float(grid[gid][param_col]),
                    'mac': mac[i],
                    'dt': dt[i],
                    'resid_mac': mac[i] - ref_mac[i],
                    'resid_dt': dt[i] - ref_dt[i],
                })
                del cleaned[gid]

    return cleaned, excluded


def group_by_sweep_tax(results_061, results_058, grid):
    by_sweep_tax = defaultdict(list)
    for gid, r in results_061.items():
        g = grid[gid]
        by_sweep_tax[(g['sweep'], float(g['tax_equity']))].append(r)
    for r in results_058.values():
        by_sweep_tax[('all', float(r['tax_equity']))].append(r)
    return by_sweep_tax


# ---------------------------------------------------------------------------
# Render functions — each draws into provided axes using given data
# ---------------------------------------------------------------------------
def render_scatter(axes, data):
    for col, sweep in enumerate(SWEEP_ORDER):
        ax = axes[col]
        for te_val in TAX_EQUITY_ORDER:
            runs = data.get((sweep, te_val), [])
            mac = np.array([float(r['mac_2030']) for r in runs])
            dt = np.array([float(r['delta_t_2100']) for r in runs])
            ax.scatter(mac, dt, c=TAX_EQUITY_COLORS[te_val], s=8, alpha=0.3,
                       edgecolors='none', rasterized=True)
        ax.set_title(SWEEP_TITLES[sweep], fontsize=10, weight='bold')
        ax.set_xlabel('OCP in 2030 ($/tCO\u2082)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Temperature Change in 2100 (\u00b0C)', fontsize=9)
        ax.set_xlim(0, 500)
        ax.set_ylim(1, 4)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)


def render_ridgeline(axes, data):
    row_spacing = 1.0
    kde_height = 0.8
    for panel_idx, (metric_key, metric_label, xmin, xmax) in enumerate(METRICS):
        ax = axes[panel_idx]
        x_grid = np.linspace(xmin, xmax, 500)
        for row_idx, sweep in enumerate(reversed(SWEEP_ORDER)):
            y_base = row_idx * row_spacing
            for te_val in TAX_EQUITY_ORDER:
                vals = np.array([float(r[metric_key])
                                 for r in data.get((sweep, te_val), [])])
                if len(vals) < 3:
                    continue
                kde = gaussian_kde(vals, bw_method=0.15)
                density = kde(x_grid)
                density_scaled = density / density.max() * kde_height
                ax.fill_between(x_grid, y_base, y_base + density_scaled,
                                color=TAX_EQUITY_COLORS[te_val], alpha=0.4, linewidth=0)
                ax.plot(x_grid, y_base + density_scaled,
                        color=TAX_EQUITY_COLORS[te_val], linewidth=1.2)
                median_val = np.median(vals)
                median_density = kde(np.array([median_val]))[0]
                median_height = median_density / density.max() * kde_height
                ax.vlines(median_val, y_base, y_base + median_height,
                          color=TAX_EQUITY_COLORS[te_val], linewidth=1.5,
                          linestyle='--', alpha=0.8)
            ax.axhline(y_base, color='grey', linewidth=0.3, alpha=0.5)
        sweep_labels = list(reversed(SWEEP_ORDER))
        ax.set_yticks([i * row_spacing for i in range(len(SWEEP_ORDER))])
        ax.set_yticklabels([SWEEP_TITLES[s] for s in sweep_labels], fontsize=9)
        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.2, len(SWEEP_ORDER) * row_spacing + 0.1)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def render_boxjitter(axes, data):
    rng = np.random.default_rng(42)
    group_spacing = 1.6
    box_spacing = 0.28
    box_width = 0.22
    n_sweeps = len(SWEEP_ORDER)
    for panel_idx, (metric_key, metric_label, _, _) in enumerate(METRICS):
        ax = axes[panel_idx]
        group_centers = []
        for gi, te_val in enumerate(TAX_EQUITY_ORDER):
            group_center = gi * group_spacing
            group_centers.append(group_center)
            all_vals = []
            all_positions = []
            for si, sweep in enumerate(SWEEP_ORDER):
                runs = data.get((sweep, te_val), [])
                vals = np.array([float(r[metric_key]) for r in runs])
                if len(vals) < 3:
                    continue
                pos = group_center + (si - (n_sweeps - 1) / 2) * box_spacing
                all_vals.append(vals)
                all_positions.append(pos)
                jitter = rng.uniform(-box_width * 0.35, box_width * 0.35, size=len(vals))
                ax.scatter(pos + jitter, vals, c=TAX_EQUITY_COLORS[te_val],
                           s=3, alpha=0.12, edgecolors='none', zorder=1, rasterized=True)
            if all_vals:
                bp = ax.boxplot(all_vals, positions=all_positions, widths=box_width,
                                showfliers=False, patch_artist=True, zorder=2,
                                medianprops=dict(color='white', linewidth=1.5),
                                boxprops=dict(linewidth=0.8),
                                whiskerprops=dict(linewidth=0.8),
                                capprops=dict(linewidth=0.8))
                for si, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(TAX_EQUITY_COLORS[te_val])
                    patch.set_alpha(0.7)
                    patch.set_hatch(SWEEP_HATCHES[SWEEP_ORDER[si]])
        ax.set_title(metric_label, fontsize=10, weight='bold')
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        if panel_idx == 0:
            ax.set_ylim(0, 500)
        else:
            ax.set_ylim(1, 4)
        x_pad = 0.7
        ax.set_xlim(min(group_centers) - x_pad, max(group_centers) + x_pad)


def _draw_left_half_violin(ax, vals, center, max_width, color, alpha_fill, edge_lw, hatch):
    kde = gaussian_kde(vals, bw_method=0.2)
    y_grid = np.linspace(vals.min(), vals.max(), 300)
    density = kde(y_grid)
    density = density / density.max() * max_width
    x_coords = np.concatenate([[center], center - density, [center]])
    y_coords = np.concatenate([[y_grid[0]], y_grid, [y_grid[-1]]])
    verts = np.column_stack([x_coords, y_coords])
    poly = plt.Polygon(verts, closed=True, facecolor=color, edgecolor='#444444',
                        linewidth=edge_lw, alpha=alpha_fill, zorder=2)
    ax.add_patch(poly)
    if hatch:
        poly_h = plt.Polygon(verts, closed=True, facecolor='none', edgecolor='#666666',
                              linewidth=0.0, hatch=hatch, alpha=0.35, zorder=3)
        ax.add_patch(poly_h)
    median = np.median(vals)
    q25, q75 = np.percentile(vals, [25, 75])
    stem_offset = max_width * 0.08
    ax.plot([center + stem_offset, center + stem_offset], [q25, q75],
            color='#333333', linewidth=0.8, solid_capstyle='butt', zorder=5)
    ax.scatter([center + stem_offset], [median], c='white', edgecolors='#222222',
               s=18, linewidths=0.7, zorder=6)


def render_raincloud(axes, data):
    group_spacing = 1.8
    violin_spacing = 0.27
    n_sweeps = len(SWEEP_ORDER)
    sweep_width = {'all': 0.26, 'rho': 0.18, 'beta': 0.18, 'eta': 0.18}
    sweep_alpha = {'all': 0.50, 'rho': 0.40, 'beta': 0.40, 'eta': 0.40}
    sweep_edge_lw = {'all': 0.6, 'rho': 0.4, 'beta': 0.4, 'eta': 0.4}
    for panel_idx, (metric_key, metric_label, _, _) in enumerate(METRICS):
        ax = axes[panel_idx]
        group_centers = []
        for gi, te_val in enumerate(TAX_EQUITY_ORDER):
            group_center = gi * group_spacing
            for si, sweep in enumerate(SWEEP_ORDER):
                pos = group_center + (si - (n_sweeps - 1) / 2) * violin_spacing
                runs = data.get((sweep, te_val), [])
                vals = np.array([float(r[metric_key]) for r in runs])
                if len(vals) < 3:
                    continue
                _draw_left_half_violin(ax, vals, pos, sweep_width[sweep],
                                       TAX_EQUITY_COLORS[te_val],
                                       sweep_alpha[sweep], sweep_edge_lw[sweep],
                                       SWEEP_HATCHES[sweep])
            group_centers.append(group_center)
        ax.set_title(metric_label, fontsize=10, weight='bold')
        ax.set_xticks([])
        ax.yaxis.grid(True, alpha=0.15, linewidth=0.4, color='#999999', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if panel_idx == 0:
            ax.set_ylim(0, 500)
        else:
            ax.set_ylim(1, 4)
        x_pad = 0.8
        ax.set_xlim(min(group_centers) - x_pad, max(group_centers) + x_pad)


def _confidence_ellipse(x, y, ax, n_std, color, linestyle):
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                       angle=angle, facecolor='none', edgecolor=color,
                       linewidth=1.0, linestyle=linestyle, alpha=0.7)
    ax.add_patch(ellipse)


def render_frontier(axes, data):
    for col, sweep in enumerate(SWEEP_ORDER):
        ax = axes[col]
        for te_val in TAX_EQUITY_ORDER:
            color = TAX_EQUITY_COLORS[te_val]
            runs = data.get((sweep, te_val), [])
            mac = np.array([float(r['mac_2030']) for r in runs])
            dt = np.array([float(r['delta_t_2100']) for r in runs])
            ax.scatter(mac, dt, c=color, s=4, alpha=0.15,
                       edgecolors='none', rasterized=True, zorder=1)
            if len(mac) > 3:
                _confidence_ellipse(mac, dt, ax, 1.645, color, '-')
                _confidence_ellipse(mac, dt, ax, 0.674, color, '--')
                ax.scatter([np.median(mac)], [np.median(dt)],
                           c='white', edgecolors=color, s=40, linewidths=1.2,
                           zorder=5, marker='o')
        ax.set_title(SWEEP_TITLES[sweep], fontsize=9, weight='bold', pad=8)
        ax.set_xlabel('OCP in 2030 ($/tCO\u2082)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Temperature Change in 2100 (\u00b0C)', fontsize=9)
        ax.set_xlim(0, 500)
        ax.set_ylim(1, 4)
        ax.grid(True, alpha=0.12, linewidth=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def render_violins(axes, data):
    for col, sweep in enumerate(SWEEP_ORDER):
        for row, (metric_key, metric_label, _, _) in enumerate(METRICS):
            ax = axes[row, col]
            violin_data = []
            positions = []
            colors = []
            for i, te_val in enumerate(TAX_EQUITY_ORDER):
                vals = np.array([float(r[metric_key])
                                 for r in data.get((sweep, te_val), [])])
                violin_data.append(vals)
                positions.append(i)
                colors.append(TAX_EQUITY_COLORS[te_val])
            if all(len(v) >= 3 for v in violin_data):
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


# ---------------------------------------------------------------------------
# Page builders — each creates a figure with top=original, bottom=merged
# ---------------------------------------------------------------------------
VARIANT_CONFIGS = [
    {
        'name': 'Scatter',
        'render': render_scatter,
        'ncols': 4, 'nrows': 1,
        'page_w': 18, 'panel_h': 5,
    },
    {
        'name': 'Ridgeline',
        'render': render_ridgeline,
        'ncols': 2, 'nrows': 1,
        'page_w': 16, 'panel_h': 7,
    },
    {
        'name': 'Box + Jitter',
        'render': render_boxjitter,
        'ncols': 2, 'nrows': 1,
        'page_w': 14, 'panel_h': 6,
    },
    {
        'name': 'Raincloud',
        'render': render_raincloud,
        'ncols': 2, 'nrows': 1,
        'page_w': 13, 'panel_h': 5.5,
    },
    {
        'name': 'Frontier',
        'render': render_frontier,
        'ncols': 4, 'nrows': 1,
        'page_w': 18, 'panel_h': 4.5,
    },
    {
        'name': 'Violins',
        'render': render_violins,
        'ncols': 4, 'nrows': 2,
        'page_w': 16, 'panel_h': 8,
    },
]


def build_page(variant, orig_data, merged_data):
    """Build one figure with top=original, bottom=merged."""
    render_fn = variant['render']
    ncols = variant['ncols']
    nrows = variant['nrows']
    page_w = variant['page_w']
    panel_h = variant['panel_h']

    total_rows = nrows * 2
    total_h = panel_h * 2 + 2.5

    fig, all_axes = plt.subplots(total_rows, ncols, figsize=(page_w, total_h))

    # Ensure 2D array even for single-row variants
    if total_rows == 2 and ncols > 1:
        axes_top = all_axes[0:1, :]  # shape (1, ncols) — but render expects (ncols,) for nrows=1
        axes_bot = all_axes[1:2, :]
        # For nrows=1 render functions, pass 1D array
        axes_top = all_axes[0]
        axes_bot = all_axes[1]
    else:
        # nrows=2 case (violins): top = rows 0-1, bottom = rows 2-3
        axes_top = all_axes[0:nrows, :]
        axes_bot = all_axes[nrows:, :]

    render_fn(axes_top, orig_data)
    render_fn(axes_bot, merged_data)

    # Row labels — placed in figure coordinates
    mid_top = 0.97
    mid_bot = 0.50
    fig.text(0.5, mid_top, 'Before warmstart (original)', fontsize=11,
             weight='bold', color='#555555', ha='center', va='bottom')
    fig.text(0.5, mid_bot, 'After warmstart (neighbor-selected)', fontsize=11,
             weight='bold', color='#555555', ha='center', va='bottom')

    fig.suptitle(variant['name'], fontsize=15, weight='bold', y=1.01)
    fig.subplots_adjust(hspace=0.55, wspace=0.3, top=0.93, bottom=0.05)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading original results...')
    orig_061 = load_results_061_original()
    orig_058 = load_results_058_original()
    grid = load_grid()
    print(f'  Original: {len(orig_061)} config_061, {len(orig_058)} config_058')

    print('Loading warmstart candidates...')
    ws_061 = load_results_061_warmstart_only()
    print(f'  Warmstart candidates: {len(ws_061)} config_061')

    print(f'\nSelecting by neighbor proximity (k={NEIGHBOR_K})...')
    selected_061, report = select_by_neighbors(orig_061, ws_061, grid)

    n_candidates = len(report)
    n_accepted = sum(1 for r in report if r['accepted'])
    n_rejected = n_candidates - n_accepted
    print(f'  Candidates evaluated: {n_candidates}')
    print(f'  Accepted (closer to curve): {n_accepted}')
    print(f'  Rejected (original was better): {n_rejected}')

    # Breakdown by (sweep, tax_equity)
    by_group = defaultdict(lambda: [0, 0])
    for r in report:
        key = (r['sweep'], r['tax_equity'])
        by_group[key][0] += 1
        if r['accepted']:
            by_group[key][1] += 1
    print(f'\n  Per group:')
    for key in sorted(by_group):
        total, acc = by_group[key]
        print(f'    {key}: {acc}/{total} accepted')

    # Show largest improvements
    accepted_report = sorted([r for r in report if r['accepted']],
                             key=lambda r: r['dist_orig'] - r['dist_ws'],
                             reverse=True)
    if accepted_report:
        print(f'\n  Top 5 largest improvements (by distance reduction):')
        for r in accepted_report[:5]:
            print(f'    gid={r["grid_id"]:5d}  {r["sweep"]:4s} te={r["tax_equity"]:.3f}  '
                  f'dist: {r["dist_orig"]:.3f} -> {r["dist_ws"]:.3f}  '
                  f'mac: {r["orig_mac"]:.1f} -> {r["ws_mac"]:.1f}  '
                  f'dT: {r["orig_dt"]:.3f} -> {r["ws_dt"]:.3f}')

    # Outlier exclusion — remove points still far from curve
    print(f'\nExcluding remaining outliers (|resid_mac| > {MAC_OUTLIER_THRESHOLD} or |resid_dt| > {DT_OUTLIER_THRESHOLD})...')
    cleaned_061, excluded = exclude_outliers(selected_061, grid)
    print(f'  Excluded: {len(excluded)} points')
    if excluded:
        by_group = defaultdict(int)
        for e in excluded:
            by_group[(e['sweep'], e['tax_equity'])] += 1
        for key in sorted(by_group):
            print(f'    {key}: {by_group[key]}')
        for e in sorted(excluded, key=lambda x: -(abs(x['resid_mac']) + abs(x['resid_dt'])))[:5]:
            print(f'    gid={e["grid_id"]:5d} {e["sweep"]:4s} te={e["tax_equity"]:.3f} '
                  f'mac={e["mac"]:.1f} dt={e["dt"]:.3f} '
                  f'resid_mac={e["resid_mac"]:.1f} resid_dt={e["resid_dt"]:.3f}')

    # Save selection report CSV
    report_csv = FIGURES_DIR / 'fig5_warmstart_selection_report.csv'
    with open(report_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'grid_id', 'sweep', 'tax_equity', 'param_val',
            'orig_mac_2030', 'orig_delta_t_2100',
            'ws_mac_2030', 'ws_delta_t_2100',
            'ref_mac_2030', 'ref_delta_t_2100',
            'dist_orig', 'dist_ws', 'accepted',
        ])
        writer.writeheader()
        for r in sorted(report, key=lambda x: (x['sweep'], x['tax_equity'], x['param_val'])):
            writer.writerow({
                'grid_id': r['grid_id'],
                'sweep': r['sweep'],
                'tax_equity': r['tax_equity'],
                'param_val': f"{r['param_val']:.6f}",
                'orig_mac_2030': f"{r['orig_mac']:.4f}",
                'orig_delta_t_2100': f"{r['orig_dt']:.6f}",
                'ws_mac_2030': f"{r['ws_mac']:.4f}",
                'ws_delta_t_2100': f"{r['ws_dt']:.6f}",
                'ref_mac_2030': f"{r['ref_mac']:.4f}",
                'ref_delta_t_2100': f"{r['ref_dt']:.6f}",
                'dist_orig': f"{r['dist_orig']:.6f}",
                'dist_ws': f"{r['dist_ws']:.6f}",
                'accepted': r['accepted'],
            })
    print(f'  Saved selection report: {report_csv}')

    # Save outlier exclusion CSV
    excluded_csv = FIGURES_DIR / 'fig5_warmstart_excluded_outliers.csv'
    with open(excluded_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'grid_id', 'sweep', 'tax_equity', 'param_val',
            'mac_2030', 'delta_t_2100', 'resid_mac_2030', 'resid_delta_t_2100',
        ])
        writer.writeheader()
        for e in sorted(excluded, key=lambda x: (x['sweep'], x['tax_equity'], x['param_val'])):
            writer.writerow({
                'grid_id': e['grid_id'],
                'sweep': e['sweep'],
                'tax_equity': e['tax_equity'],
                'param_val': f"{e['param_val']:.6f}",
                'mac_2030': f"{e['mac']:.4f}",
                'delta_t_2100': f"{e['dt']:.6f}",
                'resid_mac_2030': f"{e['resid_mac']:.4f}",
                'resid_delta_t_2100': f"{e['resid_dt']:.6f}",
            })
    print(f'  Saved excluded outliers: {excluded_csv}')

    # config_058: no sweep structure, keep original
    orig_grouped = group_by_sweep_tax(orig_061, orig_058, grid)
    selected_grouped = group_by_sweep_tax(cleaned_061, orig_058, grid)

    comparison_pdf = FIGURES_DIR / 'fig5_warmstart_comparison_pages.pdf'
    final_pdf = FIGURES_DIR / 'fig5_warmstart_comparison.pdf'
    print(f'\nGenerating comparison pages...')

    with PdfPages(str(comparison_pdf)) as pdf:
        for variant in VARIANT_CONFIGS:
            print(f'  Page: {variant["name"]}')
            fig = build_page(variant, orig_grouped, selected_grouped)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    # Merge: comparison pages + existing fig5 PDFs
    existing_fig5_pdfs = [
        FIGURES_DIR / 'fig5_1_scatter.pdf',
        FIGURES_DIR / 'fig5_2_ridgeline.pdf',
        FIGURES_DIR / 'fig5_3_boxjitter.pdf',
        FIGURES_DIR / 'fig5_7_raincloud.pdf',
        FIGURES_DIR / 'fig5_9_frontier.pdf',
        FIGURES_DIR / 'fig5_violin.pdf',
    ]

    print(f'\nMerging into {final_pdf}...')
    merger = PdfMerger()
    merger.append(str(comparison_pdf))
    for pdf_file in existing_fig5_pdfs:
        if pdf_file.exists():
            merger.append(str(pdf_file))
            print(f'  + {pdf_file.name}')
        else:
            print(f'  ! {pdf_file.name} not found, skipping')
    merger.write(str(final_pdf))
    merger.close()

    # Clean up intermediate file
    comparison_pdf.unlink()

    print(f'Saved {final_pdf}')


if __name__ == '__main__':
    main()
