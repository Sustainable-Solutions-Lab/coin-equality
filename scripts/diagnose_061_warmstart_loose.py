"""Post-repair diagnostics for config_061 loose warm-start.

Merges original + warmstart results by grid_id, picks best objective,
generates diagnostic figures, and checks for residual outliers.

Usage:
    python scripts/diagnose_061_warmstart_loose.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS, TAX_EQUITY_COLORS, SWEEP_TITLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_061_grid.csv'
CANDIDATES_CSV = PROJECT_ROOT / 'json' / 'config_061_warmstart_loose_candidates.csv'
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'figures-new'

SOURCE_DIRS = [
    PROJECT_ROOT / 'data' / 'output' / 'config_061',
    PROJECT_ROOT / 'data' / 'output' / 'config_061_warmstart',
    PROJECT_ROOT / 'data' / 'output' / 'config_061_warmstart_loose',
]

ROLLING_WINDOW = 21
RELATIVE_DEVIATION_THRESHOLD = 0.03
MAD_MULTIPLIER = 3.0

SWEEP_PARAM_COL = {
    'eta': 'eta',
    'rho': 'rho',
    'beta': 'y_damage_distribution_exponent',
}

SWEEP_PARAM_LABEL = {
    'eta': r'$\eta$',
    'rho': r'$\rho$',
    'beta': r'$\beta$',
}


def load_grid():
    """Load grid CSV and return dict: grid_id -> row."""
    rows = {}
    with open(GRID_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row['grid_id'])] = {
                'grid_id': int(row['grid_id']),
                'sweep': row['sweep'],
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            }
    return rows


def load_all_results():
    """Load best result per grid_id across all source directories.

    Returns dict: grid_id -> row dict with source directory info.
    """
    results = {}
    for base_dir in SOURCE_DIRS:
        if not base_dir.exists():
            continue
        for entry in sorted(base_dir.iterdir()):
            if not entry.is_dir():
                continue
            status_path = entry / 'run_status.csv'
            if not status_path.exists():
                continue
            with open(status_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['success'] != 'True':
                        continue
                    gid = int(row['grid_id'])
                    obj = float(row['objective_value'])
                    if gid not in results or obj < results[gid]['objective_value']:
                        results[gid] = {
                            'grid_id': gid,
                            'sweep': row['sweep'],
                            'sample_id': int(row['sample_id']),
                            'tax_equity': float(row['tax_equity']),
                            'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                            'eta': float(row['eta']),
                            'rho': float(row['rho']),
                            'objective_value': obj,
                            'mac_2030': float(row['mac_2030']),
                            'delta_t_2100': float(row['delta_t_2100']),
                            'source': str(base_dir.name),
                        }
    return results


def load_original_results():
    """Load results only from config_061 and config_061_warmstart (pre-loose)."""
    results = {}
    pre_dirs = SOURCE_DIRS[:2]
    for base_dir in pre_dirs:
        if not base_dir.exists():
            continue
        for entry in sorted(base_dir.iterdir()):
            if not entry.is_dir():
                continue
            status_path = entry / 'run_status.csv'
            if not status_path.exists():
                continue
            with open(status_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['success'] != 'True':
                        continue
                    gid = int(row['grid_id'])
                    obj = float(row['objective_value'])
                    if gid not in results or obj < results[gid]['objective_value']:
                        results[gid] = {
                            'grid_id': gid,
                            'objective_value': obj,
                            'mac_2030': float(row['mac_2030']),
                            'delta_t_2100': float(row['delta_t_2100']),
                        }
    return results


def load_candidates():
    """Load candidate grid_ids from detection CSV."""
    candidates = set()
    with open(CANDIDATES_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.add(int(row['grid_id']))
    return candidates


def rolling_median(values, window):
    """Compute rolling median with edge-adaptive window."""
    n = len(values)
    result = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.median(values[lo:hi])
    return result


def rolling_mad(values, medians, window):
    """Compute rolling MAD with edge-adaptive window."""
    n = len(values)
    result = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local_deviations = np.abs(values[lo:hi] - medians[lo:hi])
        result[i] = np.median(local_deviations)
    return result


def count_residual_outliers(results, grid):
    """Re-run smoothness detection on merged results and count residual outliers."""
    groups = {}
    for gid, run in results.items():
        key = (run['sweep'], run['tax_equity'])
        groups.setdefault(key, []).append(run)

    total_outliers = 0
    outlier_details = []

    for (sweep, tax_equity), group_runs in sorted(groups.items()):
        group_runs.sort(key=lambda r: r['sample_id'])
        mac_values = np.array([r['mac_2030'] for r in group_runs])

        med = rolling_median(mac_values, ROLLING_WINDOW)
        mad = rolling_mad(mac_values, med, ROLLING_WINDOW)

        for i, run in enumerate(group_runs):
            deviation = abs(run['mac_2030'] - med[i])
            flagged = False

            if med[i] != 0 and deviation / abs(med[i]) > RELATIVE_DEVIATION_THRESHOLD:
                flagged = True
            if mad[i] > 0 and deviation > MAD_MULTIPLIER * mad[i]:
                flagged = True

            if flagged:
                total_outliers += 1
                outlier_details.append({
                    'grid_id': run['grid_id'],
                    'sweep': sweep,
                    'tax_equity': tax_equity,
                    'sample_id': run['sample_id'],
                    'mac_2030': run['mac_2030'],
                })

    return total_outliers, outlier_details


def plot_diagnostics(grid, original_results, merged_results, candidates):
    """Generate diagnostic figures for each sweep."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for sweep in ['eta', 'rho', 'beta']:
        param_col = SWEEP_PARAM_COL[sweep]
        param_label = SWEEP_PARAM_LABEL[sweep]

        fig, axes = plt.subplots(3, 3, figsize=(16, 12))

        for col_idx, tax_equity in enumerate(TAX_EQUITY_ORDER):
            # Gather data for this (sweep, tax_equity) group
            orig_params = []
            orig_mac = []
            orig_dt = []
            merged_params = []
            merged_mac = []
            merged_dt = []
            cand_params = []
            cand_obj_improvement = []

            for gid, g in grid.items():
                if g['sweep'] != sweep or abs(g['tax_equity'] - tax_equity) > 1e-6:
                    continue
                pval = g[param_col]

                if gid in original_results:
                    orig_params.append(pval)
                    orig_mac.append(original_results[gid]['mac_2030'])
                    orig_dt.append(original_results[gid]['delta_t_2100'])

                if gid in merged_results:
                    merged_params.append(pval)
                    merged_mac.append(merged_results[gid]['mac_2030'])
                    merged_dt.append(merged_results[gid]['delta_t_2100'])

                if gid in candidates and gid in original_results and gid in merged_results:
                    cand_params.append(pval)
                    obj_orig = original_results[gid]['objective_value']
                    obj_merged = merged_results[gid]['objective_value']
                    cand_obj_improvement.append(obj_orig - obj_merged)

            orig_params = np.array(orig_params)
            orig_mac = np.array(orig_mac)
            orig_dt = np.array(orig_dt)
            merged_params = np.array(merged_params)
            merged_mac = np.array(merged_mac)
            merged_dt = np.array(merged_dt)
            cand_params = np.array(cand_params)
            cand_obj_improvement = np.array(cand_obj_improvement)

            color = TAX_EQUITY_COLORS[tax_equity]
            tax_label = TAX_EQUITY_LABELS[tax_equity]

            # Row 0: Parameter vs MAC@2030
            ax = axes[0, col_idx]
            ax.scatter(orig_params, orig_mac, s=2, alpha=0.15, c='gray',
                       edgecolors='none', rasterized=True, label='Before')
            ax.scatter(merged_params, merged_mac, s=2, alpha=0.3, c=color,
                       edgecolors='none', rasterized=True, label='After')
            ax.set_ylabel('OCP@2030 ($/tCO2)')
            ax.set_title(f'{tax_label}')
            ax.legend(loc='best', fontsize=7, markerscale=3)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Row 1: Parameter vs delta_T@2100
            ax = axes[1, col_idx]
            ax.scatter(orig_params, orig_dt, s=2, alpha=0.15, c='gray',
                       edgecolors='none', rasterized=True, label='Before')
            ax.scatter(merged_params, merged_dt, s=2, alpha=0.3, c=color,
                       edgecolors='none', rasterized=True, label='After')
            ax.set_ylabel(r'$\Delta T_{2100}$ ($\degree$C)')
            ax.legend(loc='best', fontsize=7, markerscale=3)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Row 2: Objective improvement per candidate
            ax = axes[2, col_idx]
            if len(cand_params) > 0:
                improved = cand_obj_improvement > 0
                ax.bar(cand_params[improved], cand_obj_improvement[improved],
                       width=(cand_params.max() - cand_params.min()) / len(merged_params) * 0.8
                       if len(merged_params) > 1 else 0.01,
                       color=color, alpha=0.6, label=f'Improved ({np.sum(improved)})')
                ax.bar(cand_params[~improved], cand_obj_improvement[~improved],
                       width=(cand_params.max() - cand_params.min()) / len(merged_params) * 0.8
                       if len(merged_params) > 1 else 0.01,
                       color='gray', alpha=0.4, label=f'No change ({np.sum(~improved)})')
                ax.axhline(0, color='k', linewidth=0.5)
                ax.legend(loc='best', fontsize=7)
            ax.set_ylabel('Objective improvement\n(orig - merged)')
            ax.set_xlabel(param_label)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Set x-label on all bottom row
            for row in range(2):
                axes[row, col_idx].set_xlabel(param_label)

        fig.suptitle(f'Config 061 Warm-Start Diagnostics: {SWEEP_TITLES[sweep]}',
                     fontsize=14, y=1.02)
        fig.tight_layout()

        out_path = FIGURES_DIR / f'diagnostics_061_warmstart_loose_{sweep}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Wrote {out_path}")


def main():
    grid = load_grid()
    print(f"Loaded {len(grid)} grid points")

    original_results = load_original_results()
    print(f"Loaded {len(original_results)} original results (config_061 + warmstart)")

    merged_results = load_all_results()
    print(f"Loaded {len(merged_results)} merged results (all sources)")

    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidate grid_ids")

    # Count how many candidates were improved
    n_improved = 0
    n_candidates_with_results = 0
    total_improvement = 0.0
    improvements = []

    for gid in candidates:
        if gid in original_results and gid in merged_results:
            n_candidates_with_results += 1
            obj_orig = original_results[gid]['objective_value']
            obj_merged = merged_results[gid]['objective_value']
            improvement = obj_orig - obj_merged
            if improvement > 0:
                n_improved += 1
                total_improvement += improvement
                improvements.append(improvement)

    # Count how many merged results come from warmstart_loose
    n_from_loose = sum(1 for r in merged_results.values()
                       if r['source'] == 'config_061_warmstart_loose')

    print(f"\n{'=' * 60}")
    print(f"WARM-START REPAIR SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Candidates flagged:        {len(candidates)}")
    print(f"  Candidates with results:   {n_candidates_with_results}")
    print(f"  Improved by warm-start:    {n_improved}")
    print(f"  Adopted from loose WS:     {n_from_loose}")
    if improvements:
        print(f"  Mean improvement:          {np.mean(improvements):.6e}")
        print(f"  Median improvement:        {np.median(improvements):.6e}")
        print(f"  Max improvement:           {np.max(improvements):.6e}")

    # Residual outlier check
    n_residual, residual_details = count_residual_outliers(merged_results, grid)
    print(f"\n  Residual outliers (merged): {n_residual}")

    if residual_details:
        # Group by sweep
        by_sweep = defaultdict(list)
        for d in residual_details:
            by_sweep[d['sweep']].append(d)
        for sweep, details in sorted(by_sweep.items()):
            print(f"    {sweep}: {len(details)}")

    # Generate diagnostic plots
    print(f"\nGenerating diagnostic figures...")
    plot_diagnostics(grid, original_results, merged_results, candidates)

    print(f"\nDone.")


if __name__ == '__main__':
    main()
