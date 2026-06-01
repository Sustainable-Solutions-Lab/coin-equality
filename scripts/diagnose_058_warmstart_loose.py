"""Post-repair diagnostics for config_058 loose warm-start.

Merges original + warmstart results by grid_id, picks best objective,
generates diagnostic figures, and re-runs KNN outlier check on merged data.

Usage:
    python scripts/diagnose_058_warmstart_loose.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import KDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.fig_style import TAX_EQUITY_ORDER, TAX_EQUITY_LABELS, TAX_EQUITY_COLORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_058_grid.csv'
CANDIDATES_CSV = PROJECT_ROOT / 'json' / 'config_058_warmstart_loose_candidates.csv'
FIGURES_DIR = PROJECT_ROOT / 'data' / 'output' / 'figures-new'

SOURCE_DIRS = [
    PROJECT_ROOT / 'data' / 'output' / 'config_058',
    PROJECT_ROOT / 'data' / 'output' / 'config_058_warmstart_loose',
]

# KNN parameters (must match detect script)
K_NEIGHBORS = 20
RELATIVE_DEVIATION_THRESHOLD = 0.05
MAD_MULTIPLIER = 3.0
ETA_MIN, ETA_MAX = 1.0, 2.0
RHO_MIN, RHO_MAX = 0.0, 0.05
BETA_MIN, BETA_MAX = 0.0, 1.0


def load_grid():
    """Load grid CSV and return dict: grid_id -> row."""
    rows = {}
    with open(GRID_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row['grid_id'])] = {
                'grid_id': int(row['grid_id']),
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            }
    return rows


def load_all_results():
    """Load best result per grid_id across all source directories."""
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
    """Load results only from config_058 (pre-warmstart)."""
    results = {}
    base_dir = SOURCE_DIRS[0]
    if not base_dir.exists():
        return results
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


def normalize_params(eta, rho, beta):
    """Normalize parameters to [0, 1]."""
    return np.array([
        (eta - ETA_MIN) / (ETA_MAX - ETA_MIN),
        (rho - RHO_MIN) / (RHO_MAX - RHO_MIN),
        (beta - BETA_MIN) / (BETA_MAX - BETA_MIN),
    ])


def count_residual_outliers(results):
    """Re-run KNN outlier detection on merged results."""
    groups = {}
    for gid, run in results.items():
        tax = run['tax_equity']
        groups.setdefault(tax, []).append(run)

    total_outliers = 0

    for tax_equity, group_runs in sorted(groups.items()):
        n = len(group_runs)
        coords = np.empty((n, 3))
        mac_values = np.empty(n)

        for i, run in enumerate(group_runs):
            coords[i] = normalize_params(run['eta'], run['rho'],
                                         run['y_damage_distribution_exponent'])
            mac_values[i] = run['mac_2030']

        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=K_NEIGHBORS + 1)

        for i in range(n):
            neighbor_indices = indices[i, 1:]
            neighbor_mac = mac_values[neighbor_indices]
            local_median = np.median(neighbor_mac)
            local_mad = np.median(np.abs(neighbor_mac - local_median))
            deviation = abs(mac_values[i] - local_median)

            flagged = False
            if local_median != 0 and deviation / abs(local_median) > RELATIVE_DEVIATION_THRESHOLD:
                flagged = True
            if local_mad > 0 and deviation > MAD_MULTIPLIER * local_mad:
                flagged = True

            if flagged:
                total_outliers += 1

    return total_outliers


def plot_diagnostics(grid, original_results, merged_results, candidates):
    """Generate diagnostic figures: 3 param axes x 3 tax_equity."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    param_axes = [
        ('eta', r'$\eta$'),
        ('rho', r'$\rho$'),
        ('y_damage_distribution_exponent', r'$\beta$'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for col_idx, tax_equity in enumerate(TAX_EQUITY_ORDER):
        for row_idx, (param_col, param_label) in enumerate(param_axes):
            ax = axes[row_idx, col_idx]
            color = TAX_EQUITY_COLORS[tax_equity]
            tax_label = TAX_EQUITY_LABELS[tax_equity]

            orig_params = []
            orig_mac = []
            merged_params = []
            merged_mac = []

            for gid, g in grid.items():
                if abs(g['tax_equity'] - tax_equity) > 1e-6:
                    continue
                pval = g[param_col]

                if gid in original_results:
                    orig_params.append(pval)
                    orig_mac.append(original_results[gid]['mac_2030'])

                if gid in merged_results:
                    merged_params.append(pval)
                    merged_mac.append(merged_results[gid]['mac_2030'])

            ax.scatter(orig_params, orig_mac, s=3, alpha=0.15, c='gray',
                       edgecolors='none', rasterized=True, label='Before')
            ax.scatter(merged_params, merged_mac, s=3, alpha=0.3, c=color,
                       edgecolors='none', rasterized=True, label='After')

            ax.set_xlabel(param_label)
            ax.set_ylabel('OCP@2030 ($/tCO2)')
            if row_idx == 0:
                ax.set_title(tax_label)
            ax.legend(loc='best', fontsize=7, markerscale=3)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

    fig.suptitle('Config 058 Warm-Start Diagnostics (All Parameters Combined)',
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = FIGURES_DIR / 'diagnostics_058_warmstart_loose.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Objective improvement histogram
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    improvements = []
    for gid in candidates:
        if gid in original_results and gid in merged_results:
            imp = original_results[gid]['objective_value'] - merged_results[gid]['objective_value']
            improvements.append(imp)

    improvements = np.array(improvements)
    n_improved = np.sum(improvements > 0)
    n_unchanged = np.sum(improvements <= 0)

    ax2.hist(improvements[improvements > 0], bins=50, color='#2980b9',
             alpha=0.7, label=f'Improved ({n_improved})')
    ax2.hist(improvements[improvements <= 0], bins=50, color='gray',
             alpha=0.5, label=f'Unchanged ({n_unchanged})')
    ax2.set_xlabel('Objective improvement (orig - merged)')
    ax2.set_ylabel('Count')
    ax2.set_title('Config 058: Objective improvement per candidate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    fig2.tight_layout()

    out_path2 = FIGURES_DIR / 'diagnostics_058_warmstart_loose_improvement.png'
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Wrote {out_path2}")


def main():
    grid = load_grid()
    print(f"Loaded {len(grid)} grid points")

    original_results = load_original_results()
    print(f"Loaded {len(original_results)} original results (config_058)")

    merged_results = load_all_results()
    print(f"Loaded {len(merged_results)} merged results (all sources)")

    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidate grid_ids")

    # Count improvements
    n_improved = 0
    n_candidates_with_results = 0
    improvements = []

    for gid in candidates:
        if gid in original_results and gid in merged_results:
            n_candidates_with_results += 1
            obj_orig = original_results[gid]['objective_value']
            obj_merged = merged_results[gid]['objective_value']
            improvement = obj_orig - obj_merged
            if improvement > 0:
                n_improved += 1
                improvements.append(improvement)

    n_from_loose = sum(1 for r in merged_results.values()
                       if r['source'] == 'config_058_warmstart_loose')

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
    n_residual = count_residual_outliers(merged_results)
    print(f"\n  Residual outliers (merged): {n_residual}")

    # Diagnostic plots
    print(f"\nGenerating diagnostic figures...")
    plot_diagnostics(grid, original_results, merged_results, candidates)

    print(f"\nDone.")


if __name__ == '__main__':
    main()
