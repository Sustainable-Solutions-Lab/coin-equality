"""
Find outliers/spikes in OCP vs rho curves for config_060 and config_061 rho sweeps.

For each dataset:
1. Loads all runs, selects the one with the best objective_value per rho
2. Sorts by rho, computes delta_OCP between consecutive rho values
3. Flags points where |delta_OCP| > 3 * median(|delta_OCP|) as outliers
4. Prints outlier details: rho, r(2030)%, OCP, expected OCP from neighbors

Usage:
    python scripts/find_ocp_outliers.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fig_style import TAX_EQUITY_LABELS, TAX_EQUITY_ORDER

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OCP_YEAR = 2030
R_YEAR = 2030
OUTLIER_MULTIPLIER = 3.0


def find_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0]


def load_best_runs(data_dirs, pattern_func):
    """Load runs from multiple directories, keep best objective per rho.

    Returns dict: rho_val -> (run_dir, obj_val)
    """
    best_per_rho = {}

    for data_dir in data_dirs:
        pattern = pattern_func(data_dir)
        dirs = sorted(glob.glob(pattern))

        for run_dir in dirs:
            status_file = os.path.join(run_dir, 'run_status.csv')
            if not os.path.exists(status_file):
                continue
            status_df = pd.read_csv(status_file)
            if not status_df['success'].iloc[0]:
                continue

            dirname = os.path.basename(run_dir)
            rho_str = dirname.split('_rho-')[-1].split('_')[0]
            rho_val = float(rho_str)
            obj_val = status_df['objective_value'].iloc[0]

            prev = best_per_rho.get(rho_val)
            if prev is None or obj_val > prev[1]:
                best_per_rho[rho_val] = (run_dir, obj_val)

    return best_per_rho


def extract_ocp_and_r(best_per_rho):
    """Extract OCP@2030 and r(2030) from the best runs.

    Returns arrays: rho, ocp, r, run_dirs (all sorted by rho).
    """
    rho_vals = []
    ocp_vals = []
    r_vals = []
    run_dirs = []

    for rho_val, (run_dir, _) in sorted(best_per_rho.items()):
        csv_files = sorted(glob.glob(os.path.join(run_dir, '*_optimization_results.csv')))
        if not csv_files:
            continue
        df = pd.read_csv(csv_files[-1])

        t_col = find_col(df, 't,')
        years = df[t_col].values
        mac = df['marginal_abatement_cost'].values
        r_consumption = df['r_consumption'].values

        idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
        idx_r = np.argmin(np.abs(years - R_YEAR))

        rho_vals.append(rho_val)
        ocp_vals.append(mac[idx_ocp])
        r_vals.append(r_consumption[idx_r] * 100)
        run_dirs.append(run_dir)

    return np.array(rho_vals), np.array(ocp_vals), np.array(r_vals), run_dirs


def find_outliers(rho, ocp, r, run_dirs, label):
    """Find outliers in OCP vs rho curve using consecutive-difference method.

    Flags points where |delta_OCP| > OUTLIER_MULTIPLIER * median(|delta_OCP|).
    """
    n = len(rho)
    if n < 3:
        print(f'  {label}: only {n} points, skipping')
        return []

    delta_ocp = np.diff(ocp)
    abs_delta = np.abs(delta_ocp)
    median_delta = np.median(abs_delta)
    threshold = OUTLIER_MULTIPLIER * median_delta

    outliers = []

    for i in range(n):
        is_outlier = False

        # Check jump INTO this point (from i-1 to i)
        if i > 0 and abs_delta[i - 1] > threshold:
            # Check jump OUT of this point (from i to i+1)
            if i < n - 1 and abs_delta[i] > threshold:
                # Both jumps are large: this point is the spike
                is_outlier = True
            elif i == n - 1:
                # Last point with large jump in: could be outlier
                # Only flag if significantly different from trend of last few points
                is_outlier = True

        if is_outlier:
            # Expected OCP from neighbors (average of neighbors)
            if 0 < i < n - 1:
                expected_ocp = (ocp[i - 1] + ocp[i + 1]) / 2
            elif i == 0:
                expected_ocp = ocp[1]
            else:
                expected_ocp = ocp[-2]

            outliers.append({
                'index': i,
                'rho': rho[i],
                'r_pct': r[i],
                'ocp': ocp[i],
                'expected_ocp': expected_ocp,
                'delta_from_expected': ocp[i] - expected_ocp,
                'run_dir': run_dirs[i],
            })

    return outliers


def print_outliers(outliers, label, rho, ocp, r, median_delta, threshold):
    """Print outlier details in a formatted table."""
    print(f'\n{"=" * 100}')
    print(f'  {label}')
    print(f'  Points: {len(rho)}, median|delta_OCP|: {median_delta:.2f}, '
          f'threshold (3x): {threshold:.2f}')
    print(f'{"=" * 100}')

    if not outliers:
        print('  No outliers found.')
        return

    print(f'  {"rho":>8}  {"r(2030)%":>9}  {"OCP":>10}  {"expected":>10}  '
          f'{"diff":>10}  {"diff%":>8}  run_dir')
    print(f'  {"-" * 8}  {"-" * 9}  {"-" * 10}  {"-" * 10}  '
          f'{"-" * 10}  {"-" * 8}  {"-" * 40}')

    for o in outliers:
        diff_pct = o['delta_from_expected'] / o['expected_ocp'] * 100
        short_dir = os.path.basename(o['run_dir'])
        print(f'  {o["rho"]:8.4f}  {o["r_pct"]:9.3f}  {o["ocp"]:10.2f}  '
              f'{o["expected_ocp"]:10.2f}  {o["delta_from_expected"]:+10.2f}  '
              f'{diff_pct:+7.1f}%  {short_dir}')


def analyze_dataset(data_dirs, pattern_func, dataset_label, tax_equity, ydmg):
    """Load, analyze, and report outliers for one dataset/tax_equity/ydmg combo."""
    label = f'{dataset_label} | {TAX_EQUITY_LABELS[tax_equity]} | ydmg={ydmg}'

    def pf(d):
        return pattern_func(d, tax_equity, ydmg)

    best_per_rho = load_best_runs(data_dirs, pf)
    rho, ocp, r, run_dirs = extract_ocp_and_r(best_per_rho)

    if len(rho) < 3:
        print(f'\n  {label}: only {len(rho)} points, skipping')
        return []

    delta_ocp = np.diff(ocp)
    abs_delta = np.abs(delta_ocp)
    median_delta = np.median(abs_delta)
    threshold = OUTLIER_MULTIPLIER * median_delta

    outliers = find_outliers(rho, ocp, r, run_dirs, label)
    print_outliers(outliers, label, rho, ocp, r, median_delta, threshold)

    # Also print the full r(2030) range for context
    r_min_idx = np.argmin(r)
    r_max_idx = np.argmax(r)
    print(f'  r(2030) range: {r[r_min_idx]:.3f}% (rho={rho[r_min_idx]:.4f}) '
          f'to {r[r_max_idx]:.3f}% (rho={rho[r_max_idx]:.4f})')

    # Print all points in the 4-7% r range for detailed inspection
    mask = (r >= 4.0) & (r <= 7.0)
    if np.any(mask):
        print(f'\n  Detail: all points with r(2030) in [4%, 7%]:')
        print(f'  {"rho":>8}  {"r(2030)%":>9}  {"OCP":>10}  {"delta_OCP":>10}')
        print(f'  {"-" * 8}  {"-" * 9}  {"-" * 10}  {"-" * 10}')
        indices = np.where(mask)[0]
        for idx in indices:
            d = ocp[idx] - ocp[idx - 1] if idx > 0 else float('nan')
            flag = ' ***' if any(o['index'] == idx for o in outliers) else ''
            print(f'  {rho[idx]:8.4f}  {r[idx]:9.3f}  {ocp[idx]:10.2f}  '
                  f'{d:+10.2f}{flag}')

    return outliers


def main():
    data_060 = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_060')
    data_060w = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_060_warmstart')
    data_061 = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_061')

    # Check which directories exist
    dirs_060 = [d for d in [data_060, data_060w] if os.path.isdir(d)]
    dirs_061 = [d for d in [data_061] if os.path.isdir(d)]

    # Also check for config_061_warmstart
    data_061w = os.path.join(PROJECT_ROOT, 'data', 'output', 'config_061_warmstart')
    if os.path.isdir(data_061w):
        dirs_061.append(data_061w)

    print(f'Data directories:')
    print(f'  config_060: {dirs_060}')
    print(f'  config_061: {dirs_061}')

    all_outliers = []

    # --- Config 060 patterns ---
    def pattern_060(data_dir, tax_equity, ydmg):
        return os.path.join(
            data_dir,
            f'config_060_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_rho-*',
        )

    # Config 060: ydmg=0.5 for all 3 tax_equity values
    print('\n' + '#' * 100)
    print('# CONFIG_060 + WARMSTART, ydmg=0.5')
    print('#' * 100)
    for te in TAX_EQUITY_ORDER:
        outliers = analyze_dataset(dirs_060, pattern_060, 'config_060', te, 0.5)
        all_outliers.extend(outliers)

    # Config 060: ydmg=0.0 for all 3 tax_equity values
    print('\n' + '#' * 100)
    print('# CONFIG_060 + WARMSTART, ydmg=0.0 (beta=0 curves)')
    print('#' * 100)
    for te in TAX_EQUITY_ORDER:
        outliers = analyze_dataset(dirs_060, pattern_060, 'config_060', te, 0.0)
        all_outliers.extend(outliers)

    # --- Config 061 patterns ---
    def pattern_061(data_dir, tax_equity, ydmg):
        return os.path.join(
            data_dir,
            f'config_061_rho_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_*',
        )

    # Config 061: ydmg=0.5 for all 3 tax_equity values
    print('\n' + '#' * 100)
    print('# CONFIG_061 + WARMSTART, ydmg=0.5')
    print('#' * 100)
    for te in TAX_EQUITY_ORDER:
        outliers = analyze_dataset(dirs_061, pattern_061, 'config_061', te, 0.5)
        all_outliers.extend(outliers)

    # --- Summary ---
    print('\n' + '=' * 100)
    print(f'TOTAL OUTLIERS FOUND: {len(all_outliers)}')
    print('=' * 100)

    if all_outliers:
        print('\nAll outlier run directories (for re-running):')
        for o in all_outliers:
            print(f'  rho={o["rho"]:.4f}  r={o["r_pct"]:.3f}%  '
                  f'OCP={o["ocp"]:.2f}  expected={o["expected_ocp"]:.2f}  '
                  f'{os.path.basename(o["run_dir"])}')


if __name__ == '__main__':
    main()
