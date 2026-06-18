"""
Shared data-loading module for production fig5 scripts.

Reads master_results_mc.csv (built by build_master_results.py) and extracts
r_consumption@2030 from each selected run's full *_results.csv.

Returns a dict keyed by tax_equity with arrays of (ocp, dt, r, beta).

Usage:
    from prod_fig5_data import load_prod_fig5_data
    data = load_prod_fig5_data()
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROD_OUTPUT = PROJECT_ROOT / 'data' / 'output' / 'prod'
MASTER_CSV_MC = PROD_OUTPUT / 'master_results_mc.csv'

MC_DIR_ORIG = PROD_OUTPUT / 'prod_mc'
MC_DIR_RERUN = PROD_OUTPUT / 'prod_mc_rerun'
MC_DIR_WARMSTART = PROD_OUTPUT / 'prod_mc_warmstart'

SOURCE_DIR_MAP = {
    'orig': MC_DIR_ORIG,
    'rerun': MC_DIR_RERUN,
    'warmstart': MC_DIR_WARMSTART,
}

OCP_YEAR = 2030
DT_YEAR = 2100
R_YEAR = 2030


def _build_source_dir_index(output_dir):
    """Build grid_id -> directory_name index by scanning run_status.csv files."""
    index = {}
    for entry in sorted(output_dir.iterdir()):
        status_path = entry / 'run_status.csv'
        if not status_path.exists():
            continue
        with open(status_path, 'r') as f:
            for row in csv.DictReader(f):
                if row['success'] == 'True':
                    index[int(row['grid_id'])] = entry.name
    return index


def _find_col(df, prefix):
    """Find a column by prefix match."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0]


def load_prod_fig5_data():
    """Load MC data with r_consumption from selected sources.

    Returns dict: tax_equity -> {ocp, dt, r, beta} numpy arrays.
    """
    # Load master CSV
    rows = []
    with open(MASTER_CSV_MC, 'r') as f:
        for row in csv.DictReader(f):
            if row['validated'] == 'True' and row['selected_source'] != '':
                rows.append(row)

    print(f'Loaded {len(rows)} validated MC rows from {MASTER_CSV_MC.name}')

    # Build source directory indexes
    print('Building source directory indexes...')
    dir_indexes = {}
    for source_label, base_dir in SOURCE_DIR_MAP.items():
        if base_dir.exists():
            idx = _build_source_dir_index(base_dir)
            dir_indexes[source_label] = idx
            print(f'  {source_label}: {len(idx)} entries')

    # Extract r_consumption from each run's results CSV
    results_by_tax = {}
    missing_count = 0

    for row in rows:
        grid_id = int(row['grid_id'])
        source = row['selected_source']
        tax_equity = float(row['tax_equity'])

        base_dir = SOURCE_DIR_MAP[source]
        dir_index = dir_indexes[source]

        dir_name = dir_index.get(grid_id)
        if dir_name is None:
            missing_count += 1
            continue

        run_dir = base_dir / dir_name
        results_files = sorted(run_dir.glob('*_results.csv'))
        if not results_files:
            missing_count += 1
            continue

        df = pd.read_csv(results_files[-1])
        t_col = _find_col(df, 't,')
        years = df[t_col].values

        idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
        idx_dt = np.argmin(np.abs(years - DT_YEAR))
        idx_r = np.argmin(np.abs(years - R_YEAR))

        mac_col = _find_col(df, 'marginal_abatement_cost')
        delta_t_col = _find_col(df, 'delta_T,')

        entry = {
            'ocp': df[mac_col].values[idx_ocp],
            'dt': df[delta_t_col].values[idx_dt],
            'r': df['r_consumption'].values[idx_r] * 100,
            'beta': float(row['y_damage_distribution_exponent']),
        }

        results_by_tax.setdefault(tax_equity, []).append(entry)

    if missing_count > 0:
        print(f'  WARNING: {missing_count} runs had no source directory or results CSV')

    # Convert to numpy arrays
    data = {}
    for tax_equity in sorted(results_by_tax):
        entries = results_by_tax[tax_equity]
        data[tax_equity] = {
            'ocp': np.array([e['ocp'] for e in entries]),
            'dt': np.array([e['dt'] for e in entries]),
            'r': np.array([e['r'] for e in entries]),
            'beta': np.array([e['beta'] for e in entries]),
        }
        print(f'  tax_equity={tax_equity:.4f}: {len(entries)} points with r_consumption')

    return data
