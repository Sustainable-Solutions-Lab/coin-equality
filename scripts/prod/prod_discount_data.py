"""
Data loader for production discount sweep (prod_discount).

Loads 42 runs sweeping rho across 6 tax_equity/ydmg combinations.
Returns organized data keyed by (tax_equity, ydmg) with arrays of sorted rho values.

Usage:
    from prod_discount_data import load_prod_discount_data
    data = load_prod_discount_data()
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROD_OUTPUT = PROJECT_ROOT / 'data' / 'output' / 'prod'
DISCOUNT_DIR = PROD_OUTPUT / 'prod_discount_exact'
DISCOUNT_GRID = PROJECT_ROOT / 'json' / 'prod_discount_grid.csv'

OCP_YEAR = 2030
DT_YEAR = 2100
R_YEAR = 2030


def _find_col(df, prefix):
    """Find a column by prefix match."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    return matches[0] if matches else None


def _extract_r_consumption(run_dir):
    """Extract r_consumption@2030 from a run's results CSV."""
    results_files = sorted(run_dir.glob('*_results.csv'))
    if not results_files:
        return None

    try:
        df = pd.read_csv(results_files[-1])
        if 'r_consumption' not in df.columns:
            return None

        t_col = _find_col(df, 't,')
        if t_col is None:
            return None

        years = df[t_col].values
        idx_r = np.argmin(np.abs(years - R_YEAR))
        return df['r_consumption'].values[idx_r] * 100  # convert to %
    except Exception:
        return None


def load_prod_discount_data():
    """Load prod_discount runs organized by tax_equity and ydmg.

    Returns dict keyed by (tax_equity, ydmg) with arrays sorted by rho:
        {
            (0.0, 0.0): {'rho': [...], 'ocp': [...], 'dt': [...], 'r': [...]},
            (0.0, 0.5): {...},
            ...
        }
    """
    # Load grid to map grid_id -> (tax, ydmg, eta, rho)
    grid_map = {}
    with open(DISCOUNT_GRID, 'r') as f:
        for row in csv.DictReader(f):
            grid_id = int(row['grid_id'])
            grid_map[grid_id] = {
                'tax_equity': float(row['tax_equity']),
                'ydmg': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            }

    # Load run_status.csv from prod_discount_exact directories
    data_by_key = {}

    for entry in sorted(DISCOUNT_DIR.iterdir()):
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

                grid_id = int(row['grid_id'])
                if grid_id not in grid_map:
                    continue

                params = grid_map[grid_id]
                tax_equity = params['tax_equity']
                ydmg = params['ydmg']
                rho = params['rho']

                key = (tax_equity, ydmg)

                # Extract r_consumption
                r_consumption = _extract_r_consumption(entry)
                if r_consumption is None:
                    continue

                # Extract from run_status
                ocp = float(row['mac_2030'])
                dt = float(row['delta_t_2100'])

                if key not in data_by_key:
                    data_by_key[key] = []

                data_by_key[key].append({
                    'rho': rho,
                    'ocp': ocp,
                    'dt': dt,
                    'r': r_consumption,
                })

    # Sort by r_consumption (discount rate) and convert to numpy arrays
    result = {}
    for key in sorted(data_by_key.keys()):
        entries = sorted(data_by_key[key], key=lambda x: x['r'])
        result[key] = {
            'rho': np.array([e['rho'] for e in entries]),
            'ocp': np.array([e['ocp'] for e in entries]),
            'dt': np.array([e['dt'] for e in entries]),
            'r': np.array([e['r'] for e in entries]),
        }
        tax_eq, ydmg = key
        print(f'  tax_equity={tax_eq:.3f}, ydmg={ydmg:.1f}: {len(entries)} runs')

    return result
