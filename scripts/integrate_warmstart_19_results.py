"""Integrate warmstart_19 results by creating run_status.csv in each output directory.

The warmstart optimization created output directories but didn't write run_status.csv.
This script reads warmstart_19_results_local.json and writes the necessary status files
so the figures can pick up these improved results.

Usage:
    python scripts/integrate_warmstart_19_results.py
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WARMSTART_JSON = PROJECT_ROOT / 'warmstart_19_results_local.json'

STATUS_COLUMNS = [
    'run_name', 'grid_id', 'tax_equity',
    'y_damage_distribution_exponent', 'eta', 'rho',
    'success', 'optimizer_status', 'objective_value',
    'n_iterations', 'n_evaluations', 'runtime_seconds',
    'mac_2030', 'mac_2100', 'delta_t_2100',
    'mu_2030', 'mu_2100', 'Omega_2100', 'y_net_2100',
    'aggregate_utility', 'error_message',
]


def load_warmstart_json():
    """Load warmstart results from JSON."""
    with open(WARMSTART_JSON, 'r') as f:
        return json.load(f)


def load_grid_params(grid_id):
    """Load grid parameters for a given grid_id from prod_discount_exact_grid.csv."""
    grid_csv = PROJECT_ROOT / 'json' / 'prod_discount_exact_grid.csv'
    with open(grid_csv, 'r') as f:
        for row in csv.DictReader(f):
            if int(row['grid_id']) == grid_id:
                return {
                    'tax_equity': float(row['tax_equity']),
                    'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                    'eta': float(row['eta']),
                    'rho': float(row['rho']),
                }
    return None


def write_status_csv(output_dir, result, grid_params):
    """Write run_status.csv for a warmstart result."""
    status_path = Path(output_dir) / 'run_status.csv'

    # Build status dict
    status = {
        'run_name': f"warmstart_gid_{result['grid_id']}",
        'grid_id': result['grid_id'],
        'tax_equity': grid_params['tax_equity'],
        'y_damage_distribution_exponent': grid_params['y_damage_distribution_exponent'],
        'eta': grid_params['eta'],
        'rho': grid_params['rho'],
        'success': 'True' if result['success'] else 'False',
        'optimizer_status': result['optimizer_status'],
        'objective_value': '',  # Not available from our data
        'n_iterations': '',
        'n_evaluations': result['n_evaluations'],
        'runtime_seconds': result['elapsed_seconds'],
        'mac_2030': result['ocp_2030'],
        'mac_2100': '',
        'delta_t_2100': result['dt_2100'],
        'mu_2030': result['mu_2030'],
        'mu_2100': result['mu_2100'],
        'Omega_2100': '',
        'y_net_2100': '',
        'aggregate_utility': '',
        'error_message': '',
    }

    with open(status_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerow(status)

    return status_path


def main():
    results = load_warmstart_json()
    print(f"Loaded {len(results)} warmstart results\n")

    successful = 0
    for result in results:
        grid_id = result['grid_id']
        output_dir = result['output_dir']

        # Get grid parameters
        grid_params = load_grid_params(grid_id)
        if grid_params is None:
            print(f"⚠ Grid {grid_id}: Could not find parameters in grid CSV")
            continue

        # Write status CSV
        try:
            status_path = write_status_csv(output_dir, result, grid_params)
            if result['success']:
                print(f"✓ Grid {grid_id}: Wrote {status_path}")
                successful += 1
            else:
                print(f"✗ Grid {grid_id}: Failed result at {status_path}")
        except Exception as e:
            print(f"✗ Grid {grid_id}: Error writing status - {e}")

    print(f"\nIntegration complete: {successful}/{len(results)} successful results")


if __name__ == '__main__':
    main()
