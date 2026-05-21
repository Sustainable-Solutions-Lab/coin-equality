"""
Generate unified grid CSV for config 061: single-parameter violin sweeps.

Three sweep types, each varying one parameter while holding the other two fixed:
  eta:  eta  in [1.0, 2.0],  rho=0.025, beta=0.5
  rho:  rho  in [0.0, 0.05], eta=1.5,   beta=0.5
  beta: beta in [0.0, 1.0],  eta=1.5,   rho=0.025

All three use uniform grids with 1000 points x 3 tax_equity cases = 3000 runs each.
Total: 9000 runs.

Grid ordering: sweep types are contiguous, within each grouped by sample_id:
  grid_id 1-3000:    sweep=eta
  grid_id 3001-6000: sweep=rho
  grid_id 6001-9000: sweep=beta
"""

import csv
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / 'json' / 'config_061_grid.csv'

N_SAMPLES = 1000
TAX_EQUITY_VALUES = [0.0, 0.5, 0.998]

ETA_FIXED = 1.5
RHO_FIXED = 0.025
BETA_FIXED = 0.5

SWEEPS = [
    {
        'sweep': 'eta',
        'eta': np.linspace(1.0, 2.0, N_SAMPLES),
        'rho': np.full(N_SAMPLES, RHO_FIXED),
        'beta': np.full(N_SAMPLES, BETA_FIXED),
    },
    {
        'sweep': 'rho',
        'eta': np.full(N_SAMPLES, ETA_FIXED),
        'rho': np.linspace(0.0, 0.05, N_SAMPLES),
        'beta': np.full(N_SAMPLES, BETA_FIXED),
    },
    {
        'sweep': 'beta',
        'eta': np.full(N_SAMPLES, ETA_FIXED),
        'rho': np.full(N_SAMPLES, RHO_FIXED),
        'beta': np.linspace(0.0, 1.0, N_SAMPLES),
    },
]


def main():
    grid_id = 1
    rows = []

    for sw in SWEEPS:
        print(f"\nSweep: {sw['sweep']}")
        print(f"  eta:  [{sw['eta'].min():.4f}, {sw['eta'].max():.4f}]")
        print(f"  rho:  [{sw['rho'].min():.6f}, {sw['rho'].max():.6f}]")
        print(f"  beta: [{sw['beta'].min():.4f}, {sw['beta'].max():.4f}]")

        for sample_id in range(1, N_SAMPLES + 1):
            idx = sample_id - 1
            for te in TAX_EQUITY_VALUES:
                rows.append({
                    'grid_id': grid_id,
                    'sweep': sw['sweep'],
                    'sample_id': sample_id,
                    'tax_equity': f'{te:.4f}',
                    'y_damage_distribution_exponent': f'{sw["beta"][idx]:.6f}',
                    'eta': f'{sw["eta"][idx]:.6f}',
                    'rho': f'{sw["rho"][idx]:.6f}',
                })
                grid_id += 1

    print(f"\nTax cases: {TAX_EQUITY_VALUES}")
    print(f"Total runs: {len(rows)}")

    fieldnames = ['grid_id', 'sweep', 'sample_id', 'tax_equity',
                  'y_damage_distribution_exponent', 'eta', 'rho']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_CSV}")

    print(f"\nVerification:")
    for label, start in [("eta start", 0), ("rho start", 3000), ("beta start", 6000)]:
        print(f"  {label}:")
        for row in rows[start:start + 3]:
            print(f"    grid_id={row['grid_id']}, sweep={row['sweep']}, "
                  f"sample_id={row['sample_id']}, tax_equity={row['tax_equity']}, "
                  f"beta={row['y_damage_distribution_exponent']}, "
                  f"eta={row['eta']}, rho={row['rho']}")


if __name__ == '__main__':
    main()
