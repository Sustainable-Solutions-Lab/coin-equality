"""
Generate grid CSV for config_058: Latin Hypercube Monte Carlo sensitivity analysis.

Sampling: Latin Hypercube (d=3, seed=42), 1000 samples
Parameters sampled: eta ~ U[1.0, 2.0], rho ~ U[0.0, 0.05], beta ~ U[0.0, 1.0]
Tax cases: tax_equity = {0.0, 0.5, 0.998} (paired across all samples)
Total runs: 3000

Grid ordering: grouped by sample_id for paired locality:
  grid_id 1-3:       sample_id=1,    tax_equity = {0.0, 0.5, 0.998}
  grid_id 4-6:       sample_id=2,    tax_equity = {0.0, 0.5, 0.998}
  ...
  grid_id 2998-3000: sample_id=1000, tax_equity = {0.0, 0.5, 0.998}
"""

import csv
import numpy as np
from scipy.stats.qmc import LatinHypercube
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / 'json' / 'config_058_grid.csv'

N_SAMPLES = 1000
TAX_EQUITY_VALUES = [0.0, 0.5, 0.998]

# Parameter ranges for LHS
ETA_MIN, ETA_MAX = 1.0, 2.0
RHO_MIN, RHO_MAX = 0.0, 0.05
BETA_MIN, BETA_MAX = 0.0, 1.0


def main():
    # Generate LHS samples in [0, 1]^3
    sampler = LatinHypercube(d=3, seed=42)
    unit_samples = sampler.random(n=N_SAMPLES)

    # Scale to physical parameter ranges
    eta_samples = ETA_MIN + unit_samples[:, 0] * (ETA_MAX - ETA_MIN)
    rho_samples = RHO_MIN + unit_samples[:, 1] * (RHO_MAX - RHO_MIN)
    beta_samples = BETA_MIN + unit_samples[:, 2] * (BETA_MAX - BETA_MIN)

    print(f"Latin Hypercube Sampling (seed=42):")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  eta:  [{ETA_MIN}, {ETA_MAX}]  -> mean={eta_samples.mean():.4f}, std={eta_samples.std():.4f}, min={eta_samples.min():.4f}, max={eta_samples.max():.4f}")
    print(f"  rho:  [{RHO_MIN}, {RHO_MAX}] -> mean={rho_samples.mean():.6f}, std={rho_samples.std():.6f}, min={rho_samples.min():.6f}, max={rho_samples.max():.6f}")
    print(f"  beta: [{BETA_MIN}, {BETA_MAX}]  -> mean={beta_samples.mean():.4f}, std={beta_samples.std():.4f}, min={beta_samples.min():.4f}, max={beta_samples.max():.4f}")
    print(f"  Tax cases: {TAX_EQUITY_VALUES}")
    print(f"  Total runs: {N_SAMPLES * len(TAX_EQUITY_VALUES)}")

    # Generate grid rows: grouped by sample_id
    grid_id = 1
    rows = []

    for sample_id in range(1, N_SAMPLES + 1):
        idx = sample_id - 1
        eta = eta_samples[idx]
        rho = rho_samples[idx]
        beta = beta_samples[idx]

        for te in TAX_EQUITY_VALUES:
            rows.append({
                'grid_id': grid_id,
                'sample_id': sample_id,
                'tax_equity': f'{te:.4f}',
                'y_damage_distribution_exponent': f'{beta:.6f}',
                'eta': f'{eta:.6f}',
                'rho': f'{rho:.6f}',
            })
            grid_id += 1

    # Write CSV
    fieldnames = ['grid_id', 'sample_id', 'tax_equity', 'y_damage_distribution_exponent', 'eta', 'rho']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Grid points: {len(rows)}")

    # Verify pairing structure
    print(f"\nVerification:")
    print(f"  First 6 rows (sample_id=1,2):")
    for row in rows[:6]:
        print(f"    grid_id={row['grid_id']}, sample_id={row['sample_id']}, "
              f"tax_equity={row['tax_equity']}, beta={row['y_damage_distribution_exponent']}, "
              f"eta={row['eta']}, rho={row['rho']}")
    print(f"  Last 3 rows (sample_id={N_SAMPLES}):")
    for row in rows[-3:]:
        print(f"    grid_id={row['grid_id']}, sample_id={row['sample_id']}, "
              f"tax_equity={row['tax_equity']}, beta={row['y_damage_distribution_exponent']}, "
              f"eta={row['eta']}, rho={row['rho']}")


if __name__ == '__main__':
    main()
