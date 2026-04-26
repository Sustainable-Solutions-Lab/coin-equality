"""
Generate grid CSV for config_052: 21×21 grid of tax_equity × y_damage_distribution_exponent.

Grid specifications:
- tax_equity: 0.00 to 0.985 (21 points, step 0.04925)
- y_damage_distribution_exponent: 0.00 to 0.41 (21 points, step 0.0205)
- Total: 441 grid points
"""

import csv
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / 'json' / 'config_052_grid.csv'


def main():
    # Define parameter ranges
    tax_equity_vals = np.linspace(0.00, 0.985, 21)
    ydmg_vals = np.linspace(0.00, 0.41, 21)

    print(f"Generating grid:")
    print(f"  tax_equity: {tax_equity_vals[0]:.4f} to {tax_equity_vals[-1]:.4f} ({len(tax_equity_vals)} points)")
    print(f"  y_damage_distribution_exponent: {ydmg_vals[0]:.4f} to {ydmg_vals[-1]:.4f} ({len(ydmg_vals)} points)")
    print(f"  Total grid points: {len(tax_equity_vals) * len(ydmg_vals)}")

    # Generate grid
    grid_id = 1
    rows = []

    for te in tax_equity_vals:
        for ydmg in ydmg_vals:
            rows.append({
                'grid_id': grid_id,
                'tax_equity': f'{te:.4f}',
                'y_damage_distribution_exponent': f'{ydmg:.4f}',
            })
            grid_id += 1

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['grid_id', 'tax_equity', 'y_damage_distribution_exponent'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Grid points: {len(rows)}")


if __name__ == '__main__':
    main()
