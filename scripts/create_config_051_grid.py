"""
Generate the 21x21 grid CSV for config_051 2D sensitivity sweep.

Grid: tax_equity (0–0.95, 21 points) x y_damage_distribution_exponent (0–0.41, 21 points) = 441 runs.
Output: json/config_051_grid.csv
"""

import csv
import os

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'json', 'config_051_grid.csv')

TAX_EQUITY_VALUES = np.linspace(0.00, 0.95, 21)
YDMG_VALUES = np.linspace(0.00, 0.41, 21)


def main():
    grid_id = 0
    rows = []

    for te in TAX_EQUITY_VALUES:
        for ydmg in YDMG_VALUES:
            grid_id += 1
            rows.append({
                'grid_id': grid_id,
                'tax_equity': f'{te:.4f}',
                'y_damage_distribution_exponent': f'{ydmg:.4f}',
            })

    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['grid_id', 'tax_equity', 'y_damage_distribution_exponent'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {grid_id} rows to {OUTPUT_PATH}")
    print(f"  tax_equity: {TAX_EQUITY_VALUES[0]:.4f} to {TAX_EQUITY_VALUES[-1]:.4f} ({len(TAX_EQUITY_VALUES)} values)")
    print(f"  y_damage_distribution_exponent: {YDMG_VALUES[0]:.4f} to {YDMG_VALUES[-1]:.4f} ({len(YDMG_VALUES)} values)")


if __name__ == '__main__':
    main()
