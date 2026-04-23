"""
Generate 1,000 random (rho, eta, ydmg) draws for config_044 Monte Carlo runs.

Uniform distributions:
  rho  ~ U(0, 0.05)
  eta  ~ U(0.5, 2.0)
  ydmg ~ U(0, 1.0)

Output: json/config_044_mc_draws.csv
"""

import numpy as np
import csv
import os

N_DRAWS = 1000
SEED = 42

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'json', 'config_044_mc_draws.csv')

rng = np.random.RandomState(SEED)

rho_draws = rng.uniform(0.0, 0.05, N_DRAWS)
eta_draws = rng.uniform(0.5, 2.0, N_DRAWS)
ydmg_draws = rng.uniform(0.0, 1.0, N_DRAWS)

with open(OUTPUT_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['draw', 'rho', 'eta', 'ydmg'])
    for i in range(N_DRAWS):
        writer.writerow([i + 1, f'{rho_draws[i]:.8f}', f'{eta_draws[i]:.8f}', f'{ydmg_draws[i]:.8f}'])

print(f"Wrote {N_DRAWS} draws to {OUTPUT_PATH}")
