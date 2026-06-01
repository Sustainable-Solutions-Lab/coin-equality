"""Detect convergence outliers in config_058 using KNN in normalized parameter space.

Config_058 is a Latin Hypercube Sampling Monte Carlo (3000 runs: 1000 samples x
3 tax_equity) where eta, rho, and beta all vary simultaneously. Since there is
no natural 1D ordering, we use K-nearest-neighbor detection: for each point,
compare its mac_2030 to the local median of its K nearest neighbors in
normalized (eta, rho, beta) space.

Outputs:
  json/config_058_warmstart_loose_candidates.csv  (one row per candidate)
  json/config_058_warmstart_loose_tasks.csv       (one row per warm-start task)

Usage:
    python scripts/detect_058_warmstart_candidates.py
"""

import csv
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_058_grid.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'config_058'
CANDIDATES_CSV = PROJECT_ROOT / 'json' / 'config_058_warmstart_loose_candidates.csv'
TASKS_CSV = PROJECT_ROOT / 'json' / 'config_058_warmstart_loose_tasks.csv'

# KNN detection parameters
K_NEIGHBORS = 20
RELATIVE_DEVIATION_THRESHOLD = 0.05
MAD_MULTIPLIER = 3.0

# Parameter ranges for normalization (from LHS grid generation)
ETA_MIN, ETA_MAX = 1.0, 2.0
RHO_MIN, RHO_MAX = 0.0, 0.05
BETA_MIN, BETA_MAX = 0.0, 1.0

# Number of nearest non-candidate neighbors to use as warm-start sources
N_WARMSTART_NEIGHBORS = 2


def load_grid():
    """Load grid CSV and return list of dicts."""
    rows = []
    with open(GRID_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'grid_id': int(row['grid_id']),
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            })
    return rows


def load_successful_runs():
    """Scan output directories and return dict: grid_id -> best run_status row.

    When multiple directories exist for the same grid_id (from resubmissions),
    the one with the best (most negative) objective_value is kept.
    """
    runs = {}
    if not OUTPUT_DIR.exists():
        return runs
    for entry in sorted(OUTPUT_DIR.iterdir()):
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
                if gid not in runs or obj < runs[gid]['objective_value']:
                    runs[gid] = {
                        'grid_id': gid,
                        'sample_id': int(row['sample_id']),
                        'tax_equity': float(row['tax_equity']),
                        'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                        'eta': float(row['eta']),
                        'rho': float(row['rho']),
                        'objective_value': obj,
                        'mac_2030': float(row['mac_2030']),
                        'delta_t_2100': float(row['delta_t_2100']),
                        'dir': str(entry),
                    }
    return runs


def normalize_params(eta, rho, beta):
    """Normalize parameters to [0, 1] for distance computation."""
    eta_n = (eta - ETA_MIN) / (ETA_MAX - ETA_MIN)
    rho_n = (rho - RHO_MIN) / (RHO_MAX - RHO_MIN)
    beta_n = (beta - BETA_MIN) / (BETA_MAX - BETA_MIN)
    return np.array([eta_n, rho_n, beta_n])


def detect_candidates(grid, runs):
    """Detect outlier candidates using KNN in normalized parameter space.

    For each tax_equity group, build a KD-tree in normalized (eta, rho, beta)
    space, find K nearest neighbors for each point, and flag points whose
    mac_2030 deviates significantly from the local median.

    Returns list of candidate dicts.
    """
    grid_by_id = {g['grid_id']: g for g in grid}

    # Group runs by tax_equity
    groups = {}
    for gid, run in runs.items():
        tax = run['tax_equity']
        groups.setdefault(tax, []).append(run)

    candidates = []

    for tax_equity, group_runs in sorted(groups.items()):
        # Build arrays for KD-tree
        n = len(group_runs)
        coords = np.empty((n, 3))
        mac_values = np.empty(n)
        gids = []

        for i, run in enumerate(group_runs):
            coords[i] = normalize_params(run['eta'], run['rho'],
                                         run['y_damage_distribution_exponent'])
            mac_values[i] = run['mac_2030']
            gids.append(run['grid_id'])

        # Build KD-tree and query K+1 neighbors (includes self)
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=K_NEIGHBORS + 1)

        for i in range(n):
            # Exclude self (index 0 is always self with distance 0)
            neighbor_indices = indices[i, 1:]
            neighbor_mac = mac_values[neighbor_indices]

            local_median = np.median(neighbor_mac)
            local_mad = np.median(np.abs(neighbor_mac - local_median))

            deviation = abs(mac_values[i] - local_median)
            reasons = []

            # Relative deviation check
            if local_median != 0 and deviation / abs(local_median) > RELATIVE_DEVIATION_THRESHOLD:
                reasons.append(f"relative_dev={deviation / abs(local_median):.4f}")

            # MAD check
            if local_mad > 0 and deviation > MAD_MULTIPLIER * local_mad:
                reasons.append(f"mad_score={deviation / local_mad:.2f}")

            if reasons:
                gid = gids[i]
                point = grid_by_id[gid]
                candidates.append({
                    'grid_id': gid,
                    'sample_id': point['sample_id'],
                    'tax_equity': tax_equity,
                    'ydmg': point['y_damage_distribution_exponent'],
                    'eta': point['eta'],
                    'rho': point['rho'],
                    'mac_2030': mac_values[i],
                    'delta_t_2100': runs[gid]['delta_t_2100'],
                    'objective_value': runs[gid]['objective_value'],
                    'detection_reason': ';'.join(reasons),
                })

    return candidates


def build_tasks(candidates, runs, grid):
    """Build warm-start tasks: for each candidate, find 2 nearest non-candidate neighbors.

    Uses KD-tree per tax_equity group to find nearest successful non-candidate
    points in normalized parameter space.
    """
    candidate_gids = {c['grid_id'] for c in candidates}

    # Group runs by tax_equity
    groups = {}
    for gid, run in runs.items():
        tax = run['tax_equity']
        groups.setdefault(tax, []).append(run)

    # Build KD-tree of non-candidates per tax_equity
    non_cand_trees = {}
    non_cand_gids = {}
    non_cand_coords = {}

    for tax_equity, group_runs in groups.items():
        nc_runs = [r for r in group_runs if r['grid_id'] not in candidate_gids]
        n = len(nc_runs)
        coords = np.empty((n, 3))
        gids = []
        for i, run in enumerate(nc_runs):
            coords[i] = normalize_params(run['eta'], run['rho'],
                                         run['y_damage_distribution_exponent'])
            gids.append(run['grid_id'])
        non_cand_trees[tax_equity] = KDTree(coords)
        non_cand_gids[tax_equity] = gids
        non_cand_coords[tax_equity] = coords

    tasks = []
    task_id = 0

    for cand in candidates:
        tax = cand['tax_equity']
        point_coord = normalize_params(cand['eta'], cand['rho'], cand['ydmg'])

        tree = non_cand_trees[tax]
        distances, indices = tree.query(point_coord, k=N_WARMSTART_NEIGHBORS)

        # Handle case where k returns scalar (only 1 neighbor)
        if np.ndim(indices) == 0:
            indices = [int(indices)]
        else:
            indices = list(indices)

        for rank, idx in enumerate(indices):
            nbr_gid = non_cand_gids[tax][idx]
            task_id += 1
            tasks.append({
                'task_id': task_id,
                'candidate_grid_id': cand['grid_id'],
                'neighbor_grid_id': nbr_gid,
                'neighbor_rank': rank + 1,
                'tax_equity': cand['tax_equity'],
                'ydmg': cand['ydmg'],
                'eta': cand['eta'],
                'rho': cand['rho'],
            })

    return tasks


def write_candidates_csv(candidates):
    """Write candidate manifest."""
    columns = [
        'grid_id', 'sample_id', 'tax_equity', 'ydmg',
        'eta', 'rho', 'mac_2030', 'delta_t_2100', 'objective_value',
        'detection_reason',
    ]
    with open(CANDIDATES_CSV, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for c in candidates:
            writer.writerow([c[col] for col in columns])


def write_tasks_csv(tasks):
    """Write task manifest for SLURM array jobs."""
    columns = [
        'task_id', 'candidate_grid_id', 'neighbor_grid_id', 'neighbor_rank',
        'tax_equity', 'ydmg', 'eta', 'rho',
    ]
    with open(TASKS_CSV, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for t in tasks:
            writer.writerow([t[col] for col in columns])


def main():
    grid = load_grid()
    print(f"Loaded {len(grid)} grid points from {GRID_CSV}")

    runs = load_successful_runs()
    print(f"Found {len(runs)} successful runs")

    candidates = detect_candidates(grid, runs)
    print(f"\nDetected {len(candidates)} outlier candidates")

    # Per-tax_equity breakdown
    tax_counts = {}
    for c in candidates:
        tax_counts[c['tax_equity']] = tax_counts.get(c['tax_equity'], 0) + 1

    print(f"\nCandidates by tax_equity:")
    for tax, count in sorted(tax_counts.items()):
        print(f"  tax={tax:.4f}: {count}")

    # Detection reason breakdown
    both = 0
    rel_only = 0
    mad_only = 0
    for c in candidates:
        reasons = c['detection_reason']
        has_rel = 'relative_dev' in reasons
        has_mad = 'mad_score' in reasons
        if has_rel and has_mad:
            both += 1
        elif has_rel:
            rel_only += 1
        elif has_mad:
            mad_only += 1

    print(f"\nBy detection reason:")
    print(f"  Both criteria:     {both}")
    print(f"  Relative dev only: {rel_only}")
    print(f"  MAD score only:    {mad_only}")

    tasks = build_tasks(candidates, runs, grid)
    print(f"\nGenerated {len(tasks)} warm-start tasks")

    write_candidates_csv(candidates)
    print(f"Wrote candidates: {CANDIDATES_CSV}")

    write_tasks_csv(tasks)
    print(f"Wrote tasks: {TASKS_CSV}")

    print(f"\nNext steps:")
    print(f"  1. python3 scripts/create_058_warmstart_loose_slurm.py")
    print(f"  2. rsync task CSV + runner + SLURM scripts to Sherlock")
    print(f"  3. sbatch scripts/sherlock_submit_058_warmstart_loose_a.sh")


if __name__ == '__main__':
    main()
