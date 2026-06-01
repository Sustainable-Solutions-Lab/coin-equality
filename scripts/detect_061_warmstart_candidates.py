"""Detect convergence outliers in config_061 and generate warm-start task manifests.

Detection method:
- For each (sweep, tax_equity) group, sort by sample_id
- Compute rolling median of mac_2030 (window=21)
- Flag if |actual - rolling_median| / rolling_median > 0.03 (3% relative deviation)
- Also flag if |actual - rolling_median| > 3 * local MAD (robust z-score)
- Either condition triggers flagging

Outputs:
  json/config_061_warmstart_loose_candidates.csv  (one row per candidate)
  json/config_061_warmstart_loose_tasks.csv       (one row per warm-start task)

Usage:
    python scripts/detect_061_warmstart_candidates.py
"""

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_061_grid.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'config_061'
WARMSTART_DIR = PROJECT_ROOT / 'data' / 'output' / 'config_061_warmstart'
CANDIDATES_CSV = PROJECT_ROOT / 'json' / 'config_061_warmstart_loose_candidates.csv'
TASKS_CSV = PROJECT_ROOT / 'json' / 'config_061_warmstart_loose_tasks.csv'

ROLLING_WINDOW = 21
RELATIVE_DEVIATION_THRESHOLD = 0.03
MAD_MULTIPLIER = 3.0


def load_grid():
    """Load grid CSV and return list of dicts."""
    rows = []
    with open(GRID_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'grid_id': int(row['grid_id']),
                'sweep': row['sweep'],
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            })
    return rows


def load_successful_runs():
    """Scan output directories and return dict: grid_id -> best run_status row.

    When multiple directories exist for the same grid_id (e.g., from resubmissions),
    the one with the best (most negative) objective_value is kept.
    """
    runs = {}
    search_dirs = [OUTPUT_DIR]
    if WARMSTART_DIR.exists():
        search_dirs.append(WARMSTART_DIR)

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for entry in base_dir.iterdir():
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
                            'sweep': row['sweep'],
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


def rolling_median(values, window):
    """Compute rolling median with edge-adaptive window (minimum 3 points)."""
    n = len(values)
    result = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.median(values[lo:hi])
    return result


def rolling_mad(values, medians, window):
    """Compute rolling MAD (median absolute deviation) with edge-adaptive window."""
    n = len(values)
    result = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local_deviations = np.abs(values[lo:hi] - medians[lo:hi])
        result[i] = np.median(local_deviations)
    return result


def detect_candidates(grid, runs):
    """Detect outlier candidates using rolling median and MAD methods.

    Returns list of candidate dicts.
    """
    # Build lookup: grid_id -> grid point
    grid_by_id = {g['grid_id']: g for g in grid}

    # Group by (sweep, tax_equity)
    groups = {}
    for gid, run in runs.items():
        key = (run['sweep'], run['tax_equity'])
        groups.setdefault(key, []).append(run)

    candidates = []

    for (sweep, tax_equity), group_runs in sorted(groups.items()):
        group_runs.sort(key=lambda r: r['sample_id'])
        sample_ids = np.array([r['sample_id'] for r in group_runs])
        mac_values = np.array([r['mac_2030'] for r in group_runs])

        med = rolling_median(mac_values, ROLLING_WINDOW)
        mad = rolling_mad(mac_values, med, ROLLING_WINDOW)

        for i, run in enumerate(group_runs):
            deviation = abs(run['mac_2030'] - med[i])
            reasons = []

            # Relative deviation check
            if med[i] != 0 and deviation / abs(med[i]) > RELATIVE_DEVIATION_THRESHOLD:
                reasons.append(f"relative_dev={deviation / abs(med[i]):.4f}")

            # MAD check (avoid division by zero for very flat regions)
            if mad[i] > 0 and deviation > MAD_MULTIPLIER * mad[i]:
                reasons.append(f"mad_score={deviation / mad[i]:.2f}")

            if reasons:
                gid = run['grid_id']
                point = grid_by_id[gid]
                candidates.append({
                    'grid_id': gid,
                    'sweep': sweep,
                    'sample_id': run['sample_id'],
                    'tax_equity': tax_equity,
                    'ydmg': point['y_damage_distribution_exponent'],
                    'eta': point['eta'],
                    'rho': point['rho'],
                    'mac_2030': run['mac_2030'],
                    'delta_t_2100': run['delta_t_2100'],
                    'objective_value': run['objective_value'],
                    'detection_reason': ';'.join(reasons),
                })

    return candidates


def build_tasks(candidates, runs, grid):
    """Build warm-start tasks: one per (candidate, neighbor) pair.

    Within a (sweep, tax_equity) group, neighbors share the same sweep and
    tax_equity but have sample_id ± 1.  Grid_ids within a sweep are spaced
    by 3 (3 tax_equity values interleaved), so the neighbor grid_id is ± 3.
    """
    grid_by_id = {g['grid_id']: g for g in grid}

    # Build lookup: (sweep, tax_equity, sample_id) -> grid_id
    sid_to_gid = {}
    for g in grid:
        key = (g['sweep'], g['tax_equity'], g['sample_id'])
        sid_to_gid[key] = g['grid_id']

    # Determine sample_id range per sweep
    sweep_sample_range = {}
    for g in grid:
        sw = g['sweep']
        sid = g['sample_id']
        if sw not in sweep_sample_range:
            sweep_sample_range[sw] = (sid, sid)
        else:
            lo, hi = sweep_sample_range[sw]
            sweep_sample_range[sw] = (min(lo, sid), max(hi, sid))

    tasks = []
    task_id = 0

    for cand in candidates:
        sweep = cand['sweep']
        tax_equity = cand['tax_equity']
        sample_id = cand['sample_id']
        sid_lo, sid_hi = sweep_sample_range[sweep]

        # Left neighbor: sample_id - 1
        if sample_id > sid_lo:
            nbr_gid = sid_to_gid.get((sweep, tax_equity, sample_id - 1))
            if nbr_gid is not None and nbr_gid in runs:
                task_id += 1
                nbr_point = grid_by_id[nbr_gid]
                tasks.append({
                    'task_id': task_id,
                    'candidate_grid_id': cand['grid_id'],
                    'neighbor_grid_id': nbr_gid,
                    'neighbor_side': 'left',
                    'sweep': sweep,
                    'tax_equity': tax_equity,
                    'ydmg': cand['ydmg'],
                    'eta': cand['eta'],
                    'rho': cand['rho'],
                })

        # Right neighbor: sample_id + 1
        if sample_id < sid_hi:
            nbr_gid = sid_to_gid.get((sweep, tax_equity, sample_id + 1))
            if nbr_gid is not None and nbr_gid in runs:
                task_id += 1
                nbr_point = grid_by_id[nbr_gid]
                tasks.append({
                    'task_id': task_id,
                    'candidate_grid_id': cand['grid_id'],
                    'neighbor_grid_id': nbr_gid,
                    'neighbor_side': 'right',
                    'sweep': sweep,
                    'tax_equity': tax_equity,
                    'ydmg': cand['ydmg'],
                    'eta': cand['eta'],
                    'rho': cand['rho'],
                })

    return tasks


def write_candidates_csv(candidates):
    """Write candidate manifest."""
    columns = [
        'grid_id', 'sweep', 'sample_id', 'tax_equity', 'ydmg',
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
        'task_id', 'candidate_grid_id', 'neighbor_grid_id', 'neighbor_side',
        'sweep', 'tax_equity', 'ydmg', 'eta', 'rho',
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

    # Per-group breakdown
    group_counts = {}
    for c in candidates:
        key = (c['sweep'], c['tax_equity'])
        group_counts[key] = group_counts.get(key, 0) + 1

    print(f"\nCandidates by (sweep, tax_equity):")
    for (sweep, tax), count in sorted(group_counts.items()):
        print(f"  {sweep}, tax={tax:.4f}: {count}")

    tasks = build_tasks(candidates, runs, grid)
    print(f"\nGenerated {len(tasks)} warm-start tasks")

    write_candidates_csv(candidates)
    print(f"Wrote candidates: {CANDIDATES_CSV}")

    write_tasks_csv(tasks)
    print(f"Wrote tasks: {TASKS_CSV}")

    print(f"\nNext steps:")
    print(f"  1. python3 scripts/create_061_warmstart_loose_slurm.py")
    print(f"  2. rsync task CSV + runner + SLURM script to Sherlock")
    print(f"  3. sbatch scripts/sherlock_submit_061_warmstart_loose.sh")


if __name__ == '__main__':
    main()
