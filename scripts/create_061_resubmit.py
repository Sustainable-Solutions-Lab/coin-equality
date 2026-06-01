"""Identify missing config_061 grid_ids and generate resubmission files.

Scans data/output/config_061/ for each grid_id in config_061_grid.csv,
checking for a run_status.csv with success=True. Only considers grid_ids
from batches a-h (1-8000); batch i (8001-9000) is excluded as it is
submitted separately via sherlock_submit_061_i.sh.

Writes:
  - json/config_061_resubmit.csv: tracking manifest of failed grid_ids
  - scripts/sherlock_submit_061_resubmit.sh: SLURM script with explicit grid_id array

Re-run after resubmission completes to verify all grid_ids are present.
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_061_grid.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'config_061'
RESUBMIT_CSV = PROJECT_ROOT / 'json' / 'config_061_resubmit.csv'
SLURM_SCRIPT = PROJECT_ROOT / 'scripts' / 'sherlock_submit_061_resubmit.sh'

# Batch i (grid_ids 8001-9000) is submitted separately; exclude from resubmission scan
MAX_GRID_ID_FOR_RESUBMIT = 8000


def load_grid():
    """Load the grid CSV and return list of dicts."""
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


def build_run_name(point):
    """Build the run_name prefix (without timestamp) for a grid point."""
    sweep = point['sweep']
    tax_equity = point['tax_equity']
    ydmg = point['y_damage_distribution_exponent']
    eta = point['eta']
    rho = point['rho']
    return f'config_061_{sweep}_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_eta-{eta:.4f}_rho-{rho:.4f}'


def find_successful_grid_ids():
    """Scan output directories and return set of grid_ids with success=True."""
    successful = set()
    if not OUTPUT_DIR.exists():
        return successful
    for entry in OUTPUT_DIR.iterdir():
        if not entry.is_dir():
            continue
        status_path = entry / 'run_status.csv'
        if not status_path.exists():
            continue
        with open(status_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['success'] == 'True':
                    successful.add(int(row['grid_id']))
    return successful


def classify_failure(point):
    """Determine why a grid_id is missing.

    Returns one of:
      'collision' - no directory found, or directory was overwritten by another grid_id
      'io_error'  - directory exists but no run_status.csv
      'failed'    - directory exists, run_status.csv exists with success=False
    """
    run_name_prefix = build_run_name(point)
    matching_dirs = [
        d for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and d.name.startswith(run_name_prefix)
    ]
    if not matching_dirs:
        return 'collision'
    for d in matching_dirs:
        status_path = d / 'run_status.csv'
        if not status_path.exists():
            return 'io_error'
        with open(status_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['success'] == 'False':
                    return 'failed'
    return 'collision'


def write_resubmit_csv(missing_points):
    """Write tracking manifest CSV."""
    columns = [
        'grid_id', 'sweep', 'sample_id', 'tax_equity',
        'y_damage_distribution_exponent', 'eta', 'rho', 'failure_reason',
    ]
    with open(RESUBMIT_CSV, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for point, reason in missing_points:
            writer.writerow([
                point['grid_id'], point['sweep'], point['sample_id'],
                point['tax_equity'], point['y_damage_distribution_exponent'],
                point['eta'], point['rho'], reason,
            ])


def write_slurm_script(grid_ids):
    """Write SLURM submission script with explicit grid_id array."""
    array_str = ','.join(str(gid) for gid in sorted(grid_ids))
    content = f"""\
#!/bin/bash
#SBATCH --job-name=coin_061_resub
#SBATCH --output=/scratch/users/lamprini/coin-output/logs/coin_061_%A_%a.out
#SBATCH --error=/scratch/users/lamprini/coin-output/logs/coin_061_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array={array_str}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lamprini@stanford.edu

ml python/3.12.1
source $HOME/coin_env/bin/activate
cd $HOME/coin-equality
python3 -u scripts/run_sweep_061.py $SLURM_ARRAY_TASK_ID
deactivate
"""
    with open(SLURM_SCRIPT, 'w', newline='\n') as f:
        f.write(content)


def main():
    grid = load_grid()
    print(f"Loaded {len(grid)} grid points from {GRID_CSV}")

    successful_grid_ids = find_successful_grid_ids()
    print(f"Found {len(successful_grid_ids)} successful runs in {OUTPUT_DIR}")

    all_grid_ids = {point['grid_id'] for point in grid if point['grid_id'] <= MAX_GRID_ID_FOR_RESUBMIT}
    missing_grid_ids = all_grid_ids - successful_grid_ids
    print(f"Missing: {len(missing_grid_ids)} grid_ids (excluding batch i, grid_ids > {MAX_GRID_ID_FOR_RESUBMIT})")

    if not missing_grid_ids:
        print("\nAll grid_ids have successful results. Nothing to resubmit.")
        return

    grid_by_id = {point['grid_id']: point for point in grid}
    missing_points = []
    for gid in sorted(missing_grid_ids):
        point = grid_by_id[gid]
        reason = classify_failure(point)
        missing_points.append((point, reason))

    reason_counts = {}
    sweep_counts = {}
    for point, reason in missing_points:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        sweep_counts[point['sweep']] = sweep_counts.get(point['sweep'], 0) + 1

    print(f"\nBy failure reason:")
    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count}")

    print(f"\nBy sweep:")
    for sweep, count in sorted(sweep_counts.items()):
        print(f"  {sweep}: {count}")

    write_resubmit_csv(missing_points)
    print(f"\nWrote tracking manifest: {RESUBMIT_CSV}")

    write_slurm_script(missing_grid_ids)
    print(f"Wrote SLURM script: {SLURM_SCRIPT}")

    print(f"\nNext steps:")
    print(f"  1. python3 scripts/create_061_resubmit.py  (already done)")
    print(f"  2. rsync -avz json/config_061_resubmit.csv lamprini@login.sherlock.stanford.edu:~/coin-equality/json/")
    print(f"  3. rsync -avz scripts/sherlock_submit_061_resubmit.sh lamprini@login.sherlock.stanford.edu:~/coin-equality/scripts/")
    print(f"  4. sbatch scripts/sherlock_submit_061_resubmit.sh")


if __name__ == '__main__':
    main()
