"""Run all 30 prod_discount_exact warm-start tasks in parallel locally.

Launches all tasks using multiprocessing, monitors progress, and reports
results including r_consumption accuracy at 2030.

Usage:
    python scripts/run_prod_discount_exact_parallel.py
    python scripts/run_prod_discount_exact_parallel.py --max-workers 4
"""

import csv
import os
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_NAME = 'prod_discount_exact'
GRID_CSV = PROJECT_ROOT / 'json' / f'{CONFIG_NAME}_grid.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / CONFIG_NAME
RUNNER_SCRIPT = PROJECT_ROOT / 'scripts' / 'run_warmstart_prod.py'


def run_task(task_id):
    """Run a single warm-start task and return (task_id, success, runtime)."""
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), CONFIG_NAME, str(task_id)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    success = result.returncode == 0
    return task_id, success, elapsed, result.stdout[-500:] if result.stdout else '', result.stderr[-500:] if result.stderr else ''


def check_results():
    """Check all completed runs for r_consumption accuracy."""
    grid = pd.read_csv(GRID_CSV)
    targets = [3.0, 4.0, 5.0, 6.0, 7.0]

    print(f'\n{"=" * 100}')
    print('RESULTS SUMMARY')
    print(f'{"=" * 100}')
    print(f'{"gid":>3} | {"tax":>5} | {"ydmg":>4} | {"rho":>8} | {"target_r":>8} | '
          f'{"actual_r":>8} | {"error":>8} | {"OCP@2030":>10} | {"dT@2100":>8} | {"status":>12} | {"evals":>7}')
    print('-' * 100)

    results = []
    for _, grow in grid.iterrows():
        gid = int(grow['grid_id'])
        tax = float(grow['tax_equity'])
        ydmg = float(grow['y_damage_distribution_exponent'])
        rho = float(grow['rho'])

        # Determine target r for this gid
        group_idx = (gid - 1) // 5
        target_idx = (gid - 1) % 5
        target_r = targets[target_idx]

        # Find output directory
        matching_dirs = sorted(OUTPUT_DIR.glob(f'{CONFIG_NAME}_gid-{gid}_*'))
        if not matching_dirs:
            print(f'{gid:3d} | {tax:5.3f} | {ydmg:4.1f} | {rho:8.6f} | {target_r:7.1f}% | '
                  f'{"N/A":>8} | {"N/A":>8} | {"N/A":>10} | {"N/A":>8} | {"NOT FOUND":>12} | {"N/A":>7}')
            continue

        run_dir = matching_dirs[-1]  # Latest timestamp
        status_file = run_dir / 'run_status.csv'
        if not status_file.exists():
            print(f'{gid:3d} | {tax:5.3f} | {ydmg:4.1f} | {rho:8.6f} | {target_r:7.1f}% | '
                  f'{"N/A":>8} | {"N/A":>8} | {"N/A":>10} | {"N/A":>8} | {"NO STATUS":>12} | {"N/A":>7}')
            continue

        with open(status_file, 'r') as f:
            status_row = list(csv.DictReader(f))[0]

        success = status_row['success'] == 'True'
        n_evals = status_row.get('n_evaluations', 'N/A')

        if not success:
            print(f'{gid:3d} | {tax:5.3f} | {ydmg:4.1f} | {rho:8.6f} | {target_r:7.1f}% | '
                  f'{"N/A":>8} | {"N/A":>8} | {"N/A":>10} | {"N/A":>8} | {"FAILED":>12} | {n_evals:>7}')
            continue

        # Read actual r_consumption
        results_files = sorted(run_dir.glob('*_results.csv'))
        df = pd.read_csv(results_files[-1])
        t_col = [c for c in df.columns if c.startswith('t,')][0]
        years = df[t_col].values
        idx_r = np.argmin(np.abs(years - 2030))
        actual_r = df['r_consumption'].values[idx_r] * 100

        ocp = float(status_row['mac_2030'])
        dt = float(status_row['delta_t_2100'])
        error_pct = actual_r - target_r
        opt_status = status_row['optimizer_status']

        flag = ' ***' if abs(error_pct) > 0.2 else ''

        print(f'{gid:3d} | {tax:5.3f} | {ydmg:4.1f} | {rho:8.6f} | {target_r:7.1f}% | '
              f'{actual_r:7.4f}% | {error_pct:+7.4f}% | ${ocp:9.2f} | {dt:7.4f}C | {opt_status:>12} | {n_evals:>7}{flag}')

        results.append({
            'gid': gid, 'tax': tax, 'ydmg': ydmg, 'rho': rho,
            'target_r': target_r, 'actual_r': actual_r,
            'error_pct': error_pct, 'ocp': ocp, 'dt': dt,
            'n_evals': int(n_evals), 'status': opt_status,
        })

    if results:
        errors = [abs(r['error_pct']) for r in results]
        print(f'\nError stats: mean={np.mean(errors):.4f}%, max={np.max(errors):.4f}%, '
              f'runs with >0.2% error: {sum(1 for e in errors if e > 0.2)}/{len(errors)}')

    return results


def main():
    max_workers = cpu_count() - 1

    # Parse --max-workers
    for i, arg in enumerate(sys.argv):
        if arg == '--max-workers' and i + 1 < len(sys.argv):
            max_workers = int(sys.argv[i + 1])

    # Check CSVs exist
    cand_csv = PROJECT_ROOT / 'json' / f'{CONFIG_NAME}_candidates.csv'
    if not GRID_CSV.exists() or not cand_csv.exists():
        print('Grid or candidates CSV not found. Run create_prod_discount_exact_csvs.py first.')
        sys.exit(1)

    grid = pd.read_csv(GRID_CSV)
    n_tasks = len(grid)

    print(f'Config: {CONFIG_NAME}')
    print(f'Tasks: {n_tasks}')
    print(f'Workers: {max_workers}')
    print(f'Output: {OUTPUT_DIR}')
    print()

    # Check if --check-only flag
    if '--check-only' in sys.argv:
        check_results()
        return

    # Launch all tasks in parallel
    task_ids = list(range(1, n_tasks + 1))

    print(f'Launching {n_tasks} warm-start runs...')
    start_all = time.time()

    with Pool(processes=max_workers) as pool:
        results = []
        for task_id, success, elapsed, stdout, stderr in pool.imap_unordered(run_task, task_ids):
            status_str = 'OK' if success else 'FAILED'
            print(f'  Task {task_id:2d}/{n_tasks}: {status_str} ({elapsed:.0f}s)')
            if not success:
                print(f'    stderr: {stderr}')
            results.append((task_id, success, elapsed))

    total_time = time.time() - start_all
    n_ok = sum(1 for _, s, _ in results if s)
    print(f'\nAll tasks complete: {n_ok}/{n_tasks} succeeded in {total_time:.0f}s')

    # Check results
    check_results()


if __name__ == '__main__':
    main()
