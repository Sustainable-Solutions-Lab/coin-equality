#!/usr/bin/env python3
"""
Run multiple optimization cases in parallel with batch control.

Launches run_optimization.py for each JSON configuration file in parallel,
with a maximum number of concurrent jobs. Waits for jobs to complete before
starting new ones.

Usage:
    python run_parallel_batched.py <pattern> [--max-parallel N]

Arguments:
    pattern: File path pattern or glob pattern for JSON config files
    --max-parallel: Maximum number of jobs to run simultaneously (default: 14)

Example:
    python run_parallel_batched.py "json/config_038_*.json" --max-parallel 14
"""

import sys
import glob
import subprocess
import time
from pathlib import Path
import os

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    pattern = sys.argv[1]
    max_parallel = 14

    if '--max-parallel' in sys.argv:
        idx = sys.argv.index('--max-parallel')
        max_parallel = int(sys.argv[idx + 1])

    # Find all config files
    config_files = sorted(glob.glob(pattern))

    if not config_files:
        print(f"Error: No files found matching pattern: {pattern}")
        sys.exit(1)

    print(f"Found {len(config_files)} config files")
    print(f"Running {max_parallel} jobs in parallel")
    print("="*80)

    # Python executable
    python_exec = sys.executable

    # Track running processes
    processes = {}
    completed = 0
    failed = 0

    # Queue of configs to run
    queue = list(config_files)

    try:
        while queue or processes:
            # Start new jobs if we have capacity
            while len(processes) < max_parallel and queue:
                config_file = queue.pop(0)
                config_name = Path(config_file).stem

                print(f"Starting: {config_name}")

                # Launch process
                proc = subprocess.Popen(
                    [python_exec, 'scripts/run_optimization.py', config_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                processes[proc.pid] = {
                    'process': proc,
                    'config': config_name,
                    'start_time': time.time()
                }

            # Check for completed processes
            for pid in list(processes.keys()):
                proc_info = processes[pid]
                proc = proc_info['process']

                if proc.poll() is not None:
                    # Process finished
                    elapsed = time.time() - proc_info['start_time']

                    if proc.returncode == 0:
                        completed += 1
                        status = "✓ COMPLETE"
                    else:
                        failed += 1
                        status = "✗ FAILED"

                    print(f"{status}: {proc_info['config']} ({elapsed/60:.1f} min)")
                    print(f"  Progress: {completed + failed}/{len(config_files)} | Running: {len(processes)-1} | Queue: {len(queue)}")

                    del processes[pid]

            # Brief sleep to avoid busy waiting
            if processes:
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Stopping all jobs...")
        for proc_info in processes.values():
            proc_info['process'].terminate()
        sys.exit(1)

    print("\n" + "="*80)
    print("ALL JOBS FINISHED")
    print("="*80)
    print(f"Completed: {completed}/{len(config_files)}")
    print(f"Failed: {failed}/{len(config_files)}")

    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
