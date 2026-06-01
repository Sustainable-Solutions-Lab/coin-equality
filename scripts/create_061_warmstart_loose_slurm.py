"""Create SLURM submission scripts for config_061 loose warm-start on Sherlock.

Reads json/config_061_warmstart_loose_tasks.csv to determine total task count,
then generates multiple array job scripts (max 1000 tasks each, Sherlock limit).

Usage:
    python scripts/create_061_warmstart_loose_slurm.py
"""

import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_CSV = PROJECT_ROOT / 'json' / 'config_061_warmstart_loose_tasks.csv'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

MAX_ARRAY_SIZE = 1000
BATCH_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def count_tasks():
    """Count rows in tasks CSV."""
    with open(TASKS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def main():
    n_tasks = count_tasks()
    n_batches = math.ceil(n_tasks / MAX_ARRAY_SIZE)
    print(f"Found {n_tasks} tasks in {TASKS_CSV}")
    print(f"Splitting into {n_batches} batch(es) of up to {MAX_ARRAY_SIZE}")

    scripts = []
    for i in range(n_batches):
        lo = i * MAX_ARRAY_SIZE + 1
        hi = min((i + 1) * MAX_ARRAY_SIZE, n_tasks)
        letter = BATCH_LETTERS[i]
        script_name = f'sherlock_submit_061_warmstart_loose_{letter}.sh'
        script_path = SCRIPTS_DIR / script_name

        content = f"""\
#!/bin/bash
#SBATCH --job-name=coin_061wsl{letter}
#SBATCH --output=/scratch/users/lamprini/coin-output/logs/coin_061wsl_%A_%a.out
#SBATCH --error=/scratch/users/lamprini/coin-output/logs/coin_061wsl_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array={lo}-{hi}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lamprini@stanford.edu

ml python/3.12.1
source $HOME/coin_env/bin/activate
cd $HOME/coin-equality
python3 -u scripts/run_061_warmstart_loose.py $SLURM_ARRAY_TASK_ID
deactivate
"""

        with open(script_path, 'w', newline='\n') as f:
            f.write(content)

        scripts.append((script_name, lo, hi))
        print(f"  {script_name}: tasks {lo}-{hi} ({hi - lo + 1} tasks)")

    # Clean up old single script if it exists
    old_script = SCRIPTS_DIR / 'sherlock_submit_061_warmstart_loose.sh'
    if old_script.exists():
        old_script.unlink()
        print(f"\nRemoved old single script: {old_script.name}")

    print(f"\nSubmit on Sherlock (max 2 at a time):")
    for name, lo, hi in scripts:
        print(f"  sbatch scripts/{name}")


if __name__ == '__main__':
    main()
