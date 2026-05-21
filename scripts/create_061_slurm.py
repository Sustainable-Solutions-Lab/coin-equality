"""Create SLURM submission scripts for config 061 (9000 runs in 9 batches of 1000)."""
from pathlib import Path

BATCHES = [
    ("a", 1, 1000),
    ("b", 1001, 2000),
    ("c", 2001, 3000),
    ("d", 3001, 4000),
    ("e", 4001, 5000),
    ("f", 5001, 6000),
    ("g", 6001, 7000),
    ("h", 7001, 8000),
    ("i", 8001, 9000),
]

for suffix, start, end in BATCHES:
    content = f"""\
#!/bin/bash
#SBATCH --job-name=coin_061_{suffix}
#SBATCH --output=/scratch/users/lamprini/coin-output/logs/coin_061_%A_%a.out
#SBATCH --error=/scratch/users/lamprini/coin-output/logs/coin_061_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-1000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lamprini@stanford.edu

ml python/3.12.1
source $HOME/coin_env/bin/activate
cd $HOME/coin-equality
TASK_ID=$(( $SLURM_ARRAY_TASK_ID + {start - 1} ))
python3 -u scripts/run_sweep_061.py $TASK_ID
deactivate
"""
    output_path = Path(__file__).resolve().parent / f"sherlock_submit_061_{suffix}.sh"
    with open(output_path, "w", newline="\n") as f:
        f.write(content)
    print(f"Wrote {output_path}")
