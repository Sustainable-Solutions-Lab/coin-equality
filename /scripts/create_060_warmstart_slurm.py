"""Create the SLURM submission script for config_060 warmstart on Sherlock."""
from pathlib import Path

content = """\
#!/bin/bash
#SBATCH --job-name=coin_060ws
#SBATCH --output=/scratch/users/lamprini/coin-output/logs/coin_060ws_%A_%a.out
#SBATCH --error=/scratch/users/lamprini/coin-output/logs/coin_060ws_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-10
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lamprini@stanford.edu

ml python/3.12.1
source $HOME/coin_env/bin/activate
cd $HOME/coin-equality
python3 -u scripts/run_sweep_060_warmstart.py $SLURM_ARRAY_TASK_ID
deactivate
"""

output_path = Path(__file__).resolve().parent / "sherlock_submit_060_warmstart.sh"
with open(output_path, "w", newline="\n") as f:
    f.write(content)
print(f"Wrote {output_path}")
