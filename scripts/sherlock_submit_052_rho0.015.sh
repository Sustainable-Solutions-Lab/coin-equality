#!/bin/bash
#SBATCH --job-name=coin_052_rho0.015
#SBATCH --output=/scratch/users/lamprini/coin-output/logs/coin_052_rho0.015_%A_%a.out
#SBATCH --error=/scratch/users/lamprini/coin-output/logs/coin_052_rho0.015_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-441
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lamprini@stanford.edu

ml python/3.12.1
source $HOME/coin_env/bin/activate

cd $HOME/coin-equality

python3 -u scripts/run_sweep_052.py 0.015 $SLURM_ARRAY_TASK_ID

deactivate
