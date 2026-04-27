#!/bin/bash
#SBATCH --job-name=coin_052_rho0.000
#SBATCH --output=/scratch/users/lamprini/coin-output/logs/coin_052_rho0.000_%A_%a.out
#SBATCH --error=/scratch/users/lamprini/coin-output/logs/coin_052_rho0.000_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=82,83,84,105,111,112,118,120,125,126,129,136,146,147,163,166,167,168,187,188,208,224,225,249,268,279,294,301,311,325,327,329,332,339,340,344,345,346,347,349,350,352,354,355,359,360,361,365,367,370,371,373,375,377,389,390,391,393,407,409,413,415,430,435
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lamprini@stanford.edu

ml python/3.12.1
source $HOME/coin_env/bin/activate

cd $HOME/coin-equality

python3 -u scripts/run_sweep_052.py 0.000 $SLURM_ARRAY_TASK_ID

deactivate
