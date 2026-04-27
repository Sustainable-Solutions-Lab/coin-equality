#!/bin/bash
#SBATCH --job-name=coin_052_rho0.005
#SBATCH --output=/scratch/users/lamprini/coin-output/logs/coin_052_rho0.005_%A_%a.out
#SBATCH --error=/scratch/users/lamprini/coin-output/logs/coin_052_rho0.005_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=89,91,93,94,98,99,102,104,105,106,108,111,113,114,116,117,118,125,126,127,128,131,132,133,134,135,137,139,140,141,142,143,144,146,147,149,151,152,153,154,157,158,159,160,163,164,165,167,168,169,172,173,174,175,179,188,208,209,210,222,223,225,226,227,228,230,231,234,237,239,242,247,249,250,259,263,296,316,346,347,348,350,351,352,365,367,378,384,385,386,387,389,390,391,392,393,394,395,396,409,410,411,415,417,418,419,439
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lamprini@stanford.edu

ml python/3.12.1
source $HOME/coin_env/bin/activate

cd $HOME/coin-equality

python3 -u scripts/run_sweep_052.py 0.005 $SLURM_ARRAY_TASK_ID

deactivate
