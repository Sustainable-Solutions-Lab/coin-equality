#!/bin/bash
#
# Submit all 6 config_052 sweeps to Sherlock (2,646 total jobs).
# Each sweep is a separate job array of 441 runs.
#
# Usage: bash sherlock_submit_052.sh

RHO_VALUES=(0.000 0.005 0.015 0.026 0.035 0.045)

echo "Submitting config_052 sweeps for ${#RHO_VALUES[@]} rho values..."
echo ""

for rho in "${RHO_VALUES[@]}"; do
    echo "Submitting rho=${rho}..."
    sbatch scripts/sherlock_submit_052_rho${rho}.sh
    echo ""
done

echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
