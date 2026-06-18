#!/usr/bin/env python3
"""
Submit 9 warm-start refinement jobs to Sherlock in parallel.
Each job runs one baseline case independently.
Total runtime: ~2 hours (all 9 cases run simultaneously)
"""

import subprocess
import sys
from pathlib import Path

# Configuration
PROJECT_DIR = '/home/users/lamprini/coin-equality'
CASES = [
    ('0.000', '0.00'),
    ('0.000', '0.50'),
    ('0.000', '1.00'),
    ('0.500', '0.00'),
    ('0.500', '0.50'),
    ('0.500', '1.00'),
    ('0.998', '0.00'),
    ('0.998', '0.50'),
    ('0.998', '1.00'),
]

def create_job_script(tax, ydmg):
    """Create individual job script for one case."""
    script = f"""#!/bin/bash
#SBATCH --job-name=warmstart_refined_{tax}_{ydmg}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=8GB
#SBATCH --partition=gpu
#SBATCH --output={PROJECT_DIR}/logs/warmstart_refined_tax-{tax}_ydmg-{ydmg}_%j.log

# Load environment
module load python/3.9

# Activate conda/venv
cd {PROJECT_DIR}
source .venv/bin/activate

# Run single case
echo "Starting warm-start refinement for tax={tax}, β={ydmg}"
python -c "
import sys
sys.path.insert(0, '{PROJECT_DIR}')
from scripts.run_prod_001_warmstart_refined import run_warmstart_case

success, elapsed = run_warmstart_case(
    'prod_001_warmstart_refined_tax-{tax}_ydmg-{ydmg}.json',
    '{tax}',
    '{ydmg}'
)

if success:
    print(f'✓ Case tax={tax}, β={ydmg} completed in {{elapsed:.0f}}s')
    sys.exit(0)
else:
    print(f'✗ Case tax={tax}, β={ydmg} FAILED')
    sys.exit(1)
"

echo "Warm-start refinement tax={tax}, β={ydmg} finished at $(date)"
"""
    return script


def main():
    print("="*80)
    print("SHERLOCK PARALLEL SUBMISSION: 9 Warm-Start Refinement Cases")
    print("="*80)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Number of cases: {len(CASES)}")
    print(f"Parallel execution: All 9 cases simultaneously")
    print(f"Expected total time: ~2 hours")
    print()

    # Create logs directory
    logs_dir = Path(PROJECT_DIR) / 'logs'
    logs_dir.mkdir(exist_ok=True)
    print(f"✓ Logs directory: {logs_dir}")
    print()

    # Submit jobs
    print("Submitting jobs...")
    print("-"*80)

    job_ids = []

    for tax, ydmg in CASES:
        # Create job script
        job_script = f'/tmp/warmstart_job_{tax}_{ydmg}.sh'
        script_content = create_job_script(tax, ydmg)

        with open(job_script, 'w') as f:
            f.write(script_content)

        # Make executable
        Path(job_script).chmod(0o755)

        # Submit job
        result = subprocess.run(
            ['sbatch', job_script],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR
        )

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f'✓ Submitted: tax={tax:>5}, β={ydmg:>5} (Job ID: {job_id})')
        else:
            print(f'✗ FAILED: tax={tax:>5}, β={ydmg:>5}')
            print(f'  Error: {result.stderr}')

    print()
    print("="*80)
    print("ALL JOBS SUBMITTED")
    print("="*80)
    print(f"Total jobs: {len(job_ids)}/9")
    print(f"Job IDs: {' '.join(job_ids)}")
    print()

    print("NEXT STEPS:")
    print("-"*80)
    print()
    print("1. Monitor job status:")
    print("   ssh login.sherlock.stanford.edu")
    print("   squeue -u lamprini")
    print()
    print("2. View real-time logs:")
    print(f"   tail -f {PROJECT_DIR}/logs/warmstart_refined_*.log")
    print()
    print("3. Wait for all jobs to complete (check with squeue)")
    print()
    print("4. Download results when complete:")
    print(f"   rsync -av lamprini@login.sherlock.stanford.edu:{PROJECT_DIR}/data/output/prod/prod_001_warmstart_refined_* ~/Code/coin-equality/data/output/prod/")
    print()
    print("="*80)


if __name__ == '__main__':
    main()
