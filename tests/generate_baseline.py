"""
Generate the golden regression baseline in tests/baseline/ from the current code.

Run once from the repository root, before refactoring:

    python -m tests.generate_baseline

The baseline is committed. Regenerate it only for an explicitly approved numerical
change, and say so in the commit message.
"""

import json
import platform
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import scipy
import pandas
import nlopt

from tests.regression_cases import (
    REPO_ROOT,
    BASELINE_DIR,
    FORWARD_CASES,
    OPTIMIZATION_CASES,
    RTOL_ROUNDOFF,
    O2_SPREAD_SAFETY_FACTOR,
    O2_CALIBRATION_PERTURBATIONS,
    run_forward_case,
    run_forward_results,
    run_optimization_case,
    run_kernels,
    results_csv_header,
    relative_difference,
)


def calibrate_o2_tolerance(o2_baseline):
    """Per-key O2 tolerance from the spread of runs with perturbed initial guesses."""
    spread = {key: 0.0 for key in o2_baseline}
    for perturbation in O2_CALIBRATION_PERTURBATIONS:
        perturbed = run_optimization_case('O2', perturbation)
        for key in o2_baseline:
            spread[key] = max(spread[key], relative_difference(perturbed[key], o2_baseline[key]))
    return spread, {key: max(O2_SPREAD_SAFETY_FACTOR * value, RTOL_ROUNDOFF) for key, value in spread.items()}


def main():
    BASELINE_DIR.mkdir(exist_ok=True)

    for name in FORWARD_CASES:
        print(f'Forward case {name}')
        np.savez_compressed(BASELINE_DIR / f'{name}.npz', **run_forward_case(name))

    optimization_outputs = {}
    for name in OPTIMIZATION_CASES:
        print(f'Optimization case {name}')
        optimization_outputs[name] = run_optimization_case(name, 0.0)
        np.savez_compressed(BASELINE_DIR / f'{name}.npz', **optimization_outputs[name])

    print('Calibrating O2 tolerance')
    o2_spread, o2_rtol = calibrate_o2_tolerance(optimization_outputs['O2'])

    print('Kernel checks')
    np.savez_compressed(BASELINE_DIR / 'K.npz', **run_kernels())

    with tempfile.TemporaryDirectory() as tmp:
        csv_header = results_csv_header(run_forward_results('R1'), tmp)

    git_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    git_dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'scripts'], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    meta = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'git_hash': git_hash,
        'src_or_scripts_modified': bool(git_dirty),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'pandas': pandas.__version__,
        'nlopt': nlopt.__version__,
        'o2_spread': o2_spread,
        'o2_rtol': o2_rtol,
        'csv_header': csv_header,
    }
    with open(BASELINE_DIR / 'baseline_meta.json', 'w', newline='\n') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')
    print(f'Baseline written to {BASELINE_DIR}')


if __name__ == '__main__':
    main()
