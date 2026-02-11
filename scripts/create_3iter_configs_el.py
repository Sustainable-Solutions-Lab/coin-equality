#!/usr/bin/env python3
"""
Create 3-iteration variants of config files with _el suffix.

Changes:
- Filename: append "_3" before .json extension
- run_name: append "_3"
- optimization_iterations: 2 → 3
"""

import json
import glob
from pathlib import Path

# Find all matching config files (config_010, 16 points, fract_gdp=1, 50k evals, _el, not BOBYQA)
pattern = 'json/*_010_*_16_1_50k_el.json'
files = glob.glob(pattern)

# Filter out BOBYQA files
files = [f for f in files if 'BOBYQA' not in f]

print(f"Found {len(files)} files matching pattern: {pattern}")
print()

for filepath in sorted(files):
    print(f"Processing: {filepath}")

    # Read the file
    with open(filepath, 'r') as f:
        config = json.load(f)

    # Update run_name
    old_run_name = config['run_name']
    new_run_name = old_run_name + "_3"
    config['run_name'] = new_run_name

    # Update optimization_iterations in optimization_parameters
    if 'optimization_parameters' in config:
        old_iter = config['optimization_parameters'].get('optimization_iterations', 'not found')
        config['optimization_parameters']['optimization_iterations'] = 3
        print(f"  optimization_parameters.optimization_iterations: {old_iter} → 3")

    # Remove redundant top-level optimization_iterations if it exists
    if 'optimization_iterations' in config:
        del config['optimization_iterations']

    # Remove redundant top-level optimize_time_points if it exists
    if 'optimize_time_points' in config:
        del config['optimize_time_points']

    # Generate new filename
    old_filename = Path(filepath).name
    new_filename = old_filename.replace('.json', '_3.json')
    new_filepath = Path('json') / new_filename

    # Write the new file
    with open(new_filepath, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  → Created: {new_filepath}")
    print(f"     run_name: {old_run_name} → {new_run_name}")
    print()

print(f"Done! Created {len(files)} new 3-iteration config files.")
