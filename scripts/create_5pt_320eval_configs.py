#!/usr/bin/env python3
"""
Create 5-point, 320-evaluation variants of config files.

Changes:
- Filename: _10_ → _5_, _1000_ → _320_
- run_name: _10_ → _5_, _1000_ → _320_
- max_evaluations_final: 1000 → 320
- n_points_final_f: 10 → 5
- n_points_final_s: 10 → 5
"""

import json
import glob
from pathlib import Path

# Find all matching config files
pattern = 'json/*_010_*-f-*-f-f_10_1_1000_el.json'
files = glob.glob(pattern)

print(f"Found {len(files)} files matching pattern: {pattern}")
print()

for filepath in sorted(files):
    print(f"Processing: {filepath}")

    # Read the file
    with open(filepath, 'r') as f:
        config = json.load(f)

    # Update run_name
    old_run_name = config['run_name']
    new_run_name = old_run_name.replace('_10_1_1000_', '_5_1_320_')
    config['run_name'] = new_run_name

    # Update optimization parameters
    config['optimization_parameters']['max_evaluations_final'] = 320
    config['optimization_parameters']['n_points_final_f'] = 5
    config['optimization_parameters']['n_points_final_s'] = 5

    # Remove redundant top-level fields if they exist
    if 'optimization_iterations' in config:
        del config['optimization_iterations']
    if 'optimize_time_points' in config:
        del config['optimize_time_points']

    # Generate new filename
    old_filename = Path(filepath).name
    new_filename = old_filename.replace('_10_1_1000_', '_5_1_320_')
    new_filepath = Path('json') / new_filename

    # Write the new file
    with open(new_filepath, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  → Created: {new_filepath}")
    print(f"     run_name: {old_run_name} → {new_run_name}")
    print()

print(f"Done! Created {len(files)} new config files.")
