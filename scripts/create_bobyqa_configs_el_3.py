#!/usr/bin/env python3
"""
Create BOBYQA variants of *_el_3.json config files.

Changes:
- Filename: append "_BOBYQA" before .json extension
- run_name: append "_BOBYQA"
- algorithm: "LN_SBPLX" → "LN_BOBYQA"
"""

import json
import glob
from pathlib import Path

# Find all matching config files
pattern = 'json/*_010_*_16_1_50k_el_3.json'
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
    new_run_name = old_run_name + "_BOBYQA"
    config['run_name'] = new_run_name

    # Update algorithm
    if 'optimization_parameters' in config and 'algorithm' in config['optimization_parameters']:
        old_algorithm = config['optimization_parameters']['algorithm']
        config['optimization_parameters']['algorithm'] = 'LN_BOBYQA'
        print(f"  Algorithm: {old_algorithm} → LN_BOBYQA")

    # Remove redundant top-level fields if they exist
    if 'optimization_iterations' in config:
        del config['optimization_iterations']
    if 'optimize_time_points' in config:
        del config['optimize_time_points']

    # Generate new filename
    old_filename = Path(filepath).name
    new_filename = old_filename.replace('.json', '_BOBYQA.json')
    new_filepath = Path('json') / new_filename

    # Write the new file
    with open(new_filepath, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"  → Created: {new_filepath}")
    print(f"     run_name: {old_run_name} → {new_run_name}")
    print()

print(f"Done! Created {len(files)} new BOBYQA config files.")
