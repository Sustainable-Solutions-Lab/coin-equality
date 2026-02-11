#!/usr/bin/env python3
"""
Create config_011_*_25_1_*.json files from config_011_*_16_1_*.json files.

Updates:
- Filename: _16_1_ → _25_1_
- run_name: _16_1_ → _25_1_
- n_points_final_f: 16 → 25
- n_points_final_s: 16 → 25 (if present)
"""

import json
from pathlib import Path


def create_25_config(input_path):
    """Create a _25_1_ config from a _16_1_ config."""
    # Read the input file
    with open(input_path, 'r') as f:
        config = json.load(f)

    # Update run_name
    if 'run_name' in config:
        config['run_name'] = config['run_name'].replace('_16_1_', '_25_1_')

    # Update optimization_parameters
    if 'optimization_parameters' in config:
        opt_params = config['optimization_parameters']

        # Update n_points_final_f
        if 'n_points_final_f' in opt_params:
            if opt_params['n_points_final_f'] == 16:
                opt_params['n_points_final_f'] = 25

        # Update n_points_final_s if present
        if 'n_points_final_s' in opt_params:
            if opt_params['n_points_final_s'] == 16:
                opt_params['n_points_final_s'] = 25

    # Generate output filename
    output_filename = input_path.name.replace('_16_1_', '_25_1_')
    output_path = input_path.parent / output_filename

    # Write the new config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    return output_path


def main():
    json_dir = Path('/home/kcaldeira/coin-equality/json')

    # Find all config_011_*_16_1_*.json files
    input_files = sorted(json_dir.glob('config_011_*_16_1_*.json'))

    print(f"Found {len(input_files)} config_011_*_16_1_*.json files")
    print("Creating corresponding _25_1_ files...")
    print()

    for input_path in input_files:
        output_path = create_25_config(input_path)
        print(f"  {input_path.name}")
        print(f"    → {output_path.name}")

    print(f"\nCreated {len(input_files)} new config_011_*_25_1_*.json files")


if __name__ == '__main__':
    main()
