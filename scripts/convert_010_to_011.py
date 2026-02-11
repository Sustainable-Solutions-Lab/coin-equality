#!/usr/bin/env python3
"""
Convert config_010_*.json files to config_011_*.json files.

Updates parameters:
- Remove chebyshev_scaling_power
- Add n_points_end (1 for _5_, 2 for _10_, 3 for _16_)
"""

import json
import os
import re
from pathlib import Path


def determine_n_points_end(filename):
    """Determine n_points_end based on filename pattern."""
    if '_5_' in filename:
        return 1
    elif '_10_' in filename:
        return 2
    elif '_16_' in filename:
        return 3
    else:
        raise ValueError(f"Cannot determine n_points_end for filename: {filename}")


def convert_config(input_path, output_path):
    """Convert a single config file from 010 to 011."""
    # Read the JSON file
    with open(input_path, 'r') as f:
        config = json.load(f)

    # Determine n_points_end from filename
    n_points_end = determine_n_points_end(input_path.name)

    # Update run_name
    if 'run_name' in config:
        config['run_name'] = config['run_name'].replace('_010_', '_011_')

    # Update optimization_parameters
    if 'optimization_parameters' in config:
        opt_params = config['optimization_parameters']

        # Remove chebyshev_scaling_power and its comment
        if 'chebyshev_scaling_power' in opt_params:
            del opt_params['chebyshev_scaling_power']
        if '_chebyshev_scaling_power' in opt_params:
            del opt_params['_chebyshev_scaling_power']

        # Add n_points_end and comment
        opt_params['n_points_end'] = n_points_end
        opt_params['_n_points_end'] = (
            f"Number of control points in wind-down region (segment 2). "
            f"Provides resolution for terminal behavior near t_end."
        )

    # Write the updated config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    return n_points_end


def main():
    json_dir = Path('/home/kcaldeira/coin-equality/json')

    # Find all config_010_*.json files
    input_files = sorted(json_dir.glob('config_010_*.json'))

    print(f"Found {len(input_files)} config_010_*.json files")
    print("Converting...")

    counts = {1: 0, 2: 0, 3: 0}

    for input_path in input_files:
        # Generate output filename
        output_filename = input_path.name.replace('config_010_', 'config_011_')
        output_path = json_dir / output_filename

        # Convert the file
        n_points_end = convert_config(input_path, output_path)
        counts[n_points_end] += 1

        print(f"  {input_path.name} -> {output_filename} (n_points_end={n_points_end})")

    print(f"\nConversion complete!")
    print(f"  Files with n_points_end=1 (_5_):  {counts[1]}")
    print(f"  Files with n_points_end=2 (_10_): {counts[2]}")
    print(f"  Files with n_points_end=3 (_16_): {counts[3]}")
    print(f"  Total files created: {sum(counts.values())}")


if __name__ == '__main__':
    main()
