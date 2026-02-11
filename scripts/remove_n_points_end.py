#!/usr/bin/env python3
"""
Remove n_points_end from config_011_*.json files.

The n_points_end parameter is now computed dynamically using the formula:
    n_points_end = max(1, round(4 * (1 - exp(-n_points/13))))
"""

import json
from pathlib import Path


def remove_n_points_end(config_path):
    """Remove n_points_end from a single config file."""
    # Read the JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Remove n_points_end from optimization_parameters
    if 'optimization_parameters' in config:
        opt_params = config['optimization_parameters']

        removed = False
        if 'n_points_end' in opt_params:
            del opt_params['n_points_end']
            removed = True
        if '_n_points_end' in opt_params:
            del opt_params['_n_points_end']
            removed = True

        if removed:
            # Write the updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            return True

    return False


def main():
    json_dir = Path('/home/kcaldeira/coin-equality/json')

    # Find all config_011_*.json files
    config_files = sorted(json_dir.glob('config_011_*.json'))

    print(f"Found {len(config_files)} config_011_*.json files")
    print("Removing n_points_end parameter...")

    count = 0
    for config_path in config_files:
        if remove_n_points_end(config_path):
            count += 1
            print(f"  {config_path.name}")

    print(f"\nRemoved n_points_end from {count} files")
    print("All config_011 files now use dynamic n_points_end formula")


if __name__ == '__main__':
    main()
