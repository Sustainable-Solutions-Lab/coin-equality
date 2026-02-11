#!/usr/bin/env python3
"""
Remove redundant top-level optimization_iterations and optimize_time_points fields from JSON config files.

These fields should only appear inside optimization_parameters, not at the top level.
This script cleans up all JSON config files in the json/ directory.
"""

import json
import glob
from pathlib import Path


def cleanup_config_file(filepath):
    """
    Remove redundant top-level fields from a config file.

    Parameters
    ----------
    filepath : Path
        Path to JSON config file

    Returns
    -------
    bool
        True if file was modified, False otherwise
    """
    with open(filepath, 'r') as f:
        config = json.load(f)

    modified = False

    # Remove top-level optimization_iterations if it exists
    if 'optimization_iterations' in config:
        del config['optimization_iterations']
        modified = True

    # Remove top-level optimize_time_points if it exists
    if 'optimize_time_points' in config:
        del config['optimize_time_points']
        modified = True

    if modified:
        # Write back to file with same formatting
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    return modified


def main():
    """Main entry point."""
    # Find all JSON config files
    pattern = 'json/**/*.json'
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Scanning {len(files)} JSON config files for redundant fields\n")

    modified_count = 0
    for filepath in sorted(files):
        filepath = Path(filepath)
        if cleanup_config_file(filepath):
            print(f"  Cleaned: {filepath}")
            modified_count += 1

    print(f"\n{'=' * 80}")
    print(f"Modified {modified_count} of {len(files)} files")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
