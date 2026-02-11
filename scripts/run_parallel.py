#!/usr/bin/env python3
"""
Run multiple optimization cases in parallel.

Launches run_optimization.py for each JSON configuration file in parallel,
with each job running on its own core. Terminal output for each job is
automatically saved to terminal_output.txt in its respective output directory.

Usage:
    python run_parallel.py <pattern1> [pattern2] [...] [--key value] [...]

Arguments:
    pattern1, pattern2, ...: File path patterns or glob patterns for JSON config files
                             (e.g., "config_COIN-equality_000*.json" or "config_DICE*.json")
                             Unlimited number of patterns supported
    --key value:            Override configuration parameters (applied to all jobs)
                             Use dot notation for nested keys (e.g., --optimization_params.max_evaluations 100)

Examples:
    python run_parallel.py "config_COIN-equality_000*.json"
    python run_parallel.py config_baseline.json config_sensitivity.json
    python run_parallel.py "config_COIN*.json" "config_DICE*.json"

    # Quick test with reduced evaluations
    python run_parallel.py "config_*.json" --optimization_params.max_evaluations 100

    # Override multiple parameters
    python run_parallel.py "config_*.json" --optimization_params.max_evaluations 100 --run_name quick_test

Notes:
    - All matching JSON files are launched simultaneously in parallel
    - Each job runs as a separate Python process
    - Overrides are applied to ALL jobs launched
    - Terminal output for each job is saved to terminal_output.txt in its output directory
    - The script exits after launching all jobs (does not wait for completion)
    - To monitor progress, check the terminal_output.txt files or use 'ps' to see running processes
"""

import sys
import glob
import subprocess
from pathlib import Path


def parse_arguments(args):
    """
    Parse command line arguments into patterns and overrides.

    Parameters
    ----------
    args : list of str
        Command line arguments (sys.argv[1:])

    Returns
    -------
    tuple
        (patterns, overrides) where:
        - patterns: list of file path patterns
        - overrides: list of override arguments to pass to run_optimization.py
    """
    patterns = []
    overrides = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            # This is an override
            overrides.append(arg)
            # Check if there's a value following
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                overrides.append(args[i + 1])
                i += 2
            else:
                i += 1
        else:
            # This is a pattern
            patterns.append(arg)
            i += 1

    return patterns, overrides


def discover_config_files(patterns):
    """
    Discover JSON configuration files from path patterns.

    Expands glob patterns and validates that each file exists and is a JSON file.

    Parameters
    ----------
    patterns : list of str
        List of file paths or glob patterns
        Examples: ['config_*.json', 'configs/baseline.json']

    Returns
    -------
    list of Path
        Sorted list of JSON configuration files

    Raises
    ------
    ValueError
        If no valid JSON files found
    """
    config_files = []

    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches:
            path = Path(match)
            if path.is_file() and path.suffix == '.json':
                config_files.append(path)

    if not config_files:
        raise ValueError(f"No valid JSON configuration files found for patterns: {patterns}")

    return sorted(set(config_files))


def launch_optimization(config_file, overrides):
    """
    Launch run_optimization.py for a single configuration file as a background process.

    Parameters
    ----------
    config_file : Path
        Path to JSON configuration file
    overrides : list of str
        Override arguments to pass to run_optimization.py (e.g., ['--optimization_params.max_evaluations', '100'])

    Returns
    -------
    subprocess.Popen
        Process handle for the launched job
    """
    cmd = ['python', 'run_optimization.py', str(config_file)] + overrides

    # Launch process in background
    # stdout and stderr are handled by run_optimization.py itself (saved to terminal_output.txt)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return process


def main():
    """Main entry point for parallel launcher."""

    if len(sys.argv) < 2:
        print("Usage: python run_parallel.py <pattern1> [pattern2] [...] [--key value] [...]")
        print("\nArguments:")
        print("  pattern1, pattern2, ...: File path patterns or glob patterns for JSON files")
        print("  --key value:            Override configuration parameters (applied to all jobs)")
        print("\nExamples:")
        print('  python run_parallel.py "config_COIN-equality_000*.json"')
        print('  python run_parallel.py config_baseline.json config_sensitivity.json')
        print('  python run_parallel.py "config_COIN*.json" "config_DICE*.json"')
        print('  python run_parallel.py "config_*.json" --optimization_params.max_evaluations 100')
        sys.exit(1)

    # Parse patterns and overrides
    patterns, overrides = parse_arguments(sys.argv[1:])

    if not patterns:
        print("Error: No file patterns specified")
        sys.exit(1)

    print(f"Discovering configuration files from patterns: {patterns}")
    if overrides:
        print(f"Overrides to apply: {' '.join(overrides)}")

    # Discover config files
    config_files = discover_config_files(patterns)

    print(f"\nFound {len(config_files)} configuration files:")
    for f in config_files:
        print(f"  - {f}")

    # Launch all jobs in parallel
    print(f"\n{'=' * 80}")
    print(f"Launching {len(config_files)} optimization jobs in parallel...")
    if overrides:
        print(f"With overrides: {' '.join(overrides)}")
    print(f"{'=' * 80}\n")

    processes = []
    for config_file in config_files:
        print(f"Starting: {config_file}")
        process = launch_optimization(config_file, overrides)
        processes.append((config_file, process))

    print(f"\n{'=' * 80}")
    print(f"All {len(config_files)} jobs launched successfully!")
    print(f"{'=' * 80}\n")

    print("Process IDs:")
    for config_file, process in processes:
        print(f"  {config_file.name}: PID {process.pid}")

    print("\nNotes:")
    print("  - Jobs are running in the background")
    print("  - Terminal output for each job is saved to terminal_output.txt in its output directory")
    print("\nMonitoring:")
    print("  - Count running jobs:    ps aux | grep run_optimization | grep -v grep | wc -l")
    print("  - View running jobs:     ps aux | grep run_optimization")
    print("  - Monitor job progress:  tail -f data/output/<run_name>_*/terminal_output.txt")
    print("\nControl:")
    print("  - Kill specific job:     kill <PID>")
    print("  - Kill range of PIDs:    kill $(seq 1200 1300)")
    print("  - Kill ALL these jobs:   pkill -f run_optimization.py")


if __name__ == '__main__':
    main()
