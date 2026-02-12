#!/usr/bin/env python3
"""
Batch calculate Social Cost of Carbon (SCC) for multiple result directories.

Usage:
    python scripts/batch_scc.py [directory_patterns...] [options]

Arguments:
    directory_patterns: Glob patterns for result directories to process.
                       If not specified, defaults to data/output/reference/*

Options:
    --pulse-year YEAR              Year to apply perturbation pulse (default: 2025)
    --emission-amount AMOUNT       Emission pulse amount in tCO2 (default: from config)
    --consumption-amount AMOUNT    Consumption pulse amount in $ (default: from config)
    --sensitivity-test             Run pulse-size and time-shift invariance tests

Examples:
    # Use default reference directories
    python scripts/batch_scc.py

    # Process specific directories
    python scripts/batch_scc.py "data/output/config_022_*_20260211-133736"

    # Process multiple patterns
    python scripts/batch_scc.py "data/output/config_020_*" "data/output/config_021_*"

    # Run with sensitivity test
    python scripts/batch_scc.py --sensitivity-test

    # Custom pulse year and amounts
    python scripts/batch_scc.py --pulse-year 2030 --emission-amount 1e9
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REFERENCE_DIR = "data/output/reference/*"
DEFAULT_PULSE_YEAR = 2025


def parse_arguments(args):
    """
    Parse command line arguments into directory patterns and options.

    Returns
    -------
    tuple
        (patterns, options_dict)
    """
    patterns = []
    options = {
        'pulse_year': DEFAULT_PULSE_YEAR,
        'emission_amount': None,
        'consumption_amount': None,
        'sensitivity_test': False,
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--pulse-year':
            if i + 1 < len(args):
                options['pulse_year'] = int(args[i + 1])
                i += 2
            else:
                print("Error: --pulse-year requires a value")
                sys.exit(1)
        elif arg == '--emission-amount':
            if i + 1 < len(args):
                options['emission_amount'] = float(args[i + 1])
                i += 2
            else:
                print("Error: --emission-amount requires a value")
                sys.exit(1)
        elif arg == '--consumption-amount':
            if i + 1 < len(args):
                options['consumption_amount'] = float(args[i + 1])
                i += 2
            else:
                print("Error: --consumption-amount requires a value")
                sys.exit(1)
        elif arg == '--sensitivity-test':
            options['sensitivity_test'] = True
            i += 1
        elif arg in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)
        elif arg.startswith('--'):
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
        else:
            patterns.append(arg)
            i += 1

    return patterns, options


def discover_directories(patterns):
    """
    Discover result directories from glob patterns.

    Parameters
    ----------
    patterns : list of str
        Glob patterns for directories

    Returns
    -------
    list of Path
        Sorted list of directories that exist and contain results
    """
    directories = set()

    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches:
            path = Path(match)
            if path.is_dir():
                # Check if it looks like a results directory (has results.csv or similar)
                if (path / 'results.csv').exists() or any(path.glob('*results*.csv')):
                    directories.add(path)

    return sorted(directories)


def run_scc_calculation(directory, options):
    """
    Run SCC calculation for a single directory and extract results.

    Parameters
    ----------
    directory : Path
        Path to results directory
    options : dict
        Options dictionary with pulse_year, emission_amount, etc.

    Returns
    -------
    dict
        Dictionary with directory name and SCC results
    """
    cmd = [
        'python', 'scripts/calculate_scc.py',
        '--baseline', str(directory),
        '--pulse-year', str(options['pulse_year'])
    ]

    if options['emission_amount'] is not None:
        cmd.extend(['--emission-amount', str(options['emission_amount'])])

    if options['consumption_amount'] is not None:
        cmd.extend(['--consumption-amount', str(options['consumption_amount'])])

    if options['sensitivity_test']:
        cmd.append('--sensitivity-test')

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    # Combine stdout and stderr
    output = result.stdout + result.stderr

    # Extract lines containing "SCC"
    scc_lines = [line for line in output.split('\n') if 'SCC' in line]

    return {
        'directory': directory.name,
        'scc_lines': scc_lines,
        'returncode': result.returncode,
        'full_output': output
    }


def main():
    """Main entry point."""

    # Parse arguments
    patterns, options = parse_arguments(sys.argv[1:])

    # Use default if no patterns specified
    if not patterns:
        patterns = [DEFAULT_REFERENCE_DIR]
        print(f"No directories specified, using default: {DEFAULT_REFERENCE_DIR}")

    # Discover directories
    directories = discover_directories(patterns)

    if not directories:
        print(f"No result directories found matching patterns: {patterns}")
        print("Directories must contain results.csv or *results*.csv files.")
        sys.exit(1)

    print(f"Found {len(directories)} result directories")
    print(f"Pulse year: {options['pulse_year']}")
    if options['emission_amount'] is not None:
        print(f"Emission amount: {options['emission_amount']:.2e} tCO2")
    if options['consumption_amount'] is not None:
        print(f"Consumption amount: ${options['consumption_amount']:.2e}")
    if options['sensitivity_test']:
        print("Sensitivity test: ENABLED")
    print("=" * 80)
    print()

    # Process each directory
    for directory in directories:
        print(f"=== {directory.name} ===")

        result = run_scc_calculation(directory, options)

        if result['scc_lines']:
            for line in result['scc_lines']:
                print(line)
        elif result['returncode'] != 0:
            print(f"  Error running SCC calculation (exit code {result['returncode']})")
            # Show last few lines of output for debugging
            lines = result['full_output'].strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
        else:
            print("  No SCC results found")

        print()


if __name__ == '__main__':
    main()
