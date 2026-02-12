#!/usr/bin/env python3
"""
Batch calculate Social Cost of Carbon (SCC) for multiple result directories.

Creates a timestamped output directory with consolidated results in xlsx format.
Runs cross-product of pulse years × scaling factors for comprehensive sensitivity analysis.

Usage:
    python scripts/batch_scc.py [directory_patterns...] [options]

Arguments:
    directory_patterns: Glob patterns for result directories to process.
                       If not specified, defaults to data/output/reference/*

Options:
    --pulse-years YEARS            Comma-separated years to apply pulses (default: "2025")
    --pulse-year YEAR              Single year (legacy, same as --pulse-years with one value)
    --scaling-factors FACTORS      Comma-separated scaling factors (default: "1")
    --emission-amount AMOUNT       Emission pulse amount in tCO2 (default: from config)
    --consumption-amount AMOUNT    Consumption pulse amount in $ (default: from config)
    --save-integration-results     Save full integration results as CSV files

Examples:
    # Use default reference directories
    python scripts/batch_scc.py

    # Process specific directories
    python scripts/batch_scc.py "data/output/config_022_*_20260211-133736"

    # Multiple pulse years and scaling factors (runs 3×3 = 9 combinations per directory)
    python scripts/batch_scc.py --pulse-years "2025,2050,2100" --scaling-factors "0.001,1,1000"

    # Save detailed integration results as CSV files
    python scripts/batch_scc.py --save-integration-results

    # Custom pulse year and amounts
    python scripts/batch_scc.py --pulse-year 2030 --emission-amount 1e9

Output:
    Creates ./data/output/scc_YYYYMMDD-HHMMSS/ directory containing:
    - scc_results.xlsx: Consolidated SCC results from all directories
    - *_results.csv: Integration results for each scenario (if --save-integration-results)
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scripts.calculate_scc import calculate_scc_for_directory
from src.output import write_results_csv

DEFAULT_REFERENCE_DIR = "data/output/reference/*"
DEFAULT_PULSE_YEARS = [2025]
DEFAULT_SCALING_FACTORS = [1]


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
        'pulse_years': DEFAULT_PULSE_YEARS,
        'emission_amount': None,
        'consumption_amount': None,
        'scaling_factors': DEFAULT_SCALING_FACTORS,
        'save_integration_results': False,
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--pulse-year':
            # Legacy single-year option
            if i + 1 < len(args):
                options['pulse_years'] = [int(args[i + 1])]
                i += 2
            else:
                print("Error: --pulse-year requires a value")
                sys.exit(1)
        elif arg == '--pulse-years':
            if i + 1 < len(args):
                options['pulse_years'] = [int(x.strip()) for x in args[i + 1].split(',')]
                i += 2
            else:
                print("Error: --pulse-years requires a value")
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
            # Legacy option - just print warning and ignore
            print("Warning: --sensitivity-test is deprecated. Use --scaling-factors instead.")
            i += 1
        elif arg == '--scaling-factors':
            if i + 1 < len(args):
                options['scaling_factors'] = [float(x.strip()) for x in args[i + 1].split(',')]
                i += 2
            else:
                print("Error: --scaling-factors requires a value")
                sys.exit(1)
        elif arg == '--save-integration-results':
            options['save_integration_results'] = True
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


def create_output_directory():
    """
    Create timestamped output directory.

    Returns
    -------
    Path
        Path to the created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path('data/output') / f'scc_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def results_to_dataframe(all_results, directory_name):
    """
    Convert results list to DataFrame with directory name added.

    Parameters
    ----------
    all_results : list of dict
        Results from calculate_scc_for_directory
    directory_name : str
        Name of the source directory

    Returns
    -------
    pd.DataFrame
        DataFrame with results
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    rows = []
    for result in all_results:
        row = {
            'timestamp': timestamp,
            'directory': directory_name,
            'test_type': result.get('test_type', 'main'),
            'pulse_year': result.get('pulse_year', ''),
            'scale_factor': result.get('scale', 1.0),
            'emission_amount': result.get('emission_amount', ''),
            'consumption_amount': result.get('consumption_amount', ''),
            'W_baseline': result.get('W_baseline', ''),
            'W_emission': result.get('W_emission', ''),
            'W_consumption': result.get('W_consumption', ''),
            'dW_emission': result.get('dW_emission', ''),
            'dW_consumption': result.get('dW_consumption', ''),
            'm_E': result.get('m_E', ''),
            'm_C': result.get('m_C', ''),
            'SCC': result.get('SCC', ''),
            'emission_check': result.get('emission_check', ''),
            'consumption_check': result.get('consumption_check', ''),
            'scc_check': result.get('scc_check', ''),
            'baseline_path': result.get('baseline_path', ''),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Main entry point."""

    # Parse arguments
    patterns, options = parse_arguments(sys.argv[1:])

    # Use default if no patterns specified
    if not patterns:
        patterns = [DEFAULT_REFERENCE_DIR]

    # Discover directories
    directories = discover_directories(patterns)

    if not directories:
        print(f"No result directories found matching patterns: {patterns}")
        print("Directories must contain results.csv or *results*.csv files.")
        sys.exit(1)

    # Create output directory
    output_dir = create_output_directory()

    # Print minimal runtime info
    print(f"SCC Batch Calculation")
    print(f"  Directories: {len(directories)}")
    print(f"  Pulse years: {options['pulse_years']}")
    print(f"  Scaling factors: {options['scaling_factors']}")
    print(f"  Save integration results: {'Yes' if options['save_integration_results'] else 'No'}")
    print(f"  Output: {output_dir}")
    print()

    # Collect all results
    all_dataframes = []

    for i, directory in enumerate(directories, 1):
        dir_name = directory.name
        print(f"  [{i}/{len(directories)}] {dir_name}")

        # Loop over pulse years
        for pulse_year in options['pulse_years']:
            try:
                if options['save_integration_results']:
                    results, integration_results = calculate_scc_for_directory(
                        str(directory),
                        pulse_year,
                        emission_amount=options['emission_amount'],
                        consumption_amount=options['consumption_amount'],
                        run_sensitivity=True,
                        scaling_factors=options['scaling_factors'],
                        quiet=True,
                        return_integration_results=True
                    )

                    # Save integration results to CSV files
                    for scenario_name, int_results in integration_results.items():
                        csv_filename = f"{dir_name}_year{pulse_year}_{scenario_name}_results.csv"
                        write_results_csv(int_results, str(output_dir), filename=csv_filename)

                else:
                    results = calculate_scc_for_directory(
                        str(directory),
                        pulse_year,
                        emission_amount=options['emission_amount'],
                        consumption_amount=options['consumption_amount'],
                        run_sensitivity=True,
                        scaling_factors=options['scaling_factors'],
                        quiet=True
                    )

                # Get main SCC result for this year
                main_scc = results[0]['SCC']
                print(f"    Year {pulse_year}: SCC = ${main_scc:.2f}/tCO2")

                # Convert to DataFrame and collect
                df = results_to_dataframe(results, dir_name)
                all_dataframes.append(df)

            except Exception as e:
                print(f"    Year {pulse_year}: ERROR: {e}")

    print()

    if not all_dataframes:
        print("No results collected. Exiting.")
        sys.exit(1)

    # Combine all results
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Save to xlsx
    xlsx_path = output_dir / 'scc_results.xlsx'
    combined_df.to_excel(xlsx_path, index=False, sheet_name='SCC Results')

    print(f"Results saved to: {xlsx_path}")
    print(f"  Total rows: {len(combined_df)}")
    if options['save_integration_results']:
        print(f"  Integration result CSVs saved to: {output_dir}")


if __name__ == '__main__':
    main()
