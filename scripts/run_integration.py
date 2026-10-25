"""
Run forward integration using control trajectories.

Two modes of operation:
1. From optimization output: Load config and control trajectories from an output directory
2. From JSON config: Use initial_guess_f and initial_guess_s as constant control values

Usage:
    python run_integration.py <path> [--param.path value ...]

Arguments:
    path: Either:
          - Path to optimization output directory (containing JSON config and results CSV)
          - Path to JSON config file (uses initial guesses as constant controls)
    --param.path value: Override configuration parameters (e.g., --scalar_parameters.eta 1.45)

Examples:
    # From optimization output directory:
    python run_integration.py data/output/config_010_t-t-t-f-f_10_1_1000_el_20260105-165554

    # From JSON config file (uses initial_guess_f and initial_guess_s):
    python run_integration.py json/config_934_f-f-f-f-f_1_100k_hyp-0.3.json

    # With parameter overrides:
    python run_integration.py json/config_934_f-f-f-f-f_1_100k_hyp-0.3.json --scalar_parameters.eta 2.0
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.config_cli import parse_override_arguments, load_config_dict_with_overrides
from src.parameters import configuration_from_dict
from src.economic_model import integrate_model, create_derived_variables, time_grid
from src.output import save_results
from scipy.interpolate import PchipInterpolator


def parse_arguments():
    """
    Parse command line arguments including input path and overrides.

    Returns
    -------
    tuple
        (input_path, overrides_dict) where input_path is directory or JSON file
    """
    parser = argparse.ArgumentParser(
        description='Run forward integration with optional parameter overrides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From optimization output directory:
  python run_integration.py data/output/config_010_f-f-f-f-f_10_1_1000_el_20260105-165554

  # From JSON config file (uses initial_guess_f and initial_guess_s):
  python run_integration.py json/config_934_f-f-f-f-f_1_100k_hyp-0.3.json

  # With parameter overrides:
  python run_integration.py json/config_934_f-f-f-f-f_1_100k_hyp-0.3.json --scalar_parameters.eta 1.45

Override format:
  --key.subkey.subsubkey value

Common overrides:
  --run_name <name>
  --scalar_parameters.income_dependent_damage_distribution <true/false>
  --scalar_parameters.income_dependent_tax_policy <true/false>
  --scalar_parameters.income_redistribution <true/false>
  --scalar_parameters.income_dependent_redistribution_policy <true/false>
  --scalar_parameters.eta <value>
  --scalar_parameters.rho <value>
        """
    )

    parser.add_argument('input_path', help='Path to output directory or JSON config file')

    # Use parse_known_args to allow arbitrary --key value pairs
    args, unknown = parser.parse_known_args()
    return args.input_path, parse_override_arguments(unknown)


def constant_control_arrays(initial_f, initial_s, bounds_f, bounds_s, n_steps):
    """Constant f and s (clipped to their bounds) at each of n_steps integration times."""
    return (np.full(n_steps, np.clip(initial_f, bounds_f[0], bounds_f[1])),
            np.full(n_steps, np.clip(initial_s, bounds_s[0], bounds_s[1])))


def load_control_trajectory_from_csv(csv_path):
    """
    Load control trajectories (f and s) from results CSV file.

    Parameters
    ----------
    csv_path : str
        Path to results CSV file

    Returns
    -------
    tuple
        (t_values, f_values, s_values) arrays
    """
    df = pd.read_csv(csv_path)

    # Extract column names (first part before comma)
    df.columns = [col.split(',')[0].strip() for col in df.columns]

    t_values = df['t'].values
    f_values = df['f'].values
    s_values = df['s'].values

    return t_values, f_values, s_values


def control_arrays_from_csv(t_values, f_values, s_values, t_grid, bounds_f, bounds_s):
    """
    f and s on the integration grid from a results CSV, by PCHIP interpolation.

    Parameters
    ----------
    t_values, f_values, s_values : array
        Calendar years and control values from the CSV
    t_grid : array
        Calendar years of the integration time steps
    bounds_f, bounds_s : list
        [min, max] bounds the interpolated values are clipped to

    Returns
    -------
    tuple of ndarray
        (f, s) at the times t_grid
    """
    f_interpolator = PchipInterpolator(t_values, f_values, extrapolate=True)
    s_interpolator = PchipInterpolator(t_values, s_values, extrapolate=True)
    return (np.clip(f_interpolator(t_grid), bounds_f[0], bounds_f[1]),
            np.clip(s_interpolator(t_grid), bounds_s[0], bounds_s[1]))


def find_files_in_output_dir(output_dir):
    """
    Find JSON config and results CSV files in output directory.

    Parameters
    ----------
    output_dir : str
        Path to output directory

    Returns
    -------
    tuple
        (config_path, csv_path)
    """
    output_path = Path(output_dir)

    # Find JSON config file
    json_files = list(output_path.glob('*.json'))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON config file found in {output_dir}")
    if len(json_files) > 1:
        print(f"Warning: Multiple JSON files found in {output_dir}:")
        for f in json_files:
            print(f"  {f.name}")
        print(f"Using: {json_files[0].name}")
    config_path = json_files[0]

    # Find results CSV file (want *_results.csv, not *_summary.csv)
    csv_files = [f for f in output_path.glob('*_results.csv')]
    if len(csv_files) == 0:
        # Try without _results suffix but exclude summary files
        csv_files = [f for f in output_path.glob('*.csv')
                     if 'summary' not in f.name]

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No results CSV file found in {output_dir}")
    if len(csv_files) > 1:
        print(f"Warning: Multiple CSV files found in {output_dir}:")
        for f in csv_files:
            print(f"  {f.name}")
        print(f"Using: {csv_files[0].name}")
    csv_path = csv_files[0]

    return str(config_path), str(csv_path)


def main():
    # Parse command line arguments
    input_path, overrides = parse_arguments()

    # Determine if input is a JSON file or a directory
    is_json_file = input_path.endswith('.json') and os.path.isfile(input_path)
    is_directory = os.path.isdir(input_path)

    if not is_json_file and not is_directory:
        print(f"Error: Path not found or invalid: {input_path}")
        print("  Expected either a JSON config file or an output directory")
        sys.exit(1)

    print(f'=' * 80)
    if is_json_file:
        print(f'COIN_equality Forward Integration from JSON Config')
        print(f'=' * 80)
        print(f'Config file: {input_path}')
        print(f'Mode: Constant controls (using initial_guess_f and initial_guess_s)\n')
        config_path = input_path
    else:
        print(f'COIN_equality Forward Integration from Optimized Controls')
        print(f'=' * 80)
        print(f'Source directory: {input_path}\n')
        # Find config and CSV files
        print('Searching for files...')
        config_path, csv_path = find_files_in_output_dir(input_path)
        print(f'  Config file: {os.path.basename(config_path)}')
        print(f'  Results CSV: {os.path.basename(csv_path)}')

    # Load configuration from JSON file and apply command line overrides
    if overrides:
        print(f'\n' + '=' * 80)
        print('APPLYING COMMAND LINE OVERRIDES')
        print('=' * 80)
    config = configuration_from_dict(load_config_dict_with_overrides(config_path, overrides))

    print(f'\n' + '=' * 80)
    print('CONFIGURATION')
    print('=' * 80)
    print(f'  Run name: {config.run_name}')
    print(f'  Time span: {config.integration_params.t_start} to {config.integration_params.t_end} yr')
    print(f'  Time step: {config.integration_params.dt} yr')

    # Get bounds from config
    bounds_f = config.optimization_params.bounds_f if config.optimization_params.bounds_f is not None else [0.0, 1.0]
    bounds_s = config.optimization_params.bounds_s if config.optimization_params.bounds_s is not None else [0.0, 1.0]

    if is_json_file:
        # Use constant controls from initial guesses
        initial_f = config.optimization_params.initial_guess_f
        initial_s = config.optimization_params.initial_guess_s

        print(f'\nUsing constant control values from config:')
        print(f'  initial_guess_f: {initial_f}')
        print(f'  initial_guess_s: {initial_s}')
        print(f'  bounds_f: {bounds_f}')
        print(f'  bounds_s: {bounds_s}')

        f_values, s_values = constant_control_arrays(initial_f, initial_s, bounds_f, bounds_s,
                                                     len(time_grid(config.integration_params)))
    else:
        # Load control trajectories from CSV
        print(f'\nLoading control trajectories from CSV...')
        t_csv, f_csv, s_csv = load_control_trajectory_from_csv(csv_path)
        print(f'  Time points: {len(t_csv)}')
        print(f'  f range: [{f_csv.min():.6f}, {f_csv.max():.6f}]')
        print(f'  s range: [{s_csv.min():.6f}, {s_csv.max():.6f}]')

        # Show sample values
        print(f'\n  Sample control values:')
        sample_indices = [0, len(t_csv)//4, len(t_csv)//2, 3*len(t_csv)//4, -1]
        for idx in sample_indices:
            print(f'    t={t_csv[idx]:6.1f} yr: f={f_csv[idx]:.6f}, s={s_csv[idx]:.6f}')

        # Interpolate onto the integration grid, clipped to the bounds from config
        print(f'\nInterpolating controls onto the integration grid (PCHIP)...')
        print(f'  bounds_f: {bounds_f}')
        print(f'  bounds_s: {bounds_s}')
        f_values, s_values = control_arrays_from_csv(t_csv, f_csv, s_csv, time_grid(config.integration_params),
                                                     bounds_f, bounds_s)

    # Run integration
    print(f'\nRunning forward integration...')
    results = create_derived_variables(integrate_model(config, f_values, s_values))
    run_name = f"{config.run_name}{'_const' if is_json_file else '_forward'}"

    # Display results
    print(f'\nIntegration complete!')
    print(f'Number of time points: {len(results["t"])}')

    print(f'\n' + '=' * 80)
    print(f'Results Summary')
    print(f'=' * 80)

    print(f'\nInitial State (t={results["t"][0]:.1f} yr):')
    print(f'  Capital stock (K):            {results["K"][0]:.3e} $')
    print(f'  Cumulative emissions (Ecum):  {results["Ecum"][0]:.3e} tCO2')
    print(f'  Temperature change (ΔT):      {results["delta_T"][0]:.3f} °C')
    print(f'  Gini index:                   {results["Gini"][0]:.4f}')
    print(f'  Mean utility (U):             {results["U"][0]:.6f}')

    print(f'\nFinal State (t={results["t"][-1]:.1f} yr):')
    print(f'  Capital stock (K):            {results["K"][-1]:.3e} $')
    print(f'  Cumulative emissions (Ecum):  {results["Ecum"][-1]:.3e} tCO2')
    print(f'  Temperature change (ΔT):      {results["delta_T"][-1]:.3f} °C')
    print(f'  Gini index:                   {results["Gini"][-1]:.4f}')
    print(f'  Mean utility (U):             {results["U"][-1]:.6f}')

    # Save results
    print(f'\n' + '=' * 80)
    print(f'Saving Results')
    print(f'=' * 80)
    config_filename_base = os.path.basename(config_path)
    output_paths = save_results(results, run_name, 'integration',
                                config_filename=config_filename_base)
    print(f'Output directory: {output_paths["output_dir"]}')
    print(f'CSV file:         {output_paths["csv_file"]}')
    print(f'PDF file:         {output_paths["pdf_file"]}')
    print(f'2025-2100 PDF:    {output_paths["pdf_file_2025_2100"]}')

    print(f'\n' + '=' * 80)
    print(f'Forward Integration Complete')
    print(f'=' * 80)


if __name__ == '__main__':
    main()
