"""
Forward integration of the model with given control trajectories.

Two modes:
1. From an optimization output directory: its configuration file and the f(t), s(t) of its
   results CSV, PCHIP-interpolated onto the integration grid
2. From a JSON configuration: the optimized controls held constant at their initial guesses
   (a prescribed savings rate follows time_functions['s'])

Usage:
    python scripts/run_integration.py <output directory or config.json> [--key.subkey value ...]

Examples:
    python scripts/run_integration.py data/output/prod/9cases_s/9cases_s_gid-1_20260709-231226
    python scripts/run_integration.py config.json --scalar_parameters.eta 2.0

Results are written to data/output/prod/integration/{run_name}_forward (or _const)_YYYYMMDD-HHMMSS/.
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from src.config_cli import parse_override_arguments, load_config_dict_with_overrides
from src.parameters import configuration_from_dict
from src.economic_model import integrate_model, create_derived_variables, time_grid
from src.optimization import control_arrays
from src.output import save_results, create_output_directory


def parse_arguments():
    """
    Parse the input path and the overrides.

    Returns
    -------
    tuple
        (input_path, overrides) where input_path is an output directory or a JSON file
    """
    parser = argparse.ArgumentParser(
        description='Run forward integration with optional parameter overrides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Override format:
  --key.subkey.subsubkey value     e.g. --scalar_parameters.eta 1.45
        """
    )
    parser.add_argument('input_path', help='Path to output directory or JSON config file')

    # parse_known_args leaves the --key.subkey value overrides
    args, unknown = parser.parse_known_args()
    return args.input_path, parse_override_arguments(unknown)


def control_arrays_from_csv(config, csv_path):
    """
    f and s on the integration grid by PCHIP interpolation of a results CSV.

    Optimized controls are clipped to their optimizer bounds.
    """
    df = pd.read_csv(csv_path)
    df.columns = [col.split(',')[0].strip() for col in df.columns]
    t_grid = time_grid(config.integration_params)
    controls = config.optimization_params.controls
    arrays = {}
    for name in ['f', 's']:
        arrays[name] = PchipInterpolator(df['t'].values, df[name].values, extrapolate=True)(t_grid)
        if name in controls:
            arrays[name] = np.clip(arrays[name], *controls[name].bounds)
    return arrays['f'], arrays['s']


def constant_control_arrays(config):
    """Optimized controls held at their initial guesses (clipped to their bounds)."""
    constant_controls = {name: (np.array([0.0]), np.array([np.clip(control.initial_guess, *control.bounds)]))
                         for name, control in config.optimization_params.controls.items()}
    return control_arrays(config, constant_controls)


def print_state(label, results, i):
    print(f'\n{label} (t={results["t"][i]:.1f} yr):')
    print(f'  Capital stock (K):            {results["K"][i]:.3e} $')
    print(f'  Cumulative emissions (Ecum):  {results["Ecum"][i]:.3e} tCO2')
    print(f'  Temperature change (ΔT):      {results["delta_T"][i]:.3f} °C')
    print(f'  Gini index:                   {results["gini"][i]:.4f}')
    print(f'  Aggregate utility (U):        {results["U"][i]:.6e}')


def main():
    input_path, overrides = parse_arguments()
    from_config_file = input_path.endswith('.json')

    print('=' * 80)
    if from_config_file:
        print('COIN_equality forward integration: controls held at their initial guesses')
        config_path = input_path
    else:
        print('COIN_equality forward integration: controls of an optimization output')
        # The output directory holds exactly one configuration and one results CSV
        (config_path,) = Path(input_path).glob('*.json')
        (csv_path,) = Path(input_path).glob('*_results.csv')
        print(f'Results CSV: {csv_path}')
    print(f'Configuration: {config_path}')
    for key_path, value in overrides.items():
        print(f'Override: {key_path} = {value}')
    print('=' * 80)

    config = configuration_from_dict(load_config_dict_with_overrides(config_path, overrides))
    print(f'Run name: {config.run_name}')
    print(f'Time span: {config.integration_params.t_start} to {config.integration_params.t_end} yr, '
          f'time step {config.integration_params.dt} yr')

    if from_config_file:
        f_values, s_values = constant_control_arrays(config)
    else:
        f_values, s_values = control_arrays_from_csv(config, csv_path)
    print(f'f range: [{f_values.min():.6f}, {f_values.max():.6f}], s range: [{s_values.min():.6f}, {s_values.max():.6f}]')

    print('\nRunning forward integration...')
    results = create_derived_variables(integrate_model(config, f_values, s_values, True), config.scalar_params)
    print_state('Initial state', results, 0)
    print_state('Final state', results, -1)

    run_name = f"{config.run_name}{'_const' if from_config_file else '_forward'}"
    output_dir = create_output_directory(run_name, 'integration')
    output_paths = save_results(results, run_name, output_dir, os.path.basename(config_path))
    print(f'\nResults saved to {output_dir}:')
    for path in output_paths.values():
        print(f'  {path}')


if __name__ == '__main__':
    main()
