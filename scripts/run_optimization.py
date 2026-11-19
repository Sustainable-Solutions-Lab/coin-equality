"""
Optimize the controls of the climate-economy model for one configuration.

1. Load the JSON configuration and apply command-line overrides
2. Optimize f(t) = log10(marginal abatement cost), and the savings rate s(t) when it is
   optimized, by iterative refinement
3. Integrate forward with the optimal controls
4. Write the results CSV, distributions workbook, PDF reports, optimization summary, the
   configuration actually run and the terminal output to
   data/output/prod/{prod_category}/{run_name}_YYYYMMDD-HHMMSS/
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from src.config_cli import parse_override_arguments, load_config_dict_with_overrides
from src.parameters import configuration_from_dict
from src.optimization import optimize, control_arrays
from src.economic_model import integrate_model, create_derived_variables
from src.output import save_results, write_optimization_summary, write_config_file, create_output_directory


class TeeOutput:
    """Write output to both console and file (like Unix tee command)."""

    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w')
        self.original_stream = original_stream

    def write(self, data):
        self.original_stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.original_stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def parse_arguments():
    """
    Parse the configuration file, the production category and the overrides.

    Returns
    -------
    tuple
        (config_path, prod_category, overrides) where overrides maps key paths to values
    """
    parser = argparse.ArgumentParser(
        description='Run optimization with optional parameter overrides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_optimization.py config.json 9cases_s
  python scripts/run_optimization.py config.json test --scalar_parameters.rho 0.02
  python scripts/run_optimization.py config.json test --optimization_parameters.controls.f.initial_guess 2.0 --run_name test_run

Override format:
  --key.subkey.subsubkey value

Common overrides:
  --run_name <name>
  --scalar_parameters.rho <value>
  --scalar_parameters.tax_equity <value>
  --scalar_parameters.y_damage_distribution_exponent <value>
  --optimization_parameters.max_evaluations_final <value>
  --optimization_parameters.controls.f.n_points_final <value>
  --time_functions.A.adjustment_coefficient <value>
        """
    )
    parser.add_argument('config_file', help='Path to JSON configuration file')
    parser.add_argument('prod_category', help='Production category path for output routing, e.g. "9cases" or "discount/prod_discount_exact"')

    # parse_known_args leaves the --key.subkey value overrides
    args, unknown = parser.parse_known_args()
    return args.config_file, args.prod_category, parse_override_arguments(unknown)


def print_header(text):
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_optimization_results(opt_results, t_base):
    """Print the optimized control points (calendar years), objective and evaluations."""
    for name, (times, values) in opt_results['controls'].items():
        print(f"Optimal {name} control values:")
        for t_rel, value in zip(times, values):
            print(f"  t={t_rel + t_base:6.1f} yr: {name}={value:.6f}")
        print()
    print(f"Optimal objective:      {opt_results['objective']:.12e}")
    print(f"Function evaluations:   {opt_results['n_evaluations']}")


def main():
    start_time = time.time()
    config_path, prod_category, overrides = parse_arguments()
    config_dict = load_config_dict_with_overrides(config_path, overrides)
    config = configuration_from_dict(config_dict)
    opt_params = config.optimization_params

    # Log everything from here on to the run directory as well as the console
    output_dir = create_output_directory(config.run_name, prod_category)
    terminal_output_path = os.path.join(output_dir, 'terminal_output.txt')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = TeeOutput(terminal_output_path, original_stdout)

    print_header(f"ITERATIVE REFINEMENT OPTIMIZATION ({opt_params.optimization_iterations} ITERATIONS)")
    print(f"Configuration file: {config_path}")
    for key_path, value in overrides.items():
        print(f"Override: {key_path} = {value}")
    print(f"Run name: {config.run_name}")
    print(f"Time span: {config.integration_params.t_start} to {config.integration_params.t_end} years")
    print(f"Time step: {config.integration_params.dt} years")
    for name, control in opt_params.controls.items():
        print(f"  {name}: initial guess {control.initial_guess}, bounds {control.bounds}, "
              f"control points {control.n_points_initial} -> {control.n_points_final}")
    if 's' not in opt_params.controls:
        print("  s: prescribed by time_functions['s']")

    print_header("OPTIMIZATION")
    print(f"Algorithm: {opt_params.algorithm}")
    print(f"Tolerances: ftol_rel={opt_params.ftol_rel}, ftol_abs={opt_params.ftol_abs}, "
          f"xtol_rel={opt_params.xtol_rel}, xtol_abs={opt_params.xtol_abs}")

    opt_results = optimize(config)
    t_base = config.scalar_params.t_base
    print_optimization_results(opt_results, t_base)

    print_header("SAVING RESULTS")
    print("Running forward model with optimal control...")
    f_values, s_values = control_arrays(config, opt_results['controls'])
    optimal_results = create_derived_variables(integrate_model(config, f_values, s_values, True), config.scalar_params)

    print("\nSaving results (CSV, distributions, PDF reports)...")
    config_filename = os.path.basename(config_path)
    output_paths = save_results(optimal_results, f"{config.run_name}_optimization", output_dir, config_filename)
    output_paths['summary_file'] = os.path.join(output_dir, f"{config.run_name}_optimization_summary.csv")
    write_optimization_summary(opt_results, output_paths['summary_file'], t_base)
    # The configuration actually run, including command line overrides
    output_paths['config_file'] = os.path.join(output_dir, config_filename)
    write_config_file(config_dict, output_paths['config_file'])
    for path in output_paths.values():
        print(f"  {path}")

    print_header("SUMMARY")
    print(f"Iterations performed: {opt_results['n_iterations']}")
    print(f"Total evaluations: {opt_results['n_evaluations']}")
    print("\nIteration history:")
    for iter_result in opt_results['iteration_history']:
        print(f"  Iteration {iter_result['iteration']:2d}: {len(iter_result['controls']['f'][0]):3d} points, "
              f"objective = {iter_result['objective']:.12e}, "
              f"evals = {iter_result['n_evaluations']}, {iter_result['termination_name']}")
    print()
    print_optimization_results(opt_results, t_base)

    print(f"\nAll results saved to: {output_dir}")
    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print_header("OPTIMIZATION COMPLETE")

    sys.stdout.close()
    sys.stdout, sys.stderr = original_stdout, original_stderr
    print(f"\nTerminal output saved to: {terminal_output_path}")


if __name__ == '__main__':
    main()
