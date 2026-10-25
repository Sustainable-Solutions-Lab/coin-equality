"""
Main optimization script for climate-economy model.

This script runs the complete optimization workflow:
1. Load configuration from JSON file
2. Run optimization with iterative refinement (handles both single and multiple control points)
3. Run forward integration with optimized controls
4. Generate output files (CSV results, optimization summary, PDF visualizations)
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from pathlib import Path
from io import StringIO
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
    Parse command line arguments including config file and overrides.

    Returns
    -------
    tuple
        (config_path, overrides_dict) where overrides_dict maps key paths to values
    """
    parser = argparse.ArgumentParser(
        description='Run optimization with optional parameter overrides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimization.py config_baseline.json
  python run_optimization.py config_baseline.json --scalar_parameters.alpha 0.35
  python run_optimization.py config_baseline.json --optimization_parameters.controls.f.initial_guess 2.0 --run_name "test_run"

Override format:
  --key.subkey.subsubkey value

Common overrides:
  --run_name <name>
  --scalar_parameters.alpha <value>
  --scalar_parameters.rho <value>
  --optimization_parameters.controls.f.initial_guess <value>
  --optimization_parameters.controls.s.initial_guess <value>
  --optimization_parameters.max_evaluations_final <value>
  --optimization_parameters.controls.f.n_points_final <value>
  --optimization_parameters.controls.s.n_points_final <value>
  --time_functions.A.growth_rate <value>
        """
    )

    parser.add_argument('config_file', help='Path to JSON configuration file')
    parser.add_argument('prod_category', help='Production category path for output routing, e.g. "9cases" or "discount/prod_discount_exact"')

    # Use parse_known_args to allow arbitrary --key value pairs
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
    """Main execution function."""
    start_time = time.time()

    # Capture output to buffer initially (until output directory is created)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    output_buffer = StringIO()
    sys.stdout = output_buffer
    sys.stderr = output_buffer

    try:
        # Parse command line arguments
        config_path, prod_category, overrides = parse_arguments()

        # Load configuration from JSON file and apply command line overrides
        if overrides:
            print_header("APPLYING COMMAND LINE OVERRIDES")
        config_dict = load_config_dict_with_overrides(config_path, overrides)
        config = configuration_from_dict(config_dict)
        opt_params = config.optimization_params

        # Create output directory immediately and start logging to file
        output_dir = create_output_directory(config.run_name, prod_category)
        terminal_output_path = Path(output_dir) / 'terminal_output.txt'

        # Write buffered output to file and switch to tee mode
        buffered_output = output_buffer.getvalue()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout = TeeOutput(terminal_output_path, original_stdout)
        sys.stdout = tee_stdout
        sys.stderr = tee_stdout

        # Display buffered output
        print(buffered_output, end='')
    except:
        # If anything fails during startup, restore stdout/stderr so error is visible
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        raise

    n_iterations = opt_params.optimization_iterations
    print_header(f"ITERATIVE REFINEMENT OPTIMIZATION ({n_iterations} ITERATIONS)")
    print(f"Configuration file: {config_path}")
    print(f"Run name: {config.run_name}")
    print(f"Time span: {config.integration_params.t_start} to {config.integration_params.t_end} years")
    print(f"Time step: {config.integration_params.dt} years")
    print(f"\nIterative refinement mode:")
    print(f"  Number of iterations: {n_iterations}")
    for name, control in opt_params.controls.items():
        print(f"  {name}: initial guess {control.initial_guess}, bounds {control.bounds}, "
              f"control points {control.n_points_initial} -> {control.n_points_final}")
    if 's' not in opt_params.controls:
        print(f"  s: prescribed by time_functions['s']")

    print_header("OPTIMIZATION")
    print(f"Algorithm: {opt_params.algorithm}")
    for tolerance_name in ['ftol_rel', 'ftol_abs', 'xtol_rel', 'xtol_abs']:
        if getattr(opt_params, tolerance_name) is not None:
            print(f"{tolerance_name}: {getattr(opt_params, tolerance_name)}")

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
    print(f"Iterative refinement optimization complete:")
    print(f"  Iterations performed: {opt_results['n_iterations']}")
    print(f"  Total evaluations: {opt_results['n_evaluations']}")
    print(f"\nIteration history:")
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

    # Close output file and restore original stdout/stderr
    if isinstance(sys.stdout, TeeOutput):
        sys.stdout.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"\nTerminal output saved to: {terminal_output_path}")


if __name__ == '__main__':
    main()
