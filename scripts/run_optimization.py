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
import pandas as pd
from pathlib import Path
from io import StringIO
from src.config_cli import parse_override_arguments, load_config_dict_with_overrides
from src.parameters import configuration_from_dict, ModelConfiguration, create_f_and_s_control_from_single
from src.optimization import UtilityOptimizer, create_control_function_from_points, create_f_and_s_control_function_from_points
from src.economic_model import integrate_model, create_derived_variables
from src.output import save_results, write_optimization_summary, write_config_file, create_output_directory
from src.visualization_utils import create_results_report_pdf


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
  python run_optimization.py config_baseline.json --optimization_parameters.initial_guess_f 0.3 --run_name "test_run"

Override format:
  --key.subkey.subsubkey value

Common overrides:
  --run_name <name>
  --scalar_parameters.alpha <value>
  --scalar_parameters.rho <value>
  --optimization_parameters.initial_guess_f <value>
  --optimization_parameters.initial_guess_s <value>
  --optimization_parameters.max_evaluations <value>
  --optimization_parameters.n_points_final_f <value>
  --optimization_parameters.n_points_final_s <value>
  --optimization_parameters.n_points_initial_f <value>
  --optimization_parameters.n_points_initial_s <value>
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


def print_optimization_results(opt_results, n_control_points, t_base=0.0):
    """Print formatted optimization results.

    Parameters
    ----------
    opt_results : dict
        Optimization results dictionary
    n_control_points : int
        Number of control points
    t_base : float
        Base year for time functions (added to relative times for display)
    """
    # Check if s was also optimized
    has_s_optimization = 's_control_points' in opt_results and opt_results['s_control_points'] is not None

    if n_control_points == 1:
        print(f"Optimal f₀:             {opt_results['optimal_values'][0]:.6f}")
        if has_s_optimization:
            print(f"Optimal s₀:             {opt_results['s_optimal_values'][0]:.6f}")
    else:
        print(f"Optimal f control values:")
        for t_rel, f_val in opt_results['control_points']:
            print(f"  t={t_rel + t_base:6.1f} yr: f={f_val:.6f}")

        if has_s_optimization:
            print(f"\nOptimal s control values:")
            for t_rel, s_val in opt_results['s_control_points']:
                print(f"  t={t_rel + t_base:6.1f} yr: s={s_val:.6f}")

    print(f"\nOptimal objective:      {opt_results['optimal_objective']:.6e}")
    print(f"Function evaluations:   {opt_results['n_evaluations']}")
    print(f"Status:                 {opt_results['status']}")


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
    print(f"  Initial f control points: {opt_params.n_points_initial_f}")
    if opt_params.n_points_final_f is not None:
        print(f"  Final f control points: {opt_params.n_points_final_f}")
    else:
        print(f"  Final f control points: {round(1 + (opt_params.n_points_initial_f - 1) * 2.0**(n_iterations - 1))}")
    print(f"  Initial guess: f = {opt_params.initial_guess_f}")
    if opt_params.initial_guess_s is not None:
        print(f"  Initial guess: s = {opt_params.initial_guess_s}")
        print(f"  Initial s control points: {opt_params.n_points_initial_s}")
        if opt_params.n_points_final_s is not None:
            print(f"  Final s control points: {opt_params.n_points_final_s}")
        else:
            print(f"  Final s control points: same as f")
        print(f"  Optimizing both f and s: YES")

    optimizer = UtilityOptimizer(config)

    print_header("OPTIMIZATION")

    algorithm = opt_params.algorithm if opt_params.algorithm is not None else 'LN_SBPLX'
    print(f"Algorithm: {algorithm}")

    if opt_params.ftol_rel is not None:
        print(f"ftol_rel: {opt_params.ftol_rel}")
    if opt_params.ftol_abs is not None:
        print(f"ftol_abs: {opt_params.ftol_abs}")
    if opt_params.xtol_rel is not None:
        print(f"xtol_rel: {opt_params.xtol_rel}")
    if opt_params.xtol_abs is not None:
        print(f"xtol_abs: {opt_params.xtol_abs}")

    print(f"\nRunning {algorithm} optimization...\n")

    opt_results = optimizer.optimize_with_iterative_refinement(
        n_iterations=opt_params.optimization_iterations,
        initial_guess_scalar=opt_params.initial_guess_f,
        max_evaluations_initial=opt_params.max_evaluations_initial,
        max_evaluations_final=opt_params.max_evaluations_final,
        algorithm=opt_params.algorithm,
        ftol_rel=opt_params.ftol_rel,
        ftol_abs=opt_params.ftol_abs,
        xtol_rel=opt_params.xtol_rel,
        xtol_abs=opt_params.xtol_abs,
        n_points_final=opt_params.n_points_final_f,
        n_points_initial=opt_params.n_points_initial_f,
        initial_guess_s_scalar=opt_params.initial_guess_s,
        n_points_final_s=opt_params.n_points_final_s,
        n_points_initial_s=opt_params.n_points_initial_s
    )
    n_final_control_points = len(opt_results['control_points'])

    t_base = config.scalar_params.t_base
    print_optimization_results(opt_results, n_final_control_points, t_base)

    print_header("SAVING RESULTS")

    # Create control function from optimization results
    if 's_control_points' in opt_results:
        # Both f and s were optimized
        control_func = create_f_and_s_control_function_from_points(
            opt_results['control_points'],
            opt_results['s_control_points']
        )
    else:
        # Only f was optimized; s follows its prescribed time function
        f_control = create_control_function_from_points(opt_results['control_points'])
        control_func = create_f_and_s_control_from_single(f_control, config.time_functions['s'])

    optimal_config = ModelConfiguration(
        run_name=f"{config.run_name}_optimization",
        scalar_params=config.scalar_params,
        time_functions=config.time_functions,
        integration_params=config.integration_params,
        optimization_params=config.optimization_params,
        control_function=control_func
    )

    print("Running forward model with optimal control...")
    optimal_results = integrate_model(optimal_config)
    optimal_results = create_derived_variables(optimal_results)

    print("\nSaving integration results (CSV and PDF)...")
    # Pass the pre-created output_dir to save_results
    config_filename_base = os.path.basename(config_path)
    output_paths = save_results(optimal_results, optimal_config.run_name, prod_category, output_dir=output_dir, config_filename=config_filename_base)

    print(f"  Output directory: {output_paths['output_dir']}")
    print(f"  Results CSV:      {output_paths['csv_file']}")
    print(f"  Results PDF:      {output_paths['pdf_file']}")
    if 'pdf_file_short' in output_paths:
        print(f"  Short-term PDF:   {output_paths['pdf_file_short']}")

    # Create comprehensive results visualization using unified plotting
    print("\nCreating comprehensive results visualization...")
    results_df = pd.read_csv(output_paths['csv_file'])
    comprehensive_pdf = Path(output_dir) / 'plots_full.pdf'
    create_results_report_pdf({config.run_name: results_df}, comprehensive_pdf, config_filename=config_filename_base)
    print(f"  Comprehensive PDF: {comprehensive_pdf}")

    print("\nWriting optimization summary CSV...")
    opt_csv_path = write_optimization_summary(opt_results, output_dir, config.run_name, 'optimization_summary.csv', t_base)
    print(f"  Optimization CSV: {opt_csv_path}")

    print("\nWriting configuration file (with any command line overrides)...")
    config_copy_path = write_config_file(config_dict, config_filename_base, output_dir)
    print(f"  Configuration:    {config_copy_path}")

    print_header("SUMMARY")
    print(f"Iterative refinement optimization complete:")
    print(f"  Iterations performed: {opt_results['n_iterations']}")
    print(f"  Total evaluations: {opt_results['n_evaluations']}")
    print(f"  Final control points: {n_final_control_points}")
    print(f"\nOptimal objective value: {opt_results['optimal_objective']:.12e}")
    print(f"\nIteration history:")
    for iter_result in opt_results['iteration_history']:
        print(f"  Iteration {iter_result['iteration']:2d}: {iter_result['n_control_points']:3d} points, "
              f"objective = {iter_result['optimal_objective']:.6e}, "
              f"evals = {iter_result['n_evaluations']}")
    print(f"\nFinal control trajectory (f - log10 of MAC):")
    for t_rel, f_val in opt_results['control_points']:
        print(f"  t={t_rel + t_base:6.1f} yr: f={f_val:.6f}")

    if 's_control_points' in opt_results:
        print(f"\nFinal control trajectory (s - savings rate):")
        for t_rel, s_val in opt_results['s_control_points']:
            print(f"  t={t_rel + t_base:6.1f} yr: s={s_val:.6f}")

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
