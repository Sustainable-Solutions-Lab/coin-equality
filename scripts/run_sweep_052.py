"""
2D sweep runner for config_052 on Sherlock HPC.

Takes two arguments: RHO_VALUE and SLURM_ARRAY_TASK_ID (1-441).
Maps task_id to row in grid CSV (tax_equity, y_damage_distribution_exponent),
loads base config for specified rho, overrides parameters, runs optimization,
and writes results + machine-readable run_status.csv.

Usage: python run_sweep_052.py <rho_value> <task_id>
Example: python run_sweep_052.py 0.000 1

No PDF/XLSX generation -- only CSV results and optimization summary.
"""

import sys
import os
import csv
import time
import json
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from src.parameters import load_configuration, ModelConfiguration, create_f_and_s_control_from_single
from src.optimization import UtilityOptimizer, create_control_function_from_points, create_f_and_s_control_function_from_points
from src.economic_model import integrate_model, create_derived_variables
from src.output import create_output_directory, write_results_csv, write_optimization_summary, copy_config_file


N_GRID = 441

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_052_grid.csv'


class TeeOutput:
    """Write output to both console and file."""

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


def load_grid():
    """Load the grid CSV and return list of dicts."""
    rows = []
    with open(GRID_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'grid_id': int(row['grid_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
            })
    return rows


def extract_focal_metrics(optimal_results):
    """Extract MAC@2030, MAC@2100, and delta_T@2100 from forward integration results."""
    t = optimal_results['t']
    mac = optimal_results['marginal_abatement_cost']
    delta_T = optimal_results['delta_T']

    idx_2030 = np.argmin(np.abs(t - 2030.0))
    idx_2100 = np.argmin(np.abs(t - 2100.0))

    return mac[idx_2030], mac[idx_2100], delta_T[idx_2100]


def write_status_csv(output_dir, run_name, grid_id, rho, tax_equity, y_damage_distribution_exponent,
                     success, optimizer_status, objective_value,
                     n_iterations, n_evaluations, runtime_seconds, error_message,
                     mac_2030, mac_2100, delta_t_2100):
    """Write machine-readable run_status.csv."""
    status_path = os.path.join(output_dir, 'run_status.csv')
    with open(status_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'grid_id', 'rho', 'tax_equity', 'y_damage_distribution_exponent',
            'success', 'optimizer_status', 'objective_value',
            'n_iterations', 'n_evaluations', 'runtime_seconds',
            'mac_2030', 'mac_2100', 'delta_t_2100', 'error_message'
        ])
        writer.writerow([
            run_name, grid_id, rho, tax_equity, y_damage_distribution_exponent,
            success, optimizer_status, objective_value,
            n_iterations, n_evaluations, f'{runtime_seconds:.2f}',
            mac_2030, mac_2100, delta_t_2100, error_message
        ])
    return status_path


def main():
    start_time = time.time()

    if len(sys.argv) != 3:
        print("Usage: python run_sweep_052.py <rho_value> <task_id>")
        print("Example: python run_sweep_052.py 0.000 1")
        sys.exit(1)

    rho = float(sys.argv[1])
    task_id = int(sys.argv[2])

    if task_id < 1 or task_id > N_GRID:
        raise ValueError(f"task_id must be between 1 and {N_GRID}, got {task_id}")

    # Determine base config file
    BASE_CONFIG = PROJECT_ROOT / 'json' / f'config_052_rho-{rho:.3f}_tax-0.00_ydmg-0.00.json'

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG}")

    # Load grid and get this task's parameters
    grid = load_grid()
    if len(grid) != N_GRID:
        raise ValueError(f"Expected {N_GRID} rows in {GRID_CSV}, found {len(grid)}")

    point = grid[task_id - 1]
    grid_id = point['grid_id']
    tax_equity = point['tax_equity']
    ydmg = point['y_damage_distribution_exponent']

    run_name = f'config_052_rho-{rho:.3f}_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}'

    print(f"Task ID: {task_id}")
    print(f"Grid ID: {grid_id}")
    print(f"Parameters: rho={rho:.3f}, tax_equity={tax_equity}, y_damage_distribution_exponent={ydmg}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Run name: {run_name}")

    # Load base config dict and override parameters
    with open(BASE_CONFIG, 'r') as f:
        config_dict = json.load(f)

    config_dict['run_name'] = run_name
    config_dict['scalar_parameters']['rho'] = rho
    config_dict['scalar_parameters']['tax_equity'] = tax_equity
    config_dict['scalar_parameters']['y_damage_distribution_exponent'] = ydmg

    print(f"\nOverrides applied:")
    print(f"  run_name = {run_name}")
    print(f"  rho = {rho:.3f}")
    print(f"  tax_equity = {tax_equity}")
    print(f"  y_damage_distribution_exponent = {ydmg}")

    # Write modified config to temp file for load_configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(config_dict, tmp, indent=2)
        tmp_path = tmp.name

    # Create output directory
    output_dir = create_output_directory(run_name)
    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    # Set up tee output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = TeeOutput(terminal_output_path, original_stdout)
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout

    # Re-print header to terminal log
    print(f"Task ID: {task_id}")
    print(f"Grid ID: {grid_id}")
    print(f"Parameters: rho={rho:.3f}, tax_equity={tax_equity}, y_damage_distribution_exponent={ydmg}")
    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")

    # Initialize status fields
    success = False
    optimizer_status = ''
    objective_value = ''
    n_iterations = 0
    n_evaluations = 0
    error_message = ''
    mac_2030 = ''
    mac_2100 = ''
    delta_t_2100 = ''

    try:
        config = load_configuration(tmp_path)
        opt_params = config.optimization_params

        # Run optimization
        optimizer = UtilityOptimizer(config)
        opt_results = optimizer.optimize_with_iterative_refinement(
            n_iterations=opt_params.optimization_iterations,
            initial_guess_scalar=opt_params.initial_guess_f,
            max_evaluations_initial=opt_params.max_evaluations_initial,
            max_evaluations_final=opt_params.max_evaluations_final,
            max_evaluations_time_points=opt_params.max_evaluations_time_points,
            algorithm=opt_params.algorithm,
            ftol_rel=opt_params.ftol_rel,
            ftol_abs=opt_params.ftol_abs,
            xtol_rel=opt_params.xtol_rel,
            xtol_abs=opt_params.xtol_abs,
            n_points_final=opt_params.n_points_final_f,
            n_points_initial=opt_params.n_points_initial_f,
            initial_guess_s_scalar=opt_params.initial_guess_s,
            n_points_final_s=opt_params.n_points_final_s,
            n_points_initial_s=opt_params.n_points_initial_s,
            optimize_time_points=opt_params.optimize_time_points,
        )

        optimizer_status = opt_results.get('termination_name', opt_results.get('status', 'unknown'))
        objective_value = opt_results['optimal_objective']
        n_iterations = opt_results.get('n_iterations', 1)
        n_evaluations = opt_results['n_evaluations']

        print(f"\nOptimization complete:")
        print(f"  Status: {optimizer_status}")
        print(f"  Objective: {objective_value:.12e}")
        print(f"  Evaluations: {n_evaluations}")

        # Create control function from results
        if 's_control_points' in opt_results and opt_results['s_control_points'] is not None:
            control_func = create_f_and_s_control_function_from_points(
                opt_results['control_points'],
                opt_results['s_control_points'],
            )
        else:
            f_control = create_control_function_from_points(opt_results['control_points'])
            if 's' in config.time_functions:
                s_time_function = config.time_functions['s']
            else:
                s_time_function = lambda t: config.control_function(t)[1]
            control_func = create_f_and_s_control_from_single(f_control, s_time_function)

        optimal_config = ModelConfiguration(
            run_name=f"{run_name}_optimization",
            scalar_params=config.scalar_params,
            time_functions=config.time_functions,
            integration_params=config.integration_params,
            optimization_params=config.optimization_params,
            initial_state=config.initial_state,
            control_function=control_func,
        )

        # Forward integration
        print("\nRunning forward integration...")
        optimal_results = integrate_model(optimal_config)
        optimal_results = create_derived_variables(optimal_results)

        # Extract focal metrics
        mac_2030, mac_2100, delta_t_2100 = extract_focal_metrics(optimal_results)
        print(f"\nFocal metrics:")
        print(f"  MAC@2030:     ${mac_2030:.2f}/tCO2")
        print(f"  MAC@2100:     ${mac_2100:.2f}/tCO2")
        print(f"  delta_T@2100: {delta_t_2100:.4f} C")

        # Write results CSV
        csv_path = write_results_csv(optimal_results, output_dir, run_name, 'optimization_results.csv')
        print(f"Results CSV: {csv_path}")

        # Write optimization summary
        t_base = config.scalar_params.t_base
        opt_csv_path = write_optimization_summary(
            opt_results, None, output_dir, run_name, 'optimization_summary.csv', t_base,
        )
        print(f"Optimization summary: {opt_csv_path}")

        # Copy config to output
        config_copy_path = copy_config_file(tmp_path, output_dir)
        proper_config_path = os.path.join(output_dir, f'{run_name}.json')
        os.rename(config_copy_path, proper_config_path)
        print(f"Config copy: {proper_config_path}")

        success = True

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        print(f"\nERROR: {error_message}")
        print(traceback.format_exc())

    finally:
        os.unlink(tmp_path)

    runtime_seconds = time.time() - start_time

    # Write status CSV
    write_status_csv(
        output_dir, run_name, grid_id, rho, tax_equity, ydmg,
        success, optimizer_status, objective_value,
        n_iterations, n_evaluations, runtime_seconds, error_message,
        mac_2030, mac_2100, delta_t_2100,
    )

    print(f"\nRuntime: {runtime_seconds:.2f} seconds ({runtime_seconds / 60:.2f} minutes)")
    print(f"Success: {success}")

    # Close tee output
    if isinstance(sys.stdout, TeeOutput):
        sys.stdout.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == '__main__':
    main()
