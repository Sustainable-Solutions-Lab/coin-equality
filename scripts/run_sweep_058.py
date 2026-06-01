"""
Config 058: Monte Carlo sensitivity analysis for eta, rho, beta.

Latin Hypercube sampling (1000 draws) × 3 tax_equity cases = 3000 runs.
Uses iterative refinement (3 iterations, 4→22 control points).

Parameters varied per run:
  - eta (coefficient of relative risk aversion)
  - rho (pure rate of time preference)
  - y_damage_distribution_exponent (beta)
  - tax_equity (0.0, 0.5, 0.998)

Extended focal metrics extracted for violin plots:
  MAC@2030, MAC@2100, delta_T@2100, mu@2030, mu@2100,
  Omega@2100, y_net@2100, aggregate_utility
"""

import sys
import os
import csv
import time
import json
import tempfile
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from src.parameters import load_configuration, ModelConfiguration, create_f_and_s_control_from_single
from src.optimization import UtilityOptimizer, create_control_function_from_points, create_f_and_s_control_function_from_points
from src.economic_model import integrate_model, create_derived_variables
from src.output import write_results_csv, write_distribution_xlsx, write_optimization_summary, copy_config_file


N_GRID = 3000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_058_grid.csv'
BASE_CONFIG = PROJECT_ROOT / 'json' / 'config_058_base.json'


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
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            })
    return rows


def extract_focal_metrics(optimal_results):
    """Extract focal metrics from forward integration results.

    Returns MAC@2030, MAC@2100, delta_T@2100, mu@2030, mu@2100,
    Omega@2100, y_net@2100.
    """
    t = optimal_results['t']
    mac = optimal_results['marginal_abatement_cost']
    delta_T = optimal_results['delta_T']
    mu = optimal_results['mu']
    Omega = optimal_results['Omega']
    y_net = optimal_results['y_net']

    idx_2030 = np.argmin(np.abs(t - 2030.0))
    idx_2100 = np.argmin(np.abs(t - 2100.0))

    return (mac[idx_2030], mac[idx_2100], delta_T[idx_2100],
            mu[idx_2030], mu[idx_2100], Omega[idx_2100], y_net[idx_2100])


def compute_aggregate_utility(optimal_results, rho, t_start):
    """Compute discounted aggregate utility integral (the optimization objective)."""
    t = optimal_results['t']
    U = optimal_results['U']
    discount_factors = np.exp(-rho * (t - t_start))
    return np.trapezoid(discount_factors * U, t)


STATUS_COLUMNS = [
    'run_name', 'grid_id', 'sample_id', 'tax_equity', 'y_damage_distribution_exponent',
    'eta', 'rho', 'success', 'optimizer_status', 'objective_value',
    'n_iterations', 'n_evaluations', 'runtime_seconds',
    'mac_2030', 'mac_2100', 'delta_t_2100',
    'mu_2030', 'mu_2100', 'Omega_2100', 'y_net_2100',
    'aggregate_utility', 'error_message',
]


def write_status_csv(output_dir, run_name, grid_id, sample_id, tax_equity,
                     y_damage_distribution_exponent, eta, rho,
                     success, optimizer_status, objective_value,
                     n_iterations, n_evaluations, runtime_seconds,
                     mac_2030, mac_2100, delta_t_2100,
                     mu_2030, mu_2100, Omega_2100, y_net_2100,
                     aggregate_utility, error_message):
    """Write machine-readable run_status.csv."""
    status_path = os.path.join(output_dir, 'run_status.csv')
    with open(status_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(STATUS_COLUMNS)
        writer.writerow([
            run_name, grid_id, sample_id, tax_equity, y_damage_distribution_exponent,
            eta, rho, success, optimizer_status, objective_value,
            n_iterations, n_evaluations, f'{runtime_seconds:.2f}',
            mac_2030, mac_2100, delta_t_2100,
            mu_2030, mu_2100, Omega_2100, y_net_2100,
            aggregate_utility, error_message,
        ])
    return status_path


def main():
    start_time = time.time()

    task_id = int(sys.argv[1])
    if task_id < 1 or task_id > N_GRID:
        raise ValueError(f"task_id must be between 1 and {N_GRID}, got {task_id}")

    # Load grid and get this task's parameters
    grid = load_grid()
    if len(grid) != N_GRID:
        raise ValueError(f"Expected {N_GRID} rows in {GRID_CSV}, found {len(grid)}")
    point = grid[task_id - 1]
    grid_id = point['grid_id']
    sample_id = point['sample_id']
    tax_equity = point['tax_equity']
    ydmg = point['y_damage_distribution_exponent']
    eta = point['eta']
    rho = point['rho']

    run_name = f'config_058_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_eta-{eta:.4f}_rho-{rho:.4f}'

    print(f"Task ID: {task_id}")
    print(f"Grid ID: {grid_id}")
    print(f"Sample ID: {sample_id}")
    print(f"Parameters: tax_equity={tax_equity}, y_damage_distribution_exponent={ydmg}, eta={eta}, rho={rho}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Run name: {run_name}")

    # Load base config dict and override parameters
    with open(BASE_CONFIG, 'r') as f:
        config_dict = json.load(f)

    config_dict['run_name'] = run_name
    config_dict['scalar_parameters']['tax_equity'] = tax_equity
    config_dict['scalar_parameters']['y_damage_distribution_exponent'] = ydmg
    config_dict['scalar_parameters']['eta'] = eta
    config_dict['scalar_parameters']['rho'] = rho

    print(f"\nOverrides applied:")
    print(f"  run_name = {run_name}")
    print(f"  tax_equity = {tax_equity}")
    print(f"  y_damage_distribution_exponent = {ydmg}")
    print(f"  eta = {eta}")
    print(f"  rho = {rho}")

    # Write modified config to temp file for load_configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(config_dict, tmp, indent=2)
        tmp_path = tmp.name

    # Create output directory inside data/output/config_058/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('data', 'output', 'config_058', f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

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
    print(f"Sample ID: {sample_id}")
    print(f"Parameters: tax_equity={tax_equity}, y_damage_distribution_exponent={ydmg}, eta={eta}, rho={rho}")
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
    mu_2030 = ''
    mu_2100 = ''
    Omega_2100 = ''
    y_net_2100 = ''
    aggregate_utility = ''

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
        (mac_2030, mac_2100, delta_t_2100,
         mu_2030, mu_2100, Omega_2100, y_net_2100) = extract_focal_metrics(optimal_results)

        # Compute aggregate utility
        t_start = config.integration_params.t_start
        aggregate_utility = compute_aggregate_utility(optimal_results, rho, t_start)

        print(f"\nFocal metrics:")
        print(f"  MAC@2030:     ${mac_2030:.2f}/tCO2")
        print(f"  MAC@2100:     ${mac_2100:.2f}/tCO2")
        print(f"  delta_T@2100: {delta_t_2100:.4f} C")
        print(f"  mu@2030:      {mu_2030:.6f}")
        print(f"  mu@2100:      {mu_2100:.6f}")
        print(f"  Omega@2100:   {Omega_2100:.6f}")
        print(f"  y_net@2100:   ${y_net_2100:.2f}")
        print(f"  aggregate_utility: {aggregate_utility:.12e}")

        # Write results CSV
        csv_path = write_results_csv(optimal_results, output_dir, run_name, 'optimization_results.csv')
        print(f"Results CSV: {csv_path}")

        # Write distributions XLSX
        xlsx_path = write_distribution_xlsx(optimal_results, output_dir, run_name)
        print(f"Distributions XLSX: {xlsx_path}")

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
        output_dir, run_name, grid_id, sample_id, tax_equity, ydmg, eta, rho,
        success, optimizer_status, objective_value,
        n_iterations, n_evaluations, runtime_seconds,
        mac_2030, mac_2100, delta_t_2100,
        mu_2030, mu_2100, Omega_2100, y_net_2100,
        aggregate_utility, error_message,
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
