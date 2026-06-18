"""
Combined runner for config_059 (eta) and config_060 (rho) sensitivity sweeps.

Task IDs 1-369:   config_059 eta sweep (41 eta x 9 scenarios)
Task IDs 370-738: config_060 rho sweep (41 rho x 9 scenarios)

Uses config_055_base.json (rho=0.025, eta=1.5).
Takes a single argument: SLURM_ARRAY_TASK_ID (1-738).
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


N_GRID_059 = 369
N_GRID_060 = 369
N_GRID_TOTAL = N_GRID_059 + N_GRID_060

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV_059 = PROJECT_ROOT / 'json' / 'config_059_grid.csv'
GRID_CSV_060 = PROJECT_ROOT / 'json' / 'config_060_grid.csv'
BASE_CONFIG = PROJECT_ROOT / 'json' / 'config_055_base.json'


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


def load_grid_eta():
    """Load config_059 grid (eta sweep)."""
    rows = []
    with open(GRID_CSV_059, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'grid_id': int(row['grid_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
            })
    return rows


def load_grid_rho():
    """Load config_060 grid (rho sweep)."""
    rows = []
    with open(GRID_CSV_060, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'grid_id': int(row['grid_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'rho': float(row['rho']),
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


def write_status_csv_eta(output_dir, run_name, grid_id, tax_equity, y_damage_distribution_exponent, eta,
                         success, optimizer_status, objective_value,
                         n_iterations, n_evaluations, runtime_seconds, error_message,
                         mac_2030, mac_2100, delta_t_2100):
    """Write run_status.csv for eta sweep."""
    status_path = os.path.join(output_dir, 'run_status.csv')
    with open(status_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'grid_id', 'tax_equity', 'y_damage_distribution_exponent', 'eta',
            'success', 'optimizer_status', 'objective_value',
            'n_iterations', 'n_evaluations', 'runtime_seconds',
            'mac_2030', 'mac_2100', 'delta_t_2100', 'error_message'
        ])
        writer.writerow([
            run_name, grid_id, tax_equity, y_damage_distribution_exponent, eta,
            success, optimizer_status, objective_value,
            n_iterations, n_evaluations, f'{runtime_seconds:.2f}',
            mac_2030, mac_2100, delta_t_2100, error_message
        ])
    return status_path


def write_status_csv_rho(output_dir, run_name, grid_id, tax_equity, y_damage_distribution_exponent, rho,
                         success, optimizer_status, objective_value,
                         n_iterations, n_evaluations, runtime_seconds, error_message,
                         mac_2030, mac_2100, delta_t_2100):
    """Write run_status.csv for rho sweep."""
    status_path = os.path.join(output_dir, 'run_status.csv')
    with open(status_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'grid_id', 'tax_equity', 'y_damage_distribution_exponent', 'rho',
            'success', 'optimizer_status', 'objective_value',
            'n_iterations', 'n_evaluations', 'runtime_seconds',
            'mac_2030', 'mac_2100', 'delta_t_2100', 'error_message'
        ])
        writer.writerow([
            run_name, grid_id, tax_equity, y_damage_distribution_exponent, rho,
            success, optimizer_status, objective_value,
            n_iterations, n_evaluations, f'{runtime_seconds:.2f}',
            mac_2030, mac_2100, delta_t_2100, error_message
        ])
    return status_path


def run_optimization(config_dict, run_name, output_dir, tmp_path):
    """Run optimization and return results."""
    config = load_configuration(tmp_path)
    opt_params = config.optimization_params

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

    return optimizer_status, objective_value, n_iterations, n_evaluations, mac_2030, mac_2100, delta_t_2100


def main():
    start_time = time.time()

    task_id = int(sys.argv[1])
    if task_id < 1 or task_id > N_GRID_TOTAL:
        raise ValueError(f"task_id must be between 1 and {N_GRID_TOTAL}, got {task_id}")

    is_eta = task_id <= N_GRID_059

    if is_eta:
        grid = load_grid_eta()
        if len(grid) != N_GRID_059:
            raise ValueError(f"Expected {N_GRID_059} rows in {GRID_CSV_059}, found {len(grid)}")
        point = grid[task_id - 1]
        grid_id = point['grid_id']
        tax_equity = point['tax_equity']
        ydmg = point['y_damage_distribution_exponent']
        eta = point['eta']
        config_name = 'config_059'
        run_name = f'config_059_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_eta-{eta:.4f}'
        output_subdir = 'config_059'
        param_str = f'tax_equity={tax_equity}, y_damage_distribution_exponent={ydmg}, eta={eta}'
    else:
        rho_task_id = task_id - N_GRID_059
        grid = load_grid_rho()
        if len(grid) != N_GRID_060:
            raise ValueError(f"Expected {N_GRID_060} rows in {GRID_CSV_060}, found {len(grid)}")
        point = grid[rho_task_id - 1]
        grid_id = point['grid_id']
        tax_equity = point['tax_equity']
        ydmg = point['y_damage_distribution_exponent']
        rho = point['rho']
        config_name = 'config_060'
        run_name = f'config_060_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_rho-{rho:.4f}'
        output_subdir = 'config_060'
        param_str = f'tax_equity={tax_equity}, y_damage_distribution_exponent={ydmg}, rho={rho}'

    print(f"Task ID: {task_id} ({'eta sweep' if is_eta else 'rho sweep'})")
    print(f"Grid ID: {grid_id}")
    print(f"Parameters: {param_str}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Run name: {run_name}")

    # Load base config dict and override parameters
    with open(BASE_CONFIG, 'r') as f:
        config_dict = json.load(f)

    config_dict['run_name'] = run_name
    config_dict['scalar_parameters']['tax_equity'] = tax_equity
    config_dict['scalar_parameters']['y_damage_distribution_exponent'] = ydmg
    if is_eta:
        config_dict['scalar_parameters']['eta'] = eta
    else:
        config_dict['scalar_parameters']['rho'] = rho

    print(f"\nOverrides applied:")
    print(f"  run_name = {run_name}")
    print(f"  tax_equity = {tax_equity}")
    print(f"  y_damage_distribution_exponent = {ydmg}")
    if is_eta:
        print(f"  eta = {eta}")
    else:
        print(f"  rho = {rho}")

    # Write modified config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(config_dict, tmp, indent=2)
        tmp_path = tmp.name

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('data', 'output', output_subdir, f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    # Set up tee output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = TeeOutput(terminal_output_path, original_stdout)
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout

    # Re-print header to terminal log
    print(f"Task ID: {task_id} ({'eta sweep' if is_eta else 'rho sweep'})")
    print(f"Grid ID: {grid_id}")
    print(f"Parameters: {param_str}")
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
        optimizer_status, objective_value, n_iterations, n_evaluations, mac_2030, mac_2100, delta_t_2100 = \
            run_optimization(config_dict, run_name, output_dir, tmp_path)
        success = True

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        print(f"\nERROR: {error_message}")
        print(traceback.format_exc())

    finally:
        os.unlink(tmp_path)

    runtime_seconds = time.time() - start_time

    # Write status CSV
    if is_eta:
        write_status_csv_eta(
            output_dir, run_name, grid_id, tax_equity, ydmg, eta,
            success, optimizer_status, objective_value,
            n_iterations, n_evaluations, runtime_seconds, error_message,
            mac_2030, mac_2100, delta_t_2100,
        )
    else:
        write_status_csv_rho(
            output_dir, run_name, grid_id, tax_equity, ydmg, rho,
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
