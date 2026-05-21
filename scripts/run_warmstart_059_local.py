"""
Local warm-start rerun for 2 outlier points in config_059 eta sweep.

Warm-starts eta=1.575 and eta=1.6 for te=0.998, ydmg=0.0 from the good
eta=1.55 solution (OCP=165.01, obj=-3.708091e-04).

Usage:
    python scripts/run_warmstart_059_local.py 1   # eta=1.575
    python scripts/run_warmstart_059_local.py 2   # eta=1.6
"""

import sys
import os
import csv
import time
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.parameters import load_configuration, ModelConfiguration
from src.optimization import (
    UtilityOptimizer,
    create_f_and_s_control_function_from_points,
    interpolate_to_new_grid,
)
from src.economic_model import integrate_model, create_derived_variables
from src.output import write_results_csv, write_distribution_xlsx, write_optimization_summary, copy_config_file


RERUN_POINTS = [
    (1, 1.575, 1.55),
    (2, 1.600, 1.55),
]

TAX_EQUITY = 0.998
YDMG = 0.0
NEIGHBOR_ETA = 1.55

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / 'json' / 'config_055_base.json'
NEIGHBOR_DIR = (
    PROJECT_ROOT / 'data' / 'output' / 'config_059_warmstart'
    / 'config_059_tax-0.9980_ydmg-0.0000_eta-1.5500_20260513_155624'
)


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


def parse_control_points_from_summary(summary_path):
    """Parse f(t) and s(t) control points from optimization_summary.csv.

    Returns (f_times, f_values, s_times, s_values) as numpy arrays.
    Times are in calendar years (will be converted to relative in the caller).
    """
    f_times = []
    f_values = []
    s_times = []
    s_values = []

    section = None
    with open(summary_path, 'r') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) == 0:
                continue
            if row[0] == 'Final f(t) Control Points':
                section = 'f'
                continue
            if row[0] == 'Final s(t) Control Points':
                section = 's'
                continue
            if row[0] in ('Time',):
                continue
            if row[0].startswith('Iterative') or row[0] == 'Optimization Summary' or row[0] == 'Parameter':
                section = None
                continue

            if section == 'f' and len(row) >= 2:
                f_times.append(float(row[0]))
                f_values.append(float(row[1]))
            elif section == 's' and len(row) >= 2:
                s_times.append(float(row[0]))
                s_values.append(float(row[1]))

    return np.array(f_times), np.array(f_values), np.array(s_times), np.array(s_values)


def extract_focal_metrics(optimal_results):
    """Extract MAC@2030, MAC@2100, and delta_T@2100 from forward integration results."""
    t = optimal_results['t']
    mac = optimal_results['marginal_abatement_cost']
    delta_T = optimal_results['delta_T']

    idx_2030 = np.argmin(np.abs(t - 2030.0))
    idx_2100 = np.argmin(np.abs(t - 2100.0))

    return mac[idx_2030], mac[idx_2100], delta_T[idx_2100]


def write_status_csv(output_dir, run_name, tax_equity, y_damage_distribution_exponent, eta,
                     success, optimizer_status, objective_value,
                     n_iterations, n_evaluations, runtime_seconds, error_message,
                     mac_2030, mac_2100, delta_t_2100, neighbor_eta):
    """Write machine-readable run_status.csv."""
    status_path = os.path.join(output_dir, 'run_status.csv')
    with open(status_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'tax_equity', 'y_damage_distribution_exponent', 'eta',
            'success', 'optimizer_status', 'objective_value',
            'n_iterations', 'n_evaluations', 'runtime_seconds',
            'mac_2030', 'mac_2100', 'delta_t_2100', 'neighbor_eta', 'error_message'
        ])
        writer.writerow([
            run_name, tax_equity, y_damage_distribution_exponent, eta,
            success, optimizer_status, objective_value,
            n_iterations, n_evaluations, f'{runtime_seconds:.2f}',
            mac_2030, mac_2100, delta_t_2100, neighbor_eta, error_message
        ])
    return status_path


def main():
    start_time = time.time()

    task_id = int(sys.argv[1])
    _, eta, neighbor_eta = RERUN_POINTS[task_id - 1]

    run_name = f'config_059_tax-{TAX_EQUITY:.4f}_ydmg-{YDMG:.4f}_eta-{eta:.4f}'

    print(f"Task ID: {task_id}")
    print(f"Parameters: tax_equity={TAX_EQUITY}, y_damage_distribution_exponent={YDMG}, eta={eta}")
    print(f"Neighbor eta: {neighbor_eta}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Run name: {run_name}")

    # Find the optimization_summary.csv in neighbor dir
    neighbor_summary_files = glob(str(NEIGHBOR_DIR / '*_optimization_summary.csv'))
    neighbor_summary = neighbor_summary_files[0]
    print(f"Neighbor directory: {NEIGHBOR_DIR}")
    print(f"Neighbor summary: {neighbor_summary}")

    neighbor_f_times_cal, neighbor_f_values, neighbor_s_times_cal, neighbor_s_values = (
        parse_control_points_from_summary(neighbor_summary)
    )
    print(f"Neighbor f control points: {len(neighbor_f_times_cal)} points")
    print(f"Neighbor s control points: {len(neighbor_s_times_cal)} points")

    # Load base config dict and override parameters
    with open(BASE_CONFIG, 'r') as f:
        config_dict = json.load(f)

    t_base = config_dict['scalar_parameters']['t_base']

    # Convert neighbor calendar years to relative times
    neighbor_f_times_rel = neighbor_f_times_cal - t_base
    neighbor_s_times_rel = neighbor_s_times_cal - t_base

    config_dict['run_name'] = run_name
    config_dict['scalar_parameters']['tax_equity'] = TAX_EQUITY
    config_dict['scalar_parameters']['y_damage_distribution_exponent'] = YDMG
    config_dict['scalar_parameters']['eta'] = eta

    print(f"\nOverrides applied:")
    print(f"  run_name = {run_name}")
    print(f"  tax_equity = {TAX_EQUITY}")
    print(f"  y_damage_distribution_exponent = {YDMG}")
    print(f"  eta = {eta}")

    # Write modified config to temp file for load_configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(config_dict, tmp, indent=2)
        tmp_path = tmp.name

    # Create output directory inside data/output/config_059_warmstart/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('data', 'output', 'config_059_warmstart', f'{run_name}_{timestamp}')
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
    print(f"Parameters: tax_equity={TAX_EQUITY}, y_damage_distribution_exponent={YDMG}, eta={eta}")
    print(f"Neighbor eta: {neighbor_eta}")
    print(f"Neighbor directory: {NEIGHBOR_DIR}")
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

        # Build control grids from explicit time points (same as config_055_base)
        f_control_times = np.array(opt_params.n_points_final_f) - t_base
        s_control_times = np.array(opt_params.n_points_final_s) - t_base

        # Interpolate neighbor's solution onto our control grid
        f_initial_guess = interpolate_to_new_grid(neighbor_f_times_rel, neighbor_f_values, f_control_times)
        s_initial_guess = interpolate_to_new_grid(neighbor_s_times_rel, neighbor_s_values, s_control_times)

        print(f"\nWarm-start initial guess (f): {f_initial_guess}")
        print(f"Warm-start initial guess (s): {s_initial_guess}")

        # Call the lower-level optimizer directly with warm-start initial guess
        optimizer = UtilityOptimizer(config)
        algorithm = opt_params.algorithm

        print(f"\n{'=' * 80}")
        print(f"  WARM-START OPTIMIZATION")
        print(f"  Algorithm: {algorithm}")
        print(f"  Max evaluations: {opt_params.max_evaluations_final}")
        print(f"  f control points: {len(f_control_times)}")
        print(f"  s control points: {len(s_control_times)}")
        print(f"  Warm-started from neighbor eta={neighbor_eta}")
        print(f"{'=' * 80}\n")

        opt_result = optimizer.optimize_control_points_f_and_s(
            f_control_times,
            f_initial_guess,
            s_control_times,
            s_initial_guess,
            opt_params.max_evaluations_final,
            algorithm=algorithm,
            ftol_rel=opt_params.ftol_rel,
            ftol_abs=opt_params.ftol_abs,
            xtol_rel=opt_params.xtol_rel,
            xtol_abs=opt_params.xtol_abs,
        )

        # Package result in the same format as optimize_with_iterative_refinement
        iteration_record = {
            'iteration': 1,
            'n_control_points': len(f_control_times),
            'optimal_objective': opt_result['optimal_objective'],
            'n_evaluations': opt_result['n_evaluations'],
            'termination_name': opt_result['termination_name'],
            'optimal_values': opt_result['optimal_values'],
        }
        if 's_optimal_values' in opt_result:
            iteration_record['n_s_control_points'] = len(s_control_times)

        opt_results = {
            'optimal_values': opt_result['optimal_values'],
            'optimal_objective': opt_result['optimal_objective'],
            'n_evaluations': opt_result['n_evaluations'],
            'control_points': opt_result['control_points'],
            'status': 'success',
            'algorithm': algorithm,
            'n_iterations': 1,
            'iteration_history': [iteration_record],
            'iteration_control_grids': [f_control_times.copy()],
            'refinement_base': 1.0,
            'termination_code': opt_result['termination_code'],
            'termination_name': opt_result['termination_name'],
            's_optimal_values': opt_result['s_optimal_values'],
            's_control_points': opt_result['s_control_points'],
            'iteration_s_control_grids': [s_control_times.copy()],
            'refinement_base_s': 1.0,
        }

        optimizer_status = opt_results['termination_name']
        objective_value = opt_results['optimal_objective']
        n_iterations = 1
        n_evaluations = opt_results['n_evaluations']

        print(f"\nOptimization complete:")
        print(f"  Status: {optimizer_status}")
        print(f"  Objective: {objective_value:.12e}")
        print(f"  Evaluations: {n_evaluations}")

        # Create control function from results
        control_func = create_f_and_s_control_function_from_points(
            opt_results['control_points'],
            opt_results['s_control_points'],
        )

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
        output_dir, run_name, TAX_EQUITY, YDMG, eta,
        success, optimizer_status, objective_value,
        n_iterations, n_evaluations, runtime_seconds, error_message,
        mac_2030, mac_2100, delta_t_2100, neighbor_eta,
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
