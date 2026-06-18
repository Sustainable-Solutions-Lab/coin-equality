"""Warmstart runner for production single-param sweep.

Re-optimizes wrong-basin or missing runs using a neighbor's optimal control
points as initial guess.  Reads the warmstart grid and candidates CSVs to
find the neighbor's optimization_summary.csv, parses f(t)/s(t) control
points, and calls optimize_control_points_f_and_s() directly (single pass,
no iterative refinement).

Usage: python3 run_warmstart_prod.py <config_name> <task_id>

Example:
    python3 run_warmstart_prod.py prod_single_param_warmstart 1
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.parameters import load_configuration, ModelConfiguration
from src.optimization import UtilityOptimizer, create_f_and_s_control_function_from_points
from src.economic_model import integrate_model, create_derived_variables
from src.output import write_results_csv, write_distribution_xlsx, write_optimization_summary, copy_config_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / 'json' / 'config_template.json'


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


def load_grid(config_name):
    """Load the grid CSV and return list of dicts."""
    grid_csv = PROJECT_ROOT / 'json' / f'{config_name}_grid.csv'
    rows = []
    with open(grid_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            entry = {
                'grid_id': int(row['grid_id']),
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            }
            if 'sweep' in fieldnames:
                entry['sweep'] = row['sweep']
            rows.append(entry)
    return rows


def load_candidates(config_name):
    """Load the warmstart candidates CSV and return dict keyed by grid_id."""
    candidates_csv = PROJECT_ROOT / 'json' / f'{config_name}_candidates.csv'
    candidates = {}
    with open(candidates_csv, 'r') as f:
        for row in csv.DictReader(f):
            candidates[int(row['grid_id'])] = row
    return candidates


def parse_control_points_from_summary(summary_path):
    """Parse f(t) and s(t) control points from optimization_summary.csv."""
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


def find_neighbor_summary(candidate_row):
    """Find the neighbor's optimization_summary.csv path.

    Reconstructs the path from neighbor_source_dir and neighbor_source
    (relative to PROJECT_ROOT) so it works on both local Mac and Sherlock.
    """
    source = candidate_row['neighbor_source']
    source_dir_name = candidate_row['neighbor_source_dir']
    base_dir = PROJECT_ROOT / 'data' / 'output' / 'prod' / source
    run_dir = base_dir / source_dir_name
    summary_files = list(run_dir.glob('*_optimization_summary.csv'))
    return summary_files[0]


def build_run_name(config_name, point):
    """Build a run name from grid_id — the unique key for every run."""
    gid = int(point['grid_id'])
    return f'{config_name}_gid-{gid}'


def extract_focal_metrics(optimal_results):
    """Extract focal metrics from forward integration results."""
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
    """Compute discounted aggregate utility integral."""
    t = optimal_results['t']
    U = optimal_results['U']
    discount_factors = np.exp(-rho * (t - t_start))
    return np.trapezoid(discount_factors * U, t)


STATUS_COLUMNS = [
    'run_name', 'grid_id', 'tax_equity',
    'y_damage_distribution_exponent', 'eta', 'rho',
    'success', 'optimizer_status', 'objective_value',
    'n_iterations', 'n_evaluations', 'runtime_seconds',
    'mac_2030', 'mac_2100', 'delta_t_2100',
    'mu_2030', 'mu_2100', 'Omega_2100', 'y_net_2100',
    'aggregate_utility', 'error_message',
]


def write_status_csv(output_dir, values_dict):
    """Write machine-readable run_status.csv."""
    status_path = os.path.join(output_dir, 'run_status.csv')
    with open(status_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(STATUS_COLUMNS)
        writer.writerow([values_dict.get(col, '') for col in STATUS_COLUMNS])
    return status_path


def main():
    start_time = time.time()

    config_name = sys.argv[1]
    task_id = int(sys.argv[2])

    grid = load_grid(config_name)
    n_grid = len(grid)

    point = grid[task_id - 1]
    grid_id = point['grid_id']
    tax_equity = point['tax_equity']
    ydmg = point['y_damage_distribution_exponent']
    eta = point['eta']
    rho = point['rho']

    run_name = build_run_name(config_name, point)

    # Load candidates CSV to find neighbor info
    candidates = load_candidates(config_name)
    candidate = candidates[grid_id]
    neighbor_summary_path = find_neighbor_summary(candidate)
    neighbor_source_dir = candidate['neighbor_source_dir']

    print(f"Config: {config_name}")
    print(f"Task ID: {task_id} / {n_grid}")
    print(f"Grid ID: {grid_id}")
    print(f"Parameters: tax_equity={tax_equity}, ydmg={ydmg}, eta={eta}, rho={rho}")
    print(f"Run name: {run_name}")
    print(f"Neighbor: {neighbor_source_dir}")
    print(f"Neighbor summary: {neighbor_summary_path}")

    # Parse neighbor's control points (calendar times)
    neighbor_f_times_cal, neighbor_f_values, neighbor_s_times_cal, neighbor_s_values = (
        parse_control_points_from_summary(neighbor_summary_path)
    )
    print(f"Neighbor control points: {len(neighbor_f_times_cal)} f, {len(neighbor_s_times_cal)} s")

    # Load base config and override parameters
    with open(BASE_CONFIG, 'r') as f:
        config_dict = json.load(f)

    # Convert calendar times to relative times
    t_base = config_dict['scalar_parameters']['t_base']
    neighbor_f_times_rel = neighbor_f_times_cal - t_base
    neighbor_s_times_rel = neighbor_s_times_cal - t_base

    config_dict['run_name'] = run_name
    config_dict['scalar_parameters']['tax_equity'] = tax_equity
    config_dict['scalar_parameters']['y_damage_distribution_exponent'] = ydmg
    config_dict['scalar_parameters']['eta'] = eta
    config_dict['scalar_parameters']['rho'] = rho
    config_dict['optimization_parameters']['max_evaluations_final'] = 100000

    # Write modified config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(config_dict, tmp, indent=2)
        tmp_path = tmp.name

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('data', 'output', 'prod', config_name, f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    # Set up tee output
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = TeeOutput(terminal_output_path, original_stdout)
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout

    # Re-print header to terminal log
    print(f"Config: {config_name}")
    print(f"Task ID: {task_id} / {n_grid}")
    print(f"Grid ID: {grid_id}")
    print(f"Parameters: tax_equity={tax_equity}, ydmg={ydmg}, eta={eta}, rho={rho}")
    print(f"Run name: {run_name}")
    print(f"Neighbor: {neighbor_source_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize status fields
    status = {
        'run_name': run_name, 'grid_id': grid_id,
        'tax_equity': tax_equity, 'y_damage_distribution_exponent': ydmg,
        'eta': eta, 'rho': rho,
        'success': False, 'optimizer_status': '', 'objective_value': '',
        'n_iterations': 0, 'n_evaluations': 0, 'runtime_seconds': 0,
        'mac_2030': '', 'mac_2100': '', 'delta_t_2100': '',
        'mu_2030': '', 'mu_2100': '', 'Omega_2100': '', 'y_net_2100': '',
        'aggregate_utility': '', 'error_message': '',
    }

    try:
        loaded_config = load_configuration(tmp_path)
        opt_params = loaded_config.optimization_params

        # Use neighbor's time grid and values as initial guess
        f_control_times = neighbor_f_times_rel
        s_control_times = neighbor_s_times_rel
        f_initial_guess = neighbor_f_values.copy()
        s_initial_guess = neighbor_s_values.copy()

        print(f"\nWarm-start initial guess (f): {f_initial_guess}")
        print(f"Warm-start initial guess (s): {s_initial_guess}")

        optimizer = UtilityOptimizer(loaded_config)
        algorithm = opt_params.algorithm

        print(f"\n{'=' * 80}")
        print(f"  WARM-START OPTIMIZATION")
        print(f"  Algorithm: {algorithm}")
        print(f"  Max evaluations: {opt_params.max_evaluations_final}")
        print(f"  f control points: {len(f_control_times)}")
        print(f"  s control points: {len(s_control_times)}")
        print(f"  Warm-started from: {neighbor_source_dir}")
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

        # Package result for write_optimization_summary
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

        status['optimizer_status'] = opt_results['termination_name']
        status['objective_value'] = opt_results['optimal_objective']
        status['n_iterations'] = 1
        status['n_evaluations'] = opt_results['n_evaluations']

        print(f"\nOptimization complete:")
        print(f"  Status: {status['optimizer_status']}")
        print(f"  Objective: {status['objective_value']:.12e}")
        print(f"  Evaluations: {status['n_evaluations']}")

        # Create control function from results
        control_func = create_f_and_s_control_function_from_points(
            opt_results['control_points'],
            opt_results['s_control_points'],
        )

        optimal_config = ModelConfiguration(
            run_name=f"{run_name}_optimization",
            scalar_params=loaded_config.scalar_params,
            time_functions=loaded_config.time_functions,
            integration_params=loaded_config.integration_params,
            optimization_params=loaded_config.optimization_params,
            initial_state=loaded_config.initial_state,
            control_function=control_func,
        )

        # Forward integration
        print("\nRunning forward integration...")
        optimal_results = integrate_model(optimal_config)
        optimal_results = create_derived_variables(optimal_results)

        # Extract focal metrics
        (mac_2030, mac_2100, delta_t_2100,
         mu_2030, mu_2100, Omega_2100, y_net_2100) = extract_focal_metrics(optimal_results)
        aggregate_utility = compute_aggregate_utility(optimal_results, rho, loaded_config.integration_params.t_start)

        status.update({
            'mac_2030': mac_2030, 'mac_2100': mac_2100, 'delta_t_2100': delta_t_2100,
            'mu_2030': mu_2030, 'mu_2100': mu_2100, 'Omega_2100': Omega_2100,
            'y_net_2100': y_net_2100, 'aggregate_utility': aggregate_utility,
        })

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
        csv_path = write_results_csv(optimal_results, output_dir, run_name, 'results.csv')
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

        status['success'] = True

    except Exception as e:
        status['error_message'] = f"{type(e).__name__}: {e}"
        print(f"\nERROR: {status['error_message']}")
        print(traceback.format_exc())

    finally:
        os.unlink(tmp_path)

    status['runtime_seconds'] = f'{time.time() - start_time:.2f}'

    write_status_csv(output_dir, status)

    print(f"\nRuntime: {status['runtime_seconds']} seconds")
    print(f"Success: {status['success']}")

    # Close tee output
    if isinstance(sys.stdout, TeeOutput):
        sys.stdout.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == '__main__':
    main()
