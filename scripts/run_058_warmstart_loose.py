"""Config 058 loose warm-start: rerun outlier candidates using neighbor solutions.

Reads task from json/config_058_warmstart_loose_tasks.csv.
Finds the neighbor's output directory by scanning run_status.csv files for
the matching grid_id (picks best objective if multiple directories exist).
Uses the neighbor's solution interpolated as initial guess for a full
3-iteration refinement (4->9->22 control points), matching the original
config_058 optimization pipeline.

Outputs to data/output/config_058_warmstart_loose/ with provenance columns
(neighbor_grid_id, neighbor_rank) in run_status.csv.

Takes a single argument: task_id (1-based, from SLURM_ARRAY_TASK_ID).
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

from src.parameters import load_configuration, ModelConfiguration, create_f_and_s_control_from_single
from src.optimization import (
    UtilityOptimizer,
    calculate_control_times,
    create_control_function_from_points,
    create_f_and_s_control_function_from_points,
    interpolate_to_new_grid,
)
from src.constants import OBJECTIVE_SCALE
from src.economic_model import integrate_model, create_derived_variables
from src.output import write_results_csv, write_distribution_xlsx, write_optimization_summary, copy_config_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / 'json' / 'config_058_base.json'
GRID_CSV = PROJECT_ROOT / 'json' / 'config_058_grid.csv'
TASKS_CSV = PROJECT_ROOT / 'json' / 'config_058_warmstart_loose_tasks.csv'
CONFIG_058_OUTPUT = PROJECT_ROOT / 'data' / 'output' / 'config_058'
OUTPUT_BASE = PROJECT_ROOT / 'data' / 'output' / 'config_058_warmstart_loose'


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


def load_task(task_id):
    """Load a single task row from the tasks CSV."""
    with open(TASKS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['task_id']) == task_id:
                return {
                    'task_id': int(row['task_id']),
                    'candidate_grid_id': int(row['candidate_grid_id']),
                    'neighbor_grid_id': int(row['neighbor_grid_id']),
                    'neighbor_rank': int(row['neighbor_rank']),
                    'tax_equity': float(row['tax_equity']),
                    'ydmg': float(row['ydmg']),
                    'eta': float(row['eta']),
                    'rho': float(row['rho']),
                }
    raise ValueError(f"task_id {task_id} not found in {TASKS_CSV}")


def load_grid():
    """Load grid CSV and return dict: grid_id -> row."""
    rows = {}
    with open(GRID_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row['grid_id'])] = {
                'grid_id': int(row['grid_id']),
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            }
    return rows


def find_neighbor_dir(neighbor_grid_id):
    """Find the output directory for a neighbor grid_id.

    Scans config_058 directory for run_status.csv files matching the
    neighbor_grid_id. Returns the directory with the best (most negative)
    objective value.
    """
    best_dir = None
    best_obj = np.inf

    if not CONFIG_058_OUTPUT.exists():
        return best_dir

    for entry in CONFIG_058_OUTPUT.iterdir():
        if not entry.is_dir():
            continue
        status_path = entry / 'run_status.csv'
        if not status_path.exists():
            continue
        with open(status_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['success'] == 'True' and int(row['grid_id']) == neighbor_grid_id:
                    obj = float(row['objective_value'])
                    if obj < best_obj:
                        best_obj = obj
                        best_dir = entry

    return best_dir


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


def extract_focal_metrics(optimal_results):
    """Extract extended focal metrics from forward integration results."""
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


def write_status_csv(output_dir, run_name, grid_id, sample_id,
                     tax_equity, y_damage_distribution_exponent, eta, rho,
                     success, optimizer_status, objective_value,
                     n_iterations, n_evaluations, runtime_seconds, error_message,
                     mac_2030, mac_2100, delta_t_2100,
                     mu_2030, mu_2100, Omega_2100, y_net_2100,
                     aggregate_utility, neighbor_grid_id, neighbor_rank):
    """Write machine-readable run_status.csv with provenance columns."""
    status_path = os.path.join(output_dir, 'run_status.csv')
    with open(status_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'grid_id', 'sample_id',
            'tax_equity', 'y_damage_distribution_exponent', 'eta', 'rho',
            'success', 'optimizer_status', 'objective_value',
            'n_iterations', 'n_evaluations', 'runtime_seconds',
            'mac_2030', 'mac_2100', 'delta_t_2100',
            'mu_2030', 'mu_2100', 'Omega_2100', 'y_net_2100',
            'aggregate_utility', 'neighbor_grid_id', 'neighbor_rank',
            'error_message',
        ])
        writer.writerow([
            run_name, grid_id, sample_id,
            tax_equity, y_damage_distribution_exponent, eta, rho,
            success, optimizer_status, objective_value,
            n_iterations, n_evaluations, f'{runtime_seconds:.2f}',
            mac_2030, mac_2100, delta_t_2100,
            mu_2030, mu_2100, Omega_2100, y_net_2100,
            aggregate_utility, neighbor_grid_id, neighbor_rank,
            error_message,
        ])
    return status_path


def main():
    start_time = time.time()

    task_id = int(sys.argv[1])
    task = load_task(task_id)

    candidate_grid_id = task['candidate_grid_id']
    neighbor_grid_id = task['neighbor_grid_id']
    neighbor_rank = task['neighbor_rank']
    tax_equity = task['tax_equity']
    ydmg = task['ydmg']
    eta = task['eta']
    rho = task['rho']

    # Look up sample_id from grid
    grid_by_id = load_grid()
    sample_id = grid_by_id[candidate_grid_id]['sample_id']

    run_name = f'config_058_tax-{tax_equity:.4f}_ydmg-{ydmg:.4f}_eta-{eta:.4f}_rho-{rho:.4f}'

    # Find neighbor directory
    neighbor_dir = find_neighbor_dir(neighbor_grid_id)

    print(f"Task ID: {task_id}")
    print(f"Candidate grid_id: {candidate_grid_id}")
    print(f"Neighbor grid_id: {neighbor_grid_id} (rank {neighbor_rank})")
    print(f"Parameters: tax={tax_equity}, ydmg={ydmg}, eta={eta}, rho={rho}")
    print(f"Neighbor dir: {neighbor_dir}")

    # Parse neighbor control points
    neighbor_summaries = list(neighbor_dir.glob('*_optimization_summary.csv'))
    neighbor_summary = neighbor_summaries[0]

    neighbor_f_times_cal, neighbor_f_values, neighbor_s_times_cal, neighbor_s_values = (
        parse_control_points_from_summary(neighbor_summary)
    )
    print(f"Neighbor control points: {len(neighbor_f_times_cal)} f, {len(neighbor_s_times_cal)} s")

    # Load base config and override parameters
    with open(BASE_CONFIG, 'r') as f:
        config_dict = json.load(f)

    t_base = config_dict['scalar_parameters']['t_base']
    neighbor_f_times_rel = neighbor_f_times_cal - t_base
    neighbor_s_times_rel = neighbor_s_times_cal - t_base

    config_dict['run_name'] = run_name
    config_dict['scalar_parameters']['tax_equity'] = tax_equity
    config_dict['scalar_parameters']['y_damage_distribution_exponent'] = ydmg
    config_dict['scalar_parameters']['eta'] = eta
    config_dict['scalar_parameters']['rho'] = rho

    # Write modified config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(config_dict, tmp, indent=2)
        tmp_path = tmp.name

    # Output to config_058_warmstart_loose
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = str(OUTPUT_BASE / f'{run_name}_rank{neighbor_rank}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = TeeOutput(terminal_output_path, original_stdout)
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout

    print(f"Task ID: {task_id}")
    print(f"Candidate grid_id: {candidate_grid_id}")
    print(f"Neighbor grid_id: {neighbor_grid_id} (rank {neighbor_rank})")
    print(f"Parameters: tax={tax_equity}, ydmg={ydmg}, eta={eta}, rho={rho}")
    print(f"Output directory: {output_dir}")

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
        optimizer = UtilityOptimizer(config)
        algorithm = opt_params.algorithm

        # Iterative refinement parameters (from config_058_base.json)
        n_iter = opt_params.optimization_iterations
        n_points_initial_f = opt_params.n_points_initial_f
        n_points_final_f = opt_params.n_points_final_f
        n_points_initial_s = opt_params.n_points_initial_s
        n_points_final_s = opt_params.n_points_final_s
        max_evals_initial = opt_params.max_evaluations_initial
        max_evals_final = opt_params.max_evaluations_final

        # Refinement bases
        refinement_base_f = ((n_points_final_f - 1) / (n_points_initial_f - 1)) ** (1.0 / (n_iter - 1))
        refinement_base_s = ((n_points_final_s - 1) / (n_points_initial_s - 1)) ** (1.0 / (n_iter - 1))
        refinement_base_evals = ((max_evals_final - 1) / (max_evals_initial - 1)) ** (1.0 / (n_iter - 1))

        # Time grid parameters
        t_start_relative = config.integration_params.t_start - t_base
        t_end_relative = config.integration_params.t_end - t_base
        dt = config.integration_params.dt
        delta = config.scalar_params.delta

        print(f"\nWarm-start from neighbor grid_id {neighbor_grid_id} (rank {neighbor_rank})")
        print(f"Neighbor f values: {neighbor_f_values}")
        print(f"Neighbor s values: {neighbor_s_values}")
        print(f"\nIterative refinement: {n_iter} iterations")
        print(f"  f points: {n_points_initial_f} -> {n_points_final_f} (base={refinement_base_f:.4f})")
        print(f"  s points: {n_points_initial_s} -> {n_points_final_s} (base={refinement_base_s:.4f})")
        print(f"  evaluations: {max_evals_initial} -> {max_evals_final} (base={refinement_base_evals:.4f})")

        iteration_history = []
        iteration_f_control_grids = []
        iteration_s_control_grids = []
        total_evaluations = 0

        for iteration in range(1, n_iter + 1):
            # Evaluation budget for this iteration
            iter_max_evals = round(1 + (max_evals_initial - 1) * refinement_base_evals ** (iteration - 1))

            # Control point counts for this iteration
            n_points_f = round(1 + (n_points_initial_f - 1) * refinement_base_f ** (iteration - 1))
            n_points_s = round(1 + (n_points_initial_s - 1) * refinement_base_s ** (iteration - 1))

            # Generate control point time grids
            f_control_times, f_r0, f_r1 = calculate_control_times(
                n_points_f, t_start_relative, t_end_relative, dt, delta,
            )
            s_control_times, s_r0, s_r1 = calculate_control_times(
                n_points_s, t_start_relative, t_end_relative, dt, delta,
            )

            # Initial guess: interpolate from neighbor (iter 1) or previous iteration
            if iteration == 1:
                f_initial_guess = interpolate_to_new_grid(
                    neighbor_f_times_rel, neighbor_f_values, f_control_times,
                )
                s_initial_guess = interpolate_to_new_grid(
                    neighbor_s_times_rel, neighbor_s_values, s_control_times,
                )
            else:
                old_f_times = iteration_f_control_grids[-1]
                old_f_values = iteration_history[-1]['optimal_values']
                f_initial_guess = interpolate_to_new_grid(old_f_times, old_f_values, f_control_times)
                old_s_times = iteration_s_control_grids[-1]
                old_s_values = iteration_history[-1]['s_optimal_values']
                s_initial_guess = interpolate_to_new_grid(old_s_times, old_s_values, s_control_times)

            iteration_f_control_grids.append(f_control_times.copy())
            iteration_s_control_grids.append(s_control_times.copy())

            print(f"\n{'=' * 80}")
            print(f"  ITERATION {iteration}/{n_iter} (warm-started from neighbor {neighbor_grid_id})")
            print(f"  Algorithm: {algorithm}")
            print(f"  Max evaluations: {iter_max_evals}")
            print(f"  f control points: {n_points_f}")
            print(f"  s control points: {n_points_s}")
            print(f"  f initial guess: {f_initial_guess}")
            print(f"  s initial guess: {s_initial_guess}")
            print(f"{'=' * 80}\n")

            opt_result = optimizer.optimize_control_points_f_and_s(
                f_control_times,
                f_initial_guess,
                s_control_times,
                s_initial_guess,
                iter_max_evals,
                algorithm=algorithm,
                ftol_rel=opt_params.ftol_rel,
                ftol_abs=opt_params.ftol_abs,
                xtol_rel=opt_params.xtol_rel,
                xtol_abs=opt_params.xtol_abs,
            )

            opt_result['iteration'] = iteration
            opt_result['n_control_points'] = n_points_f
            opt_result['n_s_control_points'] = n_points_s
            iteration_history.append(opt_result)
            total_evaluations += opt_result['n_evaluations']

            unscaled_objective = opt_result['optimal_objective'] / OBJECTIVE_SCALE

            print(f"\nIteration {iteration} complete:")
            print(f"  Objective value: {unscaled_objective:.12e}")
            print(f"  Evaluations: {opt_result['n_evaluations']}")
            print(f"  Status: {opt_result['termination_name']}")
            print(f"  Optimized f values: {opt_result['optimal_values']}")
            print(f"  Optimized s values: {opt_result['s_optimal_values']}")
            sys.stdout.flush()

        final_result = iteration_history[-1]

        opt_results = {
            'optimal_values': final_result['optimal_values'],
            'optimal_objective': final_result['optimal_objective'],
            'n_evaluations': total_evaluations,
            'control_points': final_result['control_points'],
            'status': 'success',
            'algorithm': algorithm,
            'n_iterations': n_iter,
            'iteration_history': iteration_history,
            'iteration_control_grids': iteration_f_control_grids,
            'refinement_base': refinement_base_f,
            'termination_code': final_result['termination_code'],
            'termination_name': final_result['termination_name'],
            's_optimal_values': final_result['s_optimal_values'],
            's_control_points': final_result['s_control_points'],
            'iteration_s_control_grids': iteration_s_control_grids,
            'refinement_base_s': refinement_base_s,
        }

        optimizer_status = opt_results['termination_name']
        objective_value = opt_results['optimal_objective']
        n_iterations = n_iter
        n_evaluations = total_evaluations

        print(f"\nAll iterations complete:")
        print(f"  Status: {optimizer_status}")
        print(f"  Objective: {objective_value:.12e}")
        print(f"  Total evaluations: {n_evaluations}")

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

        print("\nRunning forward integration...")
        optimal_results = integrate_model(optimal_config)
        optimal_results = create_derived_variables(optimal_results)

        (mac_2030, mac_2100, delta_t_2100,
         mu_2030, mu_2100, Omega_2100, y_net_2100) = extract_focal_metrics(optimal_results)

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

        csv_path = write_results_csv(optimal_results, output_dir, run_name, 'optimization_results.csv')
        print(f"Results CSV: {csv_path}")

        xlsx_path = write_distribution_xlsx(optimal_results, output_dir, run_name)
        print(f"Distributions XLSX: {xlsx_path}")

        opt_csv_path = write_optimization_summary(
            opt_results, None, output_dir, run_name, 'optimization_summary.csv', t_base,
        )
        print(f"Optimization summary: {opt_csv_path}")

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

    write_status_csv(
        output_dir, run_name, candidate_grid_id, sample_id,
        tax_equity, ydmg, eta, rho,
        success, optimizer_status, objective_value,
        n_iterations, n_evaluations, runtime_seconds, error_message,
        mac_2030, mac_2100, delta_t_2100,
        mu_2030, mu_2100, Omega_2100, y_net_2100,
        aggregate_utility, neighbor_grid_id, neighbor_rank,
    )

    print(f"\nRuntime: {runtime_seconds:.2f} seconds ({runtime_seconds / 60:.2f} minutes)")
    print(f"Success: {success}")

    if isinstance(sys.stdout, TeeOutput):
        sys.stdout.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == '__main__':
    main()
