"""
Regression cases shared by tests/generate_baseline.py and tests/test_regression.py.

Each case runs the model through its public entry points and returns a flat dict of
numpy arrays. The golden baseline in tests/baseline/ was generated once from the
pre-refactor code (see tests/baseline/baseline_meta.json); the tests compare the
current code against it.

Case families
-------------
R1-R9  Replay of the nine published 9cases_s runs with their stored optimal f(t), s(t).
D1     DICE-comparison configuration (Gini = 0, negative rho) with R1's controls.
B1-B6  Branch coverage for model options not used by any published run, with R5's controls.
O1-O3  Short optimizations (full horizon with few control points, 50-yr horizon run to
       convergence, prescribed savings with explicit control years).
K      Spot checks of numerical kernels on fixed inputs.

Tolerances
----------
Arrays are compared scale-aware: max|new - base| <= rtol * max|base|.
RTOL_ROUNDOFF (about two orders above the accumulated round-off of a 400-step run) applies
where no iterative solve is involved. RTOL_ROOT_SOLVE applies where results pass through
the Brent solve for the tax parameter: that solve converges only to xtol = loose_epsilon
= 1e-9 in log-deltaU, so 1e-9 is the convergence-criterion bound. Pure restructuring is
expected to stay far below these bounds; the test report prints the actual differences.
"""

import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd

from src.parameters import load_configuration
from src.economic_model import integrate_model, create_derived_variables, time_grid
from src.optimization import optimize, control_arrays, control_trajectory, calculate_control_times
from src.distribution_utilities import (
    quadrature_grid,
    find_log_deltaU_equal_utility,
    compute_post_tax_income_equal_utility,
    L_pareto_derivative,
    L_empirical_lorenz_derivative,
)
from src.output import write_results_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / 'tests' / 'cases'
BASELINE_DIR = REPO_ROOT / 'tests' / 'baseline'
NINE_CASES_DIR = REPO_ROOT / 'data' / 'output' / 'prod_s' / '9cases_s'
PUBLISHED_CONFIGS_DIR = REPO_ROOT / 'json' / 'prod_s'
DICE_CONFIG_PATH = (PUBLISHED_CONFIGS_DIR / 'dice_comparison' / 'dice_reference_r2_2030'
                    / 'dice_reference_r2_2030_gid-1_20260720_015723.json')

RTOL_ROUNDOFF = 1e-12
RTOL_ROOT_SOLVE = 1e-9
# Published CSVs were written by the pre-release code; the release code reproduces them
# to ~1e-12, so 1e-9 leaves margin while still catching real changes.
RTOL_PUBLISHED = 1e-9
# r_consumption in the published CSVs predates the current welfare-weighted formula.
PUBLISHED_COLUMNS_NOT_COMPARED = {'r_consumption'}
# O2 tolerance per output key: this factor times the spread between runs whose initial
# guess differs by the perturbations below (floored at RTOL_ROUNDOFF).
O2_SPREAD_SAFETY_FACTOR = 10.0
O2_CALIBRATION_PERTURBATIONS = [1e-10, -1e-10]

# Output keys deliberately removed during the refactor (baseline keys minus these must
# equal the current keys). Updated in the commit that removes them.
REMOVED_OUTPUT_KEYS = {
    # Phase 1: vestigial welfare-adjusted MAC diagnostics (wMAC == MAC by construction)
    'mu_dollar_numeraire', 'mu_dollar_cost', 'welfare_adjusted_mac', 'mac_over_wmac',
    # Phase 6: duplicate of gini, and aliases of lambda_abate and E
    'Gini', 'Lambda', 'dEcum_dt',
}

# Worst relative difference per case, collected by the tests and printed by conftest.py.
DIFFERENCE_REPORT = []


def relative_difference(new, base):
    """
    Scale-aware difference max|new - base| / max|base| over the finite entries.

    Shapes and the positions and values of non-finite entries (e.g. the -inf no-tax
    sentinel of log_deltaU_norm) must match exactly; string arrays must match exactly.
    Returns inf on any mismatch of that kind.
    """
    if new.shape != base.shape:
        return np.inf
    if base.dtype.kind == 'U':
        return 0.0 if np.array_equal(new, base) else np.inf
    finite = np.isfinite(base)
    if not (np.array_equal(np.isfinite(new), finite) and np.array_equal(new[~finite], base[~finite])):
        return np.inf
    if not finite.any():
        return 0.0
    diff = np.max(np.abs(new[finite] - base[finite]))
    scale = np.max(np.abs(base[finite]))
    return diff / scale if scale > 0 else (0.0 if diff == 0 else np.inf)


def nine_case_dir(gid):
    return next(NINE_CASES_DIR.glob(f'9cases_s_gid-{gid}_*'))


def read_results_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [col.split(',')[0].strip() for col in df.columns]
    return df


def published_results(gid):
    return read_results_csv(next(nine_case_dir(gid).glob('*_results.csv')))


def stored_controls(gid):
    df = published_results(gid)
    return df['t'].values, df['f'].values, df['s'].values


def array_outputs(results):
    return {key: np.asarray(value) for key, value in results.items() if isinstance(value, np.ndarray)}


# =============================================================================
# Forward-integration cases
# =============================================================================

def _nine_case_config_path(gid):
    return next(nine_case_dir(gid).glob('*.json'))


NINE_CASE_TAX_EQUITY = {1: 0.0, 2: 0.5, 3: 0.998, 4: 0.0, 5: 0.5, 6: 0.998, 7: 0.0, 8: 0.5, 9: 0.998}

FORWARD_CASES = {
    **{f'R{gid}': {'config_path': lambda gid=gid: _nine_case_config_path(gid),
                   'controls_gid': gid,
                   'rtol': RTOL_ROUNDOFF if NINE_CASE_TAX_EQUITY[gid] == 0.0 else RTOL_ROOT_SOLVE}
       for gid in range(1, 10)},
    'D1': {'config_path': lambda: DICE_CONFIG_PATH, 'controls_gid': 1, 'rtol': RTOL_ROUNDOFF},
    'B1': {'config_path': lambda: CASES_DIR / 'B1_pareto_lorenz.json', 'controls_gid': 5, 'rtol': RTOL_ROOT_SOLVE},
    'B2': {'config_path': lambda: CASES_DIR / 'B2_income_dependent_damage.json', 'controls_gid': 5, 'rtol': RTOL_ROOT_SOLVE},
    'B3': {'config_path': lambda: CASES_DIR / 'B3_log_utility.json', 'controls_gid': 5, 'rtol': RTOL_ROUNDOFF},
    'B4': {'config_path': lambda: CASES_DIR / 'B4_eta_below_one_high_tax.json', 'controls_gid': 5, 'rtol': RTOL_ROOT_SOLVE},
    'B5': {'config_path': lambda: CASES_DIR / 'B5_pulses.json', 'controls_gid': 5, 'rtol': RTOL_ROOT_SOLVE},
    'B6': {'config_path': lambda: CASES_DIR / 'B6_time_function_types.json', 'controls_gid': 5, 'rtol': RTOL_ROOT_SOLVE},
}


def run_forward_results(case_name):
    """Integrate a forward case with stored controls; returns the full results dict."""
    case = FORWARD_CASES[case_name]
    config = load_configuration(str(case['config_path']()))
    t_grid, f_values, s_values = stored_controls(case['controls_gid'])
    assert np.array_equal(t_grid, time_grid(config.integration_params))
    return create_derived_variables(integrate_model(config, f_values, s_values, True), config.scalar_params)


def run_forward_case(case_name):
    return array_outputs(run_forward_results(case_name))


def optimization_summary_layout(summary_path):
    """
    Section titles and parameter names of an optimization summary CSV, values dropped.

    Numbers inside parameter names (control point indices) are replaced by N and
    repeated names collapsed, so runs with different numbers of control points compare.
    """
    with open(summary_path, newline='') as f:
        rows = [row for row in csv.reader(f) if row]
    sections = [row[0] for row in rows if len(row) == 1]
    start = rows.index(['Parameter', 'Value']) + 1
    end = next(i for i in range(start, len(rows)) if len(rows[i]) == 1)
    parameters = list(dict.fromkeys(re.sub(r'\d+', 'N', row[0]) for row in rows[start:end]))
    return sections, parameters


def results_csv_header(results, output_dir):
    csv_path = Path(output_dir) / 'header_check_results.csv'
    write_results_csv(results, csv_path)
    with open(csv_path) as f:
        return pd.read_csv(f, nrows=0).columns.tolist()


# =============================================================================
# Optimization cases
# =============================================================================

OPTIMIZATION_CASES = {
    'O1': {'config_path': CASES_DIR / 'O1_full_horizon_few_points.json', 'rtol': RTOL_ROUNDOFF, 'exact_evaluations': True},
    # O2 runs to xtol convergence; its tolerance is the run-to-run spread under a 1e-10
    # perturbation of the initial guess, times 10 (calibrated when the baseline was made).
    'O2': {'config_path': CASES_DIR / 'O2_short_horizon_converged.json', 'rtol': None, 'exact_evaluations': False},
    'O3': {'config_path': CASES_DIR / 'O3_prescribed_savings_explicit_times.json', 'rtol': RTOL_ROUNDOFF, 'exact_evaluations': True},
}


def run_optimization_case(case_name, initial_guess_f_perturbation):
    config = load_configuration(str(OPTIMIZATION_CASES[case_name]['config_path']))
    config.optimization_params.controls['f'].initial_guess += initial_guess_f_perturbation
    opt_results = optimize(config)

    outputs = {
        'objective': np.array(opt_results['objective']),
        'iteration_objectives': np.array([it['objective'] for it in opt_results['iteration_history']]),
        'iteration_evaluations': np.array([it['n_evaluations'] for it in opt_results['iteration_history']]),
        'iteration_termination': np.array([it['termination_name'] for it in opt_results['iteration_history']]),
    }
    for name, (times, values) in opt_results['controls'].items():
        outputs[f'{name}_times'] = np.asarray(times)
        outputs[f'{name}_values'] = np.asarray(values)

    forward = create_derived_variables(integrate_model(config, *control_arrays(config, opt_results['controls']), True),
                                       config.scalar_params)
    for key in ['f', 's', 'mu', 'delta_T', 'U', 'y_net']:
        outputs[f'forward_{key}'] = forward[key]
    return outputs


# =============================================================================
# Kernel spot checks
# =============================================================================

def run_kernels():
    grid = quadrature_grid(32)
    Fi = grid.Fi
    omega_yi = np.linspace(0.002, 0.03, len(Fi))
    y_gross = 18000.0
    Omega_calc = 0.006

    outputs = {}
    for gini in [0.0, 0.4, 0.6]:
        outputs[f'L_empirical_lorenz_derivative_G{gini}'] = L_empirical_lorenz_derivative(Fi, gini)
    for gini in [0.4, 0.6]:
        outputs[f'L_pareto_derivative_G{gini}'] = L_pareto_derivative(Fi, gini)

    y_norm = np.geomspace(0.05, 40.0, 25)
    for tax_exponent, log_deltaU in [(0.4, -3.0), (1.0, -4.0), (1.5, -5.0), (3.0, -2.0), (300.0, -900.0)]:
        outputs[f'post_tax_eta{tax_exponent}'] = compute_post_tax_income_equal_utility(y_norm, log_deltaU, tax_exponent)

    y_damaged_mean = y_gross * (1.0 - Omega_calc)
    for use_empirical_lorenz in [True, False]:
        dLdF = L_empirical_lorenz_derivative(Fi, 0.57) if use_empirical_lorenz else L_pareto_derivative(Fi, 0.57)
        y_pretax_norm = y_gross * dLdF * (1.0 - omega_yi) / y_damaged_mean
        for tax_exponent in [0.5, 1.0, 1.5, 2.0, 250.0]:
            for tax_amount in [5.0, 500.0]:
                key = f'log_deltaU_emp{int(use_empirical_lorenz)}_eta{tax_exponent}_tax{tax_amount}'
                outputs[key] = np.array(find_log_deltaU_equal_utility(
                    y_pretax_norm, tax_amount / y_damaged_mean, tax_exponent, grid.Fwi, 1e-9))

    for n_points, t_end in [(4, 400.0), (6, 400.0), (9, 400.0), (11, 400.0), (15, 400.0), (24, 400.0), (4, 50.0), (6, 50.0)]:
        times, r0 = calculate_control_times(n_points, 0.0, t_end, 1.0, 0.1)
        # Baseline layout: times, r0, then a segment-2 growth rate that is always 0
        outputs[f'control_times_n{n_points}_T{int(t_end)}'] = np.concatenate([times, [r0, 0.0]])

    old_times = np.array([0.0, 1.0, 360.0, 400.0])
    old_values = np.array([1.2, 1.5, 3.1, 0.0])
    new_times, _ = calculate_control_times(6, 0.0, 400.0, 1.0, 0.1)
    # Baseline keys name the two functions control_trajectory replaced
    outputs['interpolate_to_new_grid'] = control_trajectory(old_times, old_values, new_times)
    outputs['evaluate_control_function'] = control_trajectory(old_times, old_values, np.arange(0.0, 401.0, 1.0))
    return outputs
