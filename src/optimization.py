"""
Optimization of the control trajectories to maximize discounted aggregate utility.

Controls: f(t) = log10(marginal abatement cost in $/tCO2), always optimized, and the
savings rate s(t), optimized jointly with f when optimization_parameters.controls has an
's' entry and otherwise prescribed by time_functions['s'].

Each optimized control is parameterized by its values at control times (relative to
t_base) and interpolated with PCHIP onto the integration time grid. The optimization
proceeds by iterative refinement: successive NLopt runs on progressively finer control
grids, each initialized by interpolating the previous solution onto the new grid.
"""

import sys
import time
import numpy as np
import nlopt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from src.economic_model import integrate_model, time_grid
from src.constants import (
    NEG_BIGNUM,
    OBJECTIVE_SCALE,
    WIND_DOWN_EFOLDING_TIMES,
    MAX_POINTS_WITH_ONE_WIND_DOWN_POINT,
    GEOMETRIC_GROWTH_RATE_BRACKET,
    CONTROL_TIME_DUPLICATE_TOLERANCE,
)


NLOPT_TERMINATION_NAMES = {
    1: 'SUCCESS',
    2: 'STOPVAL_REACHED',
    3: 'FTOL_REACHED',
    4: 'XTOL_REACHED',
    5: 'MAXEVAL_REACHED',
    6: 'MAXTIME_REACHED',
    -1: 'FAILURE',
    -2: 'INVALID_ARGS',
    -3: 'OUT_OF_MEMORY',
    -4: 'ROUNDOFF_LIMITED',
    -5: 'FORCED_STOP',
}


# =============================================================================
# Control trajectories
# =============================================================================

def calculate_control_times(n_points, t_start, t_end, dt, delta):
    """
    Calculate control point times: geometric spacing followed by a wind-down region.

    The wind-down region is the last WIND_DOWN_EFOLDING_TIMES capital e-folding times
    (1/delta each) before t_end, allowing an orderly wind-down of capital as the terminal
    time approaches. It holds one control point at t_end, or two (t_end - 1/delta and
    t_end) when n_points > MAX_POINTS_WITH_ONE_WIND_DOWN_POINT.

    The remaining points span [t_start, t_join], t_join = t_end - WIND_DOWN_EFOLDING_TIMES/delta,
    with spacings dt * (1 + r0)^i; r0 is solved so that they end exactly at t_join.

    Parameters
    ----------
    n_points : int
        Total number of control points
    t_start, t_end : float
        Start and end times (relative to t_base)
    dt : float
        First spacing (the integration time step)
    delta : float
        Capital depreciation rate (yr^-1)

    Returns
    -------
    times : ndarray
        Control point times
    r0 : float
        Geometric growth rate of the spacing before the wind-down region
    """
    n_points_end = 1 if n_points <= MAX_POINTS_WITH_ONE_WIND_DOWN_POINT else 2
    t_join = t_end - WIND_DOWN_EFOLDING_TIMES / delta

    n1 = n_points - n_points_end  # points in [t_start, t_join]
    if n1 < 3:
        raise ValueError(
            f"Insufficient control points: segment 1 requires at least 3 points, but got {n1}. "
            f"With n_points={n_points}, the formula gives n_points_end={n_points_end}. "
            f"You need at least {3 + n_points_end} total points for this to work."
        )

    n_intervals_1 = n1 - 1
    target_total_1 = t_join - t_start

    def spacing_sum_minus_target(r0):
        return sum(dt * (1 + r0)**i for i in range(n_intervals_1)) - target_total_1

    try:
        r0 = brentq(spacing_sum_minus_target, *GEOMETRIC_GROWTH_RATE_BRACKET)
    except ValueError:
        # No sign change in the bracket (very short horizons): fall back to this growth rate
        r0 = (target_total_1 / (n_intervals_1 * dt)) - 1.0

    times_1 = np.zeros(n1)
    times_1[0] = t_start
    for i in range(n_intervals_1):
        times_1[i + 1] = times_1[i] + dt * (1 + r0)**i
    times_1[-1] = t_join

    times_end = np.array([t_end]) if n_points_end == 1 else np.array([t_end - 1.0 / delta, t_end])
    return np.concatenate([times_1, times_end]), r0


def control_trajectory(times, values, t):
    """
    PCHIP interpolation of control-point values, evaluated at times t.

    PCHIP is shape-preserving with continuous first derivatives and no overshoot.
    Beyond the last control point the trajectory is constant; a single control point
    gives a constant trajectory. Control points closer in time than
    CONTROL_TIME_DUPLICATE_TOLERANCE keep the first value. No clipping is applied:
    bounds are enforced by the optimizer.

    Parameters
    ----------
    times, values : array_like
        Control point times and values
    t : ndarray
        Times at which to evaluate the trajectory

    Returns
    -------
    ndarray
        Trajectory values at t
    """
    order = np.argsort(times)
    times = np.asarray(times)[order]
    values = np.asarray(values)[order]
    keep = np.concatenate([[True], np.diff(times) > CONTROL_TIME_DUPLICATE_TOLERANCE])
    times = times[keep]
    values = values[keep]

    if len(times) == 1:
        return np.full_like(t, values[0], dtype=float)
    interpolator = PchipInterpolator(times, values, extrapolate=False)
    return np.where(t <= times[-1], interpolator(t), values[-1])


def control_arrays(config, controls):
    """
    Control arrays (f, s) on the integration time grid.

    Parameters
    ----------
    config : ModelConfiguration
    controls : dict
        {'f': (times, values)} and, when s is optimized, {'s': (times, values)}; times
        are relative to t_base. Without 's', s follows time_functions['s'].

    Returns
    -------
    tuple of ndarray
        (f_values, s_values) at the times of time_grid(config.integration_params)
    """
    t_relative = time_grid(config.integration_params) - config.scalar_params.t_base
    f_values = control_trajectory(*controls['f'], t_relative)
    if 's' in controls:
        s_values = control_trajectory(*controls['s'], t_relative)
    else:
        s_values = config.time_functions['s'](t_relative)
    return f_values, s_values


def discounted_utility(config, f_values, s_values):
    """
    Objective: discounted aggregate utility, integral of exp(-rho (t - t_start)) U(t) dt.

    U(t) is aggregate (population-weighted) utility; the integral uses the trapezoidal rule
    on the integration time grid.
    """
    results = integrate_model(config, f_values, s_values, store_detailed_output=False)
    t = results['t']
    discount_factors = np.exp(-config.scalar_params.rho * (t - config.integration_params.t_start))
    return np.trapezoid(discount_factors * results['U'], t)


# =============================================================================
# Iterative-refinement optimization
# =============================================================================

def refinement_schedule(initial, final, n_iterations):
    """
    Values for each refinement iteration, growing geometrically from initial to final.

    Iteration k gets round(1 + (initial - 1) * base^(k-1)) with
    base = ((final - 1) / (initial - 1))^(1 / (n_iterations - 1)); a single iteration gets final.
    Used for the number of control points and for the evaluation budget.
    """
    if n_iterations == 1:
        return [final]
    base = ((final - 1) / (initial - 1)) ** (1.0 / (n_iterations - 1))
    return [round(1 + (initial - 1) * base**(iteration - 1)) for iteration in range(1, n_iterations + 1)]


def control_times_by_iteration(control, n_iterations, config):
    """
    Control times (relative to t_base) of one optimized control for each refinement iteration.

    Returns a list of (times, r0) with r0 the geometric growth rate of the spacing, or None
    for explicit control years.
    """
    t_base = config.scalar_params.t_base
    if isinstance(control.n_points_final, list):
        return [(np.array(control.n_points_final) - t_base, None)] * n_iterations
    ip = config.integration_params
    return [calculate_control_times(n_points, ip.t_start - t_base, ip.t_end - t_base, ip.dt,
                                    config.scalar_params.delta)
            for n_points in refinement_schedule(control.n_points_initial, control.n_points_final, n_iterations)]


def _run_nlopt(objective, x0, lower_bounds, upper_bounds, optimization_params, max_evaluations):
    """Maximize objective(x) with NLopt; returns (optimal_x, scaled optimum, termination code)."""
    op = optimization_params
    opt = nlopt.opt(getattr(nlopt, op.algorithm), len(x0))
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_max_objective(objective)
    opt.set_maxeval(max_evaluations)
    opt.set_ftol_rel(op.ftol_rel)
    opt.set_ftol_abs(op.ftol_abs)
    opt.set_xtol_rel(op.xtol_rel)
    opt.set_xtol_abs(op.xtol_abs)
    optimal_x = opt.optimize(np.clip(x0, lower_bounds, upper_bounds))
    return optimal_x, opt.last_optimum_value(), opt.last_optimize_result()


def optimize(config):
    """
    Optimize the controls by iterative refinement.

    Each refinement iteration optimizes the values of every control in
    optimization_parameters.controls at its control points for that iteration (see
    refinement_schedule and calculate_control_times; explicit control years when given).
    The first iteration starts every control at its initial guess (zero at the final
    point); later ones start from the previous solution interpolated onto the new grid.

    Parameters
    ----------
    config : ModelConfiguration

    Returns
    -------
    dict
        - 'controls': {name: (times, values)} of the final iteration (times relative to t_base)
        - 'objective': discounted utility of the final controls
        - 'n_evaluations': objective evaluations over all iterations
        - 'algorithm', 'n_iterations'
        - 'iteration_history': per iteration, a dict with 'iteration', 'controls',
          'objective', 'n_evaluations', 'termination_name', 'elapsed_time'
    """
    op = config.optimization_params
    n_iterations = op.optimization_iterations
    # The objective never fills NLopt's gradient, so derivative-based (xD_*) algorithms
    # would silently see a zero gradient.
    if op.algorithm[1] == 'D':
        raise ValueError(f"Algorithm {op.algorithm} needs gradients; use a derivative-free "
                         f"NLopt algorithm (LN_* or GN_*).")

    controls = op.controls
    control_grids = {name: control_times_by_iteration(control, n_iterations, config)
                     for name, control in controls.items()}
    max_evaluations_by_iteration = refinement_schedule(op.max_evaluations_initial, op.max_evaluations_final,
                                                       n_iterations)

    t_base = config.scalar_params.t_base
    t_join = config.integration_params.t_end - WIND_DOWN_EFOLDING_TIMES / config.scalar_params.delta
    print(f"\nIterative refinement: {n_iterations} iterations")
    for name, grids in control_grids.items():
        print(f"Control points per iteration, {name}: {[len(times) for times, _ in grids]}")
    print(f"Max evaluations per iteration: {max_evaluations_by_iteration}")
    print(f"Control point spacing: dynamic wind-down region (t > {t_join:.1f} years)")

    iteration_history = []
    for iteration in range(1, n_iterations + 1):
        max_evaluations = max_evaluations_by_iteration[iteration - 1]
        control_times = {}
        initial_guess = {}
        print(f"\n{'=' * 80}")
        print(f"  ITERATION {iteration}/{n_iterations}")
        print(f"  Algorithm: {op.algorithm}")
        print(f"  Max evaluations: {max_evaluations}")
        for name, control in controls.items():
            times, r0 = control_grids[name][iteration - 1]
            control_times[name] = times
            if iteration == 1:
                initial_guess[name] = np.full(len(times), control.initial_guess)
                initial_guess[name][-1] = 0.0
            else:
                initial_guess[name] = control_trajectory(*iteration_history[-1]['controls'][name], times)
            spacing = 'explicit control years' if r0 is None else f'segment 1 growth r0={r0:.4f}'
            print(f"\n  {name} - OPTIMIZED: {len(times)} control points ({spacing})")
            print(f"    Time points: {times + t_base}")
        print(f"{'=' * 80}\n")

        names = list(controls)
        split_points = np.cumsum([len(control_times[name]) for name in names])[:-1]
        n_evaluations = 0

        def nlopt_objective(x, grad):
            nonlocal n_evaluations
            n_evaluations += 1
            trial_controls = {name: (control_times[name], values)
                              for name, values in zip(names, np.split(x, split_points))}
            # ValueError: equal-utility-loss taxation cannot raise the required revenue.
            # RuntimeError: the tax root solve failed at extreme progressivity.
            # Either way the trajectory is infeasible.
            try:
                objective_value = discounted_utility(config, *control_arrays(config, trial_controls))
            except (ValueError, RuntimeError):
                objective_value = NEG_BIGNUM
            if not np.isfinite(objective_value):
                print(f"WARNING: Invalid objective value: {objective_value}")
                print(f"  x: {x}")
                return NEG_BIGNUM
            return objective_value * OBJECTIVE_SCALE

        iteration_start_time = time.time()
        optimal_x, scaled_optimum, termination_code = _run_nlopt(
            nlopt_objective,
            np.concatenate([initial_guess[name] for name in names]),
            np.concatenate([np.full(len(control_times[name]), controls[name].bounds[0]) for name in names]),
            np.concatenate([np.full(len(control_times[name]), controls[name].bounds[1]) for name in names]),
            op, max_evaluations,
        )
        iteration_result = {
            'iteration': iteration,
            'controls': {name: (control_times[name], values)
                         for name, values in zip(names, np.split(optimal_x, split_points))},
            'objective': scaled_optimum / OBJECTIVE_SCALE,
            'n_evaluations': n_evaluations,
            'termination_name': NLOPT_TERMINATION_NAMES[termination_code],
            'elapsed_time': time.time() - iteration_start_time,
        }
        iteration_history.append(iteration_result)

        print(f"\nIteration {iteration} complete:")
        print(f"  Objective value: {iteration_result['objective']:.12e}")
        print(f"  Evaluations: {iteration_result['n_evaluations']}")
        print(f"  Elapsed time: {iteration_result['elapsed_time']:.2f} seconds")
        print(f"  Status: {iteration_result['termination_name']}")
        for name, (_, values) in iteration_result['controls'].items():
            print(f"  Optimized {name} values: {values}")
        sys.stdout.flush()

    final_result = iteration_history[-1]
    return {
        'controls': final_result['controls'],
        'objective': final_result['objective'],
        'n_evaluations': sum(result['n_evaluations'] for result in iteration_history),
        'algorithm': op.algorithm,
        'n_iterations': n_iterations,
        'iteration_history': iteration_history,
    }
