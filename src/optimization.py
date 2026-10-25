"""
Optimization of the control trajectories to maximize discounted aggregate utility.

Controls: f(t) = log10(marginal abatement cost in $/tCO2), always optimized, and the
savings rate s(t), optimized jointly with f when optimization_parameters.initial_guess_s
is set and otherwise prescribed by time_functions['s'].

Each optimized control is parameterized by its values at control times (relative to
t_base) and interpolated with PCHIP onto the integration time grid. The optimization
proceeds by iterative refinement: successive NLopt runs on progressively finer control
grids, each initialized by interpolating the previous solution onto the new grid.
"""

import sys
import time
from dataclasses import dataclass
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
        s_values = np.array([config.time_functions['s'](t) for t in t_relative])
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

@dataclass
class ControlSchedule:
    """How one optimized control (f or s) is parameterized across refinement iterations."""
    name: str
    initial_guess: float
    bounds: list
    n_points_initial: int
    n_points_final: object  # int, list of explicit control years (calendar), or None
    refinement_base: float  # growth factor of (number of points - 1) per iteration


def _refinement_base(n_points_initial, n_points_final, n_iterations, base_if_unspecified):
    if n_points_final is None:
        return base_if_unspecified
    if n_iterations <= 1:
        return 2.0  # not used: a single iteration uses n_points_final directly
    return ((n_points_final - 1) / (n_points_initial - 1)) ** (1.0 / (n_iterations - 1))


def control_schedules(optimization_params):
    """Schedules for f and, when initial_guess_s is set, s."""
    op = optimization_params
    n_iterations = op.optimization_iterations
    f_schedule = ControlSchedule(
        name='f',
        initial_guess=op.initial_guess_f,
        bounds=op.bounds_f if op.bounds_f is not None else [0.0, 1.0],
        n_points_initial=op.n_points_initial_f,
        n_points_final=op.n_points_final_f,
        refinement_base=_refinement_base(op.n_points_initial_f, op.n_points_final_f, n_iterations, 2.0),
    )
    if op.initial_guess_s is None:
        return [f_schedule]
    s_schedule = ControlSchedule(
        name='s',
        initial_guess=op.initial_guess_s,
        bounds=op.bounds_s if op.bounds_s is not None else [0.0, 1.0],
        n_points_initial=op.n_points_initial_s,
        n_points_final=op.n_points_final_s,
        refinement_base=_refinement_base(op.n_points_initial_s, op.n_points_final_s, n_iterations,
                                         f_schedule.refinement_base),
    )
    return [f_schedule, s_schedule]


def iteration_control_times(schedule, iteration, n_iterations, config):
    """
    Control times (relative to t_base) of one control in a refinement iteration.

    Returns (times, r0); r0 is None for explicit control years.
    """
    t_base = config.scalar_params.t_base
    if isinstance(schedule.n_points_final, list):
        return np.array(schedule.n_points_final) - t_base, None
    if n_iterations == 1:
        n_points = schedule.n_points_final if schedule.n_points_final is not None else schedule.n_points_initial
    else:
        n_points = round(1 + (schedule.n_points_initial - 1) * schedule.refinement_base**(iteration - 1))
    ip = config.integration_params
    return calculate_control_times(n_points, ip.t_start - t_base, ip.t_end - t_base, ip.dt,
                                   config.scalar_params.delta)


def _run_nlopt(objective, x0, lower_bounds, upper_bounds, algorithm, optimization_params, max_evaluations):
    """Maximize objective(x) with NLopt; returns (optimal_x, scaled optimum, termination code)."""
    op = optimization_params
    opt = nlopt.opt(getattr(nlopt, algorithm), len(x0))
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_max_objective(objective)
    opt.set_maxeval(max_evaluations)
    if op.ftol_rel is not None:
        opt.set_ftol_rel(op.ftol_rel)
    if op.ftol_abs is not None:
        opt.set_ftol_abs(op.ftol_abs)
    if op.xtol_rel is not None:
        opt.set_xtol_rel(op.xtol_rel)
    if op.xtol_abs is not None:
        opt.set_xtol_abs(op.xtol_abs)
    optimal_x = opt.optimize(np.clip(x0, lower_bounds, upper_bounds))
    return optimal_x, opt.last_optimum_value(), opt.last_optimize_result()


def optimize(config):
    """
    Optimize the controls by iterative refinement.

    Iteration k uses round(1 + (n_points_initial - 1) * base^(k-1)) control points per
    control (or n_points_final, or explicit control years, when there is one iteration)
    and round(1 + (max_evaluations_initial - 1) * base_evaluations^(k-1)) evaluations.
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
    algorithm = op.algorithm if op.algorithm is not None else 'LN_SBPLX'
    # The objective never fills NLopt's gradient, so derivative-based (xD_*) algorithms
    # would silently see a zero gradient.
    if algorithm[1] == 'D':
        raise ValueError(f"Algorithm {algorithm} needs gradients; use a derivative-free "
                         f"NLopt algorithm (LN_* or GN_*).")

    schedules = control_schedules(op)
    if n_iterations <= 1:
        refinement_base_evaluations = 2.0  # not used: a single iteration uses max_evaluations_final
    else:
        refinement_base_evaluations = ((op.max_evaluations_final - 1) / (op.max_evaluations_initial - 1)) ** (1.0 / (n_iterations - 1))

    t_base = config.scalar_params.t_base
    t_join = config.integration_params.t_end - WIND_DOWN_EFOLDING_TIMES / config.scalar_params.delta
    print(f"\nIterative refinement: {n_iterations} iterations, "
          + ', '.join(f"base_{schedule.name} = {schedule.refinement_base:.4f}" for schedule in schedules)
          + f", base_evaluations = {refinement_base_evaluations:.4f}")
    print(f"Control point spacing: dynamic wind-down region (t > {t_join:.1f} years)")
    print(f"Max evaluations: initial={op.max_evaluations_initial}, final={op.max_evaluations_final}")

    iteration_history = []
    for iteration in range(1, n_iterations + 1):
        if n_iterations == 1:
            max_evaluations = op.max_evaluations_final
        else:
            max_evaluations = round(1 + (op.max_evaluations_initial - 1) * refinement_base_evaluations**(iteration - 1))

        control_times = {}
        initial_guess = {}
        print(f"\n{'=' * 80}")
        print(f"  ITERATION {iteration}/{n_iterations}")
        print(f"  Algorithm: {algorithm}")
        print(f"  Max evaluations: {max_evaluations}")
        for schedule in schedules:
            times, r0 = iteration_control_times(schedule, iteration, n_iterations, config)
            control_times[schedule.name] = times
            if iteration == 1:
                initial_guess[schedule.name] = np.full(len(times), schedule.initial_guess)
                initial_guess[schedule.name][-1] = 0.0
            else:
                initial_guess[schedule.name] = control_trajectory(*iteration_history[-1]['controls'][schedule.name], times)
            spacing = 'explicit control years' if r0 is None else f'segment 1 growth r0={r0:.4f}'
            print(f"\n  {schedule.name} - OPTIMIZED: {len(times)} control points ({spacing})")
            print(f"    Time points: {times + t_base}")
        print(f"{'=' * 80}\n")

        sizes = [len(control_times[schedule.name]) for schedule in schedules]
        split_points = np.cumsum(sizes)[:-1]
        n_evaluations = 0

        def nlopt_objective(x, grad):
            nonlocal n_evaluations
            n_evaluations += 1
            controls = {schedule.name: (control_times[schedule.name], values)
                        for schedule, values in zip(schedules, np.split(x, split_points))}
            # ValueError: equal-utility-loss taxation cannot raise the required revenue.
            # RuntimeError: the tax root solve failed at extreme progressivity.
            # Either way the trajectory is infeasible.
            try:
                objective_value = discounted_utility(config, *control_arrays(config, controls))
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
            np.concatenate([initial_guess[schedule.name] for schedule in schedules]),
            np.concatenate([np.full(size, schedule.bounds[0]) for schedule, size in zip(schedules, sizes)]),
            np.concatenate([np.full(size, schedule.bounds[1]) for schedule, size in zip(schedules, sizes)]),
            algorithm, op, max_evaluations,
        )
        iteration_result = {
            'iteration': iteration,
            'controls': {schedule.name: (control_times[schedule.name], values)
                         for schedule, values in zip(schedules, np.split(optimal_x, split_points))},
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
        'algorithm': algorithm,
        'n_iterations': n_iterations,
        'iteration_history': iteration_history,
    }
