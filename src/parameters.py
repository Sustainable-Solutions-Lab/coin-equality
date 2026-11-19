"""
Parameter definitions and configurations for COIN_equality model.

This module provides three categories of parameters:
1. Scalar parameters - time-invariant model constants
2. Time-dependent functions - evaluated at specific times
3. Integration/optimization control parameters

All parameters are required (no defaults) following the fail-fast philosophy.

Configuration is loaded from a JSON file that specifies all parameter values
and the run name (used for output directory naming).
"""

import json
import numpy as np
from dataclasses import dataclass


# =============================================================================
# Time-Dependent Function Factories
# =============================================================================

def create_constant(value):
    """
    Create a constant function that returns the same value for all times.

    Parameters
    ----------
    value : float
        The constant value to return

    Returns
    -------
    callable
        Function f(t) = value (an array of the shape of t)
    """
    return lambda t: np.full(np.shape(t), value)


def create_exponential_growth(exponential_scaling, growth_rate, additive_constant):
    """
    Create an exponential growth/decay function.

    Parameters
    ----------
    exponential_scaling : float
        Scaling factor for the exponential term
    growth_rate : float
        Growth rate (yr^-1). Positive for growth, negative for decay.
    additive_constant : float
        Constant offset added to the exponential

    Returns
    -------
    callable
        Function f(t) = exponential_scaling * exp(growth_rate * t) + additive_constant

    Examples
    --------
    Population growth at 1% per year:
    >>> L = create_exponential_growth(7e9, 0.01, 0.0)

    Carbon intensity declining at 2% per year with floor:
    >>> sigma = create_exponential_growth(0.5, -0.02, 0.01)
    """
    return lambda t: exponential_scaling * np.exp(growth_rate * t) + additive_constant


def create_logistic_growth(L0, L_inf, growth_rate):
    """
    Create a logistic (S-curve) growth function.

    Parameters
    ----------
    L0 : float
        Initial value at t=0
    L_inf : float
        Asymptotic limit as t -> infinity
    growth_rate : float
        Intrinsic growth rate (yr^-1)

    Returns
    -------
    callable
        Function f(t) = L_inf / (1 + ((L_inf/L0) - 1) * exp(-growth_rate * t))

    Examples
    --------
    Population growth from 7B to 10B:
    >>> L = create_logistic_growth(7e9, 10e9, 0.02)
    """
    return lambda t: L_inf / (1 + ((L_inf / L0) - 1) * np.exp(-growth_rate * t))


def create_piecewise_linear(time_points, values):
    """
    Create a piecewise linear function from discrete points.

    Parameters
    ----------
    time_points : array_like
        Time values (must be monotonically increasing)
    values : array_like
        Function values at each time point

    Returns
    -------
    callable
        Function that linearly interpolates between points

    Examples
    --------
    Abatement cost declining in steps:
    >>> theta1 = create_piecewise_linear([0, 50, 100], [0.1, 0.05, 0.02])
    """
    time_points = np.asarray(time_points)
    values = np.asarray(values)
    return lambda t: np.interp(t, time_points, values)


def create_double_exponential_growth(initial_value, growth_rate_1, growth_rate_2, fract_1):
    """
    Create a double exponential growth function (Barrage & Nordhaus 2023).

    Parameters
    ----------
    initial_value : float
        Value at t=0
    growth_rate_1 : float
        First growth rate (yr^-1)
    growth_rate_2 : float
        Second growth rate (yr^-1)
    fract_1 : float
        Fraction assigned to first exponential (0 ≤ fract_1 ≤ 1)

    Returns
    -------
    callable
        Function f(t) = initial_value * (fract_1 * exp(growth_rate_1 * t) + (1 - fract_1) * exp(growth_rate_2 * t))

    Examples
    --------
    Carbon intensity with two-phase decline:
    >>> sigma = create_double_exponential_growth(0.5, -0.02, -0.005, 0.7)
    """
    return lambda t: initial_value * (fract_1 * np.exp(growth_rate_1 * t) + (1 - fract_1) * np.exp(growth_rate_2 * t))


def create_gompertz_growth(initial_value, final_value, adjustment_coefficient):
    """
    Create a Gompertz growth function.

    Parameters
    ----------
    initial_value : float
        Value at t=0
    final_value : float
        Asymptotic limit as t → ∞ (for adjustment_coefficient < 0)
    adjustment_coefficient : float
        Growth rate parameter (yr^-1). Typically negative for growth from initial to final value.

    Returns
    -------
    callable
        Function L(t) = final_value * exp(ln((initial_value / final_value)) * exp(adjustment_coefficient * t)

    Notes
    -----
    At t=0: L(0) = initial_value
    As t → ∞ (with adjustment_coefficient < 0): L(t) → final_value

    Examples
    --------
    Population growing from 7B to 10B:
    >>> L = create_gompertz_growth(7e9, 10e9, -0.02)
    """
    return lambda t: final_value * np.exp(np.log(initial_value / final_value) * np.exp(adjustment_coefficient * t))


# =============================================================================
# Parameter Dataclasses
# =============================================================================

@dataclass
class ScalarParameters:
    """
    Time-invariant scalar parameters of the economic-climate model.

    Attributes
    ----------
    alpha : float
        Output elasticity of capital (capital share of income)
    delta : float
        Capital depreciation rate (yr^-1)
    psi1, psi2 : float
        Linear (°C⁻¹) and quadratic (°C⁻²) climate damage coefficients [Barrage & Nordhaus 2023]
    y_damage_distribution_exponent : float
        Climate Damage Vulnerability Exponent beta: damage fraction scales as
        (y_net / y_net_reference)^(-beta). 0 = uniform vulnerability, 1 = equal absolute damage
    y_net_reference : float
        Reference income for the damage power law ($/person)
    eta : float
        Coefficient of relative risk aversion (CRRA utility parameter)
    rho : float
        Pure rate of time preference (yr^-1)
    theta2 : float
        Abatement cost exponent
    K_initial : float
        Initial capital stock ($)
    tax_equity : float
        Tax Progressivity Parameter tau in [0, 1):
        eta_effective = 1 + tax_equity/(1-tax_equity) * (eta - 1). When eta > 1,
        tax_equity=0 gives a flat fractional rate, 0.5 equal utility loss, and -> 1
        concentrates tax on the highest incomes. At eta = 1 any value gives a flat rate.
    income_dependent_aggregate_damage : bool
        If True, aggregate damage decreases as the world gets richer; if False the
        aggregate damage fraction depends only on temperature (DICE-like)
    use_empirical_lorenz : bool
        If True, empirical Lorenz curve w0*F^p0 + ... + w3*F^p3 scaled to the Gini;
        if False, Pareto-Lorenz
    t_base : float
        Base year: time functions receive t - t_base
    mu_max : float
        Maximum abatement fraction (values > 1 allow net-negative emissions)
    year_extra_emission, amount_extra_emission : float
        Extra emission pulse (tCO2) in the time step containing that year (0 amount = none)
    year_extra_consumption, amount_extra_consumption : float
        Extra consumption pulse ($) in the time step containing that year
    loose_epsilon : float
        Convergence tolerance (in log deltaU) of the tax root solve
    """
    alpha: float
    delta: float
    psi1: float
    psi2: float
    y_damage_distribution_exponent: float
    y_net_reference: float
    eta: float
    rho: float
    theta2: float
    K_initial: float
    tax_equity: float
    income_dependent_aggregate_damage: bool
    use_empirical_lorenz: bool
    t_base: float
    mu_max: float
    year_extra_emission: float
    amount_extra_emission: float
    year_extra_consumption: float
    amount_extra_consumption: float
    loose_epsilon: float

    def __post_init__(self):
        """Validate the tax-policy domain."""
        # tax_equity = 1.0 is a singularity in the eta_eff mapping; values outside
        # [0, 1) are not part of the policy domain. Anything inside that range is
        # allowed; if the math underflows at extreme values, the solver / post-solve
        # invariant raises a descriptive RuntimeError (fail-fast).
        if not (0.0 <= self.tax_equity < 1.0):
            raise ValueError(
                f"tax_equity must be in [0.0, 1.0), got {self.tax_equity}."
            )


@dataclass
class IntegrationParameters:
    """
    Parameters controlling time integration.

    Attributes
    ----------
    t_start, t_end : float
        Start and end of integration (calendar years)
    dt : float
        Euler time step (yr)
    n_quad : int
        Number of Gauss-Legendre quadrature points over population rank.
        Typical: 16 (testing), 32 (production), 64 (high precision).
    """
    t_start: float
    t_end: float
    dt: float
    n_quad: int


@dataclass
class ControlParameters:
    """
    Parameterization of one optimized control (f or s) across refinement iterations.

    Attributes
    ----------
    initial_guess : float
        Value at every control point in the first iteration (the final point starts at 0)
    bounds : list of float
        [min, max] enforced by the optimizer
    n_points_initial : int
        Control points in the first iteration
    n_points_final : int or list of float
        Control points in the last iteration (the count grows geometrically in between), or
        an explicit list of control years (calendar), which requires optimization_iterations = 1
    """
    initial_guess: float
    bounds: list
    n_points_initial: int
    n_points_final: object

    def __post_init__(self):
        if isinstance(self.n_points_final, list):
            self.n_points_final = sorted(self.n_points_final)


@dataclass
class OptimizationParameters:
    """
    Parameters controlling optimization by iterative refinement.

    Attributes
    ----------
    max_evaluations_initial, max_evaluations_final : int
        Objective evaluations allowed in the first and last iteration; the budget grows
        geometrically in between
    optimization_iterations : int
        Number of refinement iterations
    algorithm : str
        Derivative-free NLopt algorithm used for every iteration:
        'LN_*' local (LN_SBPLX, LN_BOBYQA, LN_COBYLA, LN_NELDERMEAD) or 'GN_*' global
        (GN_ISRES, GN_DIRECT_L). Derivative-based algorithms are rejected.
    ftol_rel, ftol_abs, xtol_rel, xtol_abs : float
        NLopt stopping tolerances on the objective and on the control values; 0 disables
        a criterion
    controls : dict
        {'f': ControlParameters} for f = log10(marginal abatement cost), plus
        {'s': ControlParameters} when the savings rate is optimized jointly; otherwise
        s follows time_functions['s']
    """
    max_evaluations_initial: int
    max_evaluations_final: int
    optimization_iterations: int
    algorithm: str
    ftol_rel: float
    ftol_abs: float
    xtol_rel: float
    xtol_abs: float
    controls: dict

    def __post_init__(self):
        """Explicit control years only make sense with a single iteration."""
        if (any(isinstance(control.n_points_final, list) for control in self.controls.values())
                and self.optimization_iterations != 1):
            raise ValueError(
                "Explicit control point times (list) only supported with optimization_iterations=1. "
                f"Got optimization_iterations={self.optimization_iterations}"
            )


@dataclass
class ModelConfiguration:
    """
    Complete model configuration bundling all parameters.

    Attributes
    ----------
    run_name : str
        Name for this run (used for output directory naming)
    scalar_params : ScalarParameters
        Time-invariant scalar parameters
    time_functions : dict
        Maps names to time-dependent callables (relative time t - t_base).
        Required: EXOGENOUS_TIME_FUNCTIONS, and 's' when the savings rate is prescribed.
    integration_params : IntegrationParameters
        Integration control parameters
    optimization_params : OptimizationParameters
        Optimization control parameters

    The controls f(t) and s(t) are not part of the configuration; they are passed to
    integrate_model as arrays on the integration time grid.
    """
    run_name: str
    scalar_params: ScalarParameters
    time_functions: dict
    integration_params: IntegrationParameters
    optimization_params: OptimizationParameters


# =============================================================================
# Parameter Evaluation
# =============================================================================

# Exogenous time functions the model evaluates (the savings rate s, when prescribed,
# is evaluated with the controls)
EXOGENOUS_TIME_FUNCTIONS = ['A', 'L', 'sigma', 'theta1', 'gini', 'emission_ratio', 'Eland']


def evaluate_time_functions(config, t_array):
    """
    Evaluate the exogenous time functions at the calendar times t_array.

    Time functions receive relative time t - t_base (0 = base year).

    Returns
    -------
    dict
        Maps each name in EXOGENOUS_TIME_FUNCTIONS to an array over t_array
    """
    t_relative = t_array - config.scalar_params.t_base
    return {name: config.time_functions[name](t_relative) for name in EXOGENOUS_TIME_FUNCTIONS}


# =============================================================================
# Configuration Loading from JSON
# =============================================================================

# Time-function types in the configuration and their factories; the remaining keys of a
# time-function specification are the factory's arguments
TIME_FUNCTION_FACTORIES = {
    'constant': create_constant,
    'exponential_growth': create_exponential_growth,
    'logistic_growth': create_logistic_growth,
    'piecewise_linear': create_piecewise_linear,
    'double_exponential_growth': create_double_exponential_growth,
    'gompertz_growth': create_gompertz_growth,
}


def _create_time_function(func_spec):
    """Create a time-dependent function from its JSON specification ('type' plus arguments)."""
    arguments = {key: value for key, value in func_spec.items() if key != 'type'}
    return TIME_FUNCTION_FACTORIES[func_spec['type']](**arguments)


def _filter_description_keys(d):
    """
    Remove keys starting with '_' from dictionary (used for descriptions/comments).

    Parameters
    ----------
    d : dict
        Dictionary that may contain description keys

    Returns
    -------
    dict
        Dictionary with all keys starting with '_' removed
    """
    return {k: v for k, v in d.items() if not k.startswith('_')}


def load_configuration(config_path):
    """Load model configuration from a JSON file (see configuration_from_dict)."""
    with open(config_path, 'r') as f:
        return configuration_from_dict(json.load(f))


def configuration_from_dict(config_data):
    """
    Build the model configuration from a parsed JSON configuration dictionary.

    Parameters
    ----------
    config_data : dict
        Parsed JSON configuration with run_name, scalar_parameters, time_functions,
        integration_parameters and optimization_parameters (including its controls)

    Returns
    -------
    ModelConfiguration

    Notes
    -----
    Keys starting with '_' are comments and are ignored.
    See data/output/prod_s/9cases_s/*/9cases_s_gid-1.json for an example.
    """
    optimization_params_data = _filter_description_keys(config_data['optimization_parameters'])
    optimization_params_data['controls'] = {
        name: ControlParameters(**_filter_description_keys(spec))
        for name, spec in optimization_params_data['controls'].items()
    }
    return ModelConfiguration(
        run_name=config_data['run_name'],
        scalar_params=ScalarParameters(**_filter_description_keys(config_data['scalar_parameters'])),
        time_functions={
            name: _create_time_function(_filter_description_keys(spec))
            for name, spec in config_data['time_functions'].items()
            if not name.startswith('_')
        },
        integration_params=IntegrationParameters(**_filter_description_keys(config_data['integration_parameters'])),
        optimization_params=OptimizationParameters(**optimization_params_data),
    )
