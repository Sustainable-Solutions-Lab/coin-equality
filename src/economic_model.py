"""
Economic production, climate impacts, income distribution and system tendencies.

The model is a Solow-Swan growth model with climate damage and emissions abatement,
an income distribution over population rank (Gauss-Legendre quadrature), progressive
taxation that pays for abatement, and an impulse-response temperature model. It is
integrated forward with Euler steps.
"""

import numpy as np
from src.distribution_utilities import (
    quadrature_grid,
    find_log_deltaU_equal_utility,
    compute_post_tax_income_equal_utility,
    L_pareto_derivative,
    L_empirical_lorenz_derivative,
)
from src.parameters import evaluate_time_functions
from src.constants import EPSILON, NEG_BIGNUM, LOG_DELTAU_NO_TAX, EMPIRICAL_LORENZ_BASE_GINI, OMEGA_BASE_MAX


def invert_mac_to_mu(marginal_abatement_cost, theta1, theta2, mu_max):
    """
    Invert marginal abatement cost to get abatement fraction mu using standard DICE formula.

    mu = (marginal_abatement_cost/theta1)^(1/(theta2-1))

    Parameters
    ----------
    marginal_abatement_cost : float
        Marginal abatement cost ($/tCO2)
    theta1 : float
        DICE abatement cost coefficient
    theta2 : float
        DICE abatement cost exponent
    mu_max : float
        Maximum allowed abatement fraction (cap on mu)

    Returns
    -------
    float
        Abatement fraction mu (capped at mu_max)
    """
    # Standard DICE inversion
    mu = (marginal_abatement_cost / theta1) ** (1.0 / (theta2 - 1.0))

    # Apply mu_max cap
    mu = min(mu_max, mu)

    return mu


def calculate_tendencies(state, sp, inputs, omega_yi_Omega_base_ratio_prev, Omega_Omega_base_ratio_prev,
                         grid, dt, store_detailed_output):
    """
    Calculate the time derivatives of the state and, optionally, all diagnostics at one time step.

    Parameters
    ----------
    state : dict
        'K' capital stock ($), 'Ecum' cumulative CO2 emissions (tCO2), 'H1', 'H2', 'H3'
        heat-uptake mode temperatures (°C)
    sp : ScalarParameters
        Time-invariant parameters
    inputs : dict
        Values at this step: 't' (calendar year), the exogenous time functions 'A', 'L',
        'sigma', 'theta1', 'gini', 'emission_ratio', 'Eland', and the controls
        'f' = log10(marginal abatement cost in $/tCO2) and 's' (savings rate)
    omega_yi_Omega_base_ratio_prev : np.ndarray
        Per-quantile damage fraction divided by Omega_base, from the previous step
    Omega_Omega_base_ratio_prev : float
        Aggregate damage fraction divided by Omega_base, from the previous step
    grid : QuadratureGrid
        Quadrature over population rank
    dt : float
        Time step (yr), used to spread emission and consumption pulses over the step
    store_detailed_output : bool
        Whether to return the diagnostics

    Returns
    -------
    tendencies : dict
        'U' (aggregate utility), 'dK_dt', 'dEcum_dt', 'dH1_dt', 'dH2_dt', 'dH3_dt', and the
        damage ratios for the next step 'omega_yi_Omega_base_ratio', 'Omega_Omega_base_ratio'
    diagnostics : dict or None
        Per-step output variables (scalars and per-quantile arrays) when
        store_detailed_output, else None

    Notes
    -----
    Climate damage is lagged one step: the damage fractions used for this step's budget are
    the previous step's damage-to-Omega_base ratios times this step's Omega_base. The steps:
    1. ΔT from cumulative emissions and heat-uptake modes (impulse response); Omega_base(ΔT)
    2. y_gross = A (K/L)^α; y_damaged from the lagged aggregate damage
    3. μ from the marginal abatement cost 10^f; abatement cost, capped below y_damaged
    4. Pre-tax income per quantile from the Lorenz curve and lagged per-quantile damage;
       the equal-utility-loss tax raises the abatement cost (less any consumption pulse)
    5. Utility per quantile (CRRA in consumption (1 - s) y_net) and its population mean
    6. Damage per quantile for the next step (power law in income, rescaled to Omega_base
       unless income_dependent_aggregate_damage)
    7. Emissions e = (1 - μ) e_pot (+ pulse), dK/dt = s y_net L - δ K, heat-uptake tendencies
    """
    # Extract state variables
    K = state['K']
    Ecum = state['Ecum']
    H1 = state['H1']
    H2 = state['H2']
    H3 = state['H3']

    # Extract inputs at this step
    t = inputs['t']
    A = inputs['A']  # total factor productivity
    L = inputs['L']  # population
    sigma = inputs['sigma']
    theta1 = inputs['theta1']
    gini = inputs['gini']
    emission_ratio = inputs['emission_ratio']
    Eland = inputs['Eland']
    log10_mac = inputs['f']
    s = inputs['s']

    # Extract parameters
    alpha = sp.alpha
    delta = sp.delta
    eta = sp.eta
    theta2 = sp.theta2
    mu_max = sp.mu_max
    use_empirical_lorenz = sp.use_empirical_lorenz
    y_damage_distribution_exponent = sp.y_damage_distribution_exponent
    y_net_reference = sp.y_net_reference
    psi1 = sp.psi1
    psi2 = sp.psi2
    income_dependent_aggregate_damage = sp.income_dependent_aggregate_damage
    tax_equity = sp.tax_equity

    # Validate Gini coefficient for empirical Lorenz
    if use_empirical_lorenz and gini > EMPIRICAL_LORENZ_BASE_GINI:
        raise ValueError(
            f"Gini coefficient ({gini:.4f}) exceeds maximum allowed value "
            f"({EMPIRICAL_LORENZ_BASE_GINI:.4f}) for empirical Lorenz curve. "
            f"Either reduce Gini or set use_empirical_lorenz=false to use Pareto-Lorenz formulation."
        )

    Fi = grid.Fi
    Fwi = grid.Fwi

    # Temperature change via impulse response (Ricke & Caldeira 2014)
    # ΔT = R_∞·Ecum + H1 + H2 + H3, where R_∞ = -(a1+a2+a3)
    a1 = sp.impulse_response_a1
    a2 = sp.impulse_response_a2
    a3 = sp.impulse_response_a3
    R_inf = -(a1 + a2 + a3)
    delta_T = R_inf * Ecum + H1 + H2 + H3

    # Base damage from temperature (capped just below 1.0 to avoid division by zero)
    # Be careful when used not to produce effective Omega values >= 1.0
    Omega_base = np.minimum(OMEGA_BASE_MAX, psi1 * delta_T + psi2 * (delta_T ** 2))

    # Reconstruct damage fractions from ratios stored at previous timestep.
    # Multiply stored ratios by current Omega_base to get current damage estimates.
    Omega_calc = np.clip(Omega_Omega_base_ratio_prev * Omega_base, 0.0, 1.0 - EPSILON)
    omega_yi_calc = np.clip(omega_yi_Omega_base_ratio_prev * Omega_base, 0.0, 1.0 - EPSILON)

    # Gross production per capita (Cobb-Douglas), before climate damage and abatement cost
    if K > 0 and L > 0:
        y_gross = A * ((K / L) ** alpha)
    else:
        y_gross = 0.0

    # Gross production net of climate damage, using the previous step's damage for budgeting
    y_damaged_calc = y_gross * (1.0 - Omega_calc)
    climate_damage_calc = Omega_calc * y_gross

    if y_gross <= EPSILON or y_damaged_calc <= EPSILON:
        # Economy has collapsed - set all downstream variables to zero or appropriate values
        abateCost_amount = 0.0
        tax_amount = 0.0
        log_deltaU_norm = LOG_DELTAU_NO_TAX
        aggregate_utility = NEG_BIGNUM
        Omega = 0.0
        lambda_abate = 0.0
        y_net = 0.0
        mu = 0.0
        U = NEG_BIGNUM
        e = 0.0
        E_extra = 0.0
        consumption_extra = 0.0
        dK_dt = -delta * K
        omega_yi = np.zeros_like(Fi)
        y_net_yi = np.zeros_like(Fi)
        utility_yi = np.zeros_like(Fi)
        average_tax_rate_yi = np.zeros_like(Fi)
        marginal_tax_rate_yi = np.zeros_like(Fi)
    else:
        # Abatement: mu and its cost follow from the marginal abatement cost, and must be
        # known before the income distribution, which pays for them through the tax.

        # Land emissions per capita (Eland is total land emissions from time function)
        eland = Eland / L

        # Potential emissions per capita (unabated), including land emissions
        epot = sigma * y_gross * emission_ratio + eland

        marginal_abatement_cost = 10.0 ** log10_mac

        if epot > EPSILON and marginal_abatement_cost > EPSILON:
            mu = invert_mac_to_mu(marginal_abatement_cost, theta1, theta2, mu_max)

            # Abatement cost from mu using standard DICE formula
            abateCost_amount = (theta1 / theta2) * epot * (mu ** theta2)

            # Cap abatement cost to ensure lambda_abate < 1.0 and y_net > 0.
            # The optimizer controls marginal cost, not total cost, and high marginal
            # costs can produce abatement costs exceeding available income.
            abateCost_amount = min(abateCost_amount, (1.0 - EPSILON) * y_damaged_calc)
        else:
            mu = 0.0
            abateCost_amount = 0.0

        lambda_abate = abateCost_amount / y_damaged_calc if y_damaged_calc > EPSILON else 0.0

        # Net production per capita after abatement cost and climate damage
        # (consumption + savings = y_net)
        y_net = (1.0 - lambda_abate) * y_damaged_calc

        # Extra consumption pulse (for SCC calculation)
        if pulse_in_step(sp.year_extra_consumption, t, dt):
            consumption_extra = sp.amount_extra_consumption / (L * dt)
        else:
            consumption_extra = 0.0

        tax_amount = abateCost_amount - consumption_extra  # per capita

        # Pre-tax income per quantile from the Lorenz curve and the lagged per-quantile damage
        # (the quadrature nodes lie inside their damage intervals, so the lagged damage at
        # node i is omega_yi_calc[i])
        if use_empirical_lorenz:
            dLdF_Fi = L_empirical_lorenz_derivative(Fi, gini)
        else:
            dLdF_Fi = L_pareto_derivative(Fi, gini)
        y_pretax_yi = y_gross * dLdF_Fi * (1.0 - omega_yi_calc)
        # Work in units of mean damaged income to avoid y^(1-eta) underflow at high eta_eff
        y_pretax_yi_norm = y_pretax_yi / y_damaged_calc

        # Tax policy: continuous-knob equal-utility-loss formula
        # eta_effective = 1 + tax_equity/(1-tax_equity) * (eta - 1)
        # At eta > 1:  tax_equity=0 -> flat rate, tax_equity=0.5 -> equal utility loss,
        #              tax_equity->1 -> tax richest.
        # At eta = 1 the knob is degenerate: the log-utility branch yields a flat rate.
        # Monotonicity dc/dF > 0 is guaranteed by the closed form, so no iteration needed.
        tax_exponent = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)
        log_deltaU_norm = LOG_DELTAU_NO_TAX
        if tax_amount > EPSILON:
            log_deltaU_norm = find_log_deltaU_equal_utility(
                y_pretax_yi_norm, tax_amount / y_damaged_calc, tax_exponent, Fwi, sp.loose_epsilon)

        c_yi_norm = compute_post_tax_income_equal_utility(y_pretax_yi_norm, log_deltaU_norm, tax_exponent)
        y_net_yi = c_yi_norm * y_damaged_calc

        # Diagnostic tax rates (not used by the simulation; surfaced for plotting / spot checks).
        # Average rate = 1 - c/y; marginal rate = 1 - dc/dy = 1 - (c/y)^tax_exponent.
        # At clipped individuals (c = 0) the marginal rate is defined as 1 (any extra
        # income would also be fully taxed to zero); without this guard, 0 ** negative
        # blows up when tax_exponent < 0 (which happens for eta < 1 with large tax_equity).
        c_over_y = np.where(y_pretax_yi > EPSILON, y_net_yi / np.maximum(y_pretax_yi, EPSILON), 1.0)
        average_tax_rate_yi = 1.0 - c_over_y
        marginal_tax_rate_yi = np.where(
            c_over_y > EPSILON,
            1.0 - np.maximum(c_over_y, EPSILON) ** tax_exponent,
            1.0,
        )

        consumption_yi = y_net_yi * (1 - s)
        if eta == 1:
            utility_yi = np.log(np.maximum(consumption_yi, EPSILON))
        else:
            utility_yi = (np.maximum(consumption_yi, EPSILON) ** (1 - eta)) / (1 - eta)

        # Damage for next timestep (income-dependent distribution).
        # y_damage_distribution_exponent = 0 is the uniform vulnerability case: damage fraction Omega_base for all incomes.
        if np.abs(y_damage_distribution_exponent) < EPSILON:
            omega_yi = np.full_like(y_net_yi, Omega_base)
        else:
            omega_yi = np.where(
                y_net_yi > EPSILON,
                Omega_base * (y_net_yi / y_net_reference) ** (-y_damage_distribution_exponent),
                1.0 - EPSILON,
            )
        omega_yi = np.clip(omega_yi, 0.0, 1.0 - EPSILON)

        aggregate_utility = np.sum(Fwi * utility_yi)

        # Aggregate damage. When income_dependent_aggregate_damage=False, rescale the
        # per-quantile damage array so the aggregate damage fraction matches Omega_base.
        total_income = np.sum(Fwi * y_net_yi)
        total_damage = np.sum(Fwi * y_net_yi * omega_yi)

        if not income_dependent_aggregate_damage:
            if total_damage > EPSILON and total_income > EPSILON:
                scale_factor = (Omega_base * total_income) / total_damage
                omega_yi = np.clip(omega_yi * scale_factor, 0.0, 1.0 - EPSILON)
                total_damage = np.sum(Fwi * y_net_yi * omega_yi)

        Omega = total_damage / total_income if total_income > EPSILON else 0.0

        # CO2eq emissions per capita after abatement (non-CO2 gases and land emissions included)
        e = (1 - mu) * epot

        # Extra emission pulse (for sensitivity analysis)
        if sp.amount_extra_emission == 0.0:
            E_extra = 0.0
        elif pulse_in_step(sp.year_extra_emission, t, dt):
            E_extra = sp.amount_extra_emission / dt
        else:
            E_extra = 0.0

        # Capital tendency
        dK_dt = s * y_net * L - delta * K

        # Aggregate (population-weighted) utility
        U = aggregate_utility * L

    dEcum_dt = e * L + E_extra
    tendencies = {
        'U': U,
        'dK_dt': dK_dt,
        'dEcum_dt': dEcum_dt,
        # Impulse response heat-uptake tendencies: dHi/dt = ai·E(t) - Hi/τi
        'dH1_dt': a1 * dEcum_dt - H1 / sp.impulse_response_tau1,
        'dH2_dt': a2 * dEcum_dt - H2 / sp.impulse_response_tau2,
        'dH3_dt': a3 * dEcum_dt - H3 / sp.impulse_response_tau3,
    }
    # Damage ratios for the next step, which scales them by its own Omega_base
    if Omega_base > EPSILON:
        tendencies['omega_yi_Omega_base_ratio'] = omega_yi / Omega_base
        tendencies['Omega_Omega_base_ratio'] = Omega / Omega_base
    else:
        tendencies['omega_yi_Omega_base_ratio'] = np.zeros(len(Fi))
        tendencies['Omega_Omega_base_ratio'] = 0.0

    if not store_detailed_output:
        return tendencies, None

    diagnostics = {
        'y_gross': y_gross,
        'y_damaged': y_damaged_calc,
        'climate_damage': climate_damage_calc,
        'y_net': y_net,
        'e': e,
        'mu': mu,
        'lambda_abate': lambda_abate,
        'abateCost_amount': abateCost_amount,
        'tax_amount': tax_amount,
        'log_deltaU_norm': log_deltaU_norm,
        'aggregate_utility': aggregate_utility,
        'delta_T': delta_T,
        'Omega': Omega,
        'Omega_base': Omega_base,
        'Omega_calc': Omega_calc,
        'y_net_yi': y_net_yi,
        'omega_yi': omega_yi,
        'utility_yi': utility_yi,
        'average_tax_rate_yi': average_tax_rate_yi,
        'marginal_tax_rate_yi': marginal_tax_rate_yi,
        'max_average_tax_rate': np.max(average_tax_rate_yi),
        'max_marginal_tax_rate': np.max(marginal_tax_rate_yi),
        'dK_dt': dK_dt,
        'E_extra': E_extra,
        'consumption_extra': consumption_extra,
    }
    return tendencies, diagnostics


def pulse_in_step(year, t, dt):
    """Whether a pulse dated `year` falls in the time step [t, t + dt)."""
    return year >= t and year < t + dt


def time_grid(integration_params):
    """Calendar times of the Euler integration steps, t_start to t_end inclusive."""
    return np.arange(integration_params.t_start, integration_params.t_end + integration_params.dt,
                     integration_params.dt)


# Inputs stored in the detailed results (Eland is used but not reported)
REPORTED_INPUTS = ['A', 'L', 'sigma', 'theta1', 'gini', 'emission_ratio', 'f', 's']


def integrate_model(config, f_values, s_values, store_detailed_output):
    """
    Integrate the model forward in time using Euler's method.

    Parameters
    ----------
    config : ModelConfiguration
        Complete model configuration including parameters and time-dependent functions
    f_values : np.ndarray
        Control f = log10(marginal abatement cost) at each time of time_grid(config.integration_params)
    s_values : np.ndarray
        Savings rate at each time of time_grid(config.integration_params)
    store_detailed_output : bool
        If True, return all variables (for output files); if False, only what the
        optimization objective needs.

    Returns
    -------
    dict
        Always 't' (calendar years) and 'U' (aggregate utility). With store_detailed_output,
        also the state ('K', 'Ecum', 'H1', 'H2', 'H3'), the reported inputs
        (REPORTED_INPUTS, plus 'Gini', the same as 'gini'), the per-step diagnostics of
        calculate_tendencies (per-quantile ones as (n_steps, n_quad) arrays) and the
        quadrature grid ('xi', 'wi', 'xi_edges', 'Fi', 'Fwi', 'Fi_edges').

    Notes
    -----
    Euler integration, state(t+dt) = state(t) + dt * tendency(t), so all functional
    relationships hold exactly at the output times. The initial state is K_initial,
    Ecum_initial and H1_initial-H3_initial. Cumulative emissions are kept non-negative
    (no cooling below the initial condition).
    """
    sp = config.scalar_params
    dt = config.integration_params.dt
    grid = quadrature_grid(config.integration_params.n_quad)
    t_array = time_grid(config.integration_params)

    inputs = evaluate_time_functions(config, t_array)
    inputs.update({'t': t_array, 'f': f_values, 's': s_values})

    state = {'K': sp.K_initial, 'Ecum': sp.Ecum_initial, 'H1': sp.H1_initial, 'H2': sp.H2_initial, 'H3': sp.H3_initial}
    # First step: damage distributed uniformly, aggregate damage equal to Omega_base
    omega_yi_Omega_base_ratio = np.ones(len(grid.Fi))
    Omega_Omega_base_ratio = 1.0

    U = np.zeros(len(t_array))
    step_records = []
    for i in range(len(t_array)):
        step_inputs = {name: values[i] for name, values in inputs.items()}
        tendencies, diagnostics = calculate_tendencies(state, sp, step_inputs,
                                                       omega_yi_Omega_base_ratio, Omega_Omega_base_ratio,
                                                       grid, dt, store_detailed_output)
        U[i] = tendencies['U']
        if store_detailed_output:
            step_records.append({**state, **diagnostics})

        state = {
            'K': state['K'] + dt * tendencies['dK_dt'],
            'Ecum': max(0.0, state['Ecum'] + dt * tendencies['dEcum_dt']),
            'H1': state['H1'] + dt * tendencies['dH1_dt'],
            'H2': state['H2'] + dt * tendencies['dH2_dt'],
            'H3': state['H3'] + dt * tendencies['dH3_dt'],
        }
        omega_yi_Omega_base_ratio = tendencies['omega_yi_Omega_base_ratio']
        Omega_Omega_base_ratio = tendencies['Omega_Omega_base_ratio']

    if not store_detailed_output:
        return {'t': t_array, 'U': U}

    results = {name: np.array([record[name] for record in step_records], dtype=float) for name in step_records[0]}
    results.update({name: np.asarray(inputs[name], dtype=float) for name in REPORTED_INPUTS})
    results['Gini'] = results['gini']
    results.update({
        't': t_array,
        'U': U,
        'xi': grid.xi,
        'wi': grid.wi,
        'xi_edges': grid.xi_edges,
        'Fi': grid.Fi,
        'Fwi': grid.Fwi,
        'Fi_edges': grid.Fi_edges,
    })
    return results


def create_derived_variables(results, scalar_params):
    """
    Add derived variables to the detailed results of integrate_model.

    Parameters
    ----------
    results : dict
        Detailed results of integrate_model (modified in place and returned)
    scalar_params : ScalarParameters

    Returns
    -------
    dict
        results with:
        - totals (uppercase) from per-capita (lowercase) values: Y_gross, Y_damaged,
          Y_net, AbateCost, E, Climate_damage
        - Consumption, consumption, Savings, savings
        - marginal_abatement_cost from mu (standard DICE formula)
        - aliases Lambda (= lambda_abate) and dEcum_dt (= E)
        - r_consumption: annual effective consumption discount rate from the
          welfare-weighted Ramsey rule over the income distribution
    """
    n_steps = len(results['t'])
    s_array = results['s']
    eta = scalar_params.eta
    rho = scalar_params.rho
    L = results['L']

    # Totals (uppercase) from per-capita values
    Y_gross = results['y_gross'] * L
    Y_damaged = results['y_damaged'] * L
    Y_net = results['y_net'] * L
    AbateCost = results['abateCost_amount'] * L
    E = results['e'] * L
    Climate_damage = results['climate_damage'] * L

    Consumption = (1.0 - s_array) * Y_net
    consumption = (1.0 - s_array) * results['y_net']
    Savings = s_array * Y_net
    savings = s_array * results['y_net']

    # Marginal abatement cost using standard DICE formula
    marginal_abatement_cost = results['theta1'] * (results['mu'] ** (scalar_params.theta2 - 1.0))

    # Annual effective consumption discount rate via the welfare-weighted Ramsey rule.
    #
    # The standard Ramsey rule, r = rho + eta * g, assumes a single representative agent
    # growing at one rate g. When income groups grow at different rates there is no unique g;
    # the social discount rate for a marginal consumption increment distributed across society
    # (utilitarian welfare W = sum_i Fwi * u(c_i)) is
    #     r = rho - d/dt ln( sum_i Fwi * u'(c_i) ),   with CRRA  u'(c) = c^(-eta).
    # Its exact discrete analog replaces the single marginal utility by the population-weighted
    # social marginal utility summed over the quadrature income groups:
    #     M(t)  = sum_j Fwi_j * c_yi(t)_j^(-eta)
    #     r(t)  = exp(rho * dt) * M(t) / M(t+dt) - 1.
    # This reduces exactly to exp(rho*dt) * (c(t+dt)/c(t))^eta - 1 when the distribution is
    # uniform, so single-distribution results are unchanged.
    t_array = results['t']
    dt = t_array[1] - t_array[0] if n_steps > 1 else 1.0
    Fwi = results['Fwi']
    c_yi = np.maximum((1.0 - s_array)[:, None] * results['y_net_yi'], EPSILON)  # shape (n_steps, n_quad)

    # Social marginal utility at each timestep; strictly positive because c_yi >= EPSILON.
    social_marginal_utility = np.sum(Fwi[None, :] * c_yi ** (-eta), axis=1)

    r_consumption = np.zeros(n_steps)
    if n_steps > 1:
        r_consumption[:-1] = np.exp(rho * dt) * (
            social_marginal_utility[:-1] / social_marginal_utility[1:]
        ) - 1.0
        # Final step has no forward data; reuse the previous step's growth (as before).
        r_consumption[-1] = r_consumption[-2]

    results.update({
        'Y_gross': Y_gross,
        'Y_damaged': Y_damaged,
        'Y_net': Y_net,
        'AbateCost': AbateCost,
        'E': E,
        'Climate_damage': Climate_damage,
        'Consumption': Consumption,
        'consumption': consumption,
        'Savings': Savings,
        'savings': savings,
        'marginal_abatement_cost': marginal_abatement_cost,
        'Lambda': results['lambda_abate'],
        'dEcum_dt': E,
        'r_consumption': r_consumption,
    })
    return results
