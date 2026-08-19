"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import math
import numpy as np
import time
from scipy.special import roots_legendre, lambertw
from src.distribution_utilities import (
    find_log_deltaU_equal_utility,
    compute_post_tax_income_equal_utility,
    _get_omega_at_F,
    L_pareto,
    L_pareto_derivative,
    L_empirical_lorenz,
    L_empirical_lorenz_derivative,
    stepwise_interpolate,
    stepwise_integrate
)
from src.parameters import evaluate_params_at_time
from src.constants import EPSILON, NEG_BIGNUM, LOG_DELTAU_NO_TAX, EMPIRICAL_LORENZ_BASE_GINI, INVERSE_EPSILON, OMEGA_BASE_MAX


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


def gini_from_distribution(values_yi, Fi_edges, Fwi):
    """
    Calculate Gini coefficient from discretized distribution.

    Parameters
    ----------
    values_yi : np.ndarray
        Values at quadrature points (length N_QUAD)
    Fi_edges : np.ndarray
        Edges of bins in F space [0, 1] (length N_QUAD + 1)
    Fwi : np.ndarray
        Bin widths (length N_QUAD)

    Returns
    -------
    float
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)

    Notes
    -----
    Gini = 2 * integral from 0 to 1 of (F - L(F)) dF
    where L(F) is the Lorenz curve (cumulative fraction of total quantity
    held by bottom F fraction of population)
    """
    total = np.sum(Fwi * values_yi)

    if np.abs(total) <= EPSILON:
        return 0.0

    # Calculate Lorenz curve at each edge
    lorenz = np.zeros(len(Fi_edges))
    cumulative = 0.0

    for i in range(len(Fwi)):
        cumulative += Fwi[i] * values_yi[i]
        lorenz[i+1] = cumulative / total

    # Calculate Gini coefficient using trapezoidal rule
    # Gini = 2 * integral from 0 to 1 of (F - L(F)) dF
    gini = 0.0
    for i in range(len(Fi_edges) - 1):
        dF = Fi_edges[i+1] - Fi_edges[i]
        F_avg = (Fi_edges[i] + Fi_edges[i+1]) / 2.0
        L_avg = (lorenz[i] + lorenz[i+1]) / 2.0
        gini += (F_avg - L_avg) * dF

    gini *= 2.0

    return gini


# Global timing statistics
_timing_stats = {
    'call_count': 0,
    'total_time': 0.0,
    'setup_time': 0.0,
    'policy_calc_time': 0.0,
    'find_deltaU_time': 0.0,
    'segment2_time': 0.0,
    'utility_time': 0.0,
    'damage_agg_time': 0.0,
    'climate_time': 0.0,
    'finalize_time': 0.0,
}

def print_timing_stats():
    """Print timing statistics for calculate_tendencies."""
    stats = _timing_stats
    if stats['call_count'] == 0:
        return

    print(f"\n{'='*80}")
    print(f"TIMING STATISTICS (after {stats['call_count']} calls)")
    print(f"{'='*80}")
    print(f"Total time:          {stats['total_time']:8.2f} s  (100.0%)")
    print(f"  Setup:             {stats['setup_time']:8.2f} s  ({100*stats['setup_time']/stats['total_time']:5.1f}%)")
    print(f"  Policy calc:       {stats['policy_calc_time']:8.2f} s  ({100*stats['policy_calc_time']/stats['total_time']:5.1f}%)")
    print(f"    find_deltaU:     {stats['find_deltaU_time']:8.2f} s  ({100*stats['find_deltaU_time']/stats['total_time']:5.1f}%)")
    print(f"  Segment 2:         {stats['segment2_time']:8.2f} s  ({100*stats['segment2_time']/stats['total_time']:5.1f}%)")
    print(f"  Utility calc:      {stats['utility_time']:8.2f} s  ({100*stats['utility_time']/stats['total_time']:5.1f}%)")
    print(f"  Damage agg:        {stats['damage_agg_time']:8.2f} s  ({100*stats['damage_agg_time']/stats['total_time']:5.1f}%)")
    print(f"  Climate:           {stats['climate_time']:8.2f} s  ({100*stats['climate_time']/stats['total_time']:5.1f}%)")
    print(f"  Finalize:          {stats['finalize_time']:8.2f} s  ({100*stats['finalize_time']/stats['total_time']:5.1f}%)")
    print(f"Avg time per call:   {stats['total_time']/stats['call_count']*1000:8.3f} ms")
    print(f"{'='*80}\n")


def calculate_tendencies(state, params,
                        omega_yi_Omega_base_ratio_prev,
                        Omega_Omega_base_ratio_prev,
                        xi, xi_edges, wi, store_detailed_output):
    """
    Calculate time derivatives and all derived variables.

    Parameters
    ----------
    state : dict
        State variables:
        - 'K': Capital stock ($)
        - 'Ecum': Cumulative CO2 emissions (tCO2)
        - 'H1', 'H2', 'H3': Heat-uptake mode variables (°C)
    params : dict
        Model parameters (all must be provided):
        - 'alpha': Output elasticity of capital
        - 'delta': Capital depreciation rate (yr^-1)
        - 's': Savings rate
        - 'psi1': Linear climate damage coefficient (°C⁻¹) [Barrage & Nordhaus 2023]
        - 'psi2': Quadratic climate damage coefficient (°C⁻²) [Barrage & Nordhaus 2023]
        - 'y_damage_distribution_exponent': Exponent for income-dependent damage distribution
        - 'y_net_reference': Reference income for power-law damage scaling ($/person)
        - 'impulse_response_a1/a2/a3': Impulse response coefficients (°C/tCO₂)
        - 'impulse_response_tau1/tau2/tau3': Impulse response time constants (years)
        - 'eta': Coefficient of relative risk aversion
        - 'A': Total factor productivity (current)
        - 'L': Population (current)
        - 'sigma': Carbon intensity of GDP (current, tCO2 $^-1)
        - 'theta1': Abatement cost coefficient (current, $ tCO2^-1)
        - 'theta2': Abatement cost exponent
        - 'mu_max': Maximum allowed abatement fraction (cap on μ)
        - 'gini': Background Gini index (current, from time function)
        - 'Gini_fract': Fraction of Gini change as instantaneous step
        - 'Gini_restore': Rate of restoration to gini (yr^-1)
        - 'f': log10 of marginal abatement cost ($/tCO2)
    omega_yi_Omega_base_ratio_prev : np.ndarray
        Per-quantile ratio of climate damage to base damage from previous timestep
        (length N_QUAD, on the Gauss-Legendre grid over [0, 1]). Multiply by current
        Omega_base to get current damage fractions.
    Omega_Omega_base_ratio_prev : float
        Ratio of aggregate damage to base damage from previous timestep.
        Multiply by current Omega_base to get current aggregate damage fraction.
    xi : np.ndarray
        Gauss-Legendre quadrature nodes on [-1, 1] (length N_QUAD)
    xi_edges : np.ndarray
        Edges of quadrature intervals on [-1, 1] (length N_QUAD + 1)
    wi : np.ndarray
        Gauss-Legendre quadrature weights (length N_QUAD)
    store_detailed_output : bool, optional
        Whether to compute and return all intermediate variables. Default: True

    Returns
    -------
    dict
        Dictionary containing:
        - Tendencies: 'dK_dt', 'dEcum_dt', 'dH1_dt', 'dH2_dt', 'dH3_dt'
        - Climate damage ratios for next timestep: 'omega_yi_Omega_base_ratio', 'Omega_Omega_base_ratio'
        - All intermediate variables: Y_gross, delta_T, Omega, Y_net, y_net, redistribution,
          mu, Lambda, AbateCost, U, E

    Notes
    -----
    Calculation order follows equations 1.1-1.10, 2.1-2.2, 3.5, 4.3-4.4:
    1. Y_gross from K, L, A, α (Eq 1.1)
    2. ΔT from Ecum, H1, H2, H3 via impulse response (Eq 2.2)
    3. y_gross from Y_gross, L (mean per-capita gross income)
    4. Ω, G_climate from ΔT, Gini, y_gross, damage params (income-dependent damage)
    5. Y_damaged from Y_gross, Ω (Eq 1.3)
    6. y from Y_damaged, L, s (Eq 1.4)
    7. Δc from y, ΔL (Eq 4.3)
    8. E_pot from σ, Y_gross (Eq 2.1)
    9. AbateCost from f, Δc, L (Eq 1.5)
    10. μ from AbateCost, θ₁, θ₂, E_pot (Eq 1.6)
    11. Λ from AbateCost, Y_damaged (Eq 1.7)
    12. Y_net from Y_damaged, Λ (Eq 1.8)
    13. y_net from y, AbateCost, L (Eq 1.9)
    14. U from y_net, Gini, η (Eq 3.5)
    16. E from σ, μ, Y_gross (Eq 2.3)
    17. dK/dt from s, Y_net, δ, K (Eq 1.10)
    """
    t_start = time.time()
    _timing_stats['call_count'] += 1

    # Extract state variables
    K = state['K']
    Ecum = state['Ecum']

    # Extract parameters
    alpha = params['alpha']
    delta = params['delta']
    s = params['s']
    eta = params['eta']
    rho = params['rho']
    t = params['t']
    A = params['A'] # total factor productivity
    L = params['L'] # population
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    mu_max = params['mu_max']
    gini = params['gini']
    use_empirical_lorenz = params['use_empirical_lorenz']
    # f is now log10 of marginal abatement cost
    log10_mac = params['f']
    y_damage_distribution_exponent = params['y_damage_distribution_exponent']
    y_net_reference = params['y_net_reference']
    psi1 = params['psi1']
    psi2 = params['psi2']
    emission_ratio = params['emission_ratio']
    year_extra_emission = params['year_extra_emission']
    amount_extra_emission = params['amount_extra_emission']
    year_extra_consumption = params['year_extra_consumption']
    amount_extra_consumption = params['amount_extra_consumption']
    Eland = params['Eland']
    dt = params['dt']

    # Validate Gini coefficient for empirical Lorenz
    if use_empirical_lorenz and gini > EMPIRICAL_LORENZ_BASE_GINI:
        raise ValueError(
            f"Gini coefficient ({gini:.4f}) exceeds maximum allowed value "
            f"({EMPIRICAL_LORENZ_BASE_GINI:.4f}) for empirical Lorenz curve. "
            f"Either reduce Gini or set use_empirical_lorenz=false to use Pareto-Lorenz formulation."
        )

    # Policy switch (only remaining one)
    income_dependent_aggregate_damage = params['income_dependent_aggregate_damage']
    tax_equity = params['tax_equity']



    # Transform xi into F space. Map [-1,1] to [0,1]
    Fi = (xi + 1.0)/2.0
    # compute edges in F space
    Fi_edges = (xi_edges + 1.0)/2.0
    # Transform quadrature weights to F space [0,1] (wi is for xi space [-1,1])
    Fwi = wi / 2.0

    #========================================================================================
    # Simplified damage calculation using aggregate damage from previous timestep
    # No iteration needed - uses temperature-based Omega_base with previous damage for budgeting

    #========================================================================================
    # Main calculations

    # Eq 2.2: Temperature change via impulse response (Ricke & Caldeira 2014)
    # ΔT = R_∞·Ecum + H1 + H2 + H3, where R_∞ = -(a1+a2+a3)
    a1 = params['impulse_response_a1']
    a2 = params['impulse_response_a2']
    a3 = params['impulse_response_a3']
    R_inf = -(a1 + a2 + a3)
    H1 = state['H1']
    H2 = state['H2']
    H3 = state['H3']
    delta_T = R_inf * Ecum + H1 + H2 + H3

    # Base damage from temperature (capped just below 1.0 to avoid division by zero)
    # Be careful when used not to produce effective Omega values >= 1.0
    Omega_base = np.minimum(OMEGA_BASE_MAX, psi1 * delta_T + psi2 * (delta_T ** 2))

    # Reconstruct damage fractions from ratios stored at previous timestep.
    # Multiply stored ratios by current Omega_base to get current damage estimates.
    Omega_calc = np.clip(Omega_Omega_base_ratio_prev * Omega_base, 0.0, 1.0 - EPSILON)
    omega_yi_calc = np.clip(omega_yi_Omega_base_ratio_prev * Omega_base, 0.0, 1.0 - EPSILON)

    # Eq 1.1: Gross production per capita (Cobb-Douglas)
    # y_gross: gross production before climate damage and abatement cost
    if K > 0 and L > 0:
        y_gross = A * ((K / L) ** alpha)
    else:
        y_gross = 0.0

    # Use Omega from previous timestep for budgeting and damage calculations
    # y_damaged_calc: gross production net of climate damage (using previous timestep's damage)
    y_damaged_calc = y_gross * (1.0 - Omega_calc)
    climate_damage_calc = Omega_calc * y_gross

    t_setup = time.time()
    _timing_stats['setup_time'] += t_setup - t_start

    # -----------------------------------------------------------------------------------------
    #  Do redistribution, taxes, utility, etcc
    # -----------------------------------------------------------------------------------------

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
        eland = 0.0
        E_extra = 0.0
        consumption_extra = 0.0
        dK_dt = -delta * K
        omega_yi = np.zeros_like(xi)
        y_net_yi = np.zeros_like(xi)
        utility_yi = np.zeros_like(xi)
        average_tax_rate_yi = np.zeros_like(xi)
        marginal_tax_rate_yi = np.zeros_like(xi)
    else:
        # Economy exists - proceed with calculations

        # ============================================================================
        # marginal_abatement_cost-based optimization: compute mu and abateCost_amount from log10(marginal_abatement_cost)
        # This must happen BEFORE income distribution calculations that depend on
        # abateCost_amount (tax_amount, find_log_deltaU_equal_utility)
        # ============================================================================

        # Land emissions per capita (Eland is total land emissions from time function)
        eland = Eland / L

        # Eq 2.1: Potential emissions per capita (unabated), including land emissions
        epot = sigma * y_gross * emission_ratio + eland

        # Convert log10(marginal_abatement_cost) to linear marginal_abatement_cost
        marginal_abatement_cost = 10.0 ** log10_mac

        if epot > EPSILON and marginal_abatement_cost > EPSILON:
            # Invert marginal_abatement_cost to get mu using standard DICE formula
            mu = invert_mac_to_mu(marginal_abatement_cost, theta1, theta2, mu_max)

            # Compute abatement cost from mu using standard DICE formula
            abateCost_amount = (theta1 / theta2) * epot * (mu ** theta2)

            # Cap abatement cost to ensure lambda_abate < 1.0 and y_net > 0.
            # With MAC-based optimization, the optimizer controls marginal cost, not total cost.
            # High MAC values can produce abatement costs exceeding available income.
            abateCost_amount = min(abateCost_amount, (1.0 - EPSILON) * y_damaged_calc)
        else:
            mu = 0.0
            abateCost_amount = 0.0

        # Compute lambda_abate from the derived abateCost_amount
        lambda_abate = abateCost_amount / y_damaged_calc if y_damaged_calc > EPSILON else 0.0

        # Eq 1.8 & 1.9: Net production per capita after abatement cost and climate damage
        # y_net (aggregate): gross production net of climate damage and abatement cost
        # Note: consumption + savings = y_net
        y_net = (1.0 - lambda_abate) * y_damaged_calc

        # ============================================================================
        # Extra consumption pulse (for SCC calculation)
        # ============================================================================
        if year_extra_consumption >= t and year_extra_consumption < t + dt:
            consumption_extra = amount_extra_consumption / (L * dt)
        else:
            consumption_extra = 0.0

        tax_amount = abateCost_amount - consumption_extra  # per capita

        # ============================================================================
        # Tax policy: continuous-knob equal-utility-loss formula
        # eta_effective = 1 + tax_equity/(1-tax_equity) * (eta - 1)
        # At eta > 1:  tax_equity=0 -> flat rate, tax_equity=0.5 -> equal utility loss,
        #              tax_equity->1 -> tax richest.
        # At eta = 1 the knob is degenerate; the log-utility branch in find_K yields flat rate.
        # Monotonicity dc/dF > 0 is guaranteed by the closed form, so no iteration needed.
        # ============================================================================
        tax_exponent = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)
        log_deltaU_norm = LOG_DELTAU_NO_TAX
        if tax_amount > EPSILON:
            t_before_find_deltaU = time.time()
            log_deltaU_norm = find_log_deltaU_equal_utility(
                y_gross, gini, Omega_calc,
                omega_yi_calc, xi_edges,
                tax_exponent, tax_amount, use_empirical_lorenz,
                tol=params['loose_epsilon'],
                xi=xi, wi=wi,
            )
            _timing_stats['find_deltaU_time'] += time.time() - t_before_find_deltaU

        _timing_stats['policy_calc_time'] += time.time() - t_setup

        # ============================================================================
        # Income distribution and utility via Gauss-Legendre quadrature on [0, 1]
        # ============================================================================
        t_before_seg2 = time.time()
        if use_empirical_lorenz:
            dLdF_Fi = L_empirical_lorenz_derivative(Fi, gini)
        else:
            dLdF_Fi = L_pareto_derivative(Fi, gini)

        # Previous timestep's damage at each quadrature point (stepwise interpolation
        # of omega_yi_calc over the full [0, 1] range).
        omega_yi_prev = _get_omega_at_F(Fi, omega_yi_calc, xi_edges)

        y_pretax_yi = y_gross * dLdF_Fi * (1.0 - omega_yi_prev)
        # Evaluate the closed form in normalized units to avoid y^(1-eta) underflow at high
        # eta_eff. y_damaged_calc is the same y_damaged_mean find_log_deltaU_equal_utility normalized by.
        y_pretax_yi_norm = y_pretax_yi / y_damaged_calc
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
        # y_damage_distribution_exponent = 0 is the uniform vulnerability case: damage fraction Omega_base for all incomes (back-door escape).
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

        _timing_stats['segment2_time'] += time.time() - t_before_seg2

        # ============================================================================
        # Aggregate damage
        # When income_dependent_aggregate_damage=False, rescale the per-quantile damage array
        # so the aggregate damage fraction matches Omega_base.
        # ============================================================================
        t_before_damage_agg = time.time()
        total_income = np.sum(Fwi * y_net_yi)
        total_damage = np.sum(Fwi * y_net_yi * omega_yi)

        if not income_dependent_aggregate_damage:
            if total_damage > EPSILON and total_income > EPSILON:
                scale_factor = (Omega_base * total_income) / total_damage
                omega_yi = np.clip(omega_yi * scale_factor, 0.0, 1.0 - EPSILON)
                total_damage = np.sum(Fwi * y_net_yi * omega_yi)

        Omega = total_damage / total_income if total_income > EPSILON else 0.0

        _timing_stats['damage_agg_time'] += time.time() - t_before_damage_agg

    #========================================================================================
    # Emissions and remaining calculations
    # (mu, abateCost_amount, epot already computed at beginning of "Economy exists" section)
        t_before_climate = time.time()

        # Eq 2.3: CO2eq emissions per capita (after abatement) consider non-CO2 gases and land C emissions
        e = (1 - mu) * epot

        # Extra emission pulse (for sensitivity analysis)
        if amount_extra_emission == 0.0:
            E_extra = 0.0
        elif year_extra_emission >= t and year_extra_emission < t + dt:
            E_extra = amount_extra_emission / dt
        else:
            E_extra = 0.0

        # Eq 1.10: Capital tendency
        dK_dt = s * y_net * L - delta * K

        # aggregate utility
        U = aggregate_utility * L

        _timing_stats['climate_time'] += time.time() - t_before_climate

    #========================================================================================

    # Prepare output
    t_before_finalize = time.time()
    results = {}

    if store_detailed_output:
        # Store primary per-capita variables
        results.update({
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
            'Gini': gini,
            'gini': gini,
            'delta_T': delta_T,
            'Omega': Omega,
            'Omega_base': Omega_base,
            'Omega_calc': Omega_calc,
            'y_net_yi': y_net_yi,
            'omega_yi': omega_yi,
            'utility_yi': utility_yi,
            'average_tax_rate_yi': average_tax_rate_yi,
            'marginal_tax_rate_yi': marginal_tax_rate_yi,
            'max_average_tax_rate': float(np.max(average_tax_rate_yi)) if average_tax_rate_yi.size else 0.0,
            'max_marginal_tax_rate': float(np.max(marginal_tax_rate_yi)) if marginal_tax_rate_yi.size else 0.0,
            's': s,
            'dK_dt': dK_dt,
            'mu_max': mu_max,
            'emission_ratio': emission_ratio,
            'E_extra': E_extra,
            'consumption_extra': consumption_extra,
        })

    # Always return minimal variables needed for optimization
    dEcum_dt = e * L + E_extra
    results.update({
        'U': U,
        'mu': mu,
        'dK_dt': dK_dt,
        'dEcum_dt': dEcum_dt,
    })

    # Impulse response heat-uptake tendencies: dHi/dt = ai·E(t) - Hi/τi
    results['dH1_dt'] = a1 * dEcum_dt - H1 / params['impulse_response_tau1']
    results['dH2_dt'] = a2 * dEcum_dt - H2 / params['impulse_response_tau2']
    results['dH3_dt'] = a3 * dEcum_dt - H3 / params['impulse_response_tau3']

    # Always return climate damage ratio for use in next time step.
    # Store per-quantile ratio so next timestep can scale by its own Omega_base.
    if Omega_base > EPSILON:
        results['omega_yi_Omega_base_ratio'] = omega_yi / Omega_base
        results['Omega_Omega_base_ratio'] = Omega / Omega_base
    else:
        results['omega_yi_Omega_base_ratio'] = np.zeros(len(xi))
        results['Omega_Omega_base_ratio'] = 0.0

    t_end = time.time()
    _timing_stats['total_time'] += t_end - t_start
    _timing_stats['finalize_time'] += t_end - t_before_finalize

    # Print timing stats every 1000000 calls
    if _timing_stats['call_count'] % 1000000 == 0:
        print_timing_stats()

    return results


def integrate_model(config, store_detailed_output=True):
    """
    Integrate the model forward in time using Euler's method.

    Parameters
    ----------
    config : ModelConfiguration
        Complete model configuration including parameters and time-dependent functions
    store_detailed_output : bool, optional
        If True (default), stores all diagnostic variables for CSV/PDF output.
        If False, stores only t, U needed for optimization objective calculation.

    Returns
    -------
    dict
        Time series results with keys:
        - 't': array of time points
        - 'U': array of utility values (always stored)
        - 'L': array of population values (always stored, needed for objective function)

        If store_detailed_output=True, also includes:
        - 'K': array of capital stock values
        - 'Ecum': array of cumulative emissions values
        - 'Gini': array of Gini index values (from background)
        - 'gini': array of background Gini index values
        - 'A', 'sigma', 'theta1', 'f': time-dependent inputs
        - All derived variables: Y_gross, delta_T, Omega, Y_damaged, Y_net,
          redistribution, mu, Lambda, AbateCost, marginal_abatement_cost, y_net, E

    Notes
    -----
    Uses simple Euler integration: state(t+dt) = state(t) + dt * tendency(t)
    This ensures all functional relationships are satisfied exactly at output points.

    Initial conditions are computed automatically:
    - Ecum(0) = Ecum_initial (initial cumulative emissions from configuration)
    - K(0) = K_initial (from configuration)
    """
    # Extract integration parameters
    t_start = config.integration_params.t_start
    t_end = config.integration_params.t_end
    dt = config.integration_params.dt
    n_quad = config.integration_params.n_quad

    # Create time array
    t_array = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_array)

    # Precompute Gauss-Legendre quadrature nodes and weights (used for all timesteps)
    xi, wi = roots_legendre(n_quad)
    # Create xi_edges: cumulative weights starting at -1, ending at +1
    # wi sums to 2 (integrating over [-1,1]), so cumsum(wi) goes from wi[0] to 2
    # We want edges from -1 to +1, so: -1 + cumsum(wi) goes from -1+wi[0] to 1
    xi_edges = np.concatenate(([-1.0], -1.0 + np.cumsum(wi)))  # length n_quad + 1

    # Initialize climate damage ratios for first timestep (uniform over quadrature grid)
    omega_yi_Omega_base_ratio_prev = np.ones(n_quad)
    Omega_Omega_base_ratio_prev = 1.0

    # Initialize state variables
    state = {
        'K': config.scalar_params.K_initial,
        'Ecum': config.scalar_params.Ecum_initial,
        'H1': config.scalar_params.H1_initial,
        'H2': config.scalar_params.H2_initial,
        'H3': config.scalar_params.H3_initial,
    }

    # Initialize storage for variables
    results = {}

    if store_detailed_output:
        # Store params for create_derived_variables()
        results['params_list'] = []
        # Add storage for primary variables
        results.update({
            'A': np.zeros(n_steps),
            'sigma': np.zeros(n_steps),
            'theta1': np.zeros(n_steps),
            'f': np.zeros(n_steps),
            'y_gross': np.zeros(n_steps),
            'delta_T': np.zeros(n_steps),
            'Omega': np.zeros(n_steps),
            'Omega_base': np.zeros(n_steps),
            'Omega_calc': np.zeros(n_steps),
            'Gini': np.zeros(n_steps),
            'gini': np.zeros(n_steps),
            'y_damaged': np.zeros(n_steps),
            'climate_damage': np.zeros(n_steps),
            'tax_amount': np.zeros(n_steps),
            'log_deltaU_norm': np.full(n_steps, LOG_DELTAU_NO_TAX),
            'aggregate_utility': np.zeros(n_steps),
            'mu': np.zeros(n_steps),
            'lambda_abate': np.zeros(n_steps),
            'abateCost_amount': np.zeros(n_steps),
            'y_net': np.zeros(n_steps),
            'e': np.zeros(n_steps),
            'dK_dt': np.zeros(n_steps),
            's': np.zeros(n_steps),
            'y_net_yi': np.zeros((n_steps, n_quad)),
            'omega_yi': np.zeros((n_steps, n_quad)),
            'utility_yi': np.zeros((n_steps, n_quad)),
            'average_tax_rate_yi': np.zeros((n_steps, n_quad)),
            'marginal_tax_rate_yi': np.zeros((n_steps, n_quad)),
            'max_average_tax_rate': np.zeros(n_steps),
            'max_marginal_tax_rate': np.zeros(n_steps),
            'emission_ratio': np.zeros(n_steps),
            'E_extra': np.zeros(n_steps),
            'consumption_extra': np.zeros(n_steps),
            'H1': np.zeros(n_steps),
            'H2': np.zeros(n_steps),
            'H3': np.zeros(n_steps),
        })

    # Always store time, state variables, and objective function variables
    results.update({
        't': t_array,
        'K': np.zeros(n_steps),
        'Ecum': np.zeros(n_steps),
        'U': np.zeros(n_steps),
        'L': np.zeros(n_steps),  # Needed for objective function
    })

    # Store quadrature information (for xlsx output)
    if store_detailed_output:
        results.update({
            'xi': xi,
            'wi': wi,
            'xi_edges': xi_edges,
            'Fi': (xi + 1.0) / 2.0,
            'Fwi': wi / 2.0,
            'Fi_edges': (xi_edges + 1.0) / 2.0,
        })

    # Time stepping loop
    for i, t in enumerate(t_array):
        # Evaluate time-dependent parameters at current time
        params = evaluate_params_at_time(t, config)

        # Pass dt for extra emission pulse timing
        params['dt'] = dt

        if store_detailed_output:
            results['params_list'].append(params)

        # Calculate all variables and tendencies at current time.
        # Use previous-timestep damage ratios to avoid circular dependency.
        outputs = calculate_tendencies(state, params,
                                      omega_yi_Omega_base_ratio_prev,
                                      Omega_Omega_base_ratio_prev,
                                      xi, xi_edges, wi, store_detailed_output)

        # Always store variables needed for objective function
        results['U'][i] = outputs['U']
        results['L'][i] = params['L']

        if store_detailed_output:
            # Store state variables
            results['K'][i] = state['K']
            results['Ecum'][i] = state['Ecum']

            # Store time-dependent inputs
            results['A'][i] = params['A']
            results['sigma'][i] = params['sigma']
            results['theta1'][i] = params['theta1']
            results['f'][i] = params['f']

            # Store primary per-capita variables
            results['y_gross'][i] = outputs['y_gross']
            results['y_damaged'][i] = outputs['y_damaged']
            results['climate_damage'][i] = outputs['climate_damage']
            results['y_net'][i] = outputs['y_net']
            results['e'][i] = outputs['e']
            results['mu'][i] = outputs['mu']
            results['lambda_abate'][i] = outputs['lambda_abate']
            results['abateCost_amount'][i] = outputs['abateCost_amount']

            # Store tax variables
            results['tax_amount'][i] = outputs['tax_amount']
            results['log_deltaU_norm'][i] = outputs['log_deltaU_norm']

            # Store climate variables
            results['delta_T'][i] = outputs['delta_T']
            results['Omega'][i] = outputs['Omega']
            results['Omega_base'][i] = outputs['Omega_base']
            results['Omega_calc'][i] = outputs['Omega_calc']

            # Store scalars
            results['Gini'][i] = outputs['Gini']
            results['gini'][i] = outputs['gini']
            results['aggregate_utility'][i] = outputs['aggregate_utility']
            results['s'][i] = outputs['s']

            # Store distributions
            results['y_net_yi'][i, :] = outputs['y_net_yi']
            results['omega_yi'][i, :] = outputs['omega_yi']
            results['utility_yi'][i, :] = outputs['utility_yi']
            results['average_tax_rate_yi'][i, :] = outputs['average_tax_rate_yi']
            results['marginal_tax_rate_yi'][i, :] = outputs['marginal_tax_rate_yi']
            results['max_average_tax_rate'][i] = outputs['max_average_tax_rate']
            results['max_marginal_tax_rate'][i] = outputs['max_marginal_tax_rate']

            # Store tendencies
            results['dK_dt'][i] = outputs['dK_dt']

            results['emission_ratio'][i] = outputs['emission_ratio']
            results['E_extra'][i] = outputs['E_extra']
            results['consumption_extra'][i] = outputs['consumption_extra']

            results['H1'][i] = state['H1']
            results['H2'][i] = state['H2']
            results['H3'][i] = state['H3']

        # Euler step: update state for next iteration (skip on last step)
        if i < n_steps - 1:
            state['K'] = state['K'] + dt * outputs['dK_dt']
            # do not allow cumulative emissions to go negative, making it colder than the initial condition
            state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])

            # Euler step: impulse response heat-uptake state variables
            state['H1'] = state['H1'] + dt * outputs['dH1_dt']
            state['H2'] = state['H2'] + dt * outputs['dH2_dt']
            state['H3'] = state['H3'] + dt * outputs['dH3_dt']

            # Update damage ratios for next time step (lagged damage approach)
            omega_yi_Omega_base_ratio_prev = outputs['omega_yi_Omega_base_ratio']
            Omega_Omega_base_ratio_prev = outputs['Omega_Omega_base_ratio']

    # Print final timing statistics only when called directly (not during optimization)
    # print_timing_stats()

    return results


def create_derived_variables(results):
    """
    Create all derived variables from integration results.

    Computes total (uppercase) variables from per-capita (lowercase) variables,
    consumption/savings variables, marginal costs, and Gini coefficients.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model() containing:
        - Time series of primary per-capita variables (y_gross, y_net, e, mu, etc.)
        - params_list: list of parameter dicts at each timestep
        - L: population array
        - All distribution arrays (y_net_yi, omega_yi, utility_yi)

    Returns
    -------
    dict
        Updated results dictionary with derived variables added:
        - Total variables: Y_gross, Y_net, Y_damaged, E, AbateCost, Climate_damage
        - Consumption/savings: Consumption, consumption, Savings, savings
        - Other: marginal_abatement_cost, Lambda, redistribution, Redistribution_amount
        - Gini: (removed - gini_consumption, gini_utility, etc. no longer computed)
        - Discount rate: r_consumption (annual effective consumption discount rate from
          the welfare-weighted Ramsey rule over the income distribution)

    Notes
    -----
    All operations are vectorized over numpy arrays for efficiency.
    Modifies results dict in-place and returns it.
    """
    # Extract parameter arrays from params_list
    params_list = results['params_list']
    n_steps = len(results['t'])

    s_array = np.array([p['s'] for p in params_list])
    theta1_array = np.array([p['theta1'] for p in params_list])
    theta2_array = np.array([p['theta2'] for p in params_list])
    eta_array = np.array([p['eta'] for p in params_list])
    gini_array = np.array([p['gini'] for p in params_list])
    rho = params_list[0]['rho']

    # Extract primary arrays
    L = results['L']
    y_gross = results['y_gross']
    y_damaged = results['y_damaged']
    climate_damage = results['climate_damage']
    y_net = results['y_net']
    e = results['e']
    mu = results['mu']
    lambda_abate = results['lambda_abate']
    abateCost_amount = results['abateCost_amount']

    # Compute total (uppercase) variables from per-capita via vectorized multiplication
    Y_gross = y_gross * L
    Y_damaged = y_damaged * L
    Y_net = y_net * L
    AbateCost = abateCost_amount * L
    E = e * L
    Climate_damage = climate_damage * L

    # Compute consumption and savings variables
    Consumption = (1.0 - s_array) * Y_net
    consumption = (1.0 - s_array) * y_net
    Savings = s_array * Y_net
    savings = s_array * y_net

    # Compute marginal abatement cost using standard DICE formula
    marginal_abatement_cost = theta1_array * (mu ** (theta2_array - 1.0))

    # ---------- Welfare-adjusted MAC (wMAC) for SCC consistency ----------
    # SCC "$ pulse" in calculate_tendencies scales everyone's consumption by the same factor,
    # which corresponds to a consumption-proportional dollar numeraire. Under the single
    # continuous tax policy, abatement-cost incidence matches that numeraire (tax spread
    # across all incomes via the K formula), so wMAC == MAC and the ratio is 1.
    wi = results['wi']
    y_net_yi = results['y_net_yi']

    Fwi = wi / 2.0  # Gauss-Legendre weights remapped from [-1,1] to [0,1]
    c_yi = np.maximum((1.0 - s_array)[:, None] * y_net_yi, EPSILON)  # shape (n_steps, n_quad)

    C = np.maximum(np.sum(Fwi[None, :] * c_yi, axis=1), EPSILON)
    N = np.sum(Fwi[None, :] * (c_yi ** (1.0 - eta_array[:, None])), axis=1)
    mu_dollar_numeraire = N / C
    mu_dollar_cost = mu_dollar_numeraire
    welfare_adjusted_mac = marginal_abatement_cost.copy()
    mac_over_wmac = np.ones_like(marginal_abatement_cost)
    # --------------------------------------------------------------------

    # Create aliases
    Lambda = lambda_abate

    # Add dEcum_dt as alias to E for consistency
    dEcum_dt = E

    # Compute annual effective consumption discount rate via the welfare-weighted Ramsey rule.
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
    # uniform, so single-distribution results are unchanged. (Fwi and c_yi computed above.)
    t_array = results['t']
    dt = t_array[1] - t_array[0] if n_steps > 1 else 1.0

    # Social marginal utility at each timestep; strictly positive because c_yi >= EPSILON.
    social_marginal_utility = np.sum(Fwi[None, :] * c_yi ** (-eta_array[:, None]), axis=1)

    r_consumption = np.zeros(n_steps)
    if n_steps > 1:
        r_consumption[:-1] = np.exp(rho * dt) * (
            social_marginal_utility[:-1] / social_marginal_utility[1:]
        ) - 1.0
        # Final step has no forward data; reuse the previous step's growth (as before).
        r_consumption[-1] = r_consumption[-2]

    # Add all derived variables to results dict
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
        'mu_dollar_numeraire': mu_dollar_numeraire,
        'mu_dollar_cost': mu_dollar_cost,
        'welfare_adjusted_mac': welfare_adjusted_mac,
        'mac_over_wmac': mac_over_wmac,
        'Lambda': Lambda,
        'dEcum_dt': dEcum_dt,
        'r_consumption': r_consumption,
    })

    return results
