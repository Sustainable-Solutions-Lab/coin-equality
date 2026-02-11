"""
Functions for calculating economic production, climate impacts, and system tendencies.

This module implements the Solow-Swann growth model with climate damage
and emissions abatement costs.
"""

import numpy as np
import time
from scipy.special import roots_legendre
from distribution_utilities import (
    y_net_of_F,
    find_Fmax,
    find_Fmin,
    L_pareto,
    L_pareto_derivative,
    L_empirical_lorenz,
    L_empirical_lorenz_derivative,
    stepwise_interpolate,
    stepwise_integrate
)
from parameters import evaluate_params_at_time
from constants import EPSILON, LOOSE_EPSILON, NEG_BIGNUM, EMPIRICAL_LORENZ_BASE_GINI, INVERSE_EPSILON

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


def remap_quadrature_to_interval(xi, wi, F_lower, F_upper):
    """
    Remap Gauss-Legendre quadrature from [-1,1] to [F_lower, F_upper].

    Parameters
    ----------
    xi : ndarray
        Gauss-Legendre nodes on [-1, 1] (length n_quad)
    wi : ndarray
        Gauss-Legendre weights on [-1, 1] (length n_quad)
    F_lower : float
        Lower bound of target interval
    F_upper : float
        Upper bound of target interval

    Returns
    -------
    tuple of (ndarray, ndarray)
        (Fi_remapped, Fwi_remapped) - nodes and weights on [F_lower, F_upper]

    Notes
    -----
    Transformation formulas:
    - Fi_remapped = F_lower + (F_upper - F_lower) * (xi + 1) / 2
    - Fwi_remapped = wi * (F_upper - F_lower) / 2

    Property: sum(Fwi_remapped) = F_upper - F_lower
    """
    interval_length = F_upper - F_lower
    Fi_remapped = F_lower + interval_length * (xi + 1.0) / 2.0
    Fwi_remapped = wi * interval_length / 2.0
    return Fi_remapped, Fwi_remapped


# Global timing statistics
_timing_stats = {
    'call_count': 0,
    'total_time': 0.0,
    'setup_time': 0.0,
    'policy_calc_time': 0.0,
    'find_Fmax_time': 0.0,
    'find_Fmin_time': 0.0,
    'segment1_time': 0.0,
    'segment2_time': 0.0,
    'segment3_time': 0.0,
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
    print(f"    find_Fmax:       {stats['find_Fmax_time']:8.2f} s  ({100*stats['find_Fmax_time']/stats['total_time']:5.1f}%)")
    print(f"    find_Fmin:       {stats['find_Fmin_time']:8.2f} s  ({100*stats['find_Fmin_time']/stats['total_time']:5.1f}%)")
    print(f"  Segment 1:         {stats['segment1_time']:8.2f} s  ({100*stats['segment1_time']/stats['total_time']:5.1f}%)")
    print(f"  Segment 2:         {stats['segment2_time']:8.2f} s  ({100*stats['segment2_time']/stats['total_time']:5.1f}%)")
    print(f"  Segment 3:         {stats['segment3_time']:8.2f} s  ({100*stats['segment3_time']/stats['total_time']:5.1f}%)")
    print(f"  Utility calc:      {stats['utility_time']:8.2f} s  ({100*stats['utility_time']/stats['total_time']:5.1f}%)")
    print(f"  Damage agg:        {stats['damage_agg_time']:8.2f} s  ({100*stats['damage_agg_time']/stats['total_time']:5.1f}%)")
    print(f"  Climate:           {stats['climate_time']:8.2f} s  ({100*stats['climate_time']/stats['total_time']:5.1f}%)")
    print(f"  Finalize:          {stats['finalize_time']:8.2f} s  ({100*stats['finalize_time']/stats['total_time']:5.1f}%)")
    print(f"Avg time per call:   {stats['total_time']/stats['call_count']*1000:8.3f} ms")
    print(f"{'='*80}\n")


def calculate_tendencies(state, params,
                        omega_Fmin_Omega_base_ratio_prev, omega_yi_Omega_base_ratio_prev, omega_Fmax_Omega_base_ratio_prev,
                        Fmin_prev, Fmax_prev,
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
    params : dict
        Model parameters (all must be provided):
        - 'alpha': Output elasticity of capital
        - 'delta': Capital depreciation rate (yr^-1)
        - 's': Savings rate
        - 'psi1': Linear climate damage coefficient (°C⁻¹) [Barrage & Nordhaus 2023]
        - 'psi2': Quadratic climate damage coefficient (°C⁻²) [Barrage & Nordhaus 2023]
        - 'y_damage_distribution_exponent': Exponent for income-dependent damage distribution
        - 'y_net_reference': Reference income for power-law damage scaling ($/person)
        - 'k_climate': Temperature sensitivity (°C tCO2^-1)
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
        - 'fract_gdp': Fraction of GDP available for redistribution and abatement
        - 'f': Fraction allocated to abatement vs redistribution
    omega_Fmin_Omega_base_ratio_prev : float
        Ratio of climate damage (Region 1: F ≤ Fmin) to base damage from previous timestep.
        Units: dimensionless. Multiply by current Omega_base to get current damage fraction.
    omega_yi_Omega_base_ratio_prev : np.ndarray
        Ratio of climate damage fractions (Region 2: Fmin < F < Fmax) to base damage from
        previous timestep (length N_QUAD). Units: dimensionless. Multiply by current Omega_base
        to get current damage fractions. Grid is mapped to [Fmin_prev, Fmax_prev].
    omega_Fmax_Omega_base_ratio_prev : float
        Ratio of climate damage (Region 3: F ≥ Fmax) to base damage from previous timestep.
        Units: dimensionless. Multiply by current Omega_base to get current damage fraction.
    Fmin_prev : float
        Previous timestep's Fmin value (defines Region 1/2 boundary and damage grid mapping).
    Fmax_prev : float
        Previous timestep's Fmax value (defines Region 2/3 boundary and damage grid mapping).
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
        - Tendencies: 'dK_dt', 'dEcum_dt'
        - Climate damage ratios for next timestep: 'omega_yi_Omega_base_ratio', 'Omega_Omega_base_ratio'
        - All intermediate variables: Y_gross, delta_T, Omega, Y_net, y_net, redistribution,
          mu, Lambda, AbateCost, U, E

    Notes
    -----
    Calculation order follows equations 1.1-1.10, 2.1-2.2, 3.5, 4.3-4.4:
    1. Y_gross from K, L, A, α (Eq 1.1)
    2. ΔT from Ecum, k_climate (Eq 2.2)
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
    k_climate = params['k_climate']
    eta = params['eta']
    rho = params['rho']
    t = params['t']
    A = params['A'] # total factor productivity
    L = params['L'] # population
    sigma = params['sigma']
    theta1 = params['theta1']
    theta2 = params['theta2']
    mu_max = params['mu_max']
    fract_gdp = params['fract_gdp']
    gini = params['gini']
    use_empirical_lorenz = params['use_empirical_lorenz']
    f = params['f']
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

    # Policy switches
    income_dependent_aggregate_damage = params['income_dependent_aggregate_damage']
    income_dependent_damage_distribution = params['income_dependent_damage_distribution']
    income_dependent_tax_policy = params['income_dependent_tax_policy']
    income_redistribution = params['income_redistribution']
    income_dependent_redistribution_policy = params['income_dependent_redistribution_policy']



    # Transform xi into F space. Map [-1,1] to [0,1]
    Fi = (xi + 1.0)/2.0
    # compute edges in F space
    Fi_edges = (xi_edges + 1.0)/2.0
    # Transform quadrature weights to F space [0,1] (wi is for xi space [-1,1])
    Fwi = wi / 2.0

    #========================================================================================
    # Simplified damage calculation using aggregate damage from previous timestep
    # No iteration needed - uses temperature-based Omega_base with previous damage for budgeting

    # Initialize variables
    uniform_redistribution_amount = 0.0
    uniform_tax_rate = 0.0
    Fmin = 0.0
    Fmax = 1.0
    min_y_net = 0.0
    max_y_net = 0.0

    #========================================================================================
    # Main calculations

    # Eq 2.2: Temperature change from cumulative emissions
    delta_T = k_climate * Ecum

    # Base damage from temperature (capped just below 1.0 to avoid division by zero)
    # Be careful when used not to produce effective Omega values >= 1.0
    Omega_base = psi1 * delta_T + psi2 * (delta_T ** 2)

    if income_dependent_damage_distribution:
        # Reconstruct damage fractions from ratios stored at previous timestep
        # Multiply stored ratios by current Omega_base to get current damage estimates
        # Three-region structure: omega_Fmin, omega_yi, omega_Fmax
        Omega_calc = Omega_Omega_base_ratio_prev * Omega_base
        omega_Fmin_calc = omega_Fmin_Omega_base_ratio_prev * Omega_base
        omega_yi_calc = omega_yi_Omega_base_ratio_prev * Omega_base
        omega_Fmax_calc = omega_Fmax_Omega_base_ratio_prev * Omega_base
        # Clip scaled values to ensure they remain valid damage fractions
        Omega_calc = np.clip(Omega_calc, 0.0, 1.0 - EPSILON)
        omega_Fmin_calc = np.clip(omega_Fmin_calc, 0.0, 1.0 - EPSILON)
        omega_yi_calc = np.clip(omega_yi_calc, 0.0, 1.0 - EPSILON)
        omega_Fmax_calc = np.clip(omega_Fmax_calc, 0.0, 1.0 - EPSILON)
    elif income_dependent_aggregate_damage:
        # NEW CASE: Income-dependent calculation but uniform application
        # Use the aggregate damage ratio (from previous timestep) uniformly across all income groups
        Omega_calc = Omega_Omega_base_ratio_prev * Omega_base
        omega_Fmin_calc = Omega_calc
        omega_yi_calc = np.full_like(omega_yi_Omega_base_ratio_prev, Omega_calc)
        omega_Fmax_calc = Omega_calc
        Omega_calc = np.clip(Omega_calc, 0.0, 1.0 - EPSILON)
        omega_Fmin_calc = np.clip(omega_Fmin_calc, 0.0, 1.0 - EPSILON)
        omega_yi_calc = np.clip(omega_yi_calc, 0.0, 1.0 - EPSILON)
        omega_Fmax_calc = np.clip(omega_Fmax_calc, 0.0, 1.0 - EPSILON)
        # Keep y_damage_distribution_exponent for income-dependent calculation
    else:
        # No income-dependent damage at all, use temperature-based damage only
        Omega_calc = Omega_base
        omega_Fmin_calc = Omega_base
        omega_yi_calc = np.full_like(omega_yi_Omega_base_ratio_prev, Omega_base)
        omega_Fmax_calc = Omega_base
        y_damage_distribution_exponent = 0.0 # set damage exponent to zero if no income-dependent damage

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
        redistribution_amount = 0.0
        abateCost_amount = 0.0
        tax_amount = 0.0
        aggregate_utility = NEG_BIGNUM
        aggregate_damage_fraction = 0.0
        Omega = 0.0
        omega_yi = np.zeros_like(xi)
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
        # Initialize three-region damage variables for consistency
        omega_Fmin = 0.0
        omega_Fmax = 0.0
        omega_region2 = np.zeros_like(xi)
    else:
        # Economy exists - proceed with calculations

        available_for_redistribution_and_abatement = fract_gdp *  y_damaged_calc 
        
        if income_redistribution:
            redistribution_amount = (1 - f) * available_for_redistribution_and_abatement
        else:
            redistribution_amount = 0.0

        abateCost_amount = f * available_for_redistribution_and_abatement # per capita
        lambda_abate = abateCost_amount / y_damaged_calc # fraction of damaged production

        tax_amount = abateCost_amount + redistribution_amount # per capita
        # tax amount can be less than amount available if redistribution is turned off.

        # Eq 1.8 & 1.9: Net production per capita after abatement cost and climate damage
        # y_net (aggregate): gross production net of climate damage and abatement cost
        # Note: consumption + savings = y_net
        y_net = (1.0 - lambda_abate) * y_damaged_calc

        # Find uniform redistribution amount
        if income_redistribution and income_dependent_redistribution_policy:
            # No uniform distribution
            uniform_redistribution_amount = 0.0
        else:
            # Uniform redistribution
            uniform_redistribution_amount = redistribution_amount

        # Pre-compute uniform tax rate when policy is uniform so Fmin uses taxed income
        # Uniform tax always covers full budget (abatement + redistribution)
        # Fmin is calculated AFTER uniform tax to ensure equal consumption below Fmin
        if income_dependent_tax_policy:
            uniform_tax_rate = 0.0
        else:
            uniform_tax_rate = (abateCost_amount + redistribution_amount) / y_damaged_calc

        #------------------------------------------------------
        # Now we are going to do the income dependent part of the code
        # To simplify we are going to shift the calculation to discrete intervals of population
        # governed by the Gaussian Laegendre nodes and weights, xi and wi

        # Find Fmin using current Omega_base
        # For income-dependent redistribution, find Fmin such that redistribution matches target
        if income_redistribution and income_dependent_redistribution_policy:
            uniform_redistribution_amount = 0.0
            # Only find Fmin if there's actually something to redistribute
            if redistribution_amount > EPSILON:
                t_before_fmin = time.time()
                # Find Fmin: everyone pays uniform_tax_rate on Lorenz income,
                # then those below Fmin receive untaxed subsidy to reach Fmin income level
                # This ensures continuity at Fmin (no perverse incentives)
                Fmin = find_Fmin(1.0 - EPSILON,
                    y_gross, gini, Omega_calc,
                    omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
                    Fmin_prev, Fmax_prev, xi_edges,
                    redistribution_amount, uniform_tax_rate,
                    use_empirical_lorenz,
                    initial_guess=Fmin_prev,
                )
                _timing_stats['find_Fmin_time'] += time.time() - t_before_fmin
            else:
                # No redistribution amount, so no targeted redistribution
                Fmin = 0.0
        else:
            # Uniform redistribution
            uniform_redistribution_amount = redistribution_amount
            Fmin = 0.0

 
        # For income-dependent tax, find Fmax such that tax matches target
        if income_dependent_tax_policy:
            # Only find Fmax if there's actually something to tax
            if tax_amount > EPSILON:
                # We want to find the value of Fmax such that if everyone above Fmax made
                # the same amount of money as people at Fmax, that would generate the right amount of tax revenue.
                # The piece of resources that would come from the Lorenz curve above Fmax is:
                # lorenz_part = y_gross *( (1 - L(Fmax)) - (1.0 - Fmax) * (d L/dF)@Fmax )
                # damage_part = stepwise_integrate(Fmax, 1.0, omega_yi_prev, Fi_edges) - (1 - Fmax) * stepwise_interpolate(Fmax, omega_yi_prev, Fi_edges)
                # The challenge is to find Fmax such that:
                # tax_amount = lorenz_part - damage_part

                t_before_fmax = time.time()

                # Find Fmax based on post-damage Lorenz income only (no taxes, no redistributions)
                # This defines WHO pays progressive tax based on income rank thresholds
                # Note: find_Fmax always uses uniform_redistribution=0.0 because
                # Fmax determines "who pays" not "how much" (that's determined by tax rates)
                Fmax = find_Fmax(Fmin,
                    y_gross, gini, Omega_calc,
                    omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
                    Fmin_prev, Fmax_prev, xi_edges,
                    redistribution_amount, abateCost_amount,
                    use_empirical_lorenz,
                    initial_guess=Fmax_prev,
                )
                _timing_stats['find_Fmax_time'] += time.time() - t_before_fmax
            else:
                # No tax amount, so no income-dependent taxation
                Fmax = 1.0
        else:
            # Uniform tax already computed
            Fmax = 1.0

        # Remap quadrature points to Region 2: [Fmin, Fmax]
        if Fmax - Fmin > EPSILON:
            Fi_region2, Fwi_region2 = remap_quadrature_to_interval(xi, wi, Fmin, Fmax)
        else:
            # Degenerate case: Fmin ≈ Fmax, no middle region
            Fi_region2 = np.array([])
            Fwi_region2 = np.array([])

        # Compute consumption, aggregate utility for the Fmin and Fmax region, and at each of the Gauss-Legendre quadrature nodes
        # Also calculate climate damage for next time step.
        # Divide calculation into three segments: [0, Fmin], [Fmin, Fmax], [Fmax, 1]
        #
        # Summary of y variants (income flow):
        # 1. y_gross: gross production before climate damage and abatement cost
        # 2. y_damaged_calc: gross production net of climate damage (using previous timestep's damage)
        # 3. y_net (aggregate): gross production net of climate damage and abatement cost
        # 4. y_damaged_and_uniform: gross production net of climate damage, uniform distributions, and uniform taxes
        #                           (intermediate concept, calculated via y_of_F_after_damage for middle segment)
        # 5. y_net_yi (individual): individual income net of climate damage, abatement cost,
        #                           uniform distributions, uniform taxes, and income-dependent
        #                           redistributions and taxes
        # Note: consumption + savings = y_net
        # Fraction of income in each bin
        if use_empirical_lorenz:
            lorenz_fractions_yi = L_empirical_lorenz(Fi_edges[1:], gini) - \
                                  L_empirical_lorenz(Fi_edges[:-1], gini)
        else:
            lorenz_fractions_yi = L_pareto(Fi_edges[1:], gini) - L_pareto(Fi_edges[:-1], gini)
        lorenz_ratio_yi = lorenz_fractions_yi/Fwi # ratio of mean income in each bin to aggregate mean income

        y_net_yi = np.zeros_like(xi)
        consumption_yi = np.zeros_like(xi)
        utility_yi = np.zeros_like(xi)
        omega_yi = np.zeros_like(xi)

        # ============================================================================
        # Three-Region Income Distribution and Utility Calculation
        # ============================================================================
        # Region 1: [0, Fmin] - constant income at min_y_net
        # Region 2: [Fmin, Fmax] - quadrature points with varying income
        # Region 3: [Fmax, 1] - constant income at max_y_net

        # Initialize damage variables (will be set in Region calculations below)
        # Use consistent naming with omega_Fmin_calc, omega_Fmax_calc
        omega_Fmin = 0.0
        omega_Fmax = 0.0
        omega_region2 = np.zeros(len(xi))

        aggregate_utility = 0.0

        # ------------------------------------------------------------------------
        # Region 1: [0, Fmin] - Constant income
        # ------------------------------------------------------------------------
        t_before_seg1 = time.time()
        _timing_stats['policy_calc_time'] += t_before_seg1 - t_setup

        if Fmin > EPSILON:
            # People below Fmin pay uniform tax on Lorenz income, then receive untaxed subsidy
            # The subsidy lifts them to the income level at Fmin
            min_y_net = y_net_of_F(
                Fmin, Fmin, Fmax,
                y_gross,
                omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
                Fmin_prev, Fmax_prev, xi_edges,
                uniform_tax_rate, uniform_redistribution_amount, gini,
                use_empirical_lorenz,
            )
            min_consumption = min_y_net * (1 - s)

            # Utility at this income level
            if eta == 1:
                min_utility = np.log(np.maximum(min_consumption, EPSILON))
            else:
                min_utility = (np.maximum(min_consumption, EPSILON) ** (1 - eta)) / (1 - eta)

            # Damage at this income level (for next timestep)
            omega_Fmin = Omega_base * (min_y_net  / y_net_reference) ** (-y_damage_distribution_exponent)
            omega_Fmin = np.clip(omega_Fmin, 0.0, 1.0 - EPSILON)

            # Contribution to aggregate utility (analytical: constant × width)
            aggregate_utility += Fmin * min_utility
        else:
            min_y_net = 0.0
            omega_Fmin = 0.0
            min_utility = NEG_BIGNUM

        _timing_stats['segment1_time'] += time.time() - t_before_seg1

        # ------------------------------------------------------------------------
        # Region 3: [Fmax, 1] - Constant income
        # ------------------------------------------------------------------------
        t_before_seg3 = time.time()
        if 1.0 - Fmax > EPSILON:
            # People above Fmax pay progressive tax on income above Fmax level
            max_y_net = y_net_of_F(
                Fmax, Fmin, Fmax,
                y_gross,
                omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
                Fmin_prev, Fmax_prev, xi_edges,
                uniform_tax_rate, uniform_redistribution_amount, gini,
                use_empirical_lorenz,
            )
            max_consumption = max_y_net * (1 - s)

            # Utility at this income level
            if eta == 1:
                max_utility = np.log(np.maximum(max_consumption, EPSILON))
            else:
                max_utility = (np.maximum(max_consumption, EPSILON) ** (1 - eta)) / (1 - eta)

            # Damage at this income level (for next timestep)
            omega_Fmax = Omega_base * (max_y_net / y_net_reference) ** (-y_damage_distribution_exponent)
            omega_Fmax = np.clip(omega_Fmax, 0.0, 1.0 - EPSILON)

            # Contribution to aggregate utility (analytical: constant × width)
            aggregate_utility += (1.0 - Fmax) * max_utility
        else:
            max_y_net = 0.0
            omega_Fmax = 0.0
            max_utility = NEG_BIGNUM

        _timing_stats['segment3_time'] += time.time() - t_before_seg3

        # ------------------------------------------------------------------------
        # Region 2: [Fmin, Fmax] - Quadrature integration
        # ------------------------------------------------------------------------
        t_before_seg2 = time.time()
        if Fmax - Fmin > EPSILON and len(Fi_region2) > 0:
            # Evaluate income at each remapped quadrature point
            y_net_region2 = y_net_of_F(
                Fi_region2, Fmin, Fmax,
                y_gross,
                omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
                Fmin_prev, Fmax_prev, xi_edges,
                uniform_tax_rate, uniform_redistribution_amount, gini,
                use_empirical_lorenz,
            )

            # Compute consumption and utility
            consumption_region2 = y_net_region2 * (1 - s) 
            if eta == 1:
                utility_region2 = np.log(np.maximum(consumption_region2, EPSILON))
            else:
                utility_region2 = (np.maximum(consumption_region2, EPSILON) ** (1 - eta)) / (1 - eta)

            # Compute damage for next timestep
            if np.abs(y_damage_distribution_exponent) < EPSILON:
                omega_region2 = np.full_like(y_net_region2, Omega_base)
            else:
                omega_region2 = Omega_base * (y_net_region2 / y_net_reference) ** (-y_damage_distribution_exponent)
            omega_region2 = np.clip(omega_region2, 0.0, 1.0 - EPSILON)

            # Contribution to aggregate utility (quadrature sum)
            aggregate_utility += np.sum(Fwi_region2 * utility_region2)
        else:
            # No Region 2 (Fmin ≈ Fmax), use constant damage equal to value at Fmin
            omega_region2 = np.ones(len(xi)) * omega_Fmin

        _timing_stats['segment2_time'] += time.time() - t_before_seg2

        # ------------------------------------------------------------------------
        # Assign Region 2 arrays directly (no mapping back to [0,1] grid)
        # ------------------------------------------------------------------------
        t_before_damage_agg = time.time()
        # omega_yi, y_net_yi, utility_yi represent ONLY Region 2 values
        # (always length n_quad, mapped to [Fmin, Fmax] interval for this timestep)
        # The three-region structure is used directly for aggregate calculations

        if len(Fi_region2) > 0:
            y_net_yi = y_net_region2
            omega_yi = omega_region2
            utility_yi = utility_region2
        else:
            # Degenerate case: no Region 2 (Fmin ≈ Fmax)
            y_net_yi = np.ones(len(xi)) * min_y_net
            omega_yi = np.ones(len(xi)) * omega_Fmin
            utility_yi = np.ones(len(xi)) * min_utility

        # ------------------------------------------------------------------------
        # Compute aggregate damage using three-region structure
        # ------------------------------------------------------------------------
        # Region contributions to income and damage
        y_net_Fmin = min_y_net if Fmin > EPSILON else 0.0
        y_net_Fmax = max_y_net if 1.0 - Fmax > EPSILON else 0.0

        income_region1 = Fmin * y_net_Fmin
        income_region2 = np.sum(Fwi_region2 * y_net_region2) if len(Fwi_region2) > 0 else 0.0
        income_region3 = (1.0 - Fmax) * y_net_Fmax

        damage_region1 = Fmin * y_net_Fmin * omega_Fmin
        damage_region2 = np.sum(Fwi_region2 * y_net_region2 * omega_region2) if len(Fwi_region2) > 0 else 0.0
        damage_region3 = (1.0 - Fmax) * y_net_Fmax * omega_Fmax

        total_income = income_region1 + income_region2 + income_region3
        total_damage = damage_region1 + damage_region2 + damage_region3

        if income_dependent_aggregate_damage and not income_dependent_damage_distribution:
            # Calculate GDP-weighted mean damage across all three regions and apply uniformly
            if total_income > EPSILON:
                omega_mean = total_damage / total_income
                # Apply uniformly to all three regions
                omega_Fmin = omega_mean
                omega_region2[:] = omega_mean
                omega_Fmax = omega_mean
                omega_yi = omega_region2  # Update omega_yi to reflect the uniform damage
                # Recalculate damage with uniform omega
                damage_region1 = income_region1 * omega_mean
                damage_region2 = income_region2 * omega_mean
                damage_region3 = income_region3 * omega_mean
                total_damage = damage_region1 + damage_region2 + damage_region3
            else:
                omega_Fmin = 0.0
                omega_region2[:] = 0.0
                omega_Fmax = 0.0
                omega_yi = omega_region2
                total_damage = 0.0

        elif not income_dependent_aggregate_damage and income_dependent_damage_distribution:
            # Rescale damage distribution to match Omega_base
            if total_damage > EPSILON and total_income > EPSILON:
                target_total_damage = Omega_base * total_income
                scale_factor = target_total_damage / total_damage
                # Apply scaling to all three regions
                omega_Fmin *= scale_factor
                omega_region2 *= scale_factor
                omega_Fmax *= scale_factor
                omega_yi = omega_region2  # Update omega_yi
                # Recalculate damage after scaling
                damage_region1 = income_region1 * omega_Fmin
                damage_region2 = np.sum(Fwi_region2 * y_net_region2 * omega_region2) if len(Fwi_region2) > 0 else 0.0
                damage_region3 = income_region3 * omega_Fmax
                total_damage = damage_region1 + damage_region2 + damage_region3

        # Clip damage fractions to valid range [0.0, 1.0 - EPSILON]
        omega_Fmin = np.clip(omega_Fmin, 0.0, 1.0 - EPSILON)
        omega_region2 = np.clip(omega_region2, 0.0, 1.0 - EPSILON)
        omega_Fmax = np.clip(omega_Fmax, 0.0, 1.0 - EPSILON)
        omega_yi = omega_region2  # Keep omega_yi in sync

        # Calculate final aggregate damage fraction
        if total_income > EPSILON:
            Omega = total_damage / total_income
        else:
            Omega = 0.0

        _timing_stats['damage_agg_time'] += time.time() - t_before_damage_agg

    #========================================================================================
    # mitigation carbon climate
        t_before_climate = time.time()

        # Land emissions per capita (Eland is total land emissions from time function)
        eland = Eland / L

        # Eq 2.1: Potential emissions per capita (unabated), including land emissions
        epot = sigma * y_gross * emission_ratio + eland

        # Eq 1.6: Abatement fraction (compute uncapped value first)
        if epot > 0 and abateCost_amount > 0:
            mu = min(mu_max, (abateCost_amount * theta2 / (epot * theta1)) ** (1 / theta2))
        else:
            mu = 0.0

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

        # Extra consumption pulse (for sensitivity analysis)
        if amount_extra_consumption == 0.0:
            consumption_extra = 0.0
            pulse_utility_scaling = 1.0
        elif year_extra_consumption >= t and year_extra_consumption < t + dt:
            consumption_extra = amount_extra_consumption / (L * dt)
            fractional_extra_consumption = 1.0 + consumption_extra / (y_net *(1 - s) + EPSILON)
            pulse_utility_scaling = fractional_extra_consumption ** (1 - eta) if eta != 1 else L * np.log(fractional_extra_consumption)
        else:
            consumption_extra = 0.0
            pulse_utility_scaling = 1.0
        # fractional extra consumption is the extra consumption as a fraction of normal consumption
        # this is 1.0 unless we are adding a pulse of consumption to numerically calculate a partial derivative
        


        # aggregate utility
        U = aggregate_utility * pulse_utility_scaling 
        # pulse utility scaling is the adjustment to utility from the extra consumption pulse (if any)

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
            'redistribution_amount': redistribution_amount,
            'tax_amount': tax_amount,
            'uniform_redistribution_amount': uniform_redistribution_amount,
            'uniform_tax_rate': uniform_tax_rate,
            'Fmin': Fmin,
            'Fmax': Fmax,
            'min_y_net': min_y_net,
            'max_y_net': max_y_net,
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
            's': s,
            'dK_dt': dK_dt,
            # Mu_up cap diagnostics
            'mu_max': mu_max,
            'emission_ratio': emission_ratio,
            'E_extra': E_extra,
            'consumption_extra': consumption_extra,
        })

    # Always return minimal variables needed for optimization
    results.update({
        'U': U,
        'mu': mu,
        'dK_dt': dK_dt,
        'dEcum_dt': e * L + E_extra,
    })

    # Always return climate damage ratios for use in next time step
    # Store ratios of damage to base damage, so next timestep can scale by new Omega_base
    # Three-region structure: omega_Fmin (Region 1), omega_region2 (Region 2), omega_Fmax (Region 3)
    if Omega_base > EPSILON:
        results['omega_Fmin_Omega_base_ratio'] = omega_Fmin / Omega_base
        results['omega_yi_Omega_base_ratio'] = omega_region2 / Omega_base
        results['omega_Fmax_Omega_base_ratio'] = omega_Fmax / Omega_base
        results['Omega_Omega_base_ratio'] = Omega / Omega_base
    else:
        results['omega_Fmin_Omega_base_ratio'] = 0.0
        results['omega_yi_Omega_base_ratio'] = np.zeros(len(xi))  # FIXED length n_quad
        results['omega_Fmax_Omega_base_ratio'] = 0.0
        results['Omega_Omega_base_ratio'] = 0.0
    # Store region boundaries (defines the mapping for omega_yi to F space)
    results['Fmax'] = Fmax
    results['Fmin'] = Fmin

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

    # Extract scalar params for mu rate-of-change constraint
    sp = config.scalar_params

    # Create time array
    t_array = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_array)

    # Precompute Gauss-Legendre quadrature nodes and weights (used for all timesteps)
    xi, wi = roots_legendre(n_quad)
    # Create xi_edges: cumulative weights starting at -1, ending at +1
    # wi sums to 2 (integrating over [-1,1]), so cumsum(wi) goes from wi[0] to 2
    # We want edges from -1 to +1, so: -1 + cumsum(wi) goes from -1+wi[0] to 1
    xi_edges = np.concatenate(([-1.0], -1.0 + np.cumsum(wi)))  # length n_quad + 1

    # Initialize climate damage ratios for first timestep
    # Three-region structure: uniform damage across all regions initially
    omega_Fmin_Omega_base_ratio_prev = 1.0  # Ratio of damage to base damage for F <= Fmin (Region 1)
    omega_yi_Omega_base_ratio_prev = np.ones(n_quad)  # Ratio of damage to base damage at quadrature points (Region 2) - FIXED length n_quad
    omega_Fmax_Omega_base_ratio_prev = 1.0  # Ratio of damage to base damage for F >= Fmax (Region 3)
    Omega_Omega_base_ratio_prev = 1.0  # Ratio of aggregate damage to base damage

    # Initialize previous region boundaries (full range [0,1] for first timestep)
    Fmin_prev = 0.0
    Fmax_prev = 1.0

    # take abatement cost and initial climate damage into account for initial capital
    state = {
        'K': config.scalar_params.K_initial,
        'Ecum': config.scalar_params.Ecum_initial,
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
            'redistribution_amount': np.zeros(n_steps),
            'uniform_redistribution_amount': np.zeros(n_steps),
            'tax_amount': np.zeros(n_steps),
            'uniform_tax_rate': np.zeros(n_steps),
            'Fmin': np.zeros(n_steps),
            'Fmax': np.zeros(n_steps),
            'aggregate_utility': np.zeros(n_steps),
            'mu': np.zeros(n_steps),
            'lambda_abate': np.zeros(n_steps),
            'abateCost_amount': np.zeros(n_steps),
            'y_net': np.zeros(n_steps),
            'e': np.zeros(n_steps),
            'dK_dt': np.zeros(n_steps),
            's': np.zeros(n_steps),
            'min_y_net': np.zeros(n_steps),
            'max_y_net': np.zeros(n_steps),
            'y_net_yi': np.zeros((n_steps, n_quad)),
            'omega_yi': np.zeros((n_steps, n_quad)),
            'utility_yi': np.zeros((n_steps, n_quad)),
            'emission_ratio': np.zeros(n_steps),
            'E_extra': np.zeros(n_steps),
            'consumption_extra': np.zeros(n_steps),
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

    # Initialize mu ceiling for rate-of-change constraint
    mu_ceiling = sp.mu_max_initial

    # Time stepping loop
    for i, t in enumerate(t_array):
        # Evaluate time-dependent parameters at current time
        params = evaluate_params_at_time(t, config)

        # Rate-of-change constraint on μ: ceiling starts at mu_max_initial, grows by max_mu_increase_rate per year
        params['mu_max'] = mu_ceiling

        # Pass dt for extra emission pulse timing
        params['dt'] = dt

        if store_detailed_output:
            results['params_list'].append(params)

        # Calculate all variables and tendencies at current time
        # Pass three-region damage ratios from previous timestep to use lagged damage (avoids circular dependency)
        # Pass Fmax_prev and Fmin_prev for damage lookup transformation
        outputs = calculate_tendencies(state, params,
                                      omega_Fmin_Omega_base_ratio_prev, omega_yi_Omega_base_ratio_prev, omega_Fmax_Omega_base_ratio_prev,
                                      Fmin_prev, Fmax_prev,
                                      Omega_Omega_base_ratio_prev,
                                      xi, xi_edges, wi, store_detailed_output)

        # Update mu ceiling for next timestep: grows from actual mu by max_mu_increase_rate
        mu_ceiling = outputs['mu'] + sp.max_mu_increase_rate * dt

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

            # Store redistribution/tax variables
            results['redistribution_amount'][i] = outputs['redistribution_amount']
            results['tax_amount'][i] = outputs['tax_amount']
            results['uniform_redistribution_amount'][i] = outputs['uniform_redistribution_amount']
            results['uniform_tax_rate'][i] = outputs['uniform_tax_rate']

            # Store climate variables
            results['delta_T'][i] = outputs['delta_T']
            results['Omega'][i] = outputs['Omega']
            results['Omega_base'][i] = outputs['Omega_base']
            results['Omega_calc'][i] = outputs['Omega_calc']

            # Store policy boundaries
            results['Fmin'][i] = outputs['Fmin']
            results['Fmax'][i] = outputs['Fmax']
            results['min_y_net'][i] = outputs['min_y_net']
            results['max_y_net'][i] = outputs['max_y_net']

            # Store scalars
            results['Gini'][i] = outputs['Gini']
            results['gini'][i] = outputs['gini']
            results['aggregate_utility'][i] = outputs['aggregate_utility']
            results['s'][i] = outputs['s']

            # Store distributions
            results['y_net_yi'][i, :] = outputs['y_net_yi']
            results['omega_yi'][i, :] = outputs['omega_yi']
            results['utility_yi'][i, :] = outputs['utility_yi']

            # Store tendencies
            results['dK_dt'][i] = outputs['dK_dt']

            results['emission_ratio'][i] = outputs['emission_ratio']
            results['E_extra'][i] = outputs['E_extra']
            results['consumption_extra'][i] = outputs['consumption_extra']

        # Euler step: update state for next iteration (skip on last step)
        if i < n_steps - 1:
            state['K'] = state['K'] + dt * outputs['dK_dt']
            # do not allow cumulative emissions to go negative, making it colder than the initial condition
            state['Ecum'] = max(0.0, state['Ecum'] + dt * outputs['dEcum_dt'])

            # Update three-region damage ratios for next time step (lagged damage approach)
            omega_Fmin_Omega_base_ratio_prev = outputs['omega_Fmin_Omega_base_ratio']
            omega_yi_Omega_base_ratio_prev = outputs['omega_yi_Omega_base_ratio']  # FIXED length n_quad
            omega_Fmax_Omega_base_ratio_prev = outputs['omega_Fmax_Omega_base_ratio']
            Omega_Omega_base_ratio_prev = outputs['Omega_Omega_base_ratio']

            # Update region boundaries for next time step (defines mapping for damage lookup)
            Fmin_prev = outputs['Fmin']
            Fmax_prev = outputs['Fmax']

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
        - Discount rate: r_consumption (annual effective discount rate on consumption)

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
    redistribution_amount = results['redistribution_amount']

    # Compute total (uppercase) variables from per-capita via vectorized multiplication
    Y_gross = y_gross * L
    Y_damaged = y_damaged * L
    Y_net = y_net * L
    AbateCost = abateCost_amount * L
    E = e * L
    Climate_damage = climate_damage * L
    Redistribution_amount = redistribution_amount * L

    # Compute consumption and savings variables
    Consumption = (1.0 - s_array) * Y_net
    consumption = (1.0 - s_array) * y_net
    Savings = s_array * Y_net
    savings = s_array * y_net

    # Compute marginal abatement cost
    marginal_abatement_cost = theta1_array * (mu ** (theta2_array - 1.0))

    # ---------- Welfare-adjusted MAC (wMAC) for SCC consistency ----------
    # SCC "$ pulse" in calculate_tendencies scales everyone's consumption by the same factor,
    # which corresponds to a consumption-proportional dollar numeraire.
    # We compute MU per $ for that numeraire, and MU per $ for abatement cost incidence.

    wi = results['wi']
    Fmin = results['Fmin']
    Fmax = results['Fmax']
    min_y_net = results['min_y_net']
    max_y_net = results['max_y_net']
    y_net_yi = results['y_net_yi']

    income_dep_tax = np.array([p['income_dependent_tax_policy'] for p in params_list], dtype=bool)

    # Per-capita consumption by region
    c_min = np.maximum((1.0 - s_array) * min_y_net, EPSILON)          # shape (n_steps,)
    c_max = np.maximum((1.0 - s_array) * max_y_net, EPSILON)          # shape (n_steps,)
    c_region2 = np.maximum((1.0 - s_array)[:, None] * y_net_yi, EPSILON)  # shape (n_steps, n_quad)

    # Remap Gauss-Legendre weights to [Fmin, Fmax] each timestep:
    # Fwi_region2 = wi * (Fmax - Fmin) / 2
    interval = np.clip(Fmax - Fmin, 0.0, 1.0)
    Fwi_region2 = wi[None, :] * interval[:, None] / 2.0

    # Total consumption per capita: C = integral c(F) dF
    C = (
        Fmin * c_min +
        np.sum(Fwi_region2 * c_region2, axis=1) +
        (1.0 - Fmax) * c_max
    )
    C = np.maximum(C, EPSILON)

    # MU per $ for consumption-proportional dollar numeraire:
    # mu_dollar = (integral u'(c)*c dF) / (integral c dF) = (integral c^(1-eta) dF) / C
    N = (
        Fmin * (c_min ** (1.0 - eta_array)) +
        np.sum(Fwi_region2 * (c_region2 ** (1.0 - eta_array[:, None])), axis=1) +
        (1.0 - Fmax) * (c_max ** (1.0 - eta_array))
    )
    mu_dollar_numeraire = N / C

    # MU per $ for abatement cost incidence:
    # Increasing abatement cost by $1 per capita reduces income by $1, which splits into:
    #   (1-s) less consumption  +  s less savings (capital accumulation)
    # Under progressive tax, the consumption reduction falls on the rich (u'(c_max)),
    # while the savings reduction affects future welfare via capital — the same channel
    # as the consumption-proportional numeraire. Hence:
    #   mu_dollar_cost = (1-s) * c_max^(-eta) + s * mu_dollar_numeraire
    # Under uniform tax, the cost incidence matches the numeraire, so mu_dollar_cost = mu_dollar_numeraire.
    mu_dollar_cost = np.where(
        income_dep_tax & ((1.0 - Fmax) > 1e-12),
        (1.0 - s_array) * c_max ** (-eta_array) + s_array * mu_dollar_numeraire,
        mu_dollar_numeraire
    )

    # Welfare-adjusted MAC: this is what should match SCC at an unconstrained interior optimum
    welfare_adjusted_mac = marginal_abatement_cost * (mu_dollar_cost / mu_dollar_numeraire)

    # Diagnostic ratio (should be ~3-4x in progressive tax runs)
    mac_over_wmac = marginal_abatement_cost / np.maximum(welfare_adjusted_mac, EPSILON)
    # --------------------------------------------------------------------

    # Create aliases
    Lambda = lambda_abate
    redistribution = redistribution_amount

    # Add dEcum_dt as alias to E for consistency
    dEcum_dt = E

    # Compute annual effective discount rate on consumption
    # r_consumption(t) = exp(rho * dt) * (consumption(t+dt)/consumption(t))^eta - 1.0
    t_array = results['t']
    dt = t_array[1] - t_array[0] if n_steps > 1 else 1.0
    r_consumption = np.zeros(n_steps)

    for i in range(n_steps - 1):
        if consumption[i] > EPSILON:
            consumption_ratio = consumption[i + 1] / consumption[i]
            r_consumption[i] = np.exp(rho * dt) * (consumption_ratio ** eta_array[i]) - 1.0
        else:
            r_consumption[i] = 0.0

    # For the last time step, use the consumption ratio from the previous time step
    if n_steps > 1:
        if consumption[-2] > EPSILON:
            consumption_ratio_last = consumption[-1] / consumption[-2]
            r_consumption[-1] = np.exp(rho * dt) * (consumption_ratio_last ** eta_array[-1]) - 1.0
        else:
            r_consumption[-1] = 0.0
    else:
        r_consumption[0] = 0.0

    # Add all derived variables to results dict
    results.update({
        'Y_gross': Y_gross,
        'Y_damaged': Y_damaged,
        'Y_net': Y_net,
        'AbateCost': AbateCost,
        'E': E,
        'Climate_damage': Climate_damage,
        'Redistribution_amount': Redistribution_amount,
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
        'redistribution': redistribution,
        'dEcum_dt': dEcum_dt,
        'r_consumption': r_consumption,
    })

    return results
