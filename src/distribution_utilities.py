"""
Income distribution functions and utility integration over Lorenz distributions.

This module provides:
- Pareto-Lorenz and Empirical income distribution calculations
- Progressive taxation and targeted redistribution
- CRRA utility integration over income distributions
- Climate damage integration over income distributions
- Stepwise interpolation and integration utilities for quadrature
"""

import math
import numpy as np
from scipy.optimize import root_scalar, fsolve, newton
from src.constants import (
    EPSILON,
    MAX_ITERATIONS,
    EMPIRICAL_LORENZ_P0,
    EMPIRICAL_LORENZ_P1,
    EMPIRICAL_LORENZ_P2,
    EMPIRICAL_LORENZ_P3,
    EMPIRICAL_LORENZ_W0,
    EMPIRICAL_LORENZ_W1,
    EMPIRICAL_LORENZ_W2,
    EMPIRICAL_LORENZ_W3,
    EMPIRICAL_LORENZ_BASE_GINI,
)


#========================================================================================
# Pareto-Lorenz distribution functions
#========================================================================================

def a_from_G(G):
    """Pareto index a from Gini coefficient G."""
    if not (0 < G < 1):
        raise ValueError("G must be in (0,1).")
    return (1.0 + 1.0/G) / 2.0


def L_pareto(F, G):
    """Lorenz curve at F for Pareto-Lorenz distribution with Gini coefficient G."""
    a = a_from_G(G)
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0

    # Clip F to avoid numerical issues when F is very close to 1.0
    # When F = 1.0, L should equal 1.0 (everyone has accumulated all income)
    F_clipped = np.clip(F_arr, 0.0, 1.0 - EPSILON)
    exponent = 1.0 - 1.0/a
    result = 1.0 - (1.0 - F_clipped)**exponent

    return result[()] if scalar_input else result


def L_pareto_derivative(F, G):
    """Derivative of Lorenz curve dL/dF at F for Pareto-Lorenz distribution with Gini coefficient G."""
    a = a_from_G(G)
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0

    F_clipped = np.clip(F_arr, EPSILON, 1.0 - EPSILON)
    result = (1.0 - 1.0/a) * (1.0 - F_clipped)**(-1.0/a)

    return result[()] if scalar_input else result


def L_pareto_and_derivative(F, G):
    """
    Compute both Lorenz curve L(F) and its derivative dL/dF(F) for Pareto-Lorenz distribution.

    This is more efficient than calling L_pareto and L_pareto_derivative separately,
    as it avoids redundant computation of a_from_G(G) and (1-F) terms.

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1]
    G : float
        Gini coefficient

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        (L, dL_dF) - Lorenz curve value and its derivative at F
    """
    a = a_from_G(G)
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0

    # Clip F for L computation
    F_clipped_L = np.clip(F_arr, 0.0, 1.0 - EPSILON)
    one_minus_F_L = 1.0 - F_clipped_L
    exponent = 1.0 - 1.0/a
    L_result = 1.0 - one_minus_F_L**exponent

    # Clip F for dL/dF computation
    F_clipped_dL = np.clip(F_arr, EPSILON, 1.0 - EPSILON)
    one_minus_F_dL = 1.0 - F_clipped_dL
    dL_result = exponent * one_minus_F_dL**(-1.0/a)

    if scalar_input:
        return L_result[()], dL_result[()]
    else:
        return L_result, dL_result


def L_empirical_lorenz_base(F):
    """
    Base empirical Lorenz curve at F.

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1]

    Returns
    -------
    float or ndarray
        Base Lorenz curve value: L_base(F) = w₀·F^p₀ + w₁·F^p₁ + w₂·F^p₂ + w₃·F^p₃
        where w₀ = 1 - w₁ - w₂ - w₃
    """
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0
    F_clipped = np.clip(F_arr, 0.0, 1.0)
    result = (EMPIRICAL_LORENZ_W0 * (F_clipped ** EMPIRICAL_LORENZ_P0) +
              EMPIRICAL_LORENZ_W1 * (F_clipped ** EMPIRICAL_LORENZ_P1) +
              EMPIRICAL_LORENZ_W2 * (F_clipped ** EMPIRICAL_LORENZ_P2) +
              EMPIRICAL_LORENZ_W3 * (F_clipped ** EMPIRICAL_LORENZ_P3))
    return result[()] if scalar_input else result


def L_empirical_lorenz_base_derivative(F):
    """
    Derivative of base empirical Lorenz curve dL_base/dF at F.

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1]

    Returns
    -------
    float or ndarray
        Derivative: dL_base/dF = w₀·p₀·F^(p₀-1) + w₁·p₁·F^(p₁-1) + w₂·p₂·F^(p₂-1) + w₃·p₃·F^(p₃-1)
        where w₀ = 1 - w₁ - w₂ - w₃
    """
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0
    F_clipped = np.clip(F_arr, EPSILON, 1.0)
    result = (EMPIRICAL_LORENZ_W0 * EMPIRICAL_LORENZ_P0 * (F_clipped ** (EMPIRICAL_LORENZ_P0 - 1.0)) +
              EMPIRICAL_LORENZ_W1 * EMPIRICAL_LORENZ_P1 * (F_clipped ** (EMPIRICAL_LORENZ_P1 - 1.0)) +
              EMPIRICAL_LORENZ_W2 * EMPIRICAL_LORENZ_P2 * (F_clipped ** (EMPIRICAL_LORENZ_P2 - 1.0)) +
              EMPIRICAL_LORENZ_W3 * EMPIRICAL_LORENZ_P3 * (F_clipped ** (EMPIRICAL_LORENZ_P3 - 1.0)))
    return result[()] if scalar_input else result


def L_empirical_lorenz(F, G):
    """
    Empirical Lorenz curve at F for Gini coefficient G.

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1]
    G : float
        Gini coefficient

    Returns
    -------
    float or ndarray
        Lorenz curve value: L(F) = (1 - G/Gini_base)·F + (G/Gini_base)·L_base(F)

    Notes
    -----
    Uses linear interpolation between perfect equality (L=F) and the base empirical curve.
    """
    alpha = G / EMPIRICAL_LORENZ_BASE_GINI
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0
    L_base = L_empirical_lorenz_base(F_arr)
    result = (1.0 - alpha) * F_arr + alpha * L_base
    return result[()] if scalar_input else result


def L_empirical_lorenz_derivative(F, G):
    """
    Derivative of empirical Lorenz curve dL/dF at F for Gini coefficient G.

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1]
    G : float
        Gini coefficient

    Returns
    -------
    float or ndarray
        Derivative: dL/dF = (1 - G/Gini_base) + (G/Gini_base)·dL_base/dF

    Notes
    -----
    Derivative of the linear interpolation between perfect equality and the base curve.
    """
    alpha = G / EMPIRICAL_LORENZ_BASE_GINI
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0
    dL_base_dF = L_empirical_lorenz_base_derivative(F_arr)
    result = (1.0 - alpha) + alpha * dL_base_dF
    return result[()] if scalar_input else result


def L_empirical_lorenz_and_derivative(F, G):
    """
    Compute both empirical Lorenz curve L(F) and its derivative dL/dF(F).

    This is more efficient than calling L_empirical_lorenz and L_empirical_lorenz_derivative
    separately, as it avoids redundant computation of alpha and F array processing.

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1]
    G : float
        Gini coefficient

    Returns
    -------
    tuple of (float or ndarray, float or ndarray)
        (L, dL_dF) - Lorenz curve value and its derivative at F
    """
    alpha = G / EMPIRICAL_LORENZ_BASE_GINI
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0

    # Compute base Lorenz curve and its derivative together
    L_base = L_empirical_lorenz_base(F_arr)
    dL_base_dF = L_empirical_lorenz_base_derivative(F_arr)

    # Apply linear interpolation to both
    L_result = (1.0 - alpha) * F_arr + alpha * L_base
    dL_result = (1.0 - alpha) + alpha * dL_base_dF

    if scalar_input:
        return L_result[()], dL_result[()]
    else:
        return L_result, dL_result


def _phi(r):
    """Helper for bracketing cap; φ(r) = (r-1) r^{1/(r-1)-1}."""
    if r <= 0:
        return float("-inf")
    if abs(r - 1.0) < EPSILON:
        return 0.0
    sgn = 1.0 if r > 1.0 else -1.0
    log_abs = math.log(abs(r - 1.0)) + (1.0/(r - 1.0) - 1.0) * math.log(r)
    return sgn * math.exp(log_abs)


#========================================================================================
# Income at rank F with damage and redistribution
#========================================================================================

# Global call counter for diagnostics
_first_call_diagnostics_printed = False
_call_counter = 0


def y_net_of_F(F, Fmin, Fmax, y_gross,
               omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
               Fmin_prev, Fmax_prev, xi_edges,
               uniform_tax_rate, uniform_redistribution, gini,
               use_empirical_lorenz):
    """
    Compute net income y_net(F) at population rank F after accounting for damage, tax, and redistribution.

    Formula:
        y_net(F) = y_gross * dL/dF(F; gini) * (1.0 - omega_prev_F) * (1.0 - uniform_tax_rate) + uniform_redistribution

        Order: Lorenz → Damage → Tax → Redistribution (untaxed)

    where:
        omega_prev_F = lookup damage at F using PREVIOUS timestep's three-region structure
        dL/dF(F) = Lorenz derivative (Pareto or Empirical formulation)

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1].
    Fmin : float
        Minimum population rank for clipping F to [Fmin, Fmax].
    Fmax : float
        Maximum population rank for clipping F to [Fmin, Fmax].
    y_gross : float
        Gross income per capita before damage.
    omega_Fmin_calc : float
        Damage fraction for F <= Fmin_prev (Region 1) from previous timestep.
    omega_yi_calc : ndarray
        Damage fractions at quadrature points (Region 2) from previous timestep - FIXED length n_quad.
    omega_Fmax_calc : float
        Damage fraction for F >= Fmax_prev (Region 3) from previous timestep.
    Fmin_prev : float
        Previous timestep's Fmin (defines mapping for omega_yi_calc to F space).
    Fmax_prev : float
        Previous timestep's Fmax (defines mapping for omega_yi_calc to F space).
    xi_edges : ndarray
        Standard quadrature edges in xi space [-1, 1] for stepwise damage interpolation.
    uniform_tax_rate : float
        Uniform tax rate (fraction of gross income).
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini index (0 < gini < 1).
    use_empirical_lorenz : bool
        If True, use Empirical Lorenz formulation; if False, use Pareto.

    Returns
    -------
    y_net : float or ndarray
        Net income evaluated at the given F values.
    """
    global _first_call_diagnostics_printed, _call_counter

    # Increment call counter and print progress periodically
    _call_counter += 1
    if _call_counter % 100000 == 0:
        print(f"  [y_net_of_F call count = {_call_counter//100000} x 100,000]")

    F = np.clip(np.asarray(F), Fmin, Fmax)
    is_scalar = F.ndim == 0
    if is_scalar:
        F = F.reshape(1)

    # FAST PATH: When previous timestep covered full [0,1] range, use direct xi interpolation
    from src.constants import EPSILON
    if abs(Fmin_prev) < EPSILON and abs(Fmax_prev - 1.0) < EPSILON:
        # Previous grid was [0,1] → F maps directly to xi via: xi = 2*F - 1
        xi_vals = 2.0 * F - 1.0
        omega_prev_F = stepwise_interpolate(xi_vals, omega_yi_calc, xi_edges)
    else:
        # Three-region damage lookup using PREVIOUS timestep's structure
        # Classify all F points into three regions
        in_region1 = F <= Fmin_prev
        in_region3 = F >= Fmax_prev
        in_region2 = ~(in_region1 | in_region3)

        # Initialize damage array
        omega_prev_F = np.zeros_like(F, dtype=float)

        # Region 1 and 3: constant values
        omega_prev_F[in_region1] = omega_Fmin_calc
        omega_prev_F[in_region3] = omega_Fmax_calc

        # Region 2: vectorized transformation and interpolation
        if np.any(in_region2):
            F_region2 = F[in_region2]
            # Transform F to normalized coordinate x ∈ [0, 1] using PREVIOUS timestep's mapping
            x_region2 = (F_region2 - Fmin_prev) / (Fmax_prev - Fmin_prev)
            # Convert x from [0,1] to xi space [-1, 1]
            xi_region2 = 2.0 * x_region2 - 1.0
            # Vectorized interpolation for all region 2 points at once
            omega_prev_F[in_region2] = stepwise_interpolate(xi_region2, omega_yi_calc, xi_edges)

    # dL/dF(F) - choose formulation
    if use_empirical_lorenz:
        dLdF = L_empirical_lorenz_derivative(F, gini)
    else:
        dLdF = L_pareto_derivative(F, gini)

    # Compute net income: Lorenz → Damage → Tax → Redistribution (untaxed)
    y_net_F = y_gross * dLdF * (1.0 - omega_prev_F) * (1.0 - uniform_tax_rate) + uniform_redistribution

    return float(y_net_F[0]) if is_scalar else y_net_F


#========================================================================================
# Analytical taxation and redistribution functions
#========================================================================================


def find_Fmax(
    Fmin,
    y_gross,
    gini,
    Omega,
    omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
    Fmin_prev, Fmax_prev, xi_edges,
    redistribution_amount,
    abateCost_amount,
    use_empirical_lorenz,
    tol,
    initial_guess=None,
):
    """
    Find Fmax in [Fmin, 1) such that progressive taxation yields target tax amount.

    Fmax is calculated based on post-damage Lorenz income only (no taxes, no redistributions).
    This ensures that Fmax defines income thresholds based on the underlying distribution,
    avoiding circular dependencies with tax and redistribution calculations.

    uniform_redistribution is always 0.0 for this calculation.

    If we are here it is because there is income-dependent tax.

    Uses analytical Lorenz curve integration instead of numerical quadrature,
    with stepwise interpolation for climate damage.

    Tax revenue = ∫_{Fmax}^1 [y(F) - y(Fmax)] dF
    where y(F) = y_gross * dL/dF(F) + uniform_redistribution - damage(F)

    The tax simplifies to:
    - Lorenz part: y_gross * [(1 - L(Fmax)) - (1 - Fmax) * dL/dF(Fmax)]
    - Damage part: ∫_{Fmax}^1 damage(F) dF - (1 - Fmax) * damage(Fmax)
    - Uniform redistribution cancels out

    Parameters
    ----------
    Fmin : float
        Lower boundary for income distribution (must have Fmin < Fmax).
    y_gross : float
        Gross income per capita before damage.
    gini : float
        Gini coefficient.
    Omega : float
        Aggregate climate damage fraction.
    omega_Fmin_calc : float
        Damage fraction for F <= Fmin_prev (Region 1) from previous timestep.
    omega_yi_calc : ndarray
        Damage fractions at quadrature points (Region 2) from previous timestep - FIXED length n_quad.
    omega_Fmax_calc : float
        Damage fraction for F >= Fmax_prev (Region 3) from previous timestep.
    Fmin_prev : float
        Previous timestep's Fmin (defines mapping for omega_yi_calc to F space).
    Fmax_prev : float
        Previous timestep's Fmax (defines mapping for omega_yi_calc to F space).
    xi_edges : ndarray
        Standard quadrature edges in xi space [-1, 1] for stepwise damage interpolation.
    redistribution_amount : float
        Per-capita redistribution amount.
    abateCost_amount : float
        Per-capita abatement cost amount.
    use_empirical_lorenz : bool
        If True, use Empirical Lorenz formulation; if False, use Pareto.
    tol : float, optional
        Tolerance for root finding. Must be explicitly provided by caller.
    initial_guess : float, optional
        Initial guess for Fmax from previous timestep (speeds up convergence).

    Returns
    -------
    float
        Fmax value such that progressive taxation yields target tax amount.
    """

    # Always use 0.0 for uniform_redistribution when finding Fmax (see README.md Tax and Redistribution Logic)
    uniform_redistribution = 0.0

    # Create Fi grid in F space [0,1] for damage integration
    # Use xi_edges to create corresponding Fi_edges: Fi = (xi + 1) / 2
    Fi_edges = (xi_edges + 1.0) / 2.0

    # Pre-compute omega values at Fi grid points using VECTORIZED three-region lookup
    # Classify all Fi_edges points into three regions at once
    in_region1 = Fi_edges <= Fmin_prev
    in_region3 = Fi_edges >= Fmax_prev
    in_region2 = ~(in_region1 | in_region3)

    # Initialize omega array
    omega_at_Fi_edges = np.zeros_like(Fi_edges)

    # Region 1 and 3: constant values
    omega_at_Fi_edges[in_region1] = omega_Fmin_calc
    omega_at_Fi_edges[in_region3] = omega_Fmax_calc

    # Region 2: vectorized transformation and interpolation
    if np.any(in_region2):
        F_region2 = Fi_edges[in_region2]
        x_region2 = (F_region2 - Fmin_prev) / (Fmax_prev - Fmin_prev)
        xi_region2 = 2.0 * x_region2 - 1.0
        # Vectorized interpolation for all region 2 points at once
        omega_at_Fi_edges[in_region2] = stepwise_interpolate(xi_region2, omega_yi_calc, xi_edges)

    # Use midpoint values for omega in each bin (stepwise constant assumption)
    omega_yi_on_Fi_grid = (omega_at_Fi_edges[:-1] + omega_at_Fi_edges[1:]) / 2.0

    # Pre-compute cumulative damage integrals at bin edges for fast lookup
    # damage_cumulative[i] = integral of damage from 0 to Fi_edges[i]
    # Order: Lorenz → Damage → Tax (uniform tax rate is zero in this routine)
    # damage(F) = omega(F) * y_gross * dL/dF(F)
    # Since omega is stepwise constant in each bin, integrate dL/dF over each bin
    bin_widths = np.diff(Fi_edges)
    # Integral of dL/dF from Fi_edges[i] to Fi_edges[i+1] = L(Fi_edges[i+1]) - L(Fi_edges[i])
    if use_empirical_lorenz:
        lorenz_diff = np.diff(L_empirical_lorenz(Fi_edges, gini))
    else:
        lorenz_diff = np.diff(L_pareto(Fi_edges, gini))
    damage_per_bin = omega_yi_on_Fi_grid * y_gross * lorenz_diff
    damage_cumulative = np.concatenate(([0.0], np.cumsum(damage_per_bin)))
    total_damage_integral = damage_cumulative[-1]

    def tax_revenue_minus_target(Fmax):
        # Lorenz contribution (use combined function to avoid redundant computation)
        if use_empirical_lorenz:
            L_Fmax, dL_Fmax = L_empirical_lorenz_and_derivative(Fmax, gini)
            lorenz_part = y_gross * ((1.0 - L_Fmax) - (1.0 - Fmax) * dL_Fmax)
        else:
            L_Fmax, dL_Fmax = L_pareto_and_derivative(Fmax, gini)
            lorenz_part = y_gross * ((1.0 - L_Fmax) - (1.0 - Fmax) * dL_Fmax)

        # Fast damage calculation using pre-computed cumulative integrals
        # Find which bin Fmax is in
        bin_idx = np.searchsorted(Fi_edges, Fmax, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, len(omega_yi_on_Fi_grid) - 1)

        # Damage at Fmax using three-region lookup from previous timestep
        if Fmax <= Fmin_prev:
            omega_at_Fmax = omega_Fmin_calc
        elif Fmax >= Fmax_prev:
            omega_at_Fmax = omega_Fmax_calc
        else:
            # Transform Fmax to xi space and interpolate
            x = (Fmax - Fmin_prev) / (Fmax_prev - Fmin_prev)
            xi_val = 2.0 * x - 1.0
            omega_at_Fmax = stepwise_interpolate(xi_val, omega_yi_calc, xi_edges)

        # Integral from Fmax to 1.0 = total - integral from 0 to Fmax
        if Fmax <= Fi_edges[0]:
            damage_integral_0_to_Fmax = 0.0
        elif Fmax >= Fi_edges[-1]:
            damage_integral_0_to_Fmax = total_damage_integral
        else:
            # Cumulative up to start of bin containing Fmax
            damage_integral_0_to_Fmax = damage_cumulative[bin_idx]
            # Add partial contribution from within the bin
            partial_width = (Fmax - Fi_edges[bin_idx]) / bin_widths[bin_idx]
            damage_integral_0_to_Fmax += (damage_cumulative[bin_idx+1] - damage_cumulative[bin_idx]) * partial_width

        damage_integral = total_damage_integral - damage_integral_0_to_Fmax

        # Damage at Fmax based on income at Fmax (uniform redistribution is excluded; tax is zero here)
        if use_empirical_lorenz:
            damage_at_Fmax = y_gross * L_empirical_lorenz_derivative(Fmax, gini) * omega_at_Fmax
        else:
            damage_at_Fmax = y_gross * L_pareto_derivative(Fmax, gini) * omega_at_Fmax
        damage_part = damage_integral - (1.0 - Fmax) * damage_at_Fmax

        # Tax revenue (uniform redistribution cancels out)
        tax_revenue = lorenz_part - damage_part

        target_tax = abateCost_amount + redistribution_amount
        return tax_revenue - target_tax

    # Upper bound for F depends on whether we have singularities
    F_upper = 1.0 if use_empirical_lorenz else 1.0 - EPSILON

    # Use Newton method if we have a good initial guess, otherwise fall back to brentq
    if initial_guess is not None and Fmin < initial_guess < F_upper:
        # First, check if the initial guess is already very close to the solution
        f_guess = tax_revenue_minus_target(initial_guess)
        if abs(f_guess) < tol:
            return initial_guess

        # Use Newton's method with initial guess
        # Newton's method has quadratic convergence (4-6 iterations typical)
        # vs secant's superlinear convergence (10-20 iterations typical)
        try:
            sol = newton(tax_revenue_minus_target, initial_guess, tol=tol, maxiter=20)
            if Fmin <= sol <= F_upper:
                return sol
        except (ValueError, RuntimeError):
            pass  # Fall through to bracketing method

    # Fall back to bracketing method if Newton fails or no initial guess
    left = Fmin
    right = F_upper
    f_left = tax_revenue_minus_target(left)
    f_right = tax_revenue_minus_target(right)

    # If both endpoints have the same sign, return the appropriate boundary
    if f_left * f_right > 0:
        if f_left > 0:
            # Both positive: target_tax is too small, return right (tax almost no one)
            return right
        else:
            # Both negative: target_tax is too large, return left (tax everyone above Fmin)
            return left

    sol = root_scalar(tax_revenue_minus_target, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmax")

    return sol.root


def find_Fmin(
    Fmax,
    y_gross,
    gini,
    Omega,
    omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
    Fmin_prev, Fmax_prev, xi_edges,
    redistribution_amount,
    uniform_tax_rate,
    use_empirical_lorenz,
    tol,
    initial_guess=None,
):
    """
    Find Fmin in (0, Fmax) such that progressive redistribution yields target subsidy amount.

    Fmin is calculated considering:
    - Everyone pays tax on their Lorenz income (including those below Fmin)
    - Redistribution subsidy is NOT taxed (it comes from taxes)
    - Fmin defines the threshold where subsidy reaches zero

    This ensures continuity at Fmin and no perverse incentives.

    Uses analytical Lorenz curve integration instead of numerical quadrature,
    with stepwise interpolation for climate damage.

    Subsidy amount = ∫_0^Fmin [y(Fmin) - y(F)] dF
    where y(F) = y_gross * dL/dF(F) + uniform_redistribution - damage(F)

    The subsidy simplifies to:
    - Lorenz part: y_gross * [Fmin * dL/dF(Fmin) - L(Fmin)]
    - Damage part: Fmin * damage(Fmin) - ∫_0^Fmin damage(F) dF
    - Uniform redistribution cancels out

    Parameters
    ----------
    Fmax : float
        Upper boundary for income distribution (must have Fmin < Fmax).
    y_gross : float
        Gross income per capita before damage.
    gini : float
        Gini coefficient.
    Omega : float
        Aggregate climate damage fraction.
    omega_Fmin_calc : float
        Damage fraction for F <= Fmin_prev (Region 1) from previous timestep.
    omega_yi_calc : ndarray
        Damage fractions at quadrature points (Region 2) from previous timestep - FIXED length n_quad.
    omega_Fmax_calc : float
        Damage fraction for F >= Fmax_prev (Region 3) from previous timestep.
    Fmin_prev : float
        Previous timestep's Fmin (defines mapping for omega_yi_calc to F space).
    Fmax_prev : float
        Previous timestep's Fmax (defines mapping for omega_yi_calc to F space).
    xi_edges : ndarray
        Standard quadrature edges in xi space [-1, 1] for stepwise damage interpolation.
    redistribution_amount : float
        Per-capita redistribution amount (target subsidy).
    uniform_tax_rate : float
        Uniform tax rate (fraction of income).
    use_empirical_lorenz : bool
        If True, use Empirical Lorenz formulation; if False, use Pareto.
    tol : float, optional
        Tolerance for root finding. Must be explicitly provided by caller.
    initial_guess : float, optional
        Initial guess for Fmin from previous timestep (speeds up convergence).

    Returns
    -------
    float
        Fmin value such that progressive redistribution yields target subsidy amount.
    """

    # Redistribution subsidy is not taxed (it comes from taxes)
    # But everyone pays tax on their Lorenz income, so uniform_tax_rate is passed as parameter
    uniform_redistribution = 0.0

    # Create Fi grid in F space [0,1] for damage integration
    # Use xi_edges to create corresponding Fi_edges: Fi = (xi + 1) / 2
    Fi_edges = (xi_edges + 1.0) / 2.0

    # Pre-compute omega values at Fi grid points using VECTORIZED three-region lookup
    # Classify all Fi_edges points into three regions at once
    in_region1 = Fi_edges <= Fmin_prev
    in_region3 = Fi_edges >= Fmax_prev
    in_region2 = ~(in_region1 | in_region3)

    # Initialize omega array
    omega_at_Fi_edges = np.zeros_like(Fi_edges)

    # Region 1 and 3: constant values
    omega_at_Fi_edges[in_region1] = omega_Fmin_calc
    omega_at_Fi_edges[in_region3] = omega_Fmax_calc

    # Region 2: vectorized transformation and interpolation
    if np.any(in_region2):
        F_region2 = Fi_edges[in_region2]
        x_region2 = (F_region2 - Fmin_prev) / (Fmax_prev - Fmin_prev)
        xi_region2 = 2.0 * x_region2 - 1.0
        # Vectorized interpolation for all region 2 points at once
        omega_at_Fi_edges[in_region2] = stepwise_interpolate(xi_region2, omega_yi_calc, xi_edges)

    # Use midpoint values for omega in each bin (stepwise constant assumption)
    omega_yi_on_Fi_grid = (omega_at_Fi_edges[:-1] + omega_at_Fi_edges[1:]) / 2.0

    # Pre-compute cumulative damage integrals at bin edges for fast lookup
    # damage_cumulative[i] = integral of damage from 0 to Fi_edges[i]
    # damage(F) = omega(F) * y_gross * dL/dF(F) * (1 - tax)
    # Redistribution is added AFTER damage, so it's not included here
    # Since omega is stepwise constant in each bin, integrate dL/dF over each bin
    bin_widths = np.diff(Fi_edges)
    # Integral of dL/dF from Fi_edges[i] to Fi_edges[i+1] = L(Fi_edges[i+1]) - L(Fi_edges[i])
    if use_empirical_lorenz:
        lorenz_diff = np.diff(L_empirical_lorenz(Fi_edges, gini))
    else:
        lorenz_diff = np.diff(L_pareto(Fi_edges, gini))
    damage_per_bin = omega_yi_on_Fi_grid * y_gross * lorenz_diff * (1.0 - uniform_tax_rate)
    damage_cumulative = np.concatenate(([0.0], np.cumsum(damage_per_bin)))

    def subsidy_minus_target(Fmin):
        # Lorenz contribution (use combined function to avoid redundant computation)
        # difference if everyone were consuming at Fmin rate minus actual integrated to Fmin.
        if use_empirical_lorenz:
            L_Fmin, dL_Fmin = L_empirical_lorenz_and_derivative(Fmin, gini)
            lorenz_part = y_gross * (1.0 - uniform_tax_rate) * (Fmin * dL_Fmin - L_Fmin)
        else:
            L_Fmin, dL_Fmin = L_pareto_and_derivative(Fmin, gini)
            lorenz_part = y_gross * (1.0 - uniform_tax_rate) * (Fmin * dL_Fmin - L_Fmin)

        # Fast damage calculation using pre-computed cumulative integrals
        # Find which bin Fmin is in
        bin_idx = np.searchsorted(Fi_edges, Fmin, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, len(omega_yi_on_Fi_grid) - 1)

        # Damage at Fmin using three-region lookup from previous timestep
        if Fmin <= Fmin_prev:
            omega_at_Fmin = omega_Fmin_calc
        elif Fmin >= Fmax_prev:
            omega_at_Fmin = omega_Fmax_calc
        else:
            # Transform Fmin to xi space and interpolate
            x = (Fmin - Fmin_prev) / (Fmax_prev - Fmin_prev)
            xi_val = 2.0 * x - 1.0
            omega_at_Fmin = stepwise_interpolate(xi_val, omega_yi_calc, xi_edges)

        # Order: Lorenz → Damage → Tax
        if use_empirical_lorenz:
            damage_at_Fmin = y_gross * L_empirical_lorenz_derivative(Fmin, gini) * omega_at_Fmin * (1.0 - uniform_tax_rate)
        else:
            damage_at_Fmin = y_gross * L_pareto_derivative(Fmin, gini) * omega_at_Fmin * (1.0 - uniform_tax_rate)

        # Integral from 0 to Fmin = cumulative up to bin start + partial bin
        if Fmin <= Fi_edges[0]:
            damage_integral = 0.0
        elif Fmin >= Fi_edges[-1]:
            damage_integral = damage_cumulative[-1]
        else:
            # Cumulative up to start of bin containing Fmin
            damage_integral = damage_cumulative[bin_idx]
            # Add partial contribution from within the bin
            partial_width = (Fmin - Fi_edges[bin_idx]) / bin_widths[bin_idx]
            damage_integral += (damage_cumulative[bin_idx+1] - damage_cumulative[bin_idx]) * partial_width

        damage_part = Fmin * damage_at_Fmin - damage_integral

        # Subsidy amount (uniform redistribution cancels out)
        subsidy_amount = lorenz_part + damage_part

        return subsidy_amount - redistribution_amount

    # Use Newton method if we have a good initial guess, otherwise fall back to brentq
    if initial_guess is not None and 0.0 < initial_guess < Fmax - EPSILON:
        # First, check if the initial guess is already very close to the solution
        f_guess = subsidy_minus_target(initial_guess)
        if abs(f_guess) < tol:
            return initial_guess

        # Use Newton's method with initial guess
        # Newton's method has quadratic convergence (4-6 iterations typical)
        # vs secant's superlinear convergence (10-20 iterations typical)
        try:
            sol = newton(subsidy_minus_target, initial_guess, tol=tol, maxiter=20)
            if 0.0 <= sol <= Fmax:
                return sol
        except (ValueError, RuntimeError):
            pass  # Fall through to bracketing method

    # Fall back to bracketing method if Newton fails or no initial guess
    left = 0.0
    right = Fmax - EPSILON
    f_left = subsidy_minus_target(left)
    f_right = subsidy_minus_target(right)

    # If both endpoints have the same sign, return the appropriate boundary
    if f_left * f_right > 0:
        if f_left > 0:
            # Both positive: target_subsidy is too small, return left (redistribute to almost no one)
            return EPSILON
        else:
            # Both negative: target_subsidy is too large, return right (redistribute to almost everyone)
            return right

    sol = root_scalar(subsidy_minus_target, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmin")

    return sol.root


#========================================================================================
# Helper functions for three-region damage lookup
#========================================================================================

def _get_omega_at_F(
    F,
    omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
    Fmin_prev, Fmax_prev, xi_edges,
):
    """
    Get damage fraction at population rank F using three-region lookup.

    Uses the previous timestep's three-region structure:
    - Region 1 (F <= Fmin_prev): constant omega_Fmin_calc
    - Region 2 (Fmin_prev < F < Fmax_prev): interpolate from omega_yi_calc
    - Region 3 (F >= Fmax_prev): constant omega_Fmax_calc

    Parameters
    ----------
    F : float or ndarray
        Population rank(s) in [0, 1]
    omega_Fmin_calc : float
        Damage fraction for F <= Fmin_prev (Region 1)
    omega_yi_calc : ndarray
        Damage fractions at quadrature points (Region 2)
    omega_Fmax_calc : float
        Damage fraction for F >= Fmax_prev (Region 3)
    Fmin_prev : float
        Previous timestep's Fmin (Region 1/2 boundary)
    Fmax_prev : float
        Previous timestep's Fmax (Region 2/3 boundary)
    xi_edges : ndarray
        Standard quadrature edges in xi space [-1, 1]

    Returns
    -------
    float or ndarray
        Damage fraction(s) at F
    """
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0
    if scalar_input:
        F_arr = F_arr.reshape(1)

    # Fast path: previous timestep covered full [0,1] range
    if abs(Fmin_prev) < EPSILON and abs(Fmax_prev - 1.0) < EPSILON:
        xi_vals = 2.0 * F_arr - 1.0
        omega = stepwise_interpolate(xi_vals, omega_yi_calc, xi_edges)
        return float(omega[0]) if scalar_input else omega

    # Three-region damage lookup
    in_region1 = F_arr <= Fmin_prev
    in_region3 = F_arr >= Fmax_prev
    in_region2 = ~(in_region1 | in_region3)

    omega = np.zeros_like(F_arr, dtype=float)
    omega[in_region1] = omega_Fmin_calc
    omega[in_region3] = omega_Fmax_calc

    if np.any(in_region2):
        F_region2 = F_arr[in_region2]
        x_region2 = (F_region2 - Fmin_prev) / (Fmax_prev - Fmin_prev)
        xi_region2 = 2.0 * x_region2 - 1.0
        omega[in_region2] = stepwise_interpolate(xi_region2, omega_yi_calc, xi_edges)

    return float(omega[0]) if scalar_input else omega


#========================================================================================
# Equal utility loss taxation functions
#========================================================================================

def compute_post_tax_income_equal_utility(y_F, K_utility_loss, eta):
    """
    Compute post-tax income c(F) for equal-utility-loss taxation.

    With CRRA utility U(c) = c^(1-η)/(1-η) for η≠1, or U(c) = ln(c) for η=1:
    Equal utility loss means U(y) - U(c) = K for all incomes.

    Solving for c:
        c(F) = [y(F)^(1-η) - (1-η)K]^(1/(1-η))   for η ≠ 1
        c(F) = y(F) * exp(-K)                     for η = 1

    Parameters
    ----------
    y_F : float or ndarray
        Pre-tax income(s)
    K_utility_loss : float
        Constant utility loss across all income levels
    eta : float
        Coefficient of relative risk aversion

    Returns
    -------
    float or ndarray
        Post-tax income(s)
    """
    y_F = np.asarray(y_F)
    scalar_input = np.ndim(y_F) == 0
    if scalar_input:
        y_F = y_F.reshape(1)

    # Handle η=1 (log utility) case
    if abs(eta - 1.0) < EPSILON:
        c_F = y_F * np.exp(-K_utility_loss)
    else:
        # c = [y^(1-η) - (1-η)K]^(1/(1-η))
        inner = y_F ** (1.0 - eta) - (1.0 - eta) * K_utility_loss
        # Ensure inner > 0 for valid result
        inner = np.maximum(inner, EPSILON)
        c_F = inner ** (1.0 / (1.0 - eta))

    return float(c_F[0]) if scalar_input else c_F


def find_K_equal_utility(
    y_gross,
    gini,
    Omega_calc,
    omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
    Fmin_prev, Fmax_prev, xi_edges,
    eta,
    tax_amount,
    use_empirical_lorenz,
    tol,
    xi, wi,
):
    """
    Find utility loss constant K for equal-utility-loss taxation.

    Returns K such that taxing everyone with equal utility loss yields tax_amount.

    With CRRA utility U(c) = c^(1-η)/(1-η):
    - Post-tax income: c(F) = [y(F)^(1-η) - (1-η)K]^(1/(1-η))
    - Tax at rank F: τ(F) = y(F) - c(F)
    - Revenue constraint: ∫₀¹ τ(F) dF = tax_amount

    Special case η=1 (log utility):
    - c(F) = y(F) * exp(-K)
    - Tax rate: r = 1 - exp(-K) (constant for all F)
    - This degenerates to uniform tax rate r = tax_amount / mean_income

    Parameters
    ----------
    y_gross : float
        Gross income per capita before damage
    gini : float
        Gini coefficient
    Omega_calc : float
        Aggregate climate damage fraction
    omega_Fmin_calc : float
        Damage fraction for F <= Fmin_prev (Region 1)
    omega_yi_calc : ndarray
        Damage fractions at quadrature points (Region 2)
    omega_Fmax_calc : float
        Damage fraction for F >= Fmax_prev (Region 3)
    Fmin_prev : float
        Previous timestep's Fmin
    Fmax_prev : float
        Previous timestep's Fmax
    xi_edges : ndarray
        Standard quadrature edges in xi space [-1, 1]
    eta : float
        Coefficient of relative risk aversion
    tax_amount : float
        Target per-capita tax revenue
    use_empirical_lorenz : bool
        If True, use Empirical Lorenz; if False, use Pareto
    tol : float
        Tolerance for root finding
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1]
    wi : ndarray
        Gauss-Legendre quadrature weights

    Returns
    -------
    float
        Utility loss constant K such that equal-utility-loss taxation yields tax_amount
    """
    # If tax_amount is negligible, return K=0 (no tax)
    if tax_amount < EPSILON:
        return 0.0

    # Mean damaged income (what we're taxing from)
    y_damaged_mean = y_gross * (1.0 - Omega_calc)

    # Special case η=1 (log utility): degenerates to uniform tax rate
    # c = y * exp(-K), so tax rate r = 1 - exp(-K)
    # Total tax = r * y_mean = tax_amount
    # => r = tax_amount / y_mean
    # => K = -ln(1 - r) = -ln(1 - tax_amount / y_mean)
    if abs(eta - 1.0) < EPSILON:
        r = tax_amount / y_damaged_mean
        # Cap r to avoid log(0) or negative
        r = min(r, 1.0 - EPSILON)
        K = -np.log(1.0 - r)
        return K

    # Map quadrature from [-1, 1] to [0, 1] in F space
    Fi = (xi + 1.0) / 2.0
    Fwi = wi / 2.0

    # Get pre-tax income y(F) = y_gross * dL/dF * (1 - omega)
    if use_empirical_lorenz:
        dLdF = L_empirical_lorenz_derivative(Fi, gini)
    else:
        dLdF = L_pareto_derivative(Fi, gini)

    omega_Fi = _get_omega_at_F(
        Fi,
        omega_Fmin_calc, omega_yi_calc, omega_Fmax_calc,
        Fmin_prev, Fmax_prev, xi_edges,
    )
    y_F = y_gross * dLdF * (1.0 - omega_Fi)

    # For η < 1: utility is unbounded below as c → 0, and we need
    # y^(1-η) - (1-η)K > 0 for all F, especially the poorest.
    # So K < y_min^(1-η) / (1-η) (note: 1-η > 0 when η < 1)
    #
    # For η > 1: utility is negative and bounded, (1-η) < 0,
    # so y^(1-η) - (1-η)K > 0 means K can be arbitrarily large.

    y_min = np.min(y_F)

    if eta < 1.0:
        # K_max such that inner > 0 for y_min
        # y_min^(1-η) - (1-η)*K > 0
        # K < y_min^(1-η) / (1-η)
        K_max = (y_min ** (1.0 - eta)) / (1.0 - eta) - EPSILON
    else:
        # For η > 1, K can be very large (utility is negative)
        # But practically, limit to reasonable value
        K_max = 1e10

    def tax_revenue_minus_target(K):
        # Post-tax income from equal utility loss formula
        c_F = compute_post_tax_income_equal_utility(y_F, K, eta)

        # Tax revenue
        tax_revenue = np.sum(Fwi * (y_F - c_F))
        return tax_revenue - tax_amount

    # Check if K=0 already exceeds target (shouldn't happen, but be safe)
    f_zero = tax_revenue_minus_target(0.0)
    if f_zero >= 0:
        return 0.0

    # Check if K_max gives enough revenue
    f_max = tax_revenue_minus_target(K_max)
    if f_max < 0:
        # Even max K doesn't give enough revenue
        # Return K_max as best we can do
        return K_max

    # Bracketing: K ∈ [0, K_max], f(0) < 0 < f(K_max)
    # Use Brent's method
    sol = root_scalar(
        tax_revenue_minus_target,
        bracket=[0.0, K_max],
        method="brentq",
        xtol=tol,
    )
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_K_equal_utility")

    return sol.root


#========================================================================================
# Stepwise interpolation and integration utilities
#========================================================================================

def stepwise_interpolate(F, yi, Fi_edges):
    """
    Stepwise (piecewise constant) interpolation over quadrature intervals.

    Returns yi[i] for F in [Fi_edges[i], Fi_edges[i+1]).

    Parameters
    ----------
    F : float or ndarray
        Evaluation point(s) where interpolated values are desired.
    yi : ndarray
        Values at quadrature points (length N). yi[i] is the constant value
        in interval [Fi_edges[i], Fi_edges[i+1]).
    Fi_edges : ndarray
        Interval boundaries (length N+1), must be monotonically increasing.

    Returns
    -------
    float or ndarray
        Interpolated value(s) at F. Returns same shape as F input.

    Notes
    -----
    - For F < Fi_edges[0], returns yi[0]
    - For F >= Fi_edges[-1], returns yi[-1]
    - For Fi_edges[i] <= F < Fi_edges[i+1], returns yi[i]
    """
    F_array = np.atleast_1d(F)
    scalar_input = np.ndim(F) == 0

    # Find which interval each F value falls into
    # searchsorted returns index i such that Fi_edges[i-1] <= F < Fi_edges[i]
    # We want index for yi, so subtract 1 and clip to valid range
    indices = np.searchsorted(Fi_edges, F_array, side='right') - 1
    indices = np.clip(indices, 0, len(yi) - 1)

    result = yi[indices]

    return result[0] if scalar_input else result


def stepwise_integrate(F0, F1, yi, Fi_edges):
    """
    Integrate a stepwise (piecewise constant) function from F0 to F1.

    The stepwise function has value yi[i] over interval [Fi_edges[i], Fi_edges[i+1]).

    Parameters
    ----------
    F0 : float
        Lower integration bound.
    F1 : float
        Upper integration bound.
    yi : ndarray
        Values at quadrature points (length N). yi[i] is the constant value
        in interval [Fi_edges[i], Fi_edges[i+1]).
    Fi_edges : ndarray
        Interval boundaries (length N+1), must be monotonically increasing.

    Returns
    -------
    float
        Integral of the stepwise function from F0 to F1.

    Notes
    -----
    - Handles F0 and F1 outside [Fi_edges[0], Fi_edges[-1]] by extending boundary values
    - For F0 > F1, returns negative of integral from F1 to F0
    """
    # Handle reversed bounds
    if F1 < F0:
        return -stepwise_integrate(F1, F0, yi, Fi_edges)

    # Clip bounds to valid range
    F0_clip = np.clip(F0, Fi_edges[0], Fi_edges[-1])
    F1_clip = np.clip(F1, Fi_edges[0], Fi_edges[-1])

    # Find which intervals F0 and F1 fall into
    i0 = np.searchsorted(Fi_edges, F0_clip, side='right') - 1
    i1 = np.searchsorted(Fi_edges, F1_clip, side='right') - 1

    # Clip to valid indices
    i0 = np.clip(i0, 0, len(yi) - 1)
    i1 = np.clip(i1, 0, len(yi) - 1)

    # Same interval case
    if i0 == i1:
        return yi[i0] * (F1_clip - F0_clip)

    # Multiple intervals case
    integral = 0.0

    # Partial contribution from first interval
    integral += yi[i0] * (Fi_edges[i0 + 1] - F0_clip)

    # Full contributions from middle intervals
    for k in range(i0 + 1, i1):
        integral += yi[k] * (Fi_edges[k + 1] - Fi_edges[k])

    # Partial contribution from last interval
    integral += yi[i1] * (F1_clip - Fi_edges[i1])

    return integral


