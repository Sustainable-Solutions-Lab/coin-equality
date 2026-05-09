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
    Compute post-tax income c(F) from the equal-utility-loss closed form.

    Given a CRRA-like pseudo-utility U*(c) = c^(1-eta)/(1-eta) (or ln c at eta=1),
    equal pseudo-utility loss U*(y) - U*(c) = K yields:
        c(F) = [y(F)^(1-eta) - (1-eta) K]^(1/(1-eta))   for eta != 1
        c(F) = y(F) * exp(-K)                           for eta = 1

    Parameters
    ----------
    y_F : float or ndarray
        Pre-tax income(s).
    K_utility_loss : float
        Common pseudo-utility loss parameter (solved upstream by find_K_equal_utility).
    eta : float
        Exponent of the pseudo-utility. In the continuous-knob policy this is the
        effective exponent eta_eff = 1 + tax_equity/(1-tax_equity)*(eta_CRRA - 1),
        not necessarily the real CRRA parameter.

    Returns
    -------
    float or ndarray
        Post-tax income(s).

    Properties
    ----------
    1. Equal real utility loss at eta_eff = eta_CRRA (i.e. tax_equity = 0.5):
       U(y) - U(c) = K for every y, where U is the real CRRA utility. This is the
       defining property of the old 'u-uniform' policy and is preserved at the
       midpoint of the continuous knob. Away from that corner the pseudo-utility
       loss is still constant across y, but it is not the real-CRRA utility loss.

    2. Monotonicity: dc/dy = (c/y)^eta > 0 for every c > 0, so post-tax income
       is a strictly increasing function of pre-tax income. A richer person
       never ends up with less post-tax income than a poorer one.

    3. Boundedness: for eta >= 1, inner > 0 always, giving 0 < c <= y. For
       eta < 1 the inner expression can go negative at large K; the
       np.maximum(inner, 0.0) clip then floors c at 0 (100% tax cap, never
       below zero).

    4. No tax rate is materialized. If a caller wants the effective schedule,
       derive average rate = 1 - c/y and marginal rate = 1 - (c/y)^eta from
       the returned c_F and the input y_F.
    """
    y_F = np.asarray(y_F)
    scalar_input = np.ndim(y_F) == 0
    if scalar_input:
        y_F = y_F.reshape(1)

    # Handle η=1 (log utility) case
    if abs(eta - 1.0) < EPSILON:
        c_F = y_F * np.exp(-K_utility_loss)
    elif eta > 1.0:
        # Log-space formulation to avoid overflow in y^(1-η) at large η.
        #
        # Rewrite c = [y^(-α) + αK]^(-1/α) as c = y·(1 + αK·y^α)^(-1/α)
        # where α = η-1 > 0. Compute αK·y^α in log space:
        #   log(αKy^α) = log(α) + log(K) + α·log(y)
        # Then log(c) = log(y) - (1/α)·log(1 + αKy^α), using logaddexp for stability.
        # This avoids materializing y^(-α) which overflows for small y and large α
        # (e.g. y=0.05, α=299 → y^(-299) ≈ 10^389 > float64 max).
        alpha = eta - 1.0
        if K_utility_loss <= 0.0:
            c_F = y_F.copy()
        else:
            with np.errstate(divide='ignore'):
                log_y = np.log(y_F)
            log_aKya = np.log(alpha) + np.log(K_utility_loss) + alpha * log_y
            log_factor = np.logaddexp(0.0, log_aKya)
            log_c = log_y - log_factor / alpha
            c_F = np.where(y_F > 0, np.exp(log_c), 0.0)
    else:
        # η < 1: direct computation is numerically stable (1-η > 0, no overflow)
        # c = [y^(1-η) - (1-η)K]^(1/(1-η))
        inner = y_F ** (1.0 - eta) - (1.0 - eta) * K_utility_loss
        # Allow inner = 0 (taxed to zero income) when K exceeds max utility loss
        inner = np.maximum(inner, 0.0)
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
    Find the normalized utility-loss constant K_norm for equal-utility-loss taxation.

    Returns K_norm in normalized income units. To recover post-tax income at raw
    income y, the caller must:

        c = compute_post_tax_income_equal_utility(y / y_damaged_mean, K_norm, eta) * y_damaged_mean

    Solving in normalized units sidesteps floating-point underflow in y^(1-eta) at
    high eta and dollar-scale incomes; K_norm stays well-conditioned even at
    eta_eff in the hundreds.

    With CRRA utility U(c) = c^(1-η)/(1-η):
    - Post-tax income: c_norm(F) = [y_norm(F)^(1-η) - (1-η)K_norm]^(1/(1-η))
    - Revenue constraint (in raw units): ∫₀¹ (y(F) - c(F)) dF = tax_amount

    Special case η=1 (log utility):
    - c(F) = y(F) * exp(-K_norm)  (the formula is scale-invariant, so K_norm = K)
    - Degenerates to uniform tax rate r = tax_amount / mean_income.

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
        Coefficient of relative risk aversion (or eta_effective for the continuous tax knob)
    tax_amount : float
        Target per-capita tax revenue (in raw income units)
    use_empirical_lorenz : bool
        If True, use Empirical Lorenz; if False, use Pareto
    tol : float
        Tolerance for root finding (in log-K_norm space)
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1]
    wi : ndarray
        Gauss-Legendre quadrature weights

    Returns
    -------
    float
        Normalized utility-loss constant K_norm. See module-level note above
        on how to recover post-tax incomes from this value.
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

    # Normalize incomes so y_F_norm is O(1). Without this, y_F^(1-eta) underflows
    # to zero at large eta and realistic income scales (e.g. y~1e5, eta~10), which
    # turns the tax-revenue function into a numerical step. We return K_norm directly;
    # the caller is responsible for normalizing y by y_damaged_mean and rescaling c.
    y_F_norm = y_F / y_damaged_mean
    tax_amount_norm = tax_amount / y_damaged_mean

    def tax_revenue_minus_target_log(logK):
        # Solve in log-K space so the tax-revenue curve is smooth in the steep regime
        # that appears at high eta (the curve is near-stepwise in K itself).
        K_norm = np.exp(logK)
        c_F_norm = compute_post_tax_income_equal_utility(y_F_norm, K_norm, eta)
        return np.sum(Fwi * (y_F_norm - c_F_norm)) - tax_amount_norm

    # Bracket in log-K space. Lower bound chosen so K_norm is numerically zero
    # (no tax); upper bound so K_norm is large enough to tax all income away.
    logK_lo = -700.0  # exp(-700) ≈ 1e-304, below double-precision underflow
    logK_hi = np.log(1e10)

    f_lo = tax_revenue_minus_target_log(logK_lo)
    if f_lo >= 0:
        # Target already met at zero tax — nothing to raise
        return 0.0

    f_hi = tax_revenue_minus_target_log(logK_hi)
    if f_hi < 0:
        raise ValueError(
            f"Cannot raise required tax_amount={tax_amount:.6g} with equal utility loss taxation. "
            f"Maximum possible revenue (taxing everyone to zero) is "
            f"{(tax_amount_norm + f_hi) * y_damaged_mean:.6g}."
        )

    sol = root_scalar(
        tax_revenue_minus_target_log,
        bracket=[logK_lo, logK_hi],
        method="brentq",
        xtol=tol,
    )
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_K_equal_utility")

    K_norm = np.exp(sol.root)

    # Post-solve invariant: c_F_norm must be finite and weakly monotone in F. y_F_norm
    # is monotone in F by Lorenz construction, and dc/dy = (c/y)^eta > 0 wherever c > 0,
    # so the schedule cannot invert the income order. Zero values are allowed: for
    # eta < 1 with large K_norm, low-income people can legitimately be clipped to
    # c_norm = 0 (100% tax cap), still non-decreasing in F. A violation here signals
    # NaN, infinite output, or genuine ordering inversion — symptoms of a Brent
    # miscarriage or out-of-range eta.
    c_F_norm_check = compute_post_tax_income_equal_utility(y_F_norm, K_norm, eta)
    if not np.all(np.isfinite(c_F_norm_check)):
        bad = np.where(~np.isfinite(c_F_norm_check))[0]
        raise RuntimeError(
            f"find_K_equal_utility: post-solve c_F_norm contains non-finite values at "
            f"indices {bad[:5].tolist()} (eta={eta:.4f}, K_norm={K_norm:.3g}, "
            f"tax_amount={tax_amount:.3g})."
        )
    diffs = np.diff(c_F_norm_check)
    monotone_tol = EPSILON * 1e3  # in normalized units, c_norm is O(1)
    if not np.all(diffs >= -monotone_tol):
        bad = np.where(diffs < -monotone_tol)[0]
        raise RuntimeError(
            f"find_K_equal_utility: post-solve c_F_norm not monotone in F at indices "
            f"{bad[:5].tolist()} (min diff = {diffs.min():.3g}, tol = {monotone_tol:.3g}, "
            f"eta={eta:.4f}, K_norm={K_norm:.3g})."
        )

    return K_norm


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


