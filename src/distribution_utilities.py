"""
Income distribution functions for the quadrature over population rank F in [0, 1].

This module provides:
- Pareto-Lorenz and empirical Lorenz curves and their derivatives
- Equal-utility-loss progressive taxation (closed-form schedule and revenue root solve)
- Stepwise interpolation of per-quadrature-point values
"""

import numpy as np
from scipy.optimize import root_scalar
from src.constants import (
    EPSILON,
    LOG_DELTAU_NO_TAX,
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


def L_pareto_derivative(F, G):
    """Derivative of Lorenz curve dL/dF at F for Pareto-Lorenz distribution with Gini coefficient G."""
    a = a_from_G(G)
    F_arr = np.asarray(F)
    scalar_input = np.ndim(F) == 0

    F_clipped = np.clip(F_arr, EPSILON, 1.0 - EPSILON)
    result = (1.0 - 1.0/a) * (1.0 - F_clipped)**(-1.0/a)

    return result[()] if scalar_input else result


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


#========================================================================================
# Helper function for income-distribution damage lookup
#========================================================================================

def _get_omega_at_F(F, omega_yi_calc, xi_edges):
    """
    Get damage fraction at population rank F by stepwise interpolation of the
    per-quadrature-point damage fractions omega_yi_calc over the full [0, 1] range.

    Parameters
    ----------
    F : float or ndarray
        Population rank(s) in [0, 1]
    omega_yi_calc : ndarray
        Damage fractions at quadrature points
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

    xi_vals = 2.0 * F_arr - 1.0
    omega = stepwise_interpolate(xi_vals, omega_yi_calc, xi_edges)
    return float(omega[0]) if scalar_input else omega


#========================================================================================
# Equal utility loss taxation functions
#========================================================================================

def compute_post_tax_income_equal_utility(y_F, log_deltaU, eta):
    """
    Compute post-tax income c(F) from the equal-utility-loss closed form.

    Given a CRRA-like pseudo-utility U*(c) = c^(1-eta)/(1-eta) (or ln c at eta=1),
    equal pseudo-utility loss U*(y) - U*(c) = deltaU yields:
        c(F) = [y(F)^(1-eta) - (1-eta) deltaU]^(1/(1-eta))   for eta != 1
        c(F) = y(F) * exp(-deltaU)                             for eta = 1

    This function accepts log(deltaU) instead of deltaU to avoid float64 underflow
    when the tax exponent is large (>= ~290). At tax_exponent=500, the true solution
    is log_deltaU ≈ -1242; exp(-1242) underflows to 0 in float64, silently disabling
    progressive taxation. By keeping deltaU in log space throughout the eta > 1 branch,
    the computation remains numerically exact.

    Parameters
    ----------
    y_F : float or ndarray
        Pre-tax income(s).
    log_deltaU : float
        Log of the common pseudo-utility loss parameter. Solved upstream by
        find_log_deltaU_equal_utility. Use LOG_DELTAU_NO_TAX (= -inf) for no tax.
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
       U(y) - U(c) = deltaU for every y, where U is the real CRRA utility.

    2. Monotonicity: dc/dy = (c/y)^eta > 0 for every c > 0, so post-tax income
       is a strictly increasing function of pre-tax income.

    3. Boundedness: for eta >= 1, inner > 0 always, giving 0 < c <= y. For
       eta < 1 the inner expression can go negative at large deltaU; the
       np.maximum(inner, 0.0) clip then floors c at 0 (100% tax cap).

    4. No tax rate is materialized. If a caller wants the effective schedule,
       derive average rate = 1 - c/y and marginal rate = 1 - (c/y)^eta from
       the returned c_F and the input y_F.
    """
    y_F = np.asarray(y_F)
    scalar_input = np.ndim(y_F) == 0
    if scalar_input:
        y_F = y_F.reshape(1)

    # No-tax sentinel: log_deltaU = -inf means deltaU = 0
    if log_deltaU == LOG_DELTAU_NO_TAX:
        c_F = y_F.copy()
        return float(c_F[0]) if scalar_input else c_F

    # Handle η=1 (log utility) case: c = y * exp(-deltaU) = y * exp(-exp(log_deltaU))
    if abs(eta - 1.0) < EPSILON:
        c_F = y_F * np.exp(-np.exp(log_deltaU))
    elif eta > 1.0:
        # Log-space formulation to avoid overflow in y^(1-η) at large η.
        #
        # Rewrite c = [y^(-α) + α·deltaU]^(-1/α) as c = y·(1 + α·deltaU·y^α)^(-1/α)
        # where α = η-1 > 0. Compute α·deltaU·y^α in log space:
        #   log(α·deltaU·y^α) = log(α) + log_deltaU + α·log(y)
        # Then log(c) = log(y) - (1/α)·log(1 + α·deltaU·y^α), using logaddexp.
        # THE KEY FIX: we use log_deltaU directly instead of log(exp(logK)),
        # which avoids the underflow that occurred when logK < -708.
        alpha = eta - 1.0
        with np.errstate(divide='ignore'):
            log_y = np.log(y_F)
        log_adUya = np.log(alpha) + log_deltaU + alpha * log_y
        log_factor = np.logaddexp(0.0, log_adUya)
        log_c = log_y - log_factor / alpha
        c_F = np.where(y_F > 0, np.exp(log_c), 0.0)
    else:
        # η < 1: direct computation is numerically stable (1-η > 0, no overflow)
        # c = [y^(1-η) - (1-η)·deltaU]^(1/(1-η))
        # Safe to materialize deltaU here: small tax_exponent → moderate deltaU
        deltaU = np.exp(log_deltaU)
        inner = y_F ** (1.0 - eta) - (1.0 - eta) * deltaU
        # Allow inner = 0 (taxed to zero income) when deltaU exceeds max utility loss
        inner = np.maximum(inner, 0.0)
        c_F = inner ** (1.0 / (1.0 - eta))

    return float(c_F[0]) if scalar_input else c_F


def find_log_deltaU_equal_utility(
    y_gross,
    gini,
    Omega_calc,
    omega_yi_calc, xi_edges,
    eta,
    tax_amount,
    use_empirical_lorenz,
    tol,
    xi, wi,
):
    """
    Find log(deltaU_norm) for equal-utility-loss taxation.

    Returns log(deltaU_norm) in normalized income units. To recover post-tax income
    at raw income y, the caller must:

        c = compute_post_tax_income_equal_utility(y / y_damaged_mean, log_deltaU_norm, eta) * y_damaged_mean

    Returns log(deltaU) instead of deltaU itself to avoid float64 underflow at high
    tax exponents (>= ~290), where the true solution can be log_deltaU ≈ -1242.
    Converting back via exp() would give 0, silently disabling progressive taxation.

    With CRRA utility U(c) = c^(1-η)/(1-η):
    - Post-tax income: c_norm(F) = [y_norm(F)^(1-η) - (1-η)·deltaU_norm]^(1/(1-η))
    - Revenue constraint (in raw units): ∫₀¹ (y(F) - c(F)) dF = tax_amount

    Special case η=1 (log utility):
    - c(F) = y(F) * exp(-deltaU)  (scale-invariant)
    - Degenerates to uniform tax rate r = tax_amount / mean_income.

    Parameters
    ----------
    y_gross : float
        Gross income per capita before damage
    gini : float
        Gini coefficient
    Omega_calc : float
        Aggregate climate damage fraction
    omega_yi_calc : ndarray
        Damage fractions at quadrature points
    xi_edges : ndarray
        Standard quadrature edges in xi space [-1, 1]
    eta : float
        Coefficient of relative risk aversion (or eta_effective for the continuous tax knob)
    tax_amount : float
        Target per-capita tax revenue (in raw income units)
    use_empirical_lorenz : bool
        If True, use Empirical Lorenz; if False, use Pareto
    tol : float
        Tolerance for root finding (in log-deltaU space)
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1]
    wi : ndarray
        Gauss-Legendre quadrature weights

    Returns
    -------
    float
        log(deltaU_norm), the log of the normalized utility-loss constant.
        Returns LOG_DELTAU_NO_TAX (= -inf) when no tax is needed.
    """
    # If tax_amount is negligible, return no-tax sentinel
    if tax_amount < EPSILON:
        return LOG_DELTAU_NO_TAX

    # Mean damaged income (what we're taxing from)
    y_damaged_mean = y_gross * (1.0 - Omega_calc)

    # Special case η=1 (log utility): degenerates to uniform tax rate
    # c = y * exp(-deltaU), so tax rate r = 1 - exp(-deltaU)
    # Total tax = r * y_mean = tax_amount
    # => r = tax_amount / y_mean
    # => deltaU = -ln(1 - r)
    # => log_deltaU = log(-ln(1 - r))
    if abs(eta - 1.0) < EPSILON:
        r = tax_amount / y_damaged_mean
        # Cap r to avoid log(0) or negative
        r = min(r, 1.0 - EPSILON)
        deltaU = -np.log(1.0 - r)
        return np.log(deltaU)

    # Map quadrature from [-1, 1] to [0, 1] in F space
    Fi = (xi + 1.0) / 2.0
    Fwi = wi / 2.0

    # Get pre-tax income y(F) = y_gross * dL/dF * (1 - omega)
    if use_empirical_lorenz:
        dLdF = L_empirical_lorenz_derivative(Fi, gini)
    else:
        dLdF = L_pareto_derivative(Fi, gini)

    omega_Fi = _get_omega_at_F(Fi, omega_yi_calc, xi_edges)
    y_F = y_gross * dLdF * (1.0 - omega_Fi)

    # Normalize incomes so y_F_norm is O(1). We return log_deltaU_norm directly;
    # the caller is responsible for normalizing y by y_damaged_mean and rescaling c.
    y_F_norm = y_F / y_damaged_mean
    tax_amount_norm = tax_amount / y_damaged_mean

    def tax_revenue_minus_target(log_dU):
        # Solve in log-deltaU space: pass log_dU directly to the post-tax function.
        # No exp() conversion — this is the key fix that avoids underflow.
        c_F_norm = compute_post_tax_income_equal_utility(y_F_norm, log_dU, eta)
        return np.sum(Fwi * (y_F_norm - c_F_norm)) - tax_amount_norm

    # Dynamic bracket in log-deltaU space.
    # Lower bound: at high tax_exponent α, the tax formula involves y^α, so
    # log_deltaU scales as -α·log(y_max). Use a dynamic lower bound that
    # accounts for the actual exponent magnitude.
    alpha = abs(eta - 1.0)
    with np.errstate(divide='ignore'):
        log_y_max = np.log(np.max(y_F_norm))
    log_dU_lo = -abs(alpha * log_y_max) - 50.0
    log_dU_hi = np.log(1e10)

    f_lo = tax_revenue_minus_target(log_dU_lo)
    if f_lo >= 0:
        # Target already met at negligible tax — no tax needed
        return LOG_DELTAU_NO_TAX

    f_hi = tax_revenue_minus_target(log_dU_hi)
    if f_hi < 0:
        raise ValueError(
            f"Cannot raise required tax_amount={tax_amount:.6g} with equal utility loss taxation. "
            f"Maximum possible revenue (taxing everyone to zero) is "
            f"{(tax_amount_norm + f_hi) * y_damaged_mean:.6g}."
        )

    sol = root_scalar(
        tax_revenue_minus_target,
        bracket=[log_dU_lo, log_dU_hi],
        method="brentq",
        xtol=tol,
    )
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_log_deltaU_equal_utility")

    # sol.root IS log_deltaU_norm — return it directly, no exp() conversion
    log_deltaU_norm = sol.root

    # Post-solve invariant: c_F_norm must be finite and weakly monotone in F.
    c_F_norm_check = compute_post_tax_income_equal_utility(y_F_norm, log_deltaU_norm, eta)
    if not np.all(np.isfinite(c_F_norm_check)):
        bad = np.where(~np.isfinite(c_F_norm_check))[0]
        raise RuntimeError(
            f"find_log_deltaU_equal_utility: post-solve c_F_norm contains non-finite values at "
            f"indices {bad[:5].tolist()} (eta={eta:.4f}, log_deltaU_norm={log_deltaU_norm:.3g}, "
            f"tax_amount={tax_amount:.3g})."
        )
    diffs = np.diff(c_F_norm_check)
    monotone_tol = EPSILON * 1e3  # in normalized units, c_norm is O(1)
    if not np.all(diffs >= -monotone_tol):
        bad = np.where(diffs < -monotone_tol)[0]
        raise RuntimeError(
            f"find_log_deltaU_equal_utility: post-solve c_F_norm not monotone in F at indices "
            f"{bad[:5].tolist()} (min diff = {diffs.min():.3g}, tol = {monotone_tol:.3g}, "
            f"eta={eta:.4f}, log_deltaU_norm={log_deltaU_norm:.3g})."
        )

    return log_deltaU_norm


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
