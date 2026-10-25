"""
Income distribution functions for the quadrature over population rank F in [0, 1].

This module provides:
- Pareto-Lorenz and empirical Lorenz curves and their derivatives
- The Gauss-Legendre quadrature grid over population rank
- Equal-utility-loss progressive taxation (closed-form schedule and revenue root solve)
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import roots_legendre
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
    TAX_LOG_DELTAU_BRACKET_MARGIN,
    TAX_DELTAU_MAX,
    TAX_MONOTONICITY_TOLERANCE,
)


#========================================================================================
# Quadrature over population rank
#========================================================================================

@dataclass(frozen=True)
class QuadratureGrid:
    """
    Gauss-Legendre quadrature over population rank F in [0, 1].

    Node i represents the population interval [Fi_edges[i], Fi_edges[i+1]), whose width is
    its weight Fwi[i]; each node lies strictly inside its interval (Chebyshev-Markov-Stieltjes
    separation), so values at the nodes are also the stepwise values on these intervals.
    """
    xi: np.ndarray        # nodes on [-1, 1]
    wi: np.ndarray        # weights on [-1, 1] (sum to 2)
    xi_edges: np.ndarray  # interval edges on [-1, 1]: -1 + cumulative weights (length n + 1)
    Fi: np.ndarray        # nodes mapped to [0, 1]
    Fwi: np.ndarray       # weights on [0, 1] (sum to 1)
    Fi_edges: np.ndarray  # interval edges on [0, 1]


def quadrature_grid(n_quad):
    """Gauss-Legendre quadrature grid with n_quad nodes over population rank."""
    xi, wi = roots_legendre(n_quad)
    xi_edges = np.concatenate(([-1.0], -1.0 + np.cumsum(wi)))
    return QuadratureGrid(xi=xi, wi=wi, xi_edges=xi_edges,
                          Fi=(xi + 1.0) / 2.0, Fwi=wi / 2.0, Fi_edges=(xi_edges + 1.0) / 2.0)


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


def find_log_deltaU_equal_utility(y_pretax_norm, tax_amount_norm, tax_exponent, Fwi, tol):
    """
    Solve for log(deltaU) of the equal-utility-loss tax schedule that raises the revenue.

    Works in normalized income units (incomes divided by mean damaged income): post-tax
    income is c_norm = compute_post_tax_income_equal_utility(y_pretax_norm, log_deltaU,
    tax_exponent) and the revenue constraint is sum_i Fwi_i (y_pretax_norm_i - c_norm_i)
    = tax_amount_norm.

    Returns log(deltaU) rather than deltaU: at high tax exponents (>= ~290) the solution
    can be log_deltaU ~ -1242, which exp() would underflow to 0, silently disabling
    progressive taxation.

    With tax_exponent = 1 (log utility) the schedule c = y exp(-deltaU) is a uniform tax
    rate r = tax_amount_norm, so deltaU = -ln(1 - r) in closed form.

    Parameters
    ----------
    y_pretax_norm : ndarray
        Pre-tax income at the quadrature nodes, normalized by mean damaged income
    tax_amount_norm : float
        Required per-capita revenue, normalized by mean damaged income
    tax_exponent : float
        Exponent of the pseudo-utility, eta_eff = 1 + tax_equity/(1-tax_equity) * (eta - 1)
    Fwi : ndarray
        Quadrature weights over population rank (sum to 1)
    tol : float
        Tolerance of the root solve in log-deltaU

    Returns
    -------
    float
        log(deltaU) in normalized units; LOG_DELTAU_NO_TAX (= -inf) when a negligible
        tax already meets the target.
    """
    if abs(tax_exponent - 1.0) < EPSILON:
        # Cap r to avoid log(0) or negative
        r = min(tax_amount_norm, 1.0 - EPSILON)
        deltaU = -np.log(1.0 - r)
        return np.log(deltaU)

    def tax_revenue_minus_target(log_dU):
        # Solve in log-deltaU space: pass log_dU directly to the post-tax function.
        # No exp() conversion — this is the key fix that avoids underflow.
        c_F_norm = compute_post_tax_income_equal_utility(y_pretax_norm, log_dU, tax_exponent)
        return np.sum(Fwi * (y_pretax_norm - c_F_norm)) - tax_amount_norm

    # Dynamic bracket in log-deltaU space.
    # Lower bound: at high tax_exponent α, the tax formula involves y^α, so
    # log_deltaU scales as -α·log(y_max). Use a dynamic lower bound that
    # accounts for the actual exponent magnitude.
    alpha = abs(tax_exponent - 1.0)
    with np.errstate(divide='ignore'):
        log_y_max = np.log(np.max(y_pretax_norm))
    log_dU_lo = -abs(alpha * log_y_max) - TAX_LOG_DELTAU_BRACKET_MARGIN
    log_dU_hi = np.log(TAX_DELTAU_MAX)

    f_lo = tax_revenue_minus_target(log_dU_lo)
    if f_lo >= 0:
        # Target already met at negligible tax — no tax needed
        return LOG_DELTAU_NO_TAX

    f_hi = tax_revenue_minus_target(log_dU_hi)
    if f_hi < 0:
        raise ValueError(
            f"Cannot raise the required tax ({tax_amount_norm:.6g} of mean damaged income) with "
            f"equal utility loss taxation. Maximum possible revenue (taxing everyone to zero) is "
            f"{tax_amount_norm + f_hi:.6g} of mean damaged income."
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
    c_F_norm_check = compute_post_tax_income_equal_utility(y_pretax_norm, log_deltaU_norm, tax_exponent)
    if not np.all(np.isfinite(c_F_norm_check)):
        bad = np.where(~np.isfinite(c_F_norm_check))[0]
        raise RuntimeError(
            f"find_log_deltaU_equal_utility: post-solve c_F_norm contains non-finite values at "
            f"indices {bad[:5].tolist()} (tax_exponent={tax_exponent:.4f}, log_deltaU_norm={log_deltaU_norm:.3g}, "
            f"tax_amount_norm={tax_amount_norm:.3g})."
        )
    diffs = np.diff(c_F_norm_check)
    monotone_tol = TAX_MONOTONICITY_TOLERANCE
    if not np.all(diffs >= -monotone_tol):
        bad = np.where(diffs < -monotone_tol)[0]
        raise RuntimeError(
            f"find_log_deltaU_equal_utility: post-solve c_F_norm not monotone in F at indices "
            f"{bad[:5].tolist()} (min diff = {diffs.min():.3g}, tol = {monotone_tol:.3g}, "
            f"tax_exponent={tax_exponent:.4f}, log_deltaU_norm={log_deltaU_norm:.3g})."
        )

    return log_deltaU_norm
