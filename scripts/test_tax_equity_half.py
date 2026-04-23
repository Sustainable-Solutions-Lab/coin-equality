"""
Property tests for the continuous tax schedule.

1. At tax_equity = 0.5, the tax schedule coincides with equal utility loss in
   the real CRRA utility: U(y) - U(c) = K for every income level y, where K is
   a single scalar.

2. At tax_equity = 0, the effective tax rate is constant across incomes (flat
   fractional rate), so c / y is a single scalar.

Both tests are run against the direct find_K / compute_post_tax_income pipeline
with a meaningful tax load, so that floating-point catastrophic cancellation in
U(y) - U(c) does not dominate the measured spread.

Run: python scripts/test_tax_equity_half.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.special import roots_legendre

from src.distribution_utilities import (
    find_K_equal_utility,
    compute_post_tax_income_equal_utility,
    L_empirical_lorenz_derivative,
    _get_omega_at_F,
)


def crra_utility(c, eta):
    """Real CRRA utility U(c) = c^(1-eta)/(1-eta) for eta != 1, ln(c) for eta = 1."""
    c = np.maximum(c, 1e-300)
    if abs(eta - 1.0) < 1e-12:
        return np.log(c)
    return c ** (1.0 - eta) / (1.0 - eta)


def build_lorenz_incomes(y_gross, gini, n_quad=32):
    """Build a quadrature grid and the corresponding Lorenz-curve incomes."""
    xi, wi = roots_legendre(n_quad)
    xi_edges = np.concatenate(([-1.0], -1.0 + np.cumsum(wi)))
    Fi = (xi + 1.0) / 2.0
    Fwi = wi / 2.0
    dLdF = L_empirical_lorenz_derivative(Fi, gini)
    omega = np.zeros(n_quad)  # ignore damage for pure tax-schedule test
    y_F = y_gross * dLdF
    return xi, wi, xi_edges, Fi, Fwi, omega, y_F


def test_equal_utility_loss_at_half():
    """tax_equity = 0.5, eta > 1: utility loss is constant across incomes."""
    eta = 2.0
    tax_equity = 0.5
    eta_eff = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)
    assert abs(eta_eff - eta) < 1e-14, f"Expected eta_eff == eta, got {eta_eff}"

    y_gross = 1.0e5
    gini = 0.5
    xi, wi, xi_edges, Fi, Fwi, omega, y_F = build_lorenz_incomes(y_gross, gini)

    # Use a visible tax load (~20% of mean income) so spread is not dominated
    # by floating-point cancellation in U(y) - U(c).
    tax_amount = 0.20 * y_gross

    K = find_K_equal_utility(
        y_gross=y_gross, gini=gini, Omega_calc=0.0,
        omega_Fmin_calc=omega, omega_yi_calc=omega, omega_Fmax_calc=omega,
        Fmin_prev=0.0, Fmax_prev=1.0, xi_edges=xi_edges,
        eta=eta_eff, tax_amount=tax_amount, use_empirical_lorenz=True,
        tol=1e-12, xi=xi, wi=wi,
    )
    c_F = compute_post_tax_income_equal_utility(y_F, K, eta_eff)
    revenue = float(np.sum(Fwi * (y_F - c_F)))
    assert abs(revenue - tax_amount) / tax_amount < 1e-9, \
        f"Revenue mismatch: {revenue} vs {tax_amount}"

    # Real-CRRA utility loss, using the REAL eta (not eta_eff).
    utility_loss = crra_utility(y_F, eta) - crra_utility(c_F, eta)
    spread = float(utility_loss.max() - utility_loss.min())
    mean_loss = float(utility_loss.mean())
    relative_spread = spread / abs(mean_loss)

    print(f"Test 1: tax_equity = 0.5, eta = {eta}")
    print(f"  eta_eff          = {eta_eff}")
    print(f"  tax_amount       = {tax_amount:.3f}")
    print(f"  K (raw)          = {K:.6e}")
    print(f"  utility loss min = {utility_loss.min():.6e}")
    print(f"  utility loss max = {utility_loss.max():.6e}")
    print(f"  mean loss        = {mean_loss:.6e}")
    print(f"  spread           = {spread:.3e}")
    print(f"  relative spread  = {relative_spread:.3e}")

    assert relative_spread < 1e-9, \
        f"Expected constant utility loss, got relative spread {relative_spread:.3e}"
    print("  PASS: real-CRRA utility loss is constant across incomes.")


def test_flat_rate_at_zero():
    """tax_equity = 0: c/y is constant (flat fractional rate)."""
    eta = 2.0
    tax_equity = 0.0
    eta_eff = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)
    assert abs(eta_eff - 1.0) < 1e-14, f"Expected eta_eff == 1 at tax_equity=0"

    y_gross = 1.0e5
    gini = 0.5
    xi, wi, xi_edges, Fi, Fwi, omega, y_F = build_lorenz_incomes(y_gross, gini)
    tax_amount = 0.20 * y_gross

    K = find_K_equal_utility(
        y_gross=y_gross, gini=gini, Omega_calc=0.0,
        omega_Fmin_calc=omega, omega_yi_calc=omega, omega_Fmax_calc=omega,
        Fmin_prev=0.0, Fmax_prev=1.0, xi_edges=xi_edges,
        eta=eta_eff, tax_amount=tax_amount, use_empirical_lorenz=True,
        tol=1e-12, xi=xi, wi=wi,
    )
    c_F = compute_post_tax_income_equal_utility(y_F, K, eta_eff)
    ratio = c_F / y_F
    spread = float(ratio.max() - ratio.min())
    mean_ratio = float(ratio.mean())

    print()
    print(f"Test 2: tax_equity = 0.0, eta = {eta}")
    print(f"  eta_eff          = {eta_eff}")
    print(f"  tax_amount       = {tax_amount:.3f}")
    print(f"  K (raw)          = {K:.6e}")
    print(f"  c/y min          = {ratio.min():.10f}")
    print(f"  c/y max          = {ratio.max():.10f}")
    print(f"  mean c/y         = {mean_ratio:.10f}")
    print(f"  spread           = {spread:.3e}")

    assert spread < 1e-12, \
        f"Expected flat c/y, got spread {spread:.3e}"
    print("  PASS: c/y is constant across incomes (flat fractional rate).")


def test_monotonicity_at_high_equity():
    """tax_equity = 0.95 (near tax-richest): c_F still monotone in y_F."""
    eta = 1.45
    tax_equity = 0.95
    eta_eff = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)

    y_gross = 1.0e5
    gini = 0.5
    xi, wi, xi_edges, Fi, Fwi, omega, y_F = build_lorenz_incomes(y_gross, gini)
    tax_amount = 0.05 * y_gross

    K = find_K_equal_utility(
        y_gross=y_gross, gini=gini, Omega_calc=0.0,
        omega_Fmin_calc=omega, omega_yi_calc=omega, omega_Fmax_calc=omega,
        Fmin_prev=0.0, Fmax_prev=1.0, xi_edges=xi_edges,
        eta=eta_eff, tax_amount=tax_amount, use_empirical_lorenz=True,
        tol=1e-12, xi=xi, wi=wi,
    )
    c_F = compute_post_tax_income_equal_utility(y_F, K, eta_eff)

    monotone = bool(np.all(np.diff(c_F) >= 0.0))
    bounded = bool(np.all(c_F > 0.0) and np.all(c_F < y_F + 1e-9))

    print()
    print(f"Test 3: tax_equity = 0.95, eta = {eta}  (eta_eff = {eta_eff:.3f})")
    print(f"  c_F monotone in F  = {monotone}")
    print(f"  c_F in (0, y_F]    = {bounded}")
    print(f"  c_F range          = [{c_F.min():.2f}, {c_F.max():.2f}]")
    print(f"  y_F range          = [{y_F.min():.2f}, {y_F.max():.2f}]")

    assert monotone, "c_F must be monotone in F"
    assert bounded, "c_F must lie in (0, y_F]"
    print("  PASS: schedule is monotone and bounded even at near-u-max knob.")


def main():
    test_equal_utility_loss_at_half()
    test_flat_rate_at_zero()
    test_monotonicity_at_high_equity()
    print()
    print("All tests passed.")


if __name__ == '__main__':
    main()
