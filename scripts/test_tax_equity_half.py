"""
Property tests for the continuous tax schedule.

After the end-to-end normalization refactor, find_K_equal_utility returns the
normalized constant K_norm. Post-tax income c at raw income y is recovered via:

    c = compute_post_tax_income_equal_utility(y / y_damaged_mean, K_norm, eta) * y_damaged_mean

These tests exercise the property that the *math* is unchanged — only the
storage units shifted — and confirm that the high-tax_equity regime that
previously hit floating-point underflow now works.

1. tax_equity = 0.5 ⇒ U(y) − U(c) = K, constant across incomes (real-CRRA
   equal-utility-loss).
2. tax_equity = 0.0 ⇒ c/y is constant (flat fractional rate).
3. tax_equity = 0.95, η = 1.45 ⇒ schedule monotone and bounded in (0, y].
4. tax_equity = 0.99, η = 2.0 (η_eff = 100) ⇒ schedule monotone and bounded —
   this case used to hit silent zero-tax under the old un-normalized formula.

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
    omega = np.zeros(n_quad)
    y_F = y_gross * dLdF
    return xi, wi, xi_edges, Fi, Fwi, omega, y_F


def solve_and_evaluate(y_gross, gini, eta_eff, tax_amount):
    """
    Run find_K_equal_utility and recover c at the raw quadrature incomes via the
    normalized API. Returns (y_F, c_F, K_norm, revenue).
    """
    xi, wi, xi_edges, Fi, Fwi, omega, y_F = build_lorenz_incomes(y_gross, gini)
    y_damaged_mean = y_gross  # Omega_calc=0 in these tests
    K_norm = find_K_equal_utility(
        y_gross=y_gross, gini=gini, Omega_calc=0.0,
        omega_Fmin_calc=omega, omega_yi_calc=omega, omega_Fmax_calc=omega,
        Fmin_prev=0.0, Fmax_prev=1.0, xi_edges=xi_edges,
        eta=eta_eff, tax_amount=tax_amount, use_empirical_lorenz=True,
        tol=1e-12, xi=xi, wi=wi,
    )
    y_F_norm = y_F / y_damaged_mean
    c_F_norm = compute_post_tax_income_equal_utility(y_F_norm, K_norm, eta_eff)
    c_F = c_F_norm * y_damaged_mean
    revenue = float(np.sum(Fwi * (y_F - c_F)))
    return y_F, c_F, K_norm, revenue, Fwi


def test_equal_utility_loss_at_half():
    """tax_equity = 0.5, eta > 1: utility loss is constant across incomes."""
    eta = 2.0
    tax_equity = 0.5
    eta_eff = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)
    assert abs(eta_eff - eta) < 1e-14

    y_gross = 1.0e5
    tax_amount = 0.20 * y_gross
    y_F, c_F, K_norm, revenue, _ = solve_and_evaluate(y_gross, 0.5, eta_eff, tax_amount)

    assert abs(revenue - tax_amount) / tax_amount < 1e-9, \
        f"Revenue mismatch: {revenue} vs {tax_amount}"

    utility_loss = crra_utility(y_F, eta) - crra_utility(c_F, eta)
    spread = float(utility_loss.max() - utility_loss.min())
    mean_loss = float(utility_loss.mean())
    relative_spread = spread / abs(mean_loss)

    print(f"Test 1: tax_equity = 0.5, eta = {eta}")
    print(f"  eta_eff          = {eta_eff}")
    print(f"  K_norm           = {K_norm:.6e}")
    print(f"  utility loss     = {mean_loss:.6e}  (spread {spread:.3e}, rel {relative_spread:.3e})")
    assert relative_spread < 1e-9
    print("  PASS: real-CRRA utility loss is constant across incomes.")


def test_flat_rate_at_zero():
    """tax_equity = 0: c/y is constant (flat fractional rate)."""
    eta = 2.0
    tax_equity = 0.0
    eta_eff = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)
    assert abs(eta_eff - 1.0) < 1e-14

    y_gross = 1.0e5
    tax_amount = 0.20 * y_gross
    y_F, c_F, K_norm, revenue, _ = solve_and_evaluate(y_gross, 0.5, eta_eff, tax_amount)

    # eta=1 short-circuits to an analytic K from the *assumed* mean income, not from the
    # integrated quadrature, so revenue matches only to quadrature precision (~1e-6),
    # not Brent precision. The flat-rate property itself holds to machine precision.
    assert abs(revenue - tax_amount) / tax_amount < 1e-5
    ratio = c_F / y_F
    spread = float(ratio.max() - ratio.min())

    print()
    print(f"Test 2: tax_equity = 0.0, eta = {eta}")
    print(f"  eta_eff          = {eta_eff}")
    print(f"  K_norm           = {K_norm:.6e}")
    print(f"  c/y              = {ratio.mean():.10f}  (spread {spread:.3e})")
    assert spread < 1e-12
    print("  PASS: c/y is constant across incomes (flat fractional rate).")


def test_monotonicity_at_high_equity():
    """tax_equity = 0.95, eta = 1.45 (eta_eff ~ 9.55): monotone, bounded."""
    eta = 1.45
    tax_equity = 0.95
    eta_eff = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)

    y_gross = 1.0e5
    tax_amount = 0.05 * y_gross
    y_F, c_F, K_norm, revenue, _ = solve_and_evaluate(y_gross, 0.5, eta_eff, tax_amount)

    assert abs(revenue - tax_amount) / tax_amount < 1e-9
    monotone = bool(np.all(np.diff(c_F) >= 0.0))
    bounded = bool(np.all(c_F > 0.0) and np.all(c_F < y_F + 1e-9))

    print()
    print(f"Test 3: tax_equity = 0.95, eta = {eta}  (eta_eff = {eta_eff:.3f})")
    print(f"  K_norm           = {K_norm:.6e}")
    print(f"  c_F range        = [{c_F.min():.2f}, {c_F.max():.2f}]")
    print(f"  y_F range        = [{y_F.min():.2f}, {y_F.max():.2f}]")
    assert monotone and bounded
    print("  PASS: schedule is monotone and bounded.")


def test_extreme_tax_equity():
    """tax_equity = 0.99, eta = 2.0 (eta_eff = 100): regime that previously
    silent-zero-tax'd under the un-normalized back-conversion. Must now solve
    correctly and produce a meaningful tax schedule."""
    eta = 2.0
    tax_equity = 0.99
    eta_eff = 1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)
    assert abs(eta_eff - 100.0) < 1e-9

    y_gross = 1.0e5
    tax_amount = 0.05 * y_gross
    y_F, c_F, K_norm, revenue, Fwi = solve_and_evaluate(y_gross, 0.5, eta_eff, tax_amount)

    rev_err = abs(revenue - tax_amount) / tax_amount
    monotone = bool(np.all(np.diff(c_F) >= 0.0))
    bounded = bool(np.all(c_F > 0.0) and np.all(c_F < y_F + 1e-9))

    # u-max-shaped: low earners pay ~nothing (c ≈ y), high earners are taxed
    # toward a common asymptote ((eta_eff-1)*K_norm)^(-1/(eta_eff-1)) * y_damaged_mean.
    # The asymptote depends on the revenue target. At eta_eff=100 and 5% revenue
    # target the asymptote sits at ~5.8x mean income — high earners are not taxed
    # to zero, they're clipped to that ceiling. Post-tax income remains strictly
    # monotone in pre-tax income (verified above) — this is "u-max-shaped" but with
    # the threshold value set by the revenue calibration.
    bottom_avg_rate = 1.0 - c_F[0] / y_F[0]
    top_avg_rate = 1.0 - c_F[-1] / y_F[-1]

    print()
    print(f"Test 4: tax_equity = 0.99, eta = {eta}  (eta_eff = {eta_eff})")
    print(f"  K_norm           = {K_norm:.6e}")
    print(f"  revenue / target = {revenue:.6f} / {tax_amount:.6f}  (rel err {rev_err:.3e})")
    print(f"  c_F range        = [{c_F.min():.2f}, {c_F.max():.2f}]")
    print(f"  y_F range        = [{y_F.min():.2f}, {y_F.max():.2f}]")
    print(f"  bottom-quantile avg tax rate = {100*bottom_avg_rate:.4f}%  (should be ~0)")
    print(f"  top-quantile avg tax rate    = {100*top_avg_rate:.2f}%   (should be substantial)")
    assert rev_err < 1e-6, f"Revenue not within tolerance: rel err {rev_err}"
    assert monotone, "c_F must be monotone in F"
    assert bounded, "c_F must lie in (0, y_F]"
    assert bottom_avg_rate < 1e-6, \
        f"Bottom quantile should pay ~no tax at eta_eff=100, got rate {bottom_avg_rate:.3e}"
    assert top_avg_rate > 0.3, \
        f"Top quantile should bear substantial tax at eta_eff=100, got rate {top_avg_rate:.3e}"
    print("  PASS: schedule is monotone, bottom untaxed, top heavily taxed.")


def main():
    test_equal_utility_loss_at_half()
    test_flat_rate_at_zero()
    test_monotonicity_at_high_equity()
    test_extreme_tax_equity()
    print()
    print("All tests passed.")


if __name__ == '__main__':
    main()
