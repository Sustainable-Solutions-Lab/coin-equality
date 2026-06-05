"""
Numerical constants and tolerances for COIN_equality model.

Defines small and large values used for numerical stability and bounds checking.
Centralizes epsilon/bignum definitions to ensure consistency across the codebase.
"""

# Large negative number for utility when constraints are violated
# Used as penalty value when Gini or other variables are out of valid range
NEG_BIGNUM = -1e30

# Small epsilon for numerical comparisons and bounds
# Used for:
# - Comparing floats to unity (e.g., eta ≈ 1)
# - Checking if values are effectively zero
# - Bounding variables away from exact 0 or 1 (e.g., Gini ∈ (ε, 1-ε))
# - Ensuring values stay strictly positive (e.g., A2 ≥ ε)
# - Root finding bracket offsets
EPSILON = 1e-12

# Objective function scaling factor for numerical stability in gradient-based optimization
# Used for:
# - Scaling objective values from ~1.5e13 to ~1.5 for better numerical conditioning
# - Applied consistently to both objective values and gradients in all optimization wrappers
# - Improves stability of gradient-based algorithms (LD_SLSQP, LD_LBFGS, etc.)
OBJECTIVE_SCALE = 1e-13

# Large value for detecting effectively infinite parameters
# Used for:
# - Checking if y_damage_distribution_exponent is so small that damage is effectively uniform
# - Detecting when parameters should trigger special case handling
INVERSE_EPSILON = 1.0 / EPSILON

# Maximum iterations for convergence loops
# Used for:
# - Initial capital stock convergence in integrate_model()
# - Climate damage convergence in calculate_tendencies()
# Set to 256 to allow slow but steady convergence during optimization
MAX_ITERATIONS = 1024

# N_QUAD removed - now specified in config.integration_params.n_quad
# This enforces explicit configuration (no defaults per CLAUDE.md)

# Empirical Lorenz curve base parameters
# The base empirical Lorenz curve is: L_base(F) = w₀·F^p₀ + w₁·F^p₁ + w₂·F^p₂ + w₃·F^p₃
# where w₀ = 1 - w₁ - w₂ - w₃
# For arbitrary Gini G: L(F) = (1 - G/Gini_base)·F + (G/Gini_base)·L_base(F)

# Power parameters for each term in the base Lorenz curve
EMPIRICAL_LORENZ_P0 = 1.500036
EMPIRICAL_LORENZ_P1 = 4.367440
EMPIRICAL_LORENZ_P2 = 14.072005
EMPIRICAL_LORENZ_P3 = 135.059674

# Weight parameters for terms 1, 2, and 3 (w₀ is computed as 1 - w₁ - w₂ - w₃)
EMPIRICAL_LORENZ_W1 = 3.776187268483524e-01
EMPIRICAL_LORENZ_W2 = 3.671247620949191e-01
EMPIRICAL_LORENZ_W3 = 9.538538350961864e-02

# Derived constants (computed once at module import time for performance)
# w₀ is the weight for the first term, derived from the constraint that weights sum to 1
EMPIRICAL_LORENZ_W0 = 1.0 - EMPIRICAL_LORENZ_W1 - EMPIRICAL_LORENZ_W2 - EMPIRICAL_LORENZ_W3

# Base Gini coefficient for the empirical Lorenz curve
# Gini_base = 1 - 2·[w₀/(p₀+1) + w₁/(p₁+1) + w₂/(p₂+1) + w₃/(p₃+1)]
EMPIRICAL_LORENZ_BASE_GINI = 1.0 - 2.0 * (
    EMPIRICAL_LORENZ_W0 / (EMPIRICAL_LORENZ_P0 + 1.0) +
    EMPIRICAL_LORENZ_W1 / (EMPIRICAL_LORENZ_P1 + 1.0) +
    EMPIRICAL_LORENZ_W2 / (EMPIRICAL_LORENZ_P2 + 1.0) +
    EMPIRICAL_LORENZ_W3 / (EMPIRICAL_LORENZ_P3 + 1.0)
)

# Maximum allowed base damage fraction from temperature
# Caps Omega_base to prevent economic collapse at extreme temperatures
# At Omega_base = 0.8, 80% of GDP is destroyed by climate damage
OMEGA_BASE_MAX = 0.8

# Sentinel for "no progressive tax" in log-deltaU space
# Used by find_log_deltaU_equal_utility and compute_post_tax_income_equal_utility
# to signal that no taxation is applied (deltaU = 0, log(deltaU) = -inf)
LOG_DELTAU_NO_TAX = float('-inf')

# Temperature impulse response parameters (Caldeira & Myhrvold 2013, Table 2, median)
# doi:10.1088/1748-9326/9/12/124002
# Functional form: ΔT(t) = -(a1+a2+a3) + a1·exp(-t/τ1) + a2·exp(-t/τ2) + a3·exp(-t/τ3)
# where t is years after a pulse emission and ΔT is in mK per GtC emitted.
# Asymptotic (equilibrium) TCRE = -(a1+a2+a3) = 1.756 mK/GtC
IMPULSE_RESPONSE_A1 = -2.308       # mK/GtC (fast mode, τ ~ 2 yr)
IMPULSE_RESPONSE_A2 = 0.743        # mK/GtC (medium mode, τ ~ 36 yr)
IMPULSE_RESPONSE_A3 = -0.191       # mK/GtC (slow mode, τ ~ 97 yr)
IMPULSE_RESPONSE_TAU1 = 2.241      # years
IMPULSE_RESPONSE_TAU2 = 35.750     # years
IMPULSE_RESPONSE_TAU3 = 97.180     # years

# Unit conversion: mK/GtC → °C/tCO₂
# 1 mK = 1e-3 °C; 1 GtC = 3.6667e9 tCO₂ (molar mass ratio 44/12)
MK_PER_GTC_TO_C_PER_TCO2 = 1e-3 / 3.6667e9
