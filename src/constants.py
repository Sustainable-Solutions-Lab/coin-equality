"""
Numerical constants and tolerances for COIN_equality model.

Centralizes the model's fixed physical constants (empirical Lorenz curve, temperature
impulse response, initial climate state) and the numerical tolerances and bounds used by
the simulation and the optimizer.
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

# Scaling of the objective passed to NLopt (discounted utility is ~1e10), so the
# optimizer works with values of order 1e-3
OBJECTIVE_SCALE = 1e-13

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

# Temperature impulse response parameters (Ricke & Caldeira 2014, median)
# doi:10.1088/1748-9326/9/12/124002
# Functional form: g(t₀) = (R_inf + a1·exp(-t₀/τ1) + a2·exp(-t₀/τ2) + a3·exp(-t₀/τ3))
# where t₀ is years after a pulse emission and g is in °C/tCO₂.
# R_inf = -(a1+a2+a3) ≈ 0.479e-12 °C/tCO₂ (equilibrium TCRE)
IMPULSE_RESPONSE_A1 = -0.629e-12   # °C/tCO₂ (fast mode, τ ~ 2 yr)
IMPULSE_RESPONSE_A2 = 0.203e-12    # °C/tCO₂ (medium mode, τ ~ 36 yr)
IMPULSE_RESPONSE_A3 = -0.0521e-12  # °C/tCO₂ (slow mode, τ ~ 97 yr)
IMPULSE_RESPONSE_TAU1 = 2.241      # years
IMPULSE_RESPONSE_TAU2 = 35.750     # years
IMPULSE_RESPONSE_TAU3 = 97.180     # years

# Initial climate state (2020), from the manuscript's historical warming:
# ΔT_hist(t) = 1.1126 - 0.07022·e^((2020-t)/τ1) + 0.21036·e^((2020-t)/τ2) - 0.08308·e^((2020-t)/τ3)
# The constant term equals R_∞ × Ecum_initial, so Ecum_initial = 1.1126 / R_∞. The other
# terms are the heat-uptake mode temperatures in 2020, obtained by convolving an exponential
# emission history (633.5 GtC cumulative to 2020, from DICE-2023) with the impulse response.
DELTA_T_HIST_EQUILIBRIUM = 1.1126  # °C
ECUM_INITIAL = DELTA_T_HIST_EQUILIBRIUM / -(IMPULSE_RESPONSE_A1 + IMPULSE_RESPONSE_A2 + IMPULSE_RESPONSE_A3)
H1_INITIAL = -0.07022  # °C (fast mode)
H2_INITIAL = 0.21036   # °C (medium mode)
H3_INITIAL = -0.08308  # °C (slow mode)

# Control-point grid for the optimizer (see optimization.calculate_control_times)
# The last WIND_DOWN_EFOLDING_TIMES capital e-folding times (1/delta each) before t_end form
# the wind-down region, which holds one control point at t_end, or two (t_end - 1/delta and
# t_end) once the grid has more than MAX_POINTS_WITH_ONE_WIND_DOWN_POINT points. Before the
# wind-down region, spacing grows geometrically from dt; its growth rate is root-solved
# within GEOMETRIC_GROWTH_RATE_BRACKET.
WIND_DOWN_EFOLDING_TIMES = 4.0
MAX_POINTS_WITH_ONE_WIND_DOWN_POINT = 10
GEOMETRIC_GROWTH_RATE_BRACKET = (-0.5, 100.0)

# Control points closer in time than this are treated as duplicates (the first is kept)
# when building PCHIP control trajectories
CONTROL_TIME_DUPLICATE_TOLERANCE = 1e-10

# Root solve for the equal-utility-loss tax (distribution_utilities.find_log_deltaU_equal_utility)
# Bracket in log(deltaU), normalized income units: from -(|tax_exponent - 1| * |log(max income)|
# + TAX_LOG_DELTAU_BRACKET_MARGIN), where revenue is negligible, to log(TAX_DELTAU_MAX), where
# everyone is taxed to (nearly) zero
TAX_LOG_DELTAU_BRACKET_MARGIN = 50.0
TAX_DELTAU_MAX = 1e10
# Post-solve check: post-tax income (normalized, O(1)) may decrease with rank by at most this
TAX_MONOTONICITY_TOLERANCE = EPSILON * 1e3
