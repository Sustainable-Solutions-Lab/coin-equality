#!/usr/bin/env python3
"""
Test sinh spacing with fraction-based parameters.

Example: 25 points with
- 25% of points in first 10% of time (0-40 years)
- 50% of points before 100 years
- 25% of points in last 10% of time (360-400 years)
"""

import numpy as np
from scipy.optimize import brentq


def sinh_spacing_from_fractions(n_points, t_start, t_end,
                                 frac_first_10pct, frac_last_10pct,
                                 frac_before_100, dt):
    """
    Generate smooth spacing using sinh transformation calibrated to target fractions.

    Cold start handling: First interior point is fixed at t=dt, then remaining
    points are distributed using sinh transformation.

    Parameters:
    - frac_first_10pct: Fraction of points in [0, 0.1*t_end]
    - frac_last_10pct: Fraction of points in [0.9*t_end, t_end]
    - frac_before_100: Fraction of points in [0, 100]
    """
    # Define key time boundaries
    t_10pct = t_start + 0.1 * (t_end - t_start)  # 40 years
    t_90pct = t_start + 0.9 * (t_end - t_start)  # 360 years
    t_100 = 100.0  # Explicit 100-year boundary

    # Calculate target number of points in each region (for all n_points including fixed ones)
    n_target_first_10_total = round(n_points * frac_first_10pct)
    n_target_last_10_total = round(n_points * frac_last_10pct)
    n_target_before_100_total = round(n_points * frac_before_100)

    # Account for fixed points: t=0 and t=dt are already placed
    # Point at t=0 is in first 10% and before 100
    # Point at t=dt is in first 10% and before 100 (assuming dt < 40 and dt < 100)
    n_fixed_in_first_10 = 2 if dt <= t_10pct else 1
    n_fixed_before_100 = 2 if dt <= t_100 else 1

    # Adjust targets for the points we need to place with sinh (excluding t=0 and t=dt)
    n_remaining = n_points - 2  # Exclude t=0 and t=dt
    n_target_first_10 = max(0, n_target_first_10_total - n_fixed_in_first_10)
    n_target_before_100 = max(0, n_target_before_100_total - n_fixed_before_100)
    n_target_last_10 = n_target_last_10_total  # Not affected by dt

    # Generate uniform parameter u ∈ [0, 1] for the remaining points
    u = np.linspace(0, 1, n_remaining)

    # Find u values that correspond to target boundaries (for remaining points)
    u_at_10pct = u[n_target_first_10 - 1] if n_target_first_10 > 0 else 0.1
    u_at_100 = u[n_target_before_100 - 1] if n_target_before_100 > 0 else 0.25
    u_at_90pct = u[n_remaining - n_target_last_10] if n_target_last_10 > 0 else 0.9

    def solve_sinh_alpha(u_target, t_target_frac):
        """Solve for alpha such that sinh(alpha * u_target) / sinh(alpha) = t_target_frac"""
        def equation(alpha):
            if alpha < 1e-6:
                return u_target - t_target_frac  # Linear limit
            return np.sinh(alpha * u_target) / np.sinh(alpha) - t_target_frac

        try:
            alpha = brentq(equation, 0.01, 20.0)
        except ValueError:
            alpha = 2.0  # Default moderate concentration
        return alpha

    # Use piecewise sinh for points in (dt+dt, t_end]:
    # After cold start at t=dt, next point can be at earliest t=dt+dt
    # Early region [2*dt, 100]: concentrated start
    # Late region [100, 400]: concentrated end

    # Adjust time boundaries relative to 2*dt (the earliest time after cold start)
    t_sinh_start = dt + dt  # First point after cold start must be at least dt away
    t_range = t_end - t_sinh_start
    t_10pct_rel = (t_10pct - t_sinh_start) / t_range if t_10pct > t_sinh_start else 0
    t_100_rel = (t_100 - t_sinh_start) / t_range if t_100 > t_sinh_start else 0
    t_90pct_rel = (t_90pct - t_sinh_start) / t_range

    # Solve for alpha (early concentration)
    if n_target_first_10 > 0 and n_target_before_100 > 0:
        alpha = solve_sinh_alpha(u_at_10pct / u_at_100, t_10pct_rel / t_100_rel)
    else:
        alpha = 2.0

    # Solve for beta (late concentration)
    if n_target_last_10 > 0:
        beta = solve_sinh_alpha((1 - u_at_90pct) / (1 - u_at_100),
                               (1 - t_90pct_rel) / (1 - t_100_rel))
    else:
        beta = 2.0

    # Generate times for remaining points on interval [t_sinh_start, t_end] = [2*dt, 400]
    times_sinh = np.zeros(n_remaining)

    for i, u_val in enumerate(u):
        if u_val <= u_at_100:
            # Early region: sinh transformation concentrated at start
            u_normalized = u_val / u_at_100  # Normalize to [0, 1]
            times_sinh[i] = t_sinh_start + t_100_rel * t_range * (np.sinh(alpha * u_normalized) / np.sinh(alpha))
        else:
            # Late region: sinh transformation concentrated at end
            u_normalized = (u_val - u_at_100) / (1 - u_at_100)  # Normalize to [0, 1]
            # Reverse sinh for end concentration
            times_sinh[i] = t_end - (1 - t_100_rel) * t_range * (np.sinh(beta * (1 - u_normalized)) / np.sinh(beta))

    # Combine: [t_start, dt] + sinh_times
    times = np.zeros(n_points)
    times[0] = t_start
    times[1] = dt
    times[2:] = times_sinh

    # Enforce minimum spacing for sinh points
    # Note: spacing between times[0] and times[1] is already dt by construction
    # Spacing between times[1] and times[2] should be >= dt since sinh starts at 2*dt
    for i in range(2, n_points - 1):
        if times[i] < times[i-1] + dt:
            times[i] = times[i-1] + dt

    times[-1] = t_end

    return times, alpha, beta


# Test parameters
t_start = 0
t_end = 400
dt = 1.0

frac_first_10pct = 0.375  # 37.5% in [0, 40]
frac_before_100 = 0.75    # 75% in [0, 100]
frac_last_10pct = 0.125   # 12.5% in [360, 400]

# Test multiple point counts
test_cases = [5, 10, 16, 25]

for n_points in test_cases:
    print("="*80)
    print(f"SINH SPACING: {n_points} POINTS (COLD START)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of points: {n_points}")
    print(f"  Time range: [{t_start}, {t_end}] years")
    print(f"  Minimum spacing (dt): {dt} year")
    print(f"  Cold start: Point 2 fixed at t={dt} year")
    print(f"\nTarget fractions (for all {n_points} points including fixed cold start):")
    print(f"  First 10% (0-40 years):    {frac_first_10pct:.3f} → {round(n_points * frac_first_10pct)} points")
    print(f"  Before 100 years (0-100):  {frac_before_100:.3f} → {round(n_points * frac_before_100)} points")
    print(f"  Last 10% (360-400 years):  {frac_last_10pct:.3f} → {round(n_points * frac_last_10pct)} points")

    # Generate spacing
    times, alpha, beta = sinh_spacing_from_fractions(
        n_points, t_start, t_end,
        frac_first_10pct, frac_last_10pct, frac_before_100, dt
    )

    # Calculate actual distributions
    n_actual_first_10 = np.sum(times <= 40)
    n_actual_before_100 = np.sum(times <= 100)
    n_actual_last_10 = np.sum(times >= 360)

    print(f"\n{'─'*80}")
    print("ACTUAL RESULTS")
    print(f"{'─'*80}")
    print(f"\nSolved parameters:")
    print(f"  Alpha (early concentration): {alpha:.4f}")
    print(f"  Beta (late concentration):   {beta:.4f}")

    print(f"\nActual point distribution:")
    print(f"  First 10% (0-40 years):    {n_actual_first_10} points ({100*n_actual_first_10/n_points:.1f}%)")
    print(f"  Before 100 years (0-100):  {n_actual_before_100} points ({100*n_actual_before_100/n_points:.1f}%)")
    print(f"  Last 10% (360-400 years):  {n_actual_last_10} points ({100*n_actual_last_10/n_points:.1f}%)")

    print(f"\n{'─'*80}")
    print("CONTROL POINT TIMES")
    print(f"{'─'*80}")

    spacing = np.diff(times)

    print(f"\n{'Point':<8} {'Time (yr)':<12} {'Spacing (yr)':<15} {'Region'}")
    print(f"{'-'*8} {'-'*12} {'-'*15} {'-'*25}")

    for i in range(n_points):
        if i == 0:
            spacing_str = "—"
        else:
            spacing_str = f"{spacing[i-1]:.2f}"

        if times[i] <= 40:
            region = "First 10% (0-40)"
        elif times[i] <= 100:
            region = "Early (40-100)"
        elif times[i] <= 360:
            region = "Middle (100-360)"
        else:
            region = "Last 10% (360-400)"

        marker = ""
        if i == n_actual_first_10 - 1:
            marker = "  ← 40 years"
        elif i == n_actual_before_100 - 1:
            marker = "  ← 100 years"
        elif i == n_points - n_actual_last_10:
            marker = "  ← 360 years"

        print(f"{i+1:<8} {times[i]:<12.2f} {spacing_str:<15} {region}{marker}")

    # Spacing statistics
    print(f"\n{'─'*80}")
    print("SPACING STATISTICS")
    print(f"{'─'*80}")
    print(f"\nSpacing between points:")
    print(f"  Minimum: {np.min(spacing):.2f} years")
    print(f"  Maximum: {np.max(spacing):.2f} years")
    print(f"  Mean:    {np.mean(spacing):.2f} years")
    print(f"  Median:  {np.median(spacing):.2f} years")

    print(f"\n{'='*80}\n")
