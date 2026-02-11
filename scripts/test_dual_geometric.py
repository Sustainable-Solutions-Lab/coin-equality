#!/usr/bin/env python3
"""
Test dual geometric spacing algorithm.

Two-sided geometric growth with automatic transition via min().
"""

import numpy as np


def dual_geometric_spacing(n_points, t_start, t_end, dt, delta,
                           frac_early=0.90, r0=0.15, r1=0.10):
    """
    Generate spacing using dual geometric progressions.

    Early intervals: dx0[i] = dt * (1 + r0)**i (EXACT, no scaling)
    Late intervals:  dx1[i] = base1 * (1 + r1)**(N - i - 2) (scaled to fit)
    Actual spacing:  dx[i] = min(dx0[i], dx1[i])

    Parameters:
    - n_points: Total number of points
    - t_start, t_end: Time range
    - dt: Minimum spacing (also base for early growth)
    - delta: Depreciation rate (for late spacing base)
    - frac_early: Fraction of points in early region (default 0.90)
    - r0: Growth rate for early region (default 0.15 = 15% growth)
    - r1: Growth rate for late region (default 0.10 = 10% growth)
    """
    n_intervals = n_points - 1

    # Early region: geometric growth from dt (EXACT - no scaling)
    dx0 = np.zeros(n_intervals)
    for i in range(n_intervals):
        dx0[i] = dt * (1 + r0)**i

    # Late region: geometric growth backward from end
    # We need to solve for base1 such that sum of min(dx0, dx1) = t_end - t_start

    # Initial guess for base1
    base1_initial = (1 - np.exp(-1)) / delta  # About 6.3 years for delta=0.1

    # Compute dx1 with initial base1
    def compute_dx1(base1):
        dx1 = np.zeros(n_intervals)
        for i in range(n_intervals):
            dx1[i] = base1 * (1 + r1)**(n_intervals - i - 1)
        return dx1

    # Function to compute total with a given base1
    def total_with_base1(base1):
        dx1 = compute_dx1(base1)
        dx_combined = np.minimum(dx0, dx1)
        return np.sum(dx_combined)

    # Binary search for base1 that gives correct total
    target_total = t_end - t_start

    # Initial bounds for base1
    base1_low = 0.1
    base1_high = 100.0

    # Binary search
    for iteration in range(50):  # Max 50 iterations
        base1_mid = (base1_low + base1_high) / 2
        total_mid = total_with_base1(base1_mid)

        if abs(total_mid - target_total) < 1e-6:
            break

        if total_mid < target_total:
            base1_low = base1_mid
        else:
            base1_high = base1_mid

    base1 = base1_mid
    dx1 = compute_dx1(base1)

    # Take minimum at each position (creates automatic transition)
    dx = np.minimum(dx0, dx1)

    # Construct times from spacings
    times = np.zeros(n_points)
    times[0] = t_start
    for i in range(n_intervals):
        times[i+1] = times[i] + dx[i]

    # Ensure exact endpoint
    times[-1] = t_end

    # Find transition point (where dx0 == dx1)
    transition_idx = np.argmin(np.abs(dx0 - dx1))

    return times, dx, dx0, dx1, transition_idx, base1


def print_results(n_points, times, dx, dx0, dx1, transition_idx, base1):
    """Print detailed results for one test case."""
    print("="*80)
    print(f"DUAL GEOMETRIC SPACING: {n_points} POINTS")
    print("="*80)

    # Statistics
    n_before_100 = np.sum(times <= 100)
    n_first_40 = np.sum(times <= 40)
    n_last_40 = np.sum(times >= 360)

    print(f"\nConfiguration:")
    print(f"  Number of points: {n_points}")
    print(f"  Number of intervals: {n_points - 1}")
    print(f"  Solved base1 (late spacing multiplier): {base1:.4f}")
    print(f"  Transition at interval {transition_idx} (point {transition_idx+1})")

    print(f"\nActual point distribution:")
    print(f"  First 10% (0-40 years):   {n_first_40} points ({100*n_first_40/n_points:.1f}%)")
    print(f"  Before 100 years (0-100): {n_before_100} points ({100*n_before_100/n_points:.1f}%)")
    print(f"  Last 10% (360-400 years): {n_last_40} points ({100*n_last_40/n_points:.1f}%)")

    print(f"\n{'─'*80}")
    print("CONTROL POINT TIMES")
    print(f"{'─'*80}")

    print(f"\n{'Point':<8} {'Time (yr)':<12} {'Spacing (yr)':<15} {'Governed By':<15} {'Region'}")
    print(f"{'-'*8} {'-'*12} {'-'*15} {'-'*15} {'-'*25}")

    for i in range(n_points):
        if i == 0:
            spacing_str = "—"
            governed = "—"
        else:
            spacing_str = f"{dx[i-1]:.2f}"
            # Determine which formula governs this interval
            if dx0[i-1] < dx1[i-1]:
                governed = "r0 (early)"
            elif dx1[i-1] < dx0[i-1]:
                governed = "r1 (late)"
            else:
                governed = "transition"

        if times[i] <= 40:
            region = "First 10% (0-40)"
        elif times[i] <= 100:
            region = "Early (40-100)"
        elif times[i] <= 360:
            region = "Middle (100-360)"
        else:
            region = "Last 10% (360-400)"

        marker = ""
        if i == n_first_40 - 1:
            marker = "  ← 40 years"
        elif i == n_before_100 - 1:
            marker = "  ← 100 years"
        elif i == n_points - n_last_40:
            marker = "  ← 360 years"

        print(f"{i+1:<8} {times[i]:<12.2f} {spacing_str:<15} {governed:<15} {region}{marker}")

    # Spacing statistics
    print(f"\n{'─'*80}")
    print("SPACING STATISTICS")
    print(f"{'─'*80}")
    print(f"\nSpacing between points:")
    print(f"  Minimum: {np.min(dx):.2f} years")
    print(f"  Maximum: {np.max(dx):.2f} years")
    print(f"  Mean:    {np.mean(dx):.2f} years")
    print(f"  Median:  {np.median(dx):.2f} years")

    # Growth factors
    growth_factors = []
    for i in range(1, len(dx)):
        if dx[i-1] > 0:
            growth_factors.append(dx[i] / dx[i-1])

    if growth_factors:
        print(f"\n{'─'*80}")
        print("SPACING GROWTH FACTORS")
        print(f"{'─'*80}")
        print(f"\nFirst 10 growth factors (spacing[i+1] / spacing[i]):")
        for i in range(min(10, len(growth_factors))):
            print(f"  Spacing {i+2}/{i+1}: {growth_factors[i]:.3f}")

        if len(growth_factors) > 10:
            print(f"  ...")
            print(f"\nLast 5 growth factors:")
            for i in range(len(growth_factors) - 5, len(growth_factors)):
                print(f"  Spacing {i+2}/{i+1}: {growth_factors[i]:.3f}")

    print(f"\n{'='*80}\n")


# Test parameters
t_start = 0
t_end = 400
dt = 1.0
delta = 0.1

# Growth rates
r0 = 0.15  # 15% growth for early region
r1 = 0.10  # 10% growth for late region

print("="*80)
print("DUAL GEOMETRIC SPACING ALGORITHM")
print("="*80)
print(f"\nParameters:")
print(f"  Time range: [{t_start}, {t_end}] years")
print(f"  dt (minimum/base spacing): {dt} year")
print(f"  delta (depreciation rate): {delta} yr⁻¹")
print(f"  r0 (early growth rate): {r0} ({r0*100:.0f}% per interval)")
print(f"  r1 (late growth rate): {r1} ({r1*100:.0f}% per interval)")
print(f"  Late base spacing: (1-exp(-1))/delta = {(1-np.exp(-1))/delta:.2f} years")
print(f"\nMethod: dx[i] = min(dt*(1+r0)^i, base1*(1+r1)^(N-i-1))")
print(f"        Early segment (dx0) is EXACT (no scaling)")
print(f"        Late segment (dx1) uses solved base1 to make total = {t_end - t_start} years")
print(f"\n{'='*80}\n")

# Test cases
test_cases = [10, 16, 25]

for n_points in test_cases:
    times, dx, dx0, dx1, transition_idx, base1 = dual_geometric_spacing(
        n_points, t_start, t_end, dt, delta, r0=r0, r1=r1
    )
    print_results(n_points, times, dx, dx0, dx1, transition_idx, base1)
