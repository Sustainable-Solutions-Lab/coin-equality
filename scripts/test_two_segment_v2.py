#!/usr/bin/env python3
"""
Test two-segment geometric spacing algorithm (version 2).

Segment 1: [0, t_join] with geometric spacing starting from dt
Segment 2: (t_join, 400] with geometric spacing working backward from 400

Join point: t_join = 400 - 4/delta
"""

import numpy as np
from scipy.optimize import brentq


def two_segment_spacing_v2(n_points, t_start, t_end, dt, delta, n2):
    """
    Generate spacing using two geometric segments.

    Segment 1: [t_start, t_join]
               - At least 3 points: at t_start, dt, and t_join
               - Geometric spacing: dx[i] = dt * (1+r0)^i
               - Solve for r0 to make sum = t_join - t_start

    Segment 2: (t_join, t_end]
               - n2 points working backward from t_end
               - If n2 = 1: just t_end
               - If n2 = 2: t_end and t_end - base_spacing
               - If n2 >= 3: geometric with base_spacing and growth rate r1
                 Spacing from point i to i+1 (toward t_end): base_spacing * (1+r1)^(i-1)
                 Solve for r1 such that first point is at t_join

    Join point: t_join = t_end - 4/delta

    Parameters:
    - n_points: Total number of points
    - t_start, t_end: Time boundaries
    - dt: Minimum spacing (base for segment 1)
    - delta: Depreciation rate (determines base_spacing and t_join)
    - n2: Number of points in segment 2
    """
    # Calculate join point: x_join = 400 - 4/delta (y_join = 4/delta = 40)
    y_join = 4.0 / delta  # This is the total distance in y-space
    t_join = t_end - y_join

    # Number of points in segment 1 (includes point at t_join)
    # Segment 2 has n2 points (NOT counting the join point)
    n1 = n_points - n2

    if n1 < 3:
        raise ValueError(f"Segment 1 must have at least 3 points, got {n1}")

    # Base spacing for segment 2
    base_spacing = (1 - np.exp(-1)) / delta

    # ========== SEGMENT 1: [t_start, t_join] ==========
    n_intervals_1 = n1 - 1
    target_total_1 = t_join - t_start

    # Solve for r0 such that sum(dt * (1+r0)^i for i=0 to n1-2) = target_total_1
    def sum_geometric_1(r0):
        total = 0
        for i in range(n_intervals_1):
            total += dt * (1 + r0)**i
        return total

    def equation_1(r0):
        return sum_geometric_1(r0) - target_total_1

    # Find r0 using root finding
    try:
        r0 = brentq(equation_1, -0.5, 5.0)
    except ValueError:
        # If can't solve, use uniform spacing
        r0 = (target_total_1 / (n_intervals_1 * dt)) - 1.0

    # Construct segment 1 times
    times_1 = np.zeros(n1)
    times_1[0] = t_start
    for i in range(n_intervals_1):
        spacing = dt * (1 + r0)**i
        times_1[i+1] = times_1[i] + spacing

    # Ensure exact join point
    times_1[-1] = t_join

    # ========== SEGMENT 2: (t_join, t_end] ==========
    # Working backward from x=400 using y = 400 - x
    # n2 points in segment 2 (NOT counting join point at t_join)
    # P = n2 intervals from y=0 to y=y_join
    # Spacing: y[p+1] - y[p] = base_spacing * (1+r1)^p for p=0, 1, ..., P-1
    # Sum: y[P] = base_spacing * sum((1+r1)^j for j=0 to P-1) = y_join

    if n2 == 0:
        # No points in segment 2
        times_2 = np.array([])
        r1 = 0.0

    elif n2 == 1:
        # Just the endpoint
        times_2 = np.array([t_end])
        r1 = 0.0

    elif n2 == 2:
        # Two points: at y=0 and y=base_spacing
        # x = 400 - base_spacing and x = 400 (in increasing order)
        times_2 = np.array([t_end - base_spacing, t_end])
        r1 = 0.0

    else:
        # n2 >= 3: solve for r1
        # We want: base_spacing * sum((1+r1)^j for j=0 to n2-1) = y_join
        P = n2  # Number of intervals

        def sum_geometric_2(r1):
            total = 0
            for j in range(P):
                total += (1 + r1)**j
            return total

        def equation_2(r1):
            return base_spacing * sum_geometric_2(r1) - y_join

        # Find r1 using root finding
        try:
            r1 = brentq(equation_2, -0.5, 5.0)
        except ValueError:
            # If can't solve, use uniform spacing
            r1 = 0.0

        # Construct segment 2 points in y-space, then convert to x-space
        y_vals = np.zeros(n2 + 1)  # Includes y=0 to y=y_join
        y_vals[0] = 0  # y[0] = 0 (x = 400)

        for p in range(P):
            spacing_y = base_spacing * (1 + r1)**p
            y_vals[p+1] = y_vals[p] + spacing_y

        # Convert to x-space (but exclude the join point y[P])
        # We only want the n2 points NOT including the join
        # Store in increasing order (reverse of y order)
        times_2 = np.zeros(n2)
        for i in range(n2):
            times_2[n2 - 1 - i] = t_end - y_vals[i]

    # Combine segments
    times = np.concatenate([times_1, times_2])

    return times, times_1, times_2, r0, r1, t_join, base_spacing


def print_results(n_points, n2, times, times_1, times_2, r0, r1, t_join, base_spacing, dt, delta):
    """Print detailed results."""
    print("="*80)
    print(f"TWO-SEGMENT SPACING: {n_points} POINTS")
    print("="*80)

    n1 = len(times_1)

    print(f"\nConfiguration:")
    print(f"  Total points: {n_points}")
    print(f"  Join point: t_join = 400 - 4/delta = {t_join:.2f} years")
    print(f"  Segment 1 [0, {t_join:.0f}]: {n1} points (includes join point)")
    print(f"  Segment 2 ({t_join:.0f}, 400]: {n2} points (excludes join point, {n2} intervals)")
    print(f"\nSegment 1 parameters:")
    print(f"  Base spacing: dt = {dt} year")
    print(f"  Solved r0 (growth rate): {r0:.6f} ({r0*100:.2f}% per interval)")
    print(f"\nSegment 2 parameters:")
    print(f"  Base spacing: (1-exp(-1))/delta = {base_spacing:.4f} years")
    if n2 >= 3:
        print(f"  Solved r1 (growth rate): {r1:.6f} ({r1*100:.2f}% per interval)")
    else:
        print(f"  Growth rate: N/A (only {n2} point(s) in segment 2)")

    # Statistics
    n_before_100 = np.sum(times <= 100)
    n_first_40 = np.sum(times <= 40)
    n_last_40 = np.sum(times >= 360)

    print(f"\nActual point distribution:")
    print(f"  First 10% (0-40 years):   {n_first_40} points ({100*n_first_40/n_points:.1f}%)")
    print(f"  Before 100 years (0-100): {n_before_100} points ({100*n_before_100/n_points:.1f}%)")
    print(f"  Last 10% (360-400 years): {n_last_40} points ({100*n_last_40/n_points:.1f}%)")

    print(f"\n{'─'*80}")
    print("CONTROL POINT TIMES")
    print(f"{'─'*80}")

    spacing = np.diff(times)

    print(f"\n{'Point':<8} {'Time (yr)':<12} {'Spacing (yr)':<15} {'Segment':<10} {'Region'}")
    print(f"{'-'*8} {'-'*12} {'-'*15} {'-'*10} {'-'*25}")

    for i in range(n_points):
        if i == 0:
            spacing_str = "—"
        else:
            spacing_str = f"{spacing[i-1]:.2f}"

        if times[i] < t_join:
            segment = "1"
        elif times[i] == t_join:
            segment = "JOIN"
        else:
            segment = "2"

        if times[i] <= 40:
            region = "First 10% (0-40)"
        elif times[i] <= 100:
            region = "Early (40-100)"
        elif times[i] < t_join:
            region = f"Middle (100-{t_join:.0f})"
        else:
            region = f"Late ({t_join:.0f}-400)"

        marker = ""
        if i == n_first_40 - 1 and times[i] <= 40:
            marker = "  ← 40 years"
        elif i == n_before_100 - 1 and times[i] <= 100:
            marker = "  ← 100 years"
        elif times[i] == t_join:
            marker = f"  ← JOIN at {t_join:.0f}"

        print(f"{i+1:<8} {times[i]:<12.2f} {spacing_str:<15} {segment:<10} {region}{marker}")

    # Spacing statistics
    spacing_1 = np.diff(times_1)

    # For segment 2, we need to compute spacings including from join to first point of seg 2
    if n2 > 0:
        # Spacing from t_join to first point of segment 2, then within segment 2
        spacing_2_full = np.zeros(n2)
        spacing_2_full[0] = times_2[0] - t_join  # From join to first seg2 point
        if n2 > 1:
            spacing_2_full[1:] = np.diff(times_2)
        spacing_2 = spacing_2_full
    else:
        spacing_2 = np.array([])

    print(f"\n{'─'*80}")
    print("SPACING STATISTICS")
    print(f"{'─'*80}")
    print(f"\nSegment 1 spacing [0, {t_join:.0f}]:")
    print(f"  Minimum: {np.min(spacing_1):.2f} years")
    print(f"  Maximum: {np.max(spacing_1):.2f} years")
    print(f"  Mean:    {np.mean(spacing_1):.2f} years")

    if len(spacing_2) > 0:
        print(f"\nSegment 2 spacing [{t_join:.0f}, 400]:")
        print(f"  Minimum: {np.min(spacing_2):.2f} years")
        print(f"  Maximum: {np.max(spacing_2):.2f} years")
        print(f"  Mean:    {np.mean(spacing_2):.2f} years")

    # Growth factors
    print(f"\n{'─'*80}")
    print("SPACING GROWTH FACTORS")
    print(f"{'─'*80}")

    growth_1 = []
    for i in range(1, len(spacing_1)):
        if spacing_1[i-1] > 0:
            growth_1.append(spacing_1[i] / spacing_1[i-1])

    if growth_1:
        print(f"\nSegment 1 (first 10 factors):")
        for i in range(min(10, len(growth_1))):
            print(f"  Spacing {i+2}/{i+1}: {growth_1[i]:.6f}")
        if len(growth_1) > 10:
            print(f"  ... (all should be {1+r0:.6f})")

    if len(spacing_2) > 1:
        growth_2 = []
        for i in range(1, len(spacing_2)):
            if spacing_2[i-1] > 0:
                growth_2.append(spacing_2[i] / spacing_2[i-1])
        if growth_2:
            print(f"\nSegment 2:")
            for i in range(len(growth_2)):
                print(f"  Spacing {i+2}/{i+1}: {growth_2[i]:.6f}")

    print(f"\n{'='*80}\n")


# Test parameters
t_start = 0
t_end = 400
dt = 1.0
delta = 0.1

print("="*80)
print("TWO-SEGMENT GEOMETRIC SPACING ALGORITHM V2")
print("="*80)
print(f"\nParameters:")
print(f"  Time range: [{t_start}, {t_end}] years")
print(f"  Join point: t_join = 400 - 4/delta = {t_end - 4/delta:.0f} years")
print(f"  dt (minimum/base spacing): {dt} year")
print(f"  delta (depreciation rate): {delta} yr⁻¹")
print(f"  base_spacing = (1-exp(-1))/delta = {(1-np.exp(-1))/delta:.4f} years")
print(f"\nMethod:")
print(f"  Segment 1: dx[i] = dt * (1+r0)^i, solve for r0")
print(f"  Segment 2: backward from 400 with base_spacing*(1+r1)^i, solve for r1")
print(f"\n{'='*80}\n")

# Test cases: (n_points, n2)
test_cases = [
    (9, 3),    # 9 points: 6 in segment 1, 3 in segment 2
    (16, 3),   # 16 points: 13 in segment 1, 3 in segment 2
    (25, 3),   # 25 points: 22 in segment 1, 3 in segment 2
]

for n_points, n2 in test_cases:
    times, times_1, times_2, r0, r1, t_join, base_spacing = two_segment_spacing_v2(
        n_points, t_start, t_end, dt, delta, n2
    )
    print_results(n_points, n2, times, times_1, times_2, r0, r1, t_join, base_spacing, dt, delta)
