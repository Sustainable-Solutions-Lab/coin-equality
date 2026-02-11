#!/usr/bin/env python3
"""
Test two-segment geometric spacing algorithm.

Segment 1: [0, 360] with n1 points, geometric spacing with growth rate r0
Segment 2: [360, 400] with n2 points, geometric spacing with growth rate r1
"""

import numpy as np
from scipy.optimize import brentq


def two_segment_spacing(n_points, t_start, t_split, t_end, dt, delta, n1):
    """
    Generate spacing using two geometric segments.

    Segment 1: [t_start, t_split] with n1 points
               dx[i] = dt * (1 + r0)**i, solve for r0 to sum to (t_split - t_start)

    Segment 2: [t_split, t_end] with (n_points - n1 + 1) points
               dx[i] = base1 * (1 + r1)**i, solve for r1 to sum to (t_end - t_split)

    Parameters:
    - n_points: Total number of points
    - t_start, t_split, t_end: Time boundaries
    - dt: Minimum spacing (base for segment 1)
    - delta: Depreciation rate (determines base1 for segment 2)
    - n1: Number of points in segment 1 (including t_split)
    """
    n2 = n_points - n1 + 1  # Points in segment 2 (includes shared point at t_split)

    # Segment 1: [t_start, t_split] with n1 points
    # Has n1-1 intervals
    n_intervals_1 = n1 - 1
    target_total_1 = t_split - t_start

    # Solve for r0 such that sum(dt * (1+r0)^i for i=0 to n1-2) = target_total_1
    def sum_geometric_1(r0):
        if abs(r0) < 1e-10:
            return dt * n_intervals_1
        total = 0
        for i in range(n_intervals_1):
            total += dt * (1 + r0)**i
        return total

    def equation_1(r0):
        return sum_geometric_1(r0) - target_total_1

    # Find r0 using root finding
    try:
        r0 = brentq(equation_1, -0.5, 2.0)
    except ValueError:
        # If can't solve, use uniform spacing
        r0 = 0.0

    # Construct segment 1 times
    times_1 = np.zeros(n1)
    times_1[0] = t_start
    for i in range(n_intervals_1):
        spacing = dt * (1 + r0)**i
        times_1[i+1] = times_1[i] + spacing

    # Ensure exact split point
    times_1[-1] = t_split

    # Segment 2: [t_split, t_end] with n2 points
    # Has n2-1 intervals
    n_intervals_2 = n2 - 1
    target_total_2 = t_end - t_split

    # Base spacing for segment 2 (related to depreciation timescale)
    base1_initial = (1 - np.exp(-1)) / delta

    # Solve for r1 such that sum(base1 * (1+r1)^i for i=0 to n2-2) = target_total_2
    def sum_geometric_2(r1, base1):
        if abs(r1) < 1e-10:
            return base1 * n_intervals_2
        total = 0
        for i in range(n_intervals_2):
            total += base1 * (1 + r1)**i
        return total

    # We need to solve for base1 and r1 together
    # Let's fix r1 and solve for base1
    # Or we can specify r1 and solve for base1

    # Simpler: solve for base1 given a fixed r1
    r1 = 0.0  # Try uniform spacing for segment 2 first
    base1 = target_total_2 / n_intervals_2 if n_intervals_2 > 0 else 1.0

    # Construct segment 2 times
    times_2 = np.zeros(n2)
    times_2[0] = t_split
    for i in range(n_intervals_2):
        spacing = base1 * (1 + r1)**i
        times_2[i+1] = times_2[i] + spacing

    # Ensure exact endpoint
    times_2[-1] = t_end

    # Combine segments (skip duplicate at t_split)
    times = np.concatenate([times_1, times_2[1:]])

    return times, times_1, times_2, r0, r1, base1


def print_results(n_points, n1, times, times_1, times_2, r0, r1, base1):
    """Print detailed results."""
    print("="*80)
    print(f"TWO-SEGMENT SPACING: {n_points} POINTS")
    print("="*80)

    n2 = len(times_2)

    print(f"\nConfiguration:")
    print(f"  Total points: {n_points}")
    print(f"  Segment 1 [0, 360]: {n1} points ({n1-1} intervals)")
    print(f"  Segment 2 [360, 400]: {n2} points ({n2-1} intervals)")
    print(f"  Solved r0 (segment 1 growth rate): {r0:.6f} ({r0*100:.2f}% per interval)")
    print(f"  Fixed r1 (segment 2 growth rate): {r1:.6f}")
    print(f"  Solved base1 (segment 2 base spacing): {base1:.4f} years")

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

        if times[i] < 360:
            segment = "1 (early)"
        elif times[i] == 360:
            segment = "split"
        else:
            segment = "2 (late)"

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
        elif times[i] == 360:
            marker = "  ← SPLIT at 360"

        print(f"{i+1:<8} {times[i]:<12.2f} {spacing_str:<15} {segment:<10} {region}{marker}")

    # Spacing statistics
    spacing_1 = np.diff(times_1)
    spacing_2 = np.diff(times_2)

    print(f"\n{'─'*80}")
    print("SPACING STATISTICS")
    print(f"{'─'*80}")
    print(f"\nSegment 1 spacing [0, 360]:")
    print(f"  Minimum: {np.min(spacing_1):.2f} years")
    print(f"  Maximum: {np.max(spacing_1):.2f} years")
    print(f"  Mean:    {np.mean(spacing_1):.2f} years")

    if len(spacing_2) > 0:
        print(f"\nSegment 2 spacing [360, 400]:")
        print(f"  Minimum: {np.min(spacing_2):.2f} years")
        print(f"  Maximum: {np.max(spacing_2):.2f} years")
        print(f"  Mean:    {np.mean(spacing_2):.2f} years")

    # Growth factors for segment 1
    print(f"\n{'─'*80}")
    print("SEGMENT 1 SPACING GROWTH")
    print(f"{'─'*80}")

    growth_1 = []
    for i in range(1, len(spacing_1)):
        if spacing_1[i-1] > 0:
            growth_1.append(spacing_1[i] / spacing_1[i-1])

    if growth_1:
        print(f"\nFirst 10 growth factors:")
        for i in range(min(10, len(growth_1))):
            print(f"  Spacing {i+2}/{i+1}: {growth_1[i]:.6f}")

    print(f"\n{'='*80}\n")


# Test parameters
t_start = 0
t_split = 360
t_end = 400
dt = 1.0
delta = 0.1

print("="*80)
print("TWO-SEGMENT GEOMETRIC SPACING ALGORITHM")
print("="*80)
print(f"\nParameters:")
print(f"  Time range: [{t_start}, {t_end}] years")
print(f"  Split point: {t_split} years")
print(f"  dt (minimum/base spacing): {dt} year")
print(f"  delta (depreciation rate): {delta} yr⁻¹")
print(f"\nMethod:")
print(f"  Segment 1 [0, 360]: dx[i] = dt * (1+r0)^i, solve for r0")
print(f"  Segment 2 [360, 400]: dx[i] = base1 * (1+r1)^i, solve for base1 (r1 fixed)")
print(f"\n{'='*80}\n")

# Test cases with different splits
test_cases = [
    (10, 9),   # 10 points: 9 in [0,360], 2 in [360,400] (shared point)
    (16, 14),  # 16 points: 14 in [0,360], 3 in [360,400]
    (25, 22),  # 25 points: 22 in [0,360], 4 in [360,400]
]

for n_points, n1 in test_cases:
    times, times_1, times_2, r0, r1, base1 = two_segment_spacing(
        n_points, t_start, t_split, t_end, dt, delta, n1
    )
    print_results(n_points, n1, times, times_1, times_2, r0, r1, base1)
