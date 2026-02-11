#!/usr/bin/env python3
"""
Test the new spacing algorithm with the user's example.

This demonstrates how the new algorithm maintains full Chebyshev concentration
by shifting points forward rather than reducing the scaling power.
"""

import numpy as np

def calculate_chebyshev_times_new(n_points, t_start, t_end, scaling_power, dt):
    """New algorithm: shift points forward to enforce spacing."""
    N = n_points
    k_values = np.arange(N)
    u = (1 - np.cos(k_values * np.pi / (N - 1))) / 2
    u_scaled = u ** scaling_power
    times = t_start + (t_end - t_start) * u_scaled
    times[0] = t_start
    times[-1] = t_end

    # Shift forward to enforce minimum spacing
    for i in range(1, N - 1):
        if times[i] < times[i-1] + dt:
            times[i] = times[i-1] + dt

    return times, scaling_power


def calculate_chebyshev_times_old(n_points, t_start, t_end, scaling_power, dt):
    """Old algorithm: reduce scaling power to avoid close spacing."""
    N = n_points
    k_values = np.arange(N)
    u = (1 - np.cos(k_values * np.pi / (N - 1))) / 2

    # Calculate max scaling power that ensures minimum spacing
    if N > 1:
        max_scaling_power = np.log(dt / (t_end - t_start)) / np.log(u[1])
    else:
        max_scaling_power = scaling_power

    effective_scaling_power = min(scaling_power, max_scaling_power)
    u_scaled = u ** effective_scaling_power
    times = t_start + (t_end - t_start) * u_scaled
    times[0] = t_start
    times[-1] = t_end

    return times, effective_scaling_power


print("="*80)
print("COMPARISON: OLD vs NEW SPACING ALGORITHM")
print("="*80)

# Test case: 24 points, [0, 400], scaling_power=1.5, dt=1
n_points = 24
t_start = 0
t_end = 400
scaling_power = 1.5
dt = 1.0

print(f"\nConfiguration:")
print(f"  Points: {n_points}")
print(f"  Range: [{t_start}, {t_end}] years")
print(f"  Requested scaling power: {scaling_power}")
print(f"  Minimum spacing (dt): {dt} year")

print(f"\n{'─'*80}")
print("OLD ALGORITHM (reduce scaling power)")
print(f"{'─'*80}")

times_old, scaling_old = calculate_chebyshev_times_old(n_points, t_start, t_end, scaling_power, dt)
spacing_old = np.diff(times_old)

print(f"\nEffective scaling power: {scaling_old:.6f} (reduced from {scaling_power})")
print(f"\nFirst 10 control points:")
for i in range(10):
    print(f"  Point {i+1:2d}: t = {times_old[i]:7.2f} years")
print("  ...")

print(f"\nSpacing statistics:")
print(f"  Minimum: {np.min(spacing_old):7.2f} years")
print(f"  Maximum: {np.max(spacing_old):7.2f} years")
print(f"  Mean:    {np.mean(spacing_old):7.2f} years")

print(f"\n{'─'*80}")
print("NEW ALGORITHM (shift points forward)")
print(f"{'─'*80}")

times_new, scaling_new = calculate_chebyshev_times_new(n_points, t_start, t_end, scaling_power, dt)
spacing_new = np.diff(times_new)

print(f"\nActual scaling power: {scaling_new:.6f} (full concentration maintained)")
print(f"\nFirst 10 control points:")
for i in range(10):
    print(f"  Point {i+1:2d}: t = {times_new[i]:7.2f} years")
print("  ...")

print(f"\nSpacing statistics:")
print(f"  Minimum: {np.min(spacing_new):7.2f} years")
print(f"  Maximum: {np.max(spacing_new):7.2f} years")
print(f"  Mean:    {np.mean(spacing_new):7.2f} years")

print(f"\n{'─'*80}")
print("COMPARISON")
print(f"{'─'*80}")

# Count points in first 100 years
n_old_first100 = np.sum(times_old <= 100)
n_new_first100 = np.sum(times_new <= 100)

print(f"\nPoints in first 100 years:")
print(f"  OLD: {n_old_first100} points")
print(f"  NEW: {n_new_first100} points")
print(f"  Difference: {n_new_first100 - n_old_first100} more points with NEW algorithm")

print(f"\nWhy NEW is better for early-year focus:")
print(f"  - Maintains full scaling power ({scaling_power}) for maximum concentration")
print(f"  - Shifts points forward only where needed (early years with dense spacing)")
print(f"  - Late years naturally have wider spacing (no shifting needed)")
print(f"  - Result: More detail where we need it (first 100 years)")

print(f"\n{'='*80}")
