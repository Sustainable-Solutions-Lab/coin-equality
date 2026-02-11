#!/usr/bin/env python3
"""
Calculate the iteration schedule for optimization refinement.

Shows the number of decision points, their locations, and max evaluations
for each iteration.
"""

import os
import sys
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization import calculate_control_times


def main():
    # User's configuration
    n_iterations = 4
    n_points_initial = 6  # Minimum with n_points_end=3 is 6 (3 in segment 1 + 3 in segment 2)
    n_points_final = 24
    max_evaluations_initial = 320
    max_evaluations_final = 50000

    # Integration parameters (typical)
    t_start = 0
    t_end = 400
    dt = 1.0
    delta = 0.1  # Capital depreciation rate (yr^-1)

    t_join = t_end - 4.0 / delta

    print("="*80)
    print("OPTIMIZATION ITERATION SCHEDULE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of iterations: {n_iterations}")
    print(f"  Initial control points: {n_points_initial}")
    print(f"  Final control points: {n_points_final}")
    print(f"  Initial max evaluations: {max_evaluations_initial}")
    print(f"  Final max evaluations: {max_evaluations_final}")
    print(f"  Time range: [{t_start}, {t_end}] years")
    print(f"  Control point spacing: dynamic wind-down region (t > {t_join:.1f} years)")
    print(f"  Formula: n_points_end = max(1, round(4*(1-exp(-n_points/13))))")
    print()

    # Calculate refinement bases
    refinement_base_f = ((n_points_final - 1) / (n_points_initial - 1)) ** (1.0 / (n_iterations - 1))
    refinement_base_evaluations = ((max_evaluations_final - 1) / (max_evaluations_initial - 1)) ** (1.0 / (n_iterations - 1))

    print(f"Refinement base for control points: {refinement_base_f:.6f}")
    print(f"Refinement base for evaluations: {refinement_base_evaluations:.6f}")
    print()

    print("="*80)
    print("ITERATION-BY-ITERATION BREAKDOWN")
    print("="*80)

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─'*80}")
        print(f"ITERATION {iteration}")
        print(f"{'─'*80}")

        # Calculate number of control points
        if n_iterations == 1:
            n_points = n_points_final if n_points_final is not None else n_points_initial
        else:
            n_points = round(1 + (n_points_initial - 1) * refinement_base_f**(iteration - 1))

        # Calculate max evaluations
        if iteration == n_iterations:
            iteration_max_evaluations = max_evaluations_final
        else:
            iteration_max_evaluations = round(1 + (max_evaluations_initial - 1) * refinement_base_evaluations**(iteration - 1))

        print(f"\nNumber of control points: {n_points}")
        print(f"Maximum function evaluations: {iteration_max_evaluations:,}")

        # Calculate control point locations
        control_times, r0, r1 = calculate_control_times(
            n_points, t_start, t_end, dt, delta
        )

        # Compute n_points_end for display
        import numpy as np
        n_points_end = max(1, round(4 * (1 - np.exp(-n_points / 13))))

        print(f"Spacing: {n_points_end} wind-down points, segment 1 growth r0={r0:.4f}, segment 2 growth r1={r1:.4f}")
        print(f"\nControl point locations (years):")

        # Show control points in a readable format
        if n_points <= 10:
            for i, t in enumerate(control_times):
                print(f"  Point {i+1:2d}: t = {t:7.2f} years")
        else:
            # Show first 5, middle, and last 5
            for i in range(5):
                print(f"  Point {i+1:2d}: t = {control_times[i]:7.2f} years")
            print(f"  ...")
            for i in range(n_points - 5, n_points):
                print(f"  Point {i+1:2d}: t = {control_times[i]:7.2f} years")

        # Calculate and show time spacing
        if n_points > 1:
            spacings = np.diff(control_times)
            print(f"\nControl point spacing:")
            print(f"  Minimum spacing: {np.min(spacings):7.2f} years")
            print(f"  Maximum spacing: {np.max(spacings):7.2f} years")
            print(f"  Mean spacing: {np.mean(spacings):7.2f} years")
            print(f"  Median spacing: {np.median(spacings):7.2f} years")

    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print()
    print(f"{'Iteration':<12} {'Control Points':<18} {'Max Evaluations':<20}")
    print(f"{'-'*12} {'-'*18} {'-'*20}")

    for iteration in range(1, n_iterations + 1):
        # Calculate number of control points
        if n_iterations == 1:
            n_points = n_points_final if n_points_final is not None else n_points_initial
        else:
            n_points = round(1 + (n_points_initial - 1) * refinement_base_f**(iteration - 1))

        # Calculate max evaluations
        if iteration == n_iterations:
            iteration_max_evaluations = max_evaluations_final
        else:
            iteration_max_evaluations = round(1 + (max_evaluations_initial - 1) * refinement_base_evaluations**(iteration - 1))

        print(f"{iteration:<12} {n_points:<18} {iteration_max_evaluations:<20,}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
