#!/usr/bin/env python3
"""
Test that s(t) trajectories produce economically sensible results.
Since dual optimization methods aren't fully implemented yet, we test
the infrastructure by evaluating different s(t) trajectories.
"""

import os
import sys
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parameters import load_configuration
from src.optimization import UtilityOptimizer

def test_optimal_s_economic_sensibility():
    """
    Test different s(t) trajectories and verify results are economically reasonable.

    Economic intuition:
    - Savings rate typically 15-35% in growth models
    - Early periods: higher s when returns to capital are high
    - Later periods: lower s as economy approaches steady state
    - Higher s should increase capital accumulation and long-term Consumption
    """

    print("=" * 70)
    print("Test 6: Verify s(t) Trajectories Make Economic Sense")
    print("=" * 70)

    # Load baseline config
    baseline_config = load_configuration('config_test_DICE.json')

    # Create optimizer
    optimizer = UtilityOptimizer(baseline_config)

    # Define test scenarios with different s(t) trajectories
    scenarios = [
        {
            'name': 'Baseline (constant s=0.24)',
            's_times': [0, 400],
            's_values': [0.24, 0.24]
        },
        {
            'name': 'High savings (s=0.30)',
            's_times': [0, 400],
            's_values': [0.30, 0.30]
        },
        {
            'name': 'Low savings (s=0.18)',
            's_times': [0, 400],
            's_values': [0.18, 0.18]
        },
        {
            'name': 'Declining s (0.30→0.20)',
            's_times': [0, 200, 400],
            's_values': [0.30, 0.25, 0.20]
        },
        {
            'name': 'Increasing s (0.20→0.30)',
            's_times': [0, 200, 400],
            's_values': [0.20, 0.25, 0.30]
        }
    ]

    # Fixed f control (50-50 allocation)
    f_times = np.array([0, 400])
    f_values = np.array([0.5, 0.5])

    results = []

    print("\nEvaluating different s(t) trajectories:")
    print("-" * 70)

    for scenario in scenarios:
        s_times = np.array(scenario['s_times'])
        s_values = np.array(scenario['s_values'])

        # Calculate objective with this s(t) trajectory
        obj = optimizer.calculate_objective(
            control_values=f_values,
            control_times=f_times,
            s_control_values=s_values,
            s_control_times=s_times
        )

        results.append({
            'name': scenario['name'],
            's_times': s_times,
            's_values': s_values,
            'objective': obj
        })

        print(f"\n{scenario['name']}")
        print(f"  s trajectory: ", end="")
        for t, s in zip(s_times, s_values):
            print(f"s({t:.0f})={s:.2f} ", end="")
        print(f"\n  Objective: {obj:.6e}")

    # Economic sensibility checks
    print("\n" + "=" * 70)
    print("Economic Sensibility Checks:")
    print("=" * 70)

    checks_passed = []

    # Extract objectives
    baseline_obj = results[0]['objective']
    high_s_obj = results[1]['objective']
    low_s_obj = results[2]['objective']
    declining_s_obj = results[3]['objective']
    increasing_s_obj = results[4]['objective']

    # Check 1: Higher s should generally lead to different outcomes
    # (not necessarily always higher utility due to Consumption-investment tradeoff)
    diff_high_low = abs(high_s_obj - low_s_obj) / baseline_obj
    check1 = diff_high_low > 0.001  # At least 0.1% difference
    checks_passed.append(check1)
    print(f"\nCheck 1 - High vs Low s differ significantly:")
    print(f"  Difference: {diff_high_low*100:.2f}% of baseline")
    print(f"  Result: {'PASS' if check1 else 'FAIL'}")

    # Check 2: There should be an optimal s in the middle range
    # Neither extremely high nor extremely low should be best
    best_constant_s = max([results[0], results[1], results[2]],
                          key=lambda x: x['objective'])
    check2 = 0.20 <= best_constant_s['s_values'][0] <= 0.35
    checks_passed.append(check2)
    print(f"\nCheck 2 - Best constant s in reasonable range [0.20, 0.35]:")
    print(f"  Best s: {best_constant_s['s_values'][0]:.2f}")
    print(f"  Objective: {best_constant_s['objective']:.6e}")
    print(f"  Result: {'PASS' if check2 else 'FAIL'}")

    # Check 3: Time-varying s should be able to improve on constant s
    # (or at least be comparable if constant is near optimal)
    best_overall = max(results, key=lambda x: x['objective'])
    improvement = (best_overall['objective'] / baseline_obj - 1) * 100
    check3 = abs(improvement) < 10  # Within 10% - either direction is fine
    checks_passed.append(check3)
    print(f"\nCheck 3 - Best trajectory within 10% of baseline:")
    print(f"  Best: {best_overall['name']}")
    print(f"  Objective: {best_overall['objective']:.6e}")
    print(f"  vs Baseline: {improvement:+.2f}%")
    print(f"  Result: {'PASS' if check3 else 'FAIL'}")

    # Check 4: Calculate_objective handles different trajectory shapes correctly
    # (declining vs increasing should give different results)
    diff_trajectories = abs(declining_s_obj - increasing_s_obj) / baseline_obj
    check4 = diff_trajectories > 0.0001  # At least 0.01% difference
    checks_passed.append(check4)
    print(f"\nCheck 4 - Declining vs Increasing s differ:")
    print(f"  Declining: {declining_s_obj:.6e}")
    print(f"  Increasing: {increasing_s_obj:.6e}")
    print(f"  Difference: {diff_trajectories*100:.2f}% of baseline")
    print(f"  Result: {'PASS' if check4 else 'FAIL'}")

    # Overall assessment
    all_passed = all(checks_passed)
    print(f"\n{'=' * 70}")
    print(f"Economic sensibility: {'PASS ✓' if all_passed else 'FAIL ✗'}")
    print(f"  Checks passed: {sum(checks_passed)}/{len(checks_passed)}")
    print(f"{'=' * 70}")

    return all_passed

if __name__ == '__main__':
    test_optimal_s_economic_sensibility()
