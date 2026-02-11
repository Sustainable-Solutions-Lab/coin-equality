#!/usr/bin/env python3
"""
Quick test to verify timing statistics are working correctly.

Runs a short optimization and verifies that:
1. All timing categories add up to approximately the total time
2. No category has >100% of total time
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parameters import load_configuration
from src.optimization import UtilityOptimizer

# Use a fast configuration
config_file = 'json/config_010_f-f-f-f-f_5_1_320_el.json'

print(f"Running timing test with: {config_file}")
print(f"This will run optimization and check timing statistics...\n")

# Load configuration
config = load_configuration(config_file)

# Create optimizer
optimizer = UtilityOptimizer(config)

# Run optimization (timing stats will print automatically every 1M calls)
opt_params = config.optimization_params
results = optimizer.optimize_with_iterative_refinement(
    n_iterations=opt_params.optimization_iterations,
    initial_guess_scalar=opt_params.initial_guess_f,
    max_evaluations_initial=opt_params.max_evaluations_initial,
    max_evaluations_final=opt_params.max_evaluations_final,
    max_evaluations_time_points=opt_params.max_evaluations_time_points,
    n_points_initial=opt_params.n_points_initial_f,
    n_points_final=opt_params.n_points_final_f,
    initial_guess_s_scalar=getattr(opt_params, 'initial_guess_s', None),
    n_points_initial_s=getattr(opt_params, 'n_points_initial_s', None),
    n_points_final_s=getattr(opt_params, 'n_points_final_s', None),
)

print("\n" + "="*80)
print("TIMING VERIFICATION")
print("="*80)

# Import the timing stats and print them
from src.economic_model import _timing_stats as stats, print_timing_stats

# Force print timing stats regardless of call count
print(f"\nForcing timing stats print (call count: {stats['call_count']})")
print_timing_stats()

if stats['call_count'] == 0:
    print("ERROR: No function calls recorded!")
    sys.exit(1)

# Calculate sum of all components
# Note: find_Fmax_time and find_Fmin_time are subsets of policy_calc_time, so don't double-count
components = [
    'setup_time',
    'policy_calc_time',
    'segment1_time',
    'segment2_time',
    'segment3_time',
    'utility_time',
    'damage_agg_time',
    'climate_time',
    'finalize_time',
]

component_sum = sum(stats[key] for key in components)
total_time = stats['total_time']

print(f"\nTotal time:           {total_time:10.2f} s")
print(f"Sum of components:    {component_sum:10.2f} s")
print(f"Difference:           {abs(total_time - component_sum):10.2f} s")
print(f"Relative difference:  {abs(total_time - component_sum)/total_time*100:10.4f} %")

# Check that sum is within 1% of total (allowing for Python overhead)
tolerance = 0.01  # 1%
if abs(total_time - component_sum) / total_time > tolerance:
    print(f"\n❌ FAIL: Components don't add up to total (>{tolerance*100:.1f}% difference)")
    sys.exit(1)
else:
    print(f"\n✓ PASS: Components add up to total within tolerance (difference < {tolerance*100:.1f}%)")

# Check that each component is reasonable
print("\nComponent breakdown:")
for key in components:
    pct = 100 * stats[key] / total_time
    status = "✓" if pct <= 100.0 else "✗"
    print(f"  {status} {key:20s}: {stats[key]:8.2f} s  ({pct:5.1f}%)")
    if pct > 100.0:
        print(f"    ❌ FAIL: Component has >100% of total time!")
        sys.exit(1)

print("\n" + "="*80)
print("✓ TIMING TEST PASSED")
print("="*80)
print("\nAll timing categories properly account for total time.")
print("No double-counting detected.")
