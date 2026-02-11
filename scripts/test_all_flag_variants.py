"""
Test script to verify that all flag combinations produce identical climate and
economic outcomes when using the same control trajectories with fract_gdp=0.02.

This script runs 16 variants of the *-f-*-*-* flag pattern with control
trajectories from the f-f-f-f-f case and verifies that:
- Climate variables (Ecum, delta_T, E, Omega_base) are identical (within dt error)
- Aggregate economics (K, Y_gross, Y_net, Consumption, Savings) are identical
- Only distribution-dependent quantities differ (Gini, utility, etc.)

The 5 flags are:
  1. income_dependent_damage_distribution (varies)
  2. income_dependent_aggregate_damage (fixed to false)
  3. income_dependent_tax_policy (varies)
  4. income_redistribution (varies)
  5. income_dependent_redistribution_policy (varies)

This gives 2^4 = 16 combinations.

Usage:
    python test_all_flag_variants.py <base_output_directory>

Example:
    python test_all_flag_variants.py data/output/config_010_f-f-f-f-f_10_0.02_1000_el_20260106-070053
"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product


def run_variant(base_dir, flag_pattern, overrides):
    """
    Run a single flag variant.

    Parameters
    ----------
    base_dir : str
        Base output directory with control trajectories
    flag_pattern : str
        Flag pattern (e.g., 'f-f-f-f-f')
    overrides : list
        List of override arguments to pass to run_integration.py

    Returns
    -------
    dict
        {'pattern': str, 'output_dir': str, 'df': DataFrame} or None if failed
    """
    cmd = ['python', 'run_integration.py', base_dir] + overrides

    print(f"\n{'=' * 80}")
    print(f"Running variant: {flag_pattern}")
    print(f"{'=' * 80}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Variant {flag_pattern} failed!")
        print(result.stderr)
        return None

    # Extract output directory from stdout
    output_dir = None
    for line in result.stdout.split('\n'):
        if 'Output directory:' in line:
            output_dir = line.split('Output directory:')[1].strip()
            break

    if not output_dir:
        print(f"WARNING: Could not find output directory for {flag_pattern}")
        return None

    # Load results CSV
    csv_files = list(Path(output_dir).glob('*_results.csv'))
    if not csv_files:
        print(f"WARNING: No results CSV found in {output_dir}")
        return None

    df = pd.read_csv(csv_files[0])
    df.columns = [col.split(',')[0].strip() for col in df.columns]

    print(f"Success: {output_dir}")
    return {
        'pattern': flag_pattern,
        'output_dir': output_dir,
        'df': df
    }


def compare_results(results):
    """
    Compare results across all variants.

    Parameters
    ----------
    results : list of dict
        List of {'pattern': str, 'output_dir': str, 'df': DataFrame}
    """
    print(f"\n{'=' * 80}")
    print("COMPARING RESULTS ACROSS ALL VARIANTS")
    print(f"{'=' * 80}")

    # Variables that should be IDENTICAL (climate and aggregate economics)
    identical_vars = [
        'K', 'Ecum', 'delta_T', 'Y_gross', 'Y_net', 'Consumption', 'Savings',
        'E', 'dEcum_dt', 'Lambda', 'mu', 'AbateCost', 'dK_dt'
    ]

    # Variables that may DIFFER (distribution-dependent)
    different_vars = [
        'U', 'Gini', 'r_consumption',
        'Fmin', 'Fmax', 'min_y_net', 'max_y_net'
    ]

    # Use first result as base case
    base_case = results[0]
    base_df = base_case['df']
    other_variants = results[1:]

    print(f"\nBase case: {base_case['pattern']}")
    print(f"Comparing against {len(other_variants)} other variants")

    # Track maximum differences for each variable
    max_diffs = {}

    print(f"\n{'=' * 80}")
    print("CLIMATE AND ECONOMIC VARIABLES (should be identical)")
    print(f"{'=' * 80}")

    for var in identical_vars:
        if var not in base_df.columns:
            continue

        print(f"\n{var}:")
        max_rel_diff = 0.0
        max_abs_diff = 0.0
        worst_variant = None

        base_vals = base_df[var].values

        for result in other_variants:
            var_df = result['df']
            if var not in var_df.columns:
                print(f"  {result['pattern']}: MISSING VARIABLE")
                continue

            var_vals = var_df[var].values

            # Calculate differences
            abs_diff = np.abs(var_vals - base_vals)
            max_abs = np.max(abs_diff)

            # Relative difference (where base is non-zero)
            mask = np.abs(base_vals) > 1e-10
            if np.any(mask):
                rel_diff = abs_diff[mask] / np.abs(base_vals[mask])
                max_rel = np.max(rel_diff)
            else:
                max_rel = 0.0

            if max_rel > max_rel_diff:
                max_rel_diff = max_rel
                max_abs_diff = max_abs
                worst_variant = result['pattern']

        # Report result for this variable
        if max_rel_diff < 1e-6:
            status = "✓ IDENTICAL"
        elif max_rel_diff < 1e-3:
            status = "⚠ SMALL DIFF (likely dt error)"
        else:
            status = "✗ DIFFERENT"

        print(f"  {status}: max rel diff = {max_rel_diff:.3e}, max abs diff = {max_abs_diff:.3e}")
        if worst_variant:
            print(f"    Worst case: {worst_variant}")

        max_diffs[var] = (max_rel_diff, max_abs_diff)

    # Special check for Omega (dt-related differences expected)
    if 'Omega' in base_df.columns:
        print(f"\nOmega (climate damage fraction):")
        max_rel_diff = 0.0
        max_abs_diff = 0.0
        worst_variant = None

        base_vals = base_df['Omega'].values

        for result in other_variants:
            var_df = result['df']
            if 'Omega' not in var_df.columns:
                continue

            var_vals = var_df['Omega'].values

            abs_diff = np.abs(var_vals - base_vals)
            max_abs = np.max(abs_diff)

            mask = np.abs(base_vals) > 1e-10
            if np.any(mask):
                rel_diff = abs_diff[mask] / np.abs(base_vals[mask])
                max_rel = np.max(rel_diff)
            else:
                max_rel = 0.0

            if max_rel > max_rel_diff:
                max_rel_diff = max_rel
                max_abs_diff = max_abs
                worst_variant = result['pattern']

        print(f"  ⚠ Expected dt-related differences")
        print(f"    Max rel diff = {max_rel_diff:.3e}, max abs diff = {max_abs_diff:.3e}")
        if worst_variant:
            print(f"    Worst case: {worst_variant}")

    # Summary of distribution-dependent variables
    print(f"\n{'=' * 80}")
    print("DISTRIBUTION-DEPENDENT VARIABLES (expected to differ)")
    print(f"{'=' * 80}")

    for var in different_vars:
        if var not in base_df.columns:
            continue

        base_vals = base_df[var].values
        base_mean = np.mean(base_vals)
        base_range = (np.min(base_vals), np.max(base_vals))

        # Find range across all variants
        all_means = [base_mean]
        for result in other_variants:
            if var in result['df'].columns:
                all_means.append(np.mean(result['df'][var].values))

        mean_range = (np.min(all_means), np.max(all_means))

        print(f"{var:30s}: base mean = {base_mean:10.4f}, "
              f"range across variants = [{mean_range[0]:10.4f}, {mean_range[1]:10.4f}]")

    # Final summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    all_identical = True
    for var, (max_rel, max_abs) in max_diffs.items():
        if max_rel >= 1e-3:  # Threshold for concern (larger than dt error)
            print(f"⚠ WARNING: {var} shows differences > 0.1%")
            all_identical = False

    if all_identical:
        print("✓ All climate and economic variables identical within expected tolerance")
        print("  (Small differences < 0.1% are consistent with dt-related coupling)")
    else:
        print("✗ Some variables show unexpected differences - investigate further")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_all_flag_variants.py <base_output_directory>")
        print("Example: python test_all_flag_variants.py data/output/config_010_f-f-f-f-f_10_0.02_1000_el_20260106-070053")
        sys.exit(1)

    base_dir = sys.argv[1]

    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    print(f"{'=' * 80}")
    print("Testing All Flag Variants with Identical Control Trajectories")
    print(f"{'=' * 80}")
    print(f"Base directory: {base_dir}")
    print(f"\nGenerating 16 variants of *-f-*-*-* pattern")
    print(f"  Position 1: income_dependent_damage_distribution (varies)")
    print(f"  Position 2: income_dependent_aggregate_damage (fixed to false)")
    print(f"  Position 3: income_dependent_tax_policy (varies)")
    print(f"  Position 4: income_redistribution (varies)")
    print(f"  Position 5: income_dependent_redistribution_policy (varies)")

    # Generate all 16 combinations
    flags = [
        'income_dependent_damage_distribution',
        # 'income_dependent_aggregate_damage',  # Fixed to false
        'income_dependent_tax_policy',
        'income_redistribution',
        'income_dependent_redistribution_policy'
    ]

    results = []

    for combo in product([False, True], repeat=4):
        # Build flag pattern string
        flag_values = [combo[0], False, combo[1], combo[2], combo[3]]  # Position 2 always False
        pattern = '-'.join(['t' if f else 'f' for f in flag_values])

        # Build overrides list
        overrides = [
            '--scalar_parameters.income_dependent_damage_distribution', str(combo[0]).lower(),
            '--scalar_parameters.income_dependent_aggregate_damage', 'false',
            '--scalar_parameters.income_dependent_tax_policy', str(combo[1]).lower(),
            '--scalar_parameters.income_redistribution', str(combo[2]).lower(),
            '--scalar_parameters.income_dependent_redistribution_policy', str(combo[3]).lower(),
            '--run_name', f'test_{pattern}'
        ]

        # Run variant
        result = run_variant(base_dir, pattern, overrides)
        if result:
            results.append(result)
        else:
            print(f"ERROR: Failed to run variant {pattern}")

    if len(results) < 2:
        print("\nERROR: Need at least 2 successful runs to compare")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"Successfully ran {len(results)} out of 16 variants")
    print(f"{'=' * 80}")

    # Compare results
    compare_results(results)

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
