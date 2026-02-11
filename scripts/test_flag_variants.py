"""
Test script to verify that different flag combinations produce identical
climate and economic outcomes when using the same control trajectories.

This script runs 4 variants of the *-f-*-f-f flag pattern with control
trajectories from the f-f-f-f-f case and verifies that:
- Climate variables (Ecum, delta_T, E, Omega_base) are identical
- Aggregate economics (K, Y_gross, Y_net, Consumption, Savings) are identical
- Only distribution-dependent quantities differ (Gini, utility, etc.)

Usage:
    python test_flag_variants.py <base_output_directory>

Example:
    python test_flag_variants.py data/output/config_010_f-f-f-f-f_10_1_50k_el_20260105-165554
"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


def run_variant(base_dir, suffix, overrides):
    """
    Run a single flag variant.

    Parameters
    ----------
    base_dir : str
        Base output directory with control trajectories
    suffix : str
        Suffix for output naming (e.g., 'f-f-f-f-f')
    overrides : list
        List of override arguments to pass to run_integration.py

    Returns
    -------
    str
        Path to output directory
    """
    cmd = ['python', 'run_integration.py', base_dir] + overrides

    print(f"\n{'=' * 80}")
    print(f"Running variant: {suffix}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Variant {suffix} failed!")
        print(result.stderr)
        return None

    # Extract output directory from stdout
    for line in result.stdout.split('\n'):
        if 'Output directory:' in line:
            output_dir = line.split('Output directory:')[1].strip()
            print(f"Output: {output_dir}")
            return output_dir

    print(f"WARNING: Could not find output directory for {suffix}")
    return None


def compare_results(base_case, variants):
    """
    Compare results across variants.

    Parameters
    ----------
    base_case : dict
        {'name': str, 'dir': str, 'df': DataFrame}
    variants : list of dict
        List of {'name': str, 'dir': str, 'df': DataFrame}
    """
    print(f"\n{'=' * 80}")
    print("COMPARING RESULTS")
    print(f"{'=' * 80}")

    # Variables that should be IDENTICAL (climate and aggregate economics)
    identical_vars = [
        'K', 'Ecum', 'delta_T', 'Y_gross', 'Y_net', 'Consumption', 'Savings',
        'E', 'Omega', 'Lambda', 'mu', 'AbateCost'
    ]

    # Variables that may DIFFER (distribution-dependent)
    different_vars = [
        'U', 'Gini', 'r_consumption'
    ]

    base_df = base_case['df']

    print(f"\nBase case: {base_case['name']}")
    print(f"Number of variants: {len(variants)}")

    for var in identical_vars:
        if var not in base_df.columns:
            continue

        print(f"\n  Checking {var}:")
        max_rel_diff = 0.0
        max_abs_diff = 0.0

        for variant in variants:
            var_df = variant['df']
            if var not in var_df.columns:
                print(f"    {variant['name']}: MISSING VARIABLE")
                continue

            base_vals = base_df[var].values
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

            max_rel_diff = max(max_rel_diff, max_rel)
            max_abs_diff = max(max_abs_diff, max_abs)

            status = "✓ IDENTICAL" if max_rel < 1e-6 else "✗ DIFFERENT"
            print(f"    {variant['name']}: {status} (max rel diff: {max_rel:.2e}, max abs diff: {max_abs:.2e})")

        # Summary for this variable
        if max_rel_diff < 1e-6:
            print(f"  → {var}: ✓ All variants identical")
        else:
            print(f"  → {var}: ✗ WARNING: Differences detected (max rel: {max_rel_diff:.2e})")

    # Check variables that SHOULD differ
    print(f"\n{'=' * 80}")
    print("Distribution-dependent variables (expected to differ):")
    print(f"{'=' * 80}")

    for var in different_vars:
        if var not in base_df.columns:
            continue

        print(f"\n  {var}:")
        for variant in variants:
            var_df = variant['df']
            if var not in var_df.columns:
                print(f"    {variant['name']}: MISSING")
                continue

            base_vals = base_df[var].values
            var_vals = var_df[var].values

            mean_base = np.mean(base_vals)
            mean_var = np.mean(var_vals)

            print(f"    {variant['name']}: mean = {mean_var:.6f} (base: {mean_base:.6f})")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_flag_variants.py <base_output_directory>")
        print("Example: python test_flag_variants.py data/output/config_010_f-f-f-f-f_10_1_50k_el_20260105-165554")
        sys.exit(1)

    base_dir = sys.argv[1]

    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    print(f"{'=' * 80}")
    print("Testing Flag Variants with Identical Control Trajectories")
    print(f"{'=' * 80}")
    print(f"Base directory: {base_dir}")

    # Define the 4 variants
    variants_config = [
        {
            'name': 'f-f-f-f-f',
            'overrides': [
                '--scalar_parameters.income_dependent_damage_distribution', 'false',
                '--scalar_parameters.income_dependent_tax_policy', 'false',
                '--scalar_parameters.income_redistribution', 'false',
                '--run_name', 'test_f-f-f-f-f'
            ]
        },
        {
            'name': 't-f-f-f-f',
            'overrides': [
                '--scalar_parameters.income_dependent_damage_distribution', 'true',
                '--scalar_parameters.income_dependent_tax_policy', 'false',
                '--scalar_parameters.income_redistribution', 'false',
                '--run_name', 'test_t-f-f-f-f'
            ]
        },
        {
            'name': 'f-f-t-f-f',
            'overrides': [
                '--scalar_parameters.income_dependent_damage_distribution', 'false',
                '--scalar_parameters.income_dependent_tax_policy', 'true',
                '--scalar_parameters.income_redistribution', 'false',
                '--run_name', 'test_f-f-t-f-f'
            ]
        },
        {
            'name': 't-f-t-f-f',
            'overrides': [
                '--scalar_parameters.income_dependent_damage_distribution', 'true',
                '--scalar_parameters.income_dependent_tax_policy', 'true',
                '--scalar_parameters.income_redistribution', 'false',
                '--run_name', 'test_t-f-t-f-f'
            ]
        }
    ]

    # Run all variants
    results = []
    for var_config in variants_config:
        output_dir = run_variant(base_dir, var_config['name'], var_config['overrides'])
        if output_dir:
            # Load results CSV
            csv_files = list(Path(output_dir).glob('*_results.csv'))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                df.columns = [col.split(',')[0].strip() for col in df.columns]
                results.append({
                    'name': var_config['name'],
                    'dir': output_dir,
                    'df': df
                })
            else:
                print(f"WARNING: No results CSV found in {output_dir}")
        else:
            print(f"ERROR: Failed to run variant {var_config['name']}")

    if len(results) < 2:
        print("\nERROR: Need at least 2 successful runs to compare")
        sys.exit(1)

    # Use first result as base case
    base_case = results[0]
    variants = results[1:]

    # Compare results
    compare_results(base_case, variants)

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")
    print(f"Ran {len(results)} variants")
    print("Check output above for any differences in climate/economic variables")


if __name__ == '__main__':
    main()
