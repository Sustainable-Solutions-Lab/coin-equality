#!/usr/bin/env python3
"""
Calculate Social Cost of Carbon (SCC) using the perturbation method.

SCC(t) = - (ΔW_emission / emission_amount) / (ΔW_consumption / consumption_amount)

This script:
1. Loads baseline optimization results (config + optimized f, s trajectories)
2. Runs 3 integrations: baseline, +emission pulse, +consumption pulse
3. Computes SCC from welfare differences

Usage:
    python calculate_scc.py --baseline <results_directory> --pulse-year 2025

See SCC_IMPLEMENTATION_PLAN.md for detailed documentation.
"""

import argparse
import os
import sys
import csv
import numpy as np
from datetime import datetime
from scipy.interpolate import PchipInterpolator

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.economic_model import integrate_model
from src.parameters import load_configuration


def load_baseline(baseline_path):
    """
    Load baseline optimization results.

    Parameters
    ----------
    baseline_path : str
        Path to results directory containing results.csv and config JSON

    Returns
    -------
    dict
        Contains: config_path, t_values, f_values, s_values
    """
    # Find config JSON in the directory
    config_path = None
    for fname in os.listdir(baseline_path):
        if fname.endswith('.json') and fname.startswith('config_'):
            config_path = os.path.join(baseline_path, fname)
            break

    # If no config in results dir, look in json/ directory based on directory name
    if config_path is None:
        dir_name = os.path.basename(baseline_path.rstrip('/'))
        # Directory name format: config_019_..._YYYYMMDD-HHMMSS
        # Extract config name (everything before the timestamp)
        parts = dir_name.rsplit('_', 1)
        if len(parts) == 2:
            config_name = parts[0] + '.json'
            # Look in json/ directory relative to data/output/
            potential_path = os.path.join(
                os.path.dirname(os.path.dirname(baseline_path.rstrip('/'))),
                'json',
                config_name
            )
            if os.path.exists(potential_path):
                config_path = potential_path

    if config_path is None:
        raise FileNotFoundError(f"Could not find config JSON for {baseline_path}")

    # Load results CSV to get optimized f, s trajectories
    # Look for results.csv or *_optimization_results.csv
    results_csv_path = os.path.join(baseline_path, 'results.csv')
    if not os.path.exists(results_csv_path):
        # Try to find *_optimization_results.csv
        for fname in os.listdir(baseline_path):
            if fname.endswith('_optimization_results.csv'):
                results_csv_path = os.path.join(baseline_path, fname)
                break
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Could not find results CSV in {baseline_path}")

    # Read CSV - handle both simple and descriptive column names
    t_values = []
    f_values = []
    s_values = []

    with open(results_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Get actual column names
        fieldnames = reader.fieldnames

        # Find the right column names (may be 't' or 't, Time, (yr)' etc.)
        t_col = None
        f_col = None
        s_col = None
        for col in fieldnames:
            if col == 't' or col.startswith('t,'):
                t_col = col
            if col == 'f' or col.startswith('f,'):
                f_col = col
            if col == 's' or col.startswith('s,'):
                s_col = col

        if not all([t_col, f_col, s_col]):
            raise KeyError(f"Could not find t, f, s columns in {results_csv_path}. Found: {fieldnames[:10]}...")

        for row in reader:
            t_values.append(float(row[t_col]))
            f_values.append(float(row[f_col]))
            s_values.append(float(row[s_col]))

    return {
        'config_path': config_path,
        't_values': np.array(t_values),
        'f_values': np.array(f_values),
        's_values': np.array(s_values),
    }


def create_fixed_control_function(t_values, f_values, s_values, t_base):
    """
    Create a control function that interpolates fixed (f, s) trajectories.

    Parameters
    ----------
    t_values : np.ndarray
        Time points (calendar years from results CSV)
    f_values : np.ndarray
        Abatement fraction values at each time point
    s_values : np.ndarray
        Savings rate values at each time point
    t_base : float
        Base year for converting calendar years to relative time

    Returns
    -------
    callable
        Function that takes relative time t and returns (f, s)
    """
    # Convert calendar years to relative time (control function is called with t - t_base)
    t_relative = t_values - t_base

    # Use PCHIP interpolation for smooth monotonic interpolation
    f_interp = PchipInterpolator(t_relative, f_values, extrapolate=True)
    s_interp = PchipInterpolator(t_relative, s_values, extrapolate=True)

    def control_function(t):
        f = float(np.clip(f_interp(t), 0.0, 1.0))
        s = float(np.clip(s_interp(t), 0.0, 1.0))
        return f, s

    return control_function


def create_config_with_pulse(config_path, control_function,
                              year_extra_emission, amount_extra_emission,
                              year_extra_consumption, amount_extra_consumption):
    """
    Load config from JSON and override pulse parameters and control function.

    Parameters
    ----------
    config_path : str
        Path to config JSON file
    control_function : callable
        Fixed control function returning (f, s)
    year_extra_emission : float
        Year for emission pulse (0 to disable)
    amount_extra_emission : float
        Emission pulse amount in tCO2
    year_extra_consumption : float
        Year for consumption pulse (0 to disable)
    amount_extra_consumption : float
        Consumption pulse amount in $

    Returns
    -------
    ModelConfiguration
        Configuration object ready for integration
    """
    # Load base configuration
    config = load_configuration(config_path)

    # Override pulse parameters in scalar_params
    config.scalar_params.year_extra_emission = year_extra_emission
    config.scalar_params.amount_extra_emission = amount_extra_emission
    config.scalar_params.year_extra_consumption = year_extra_consumption
    config.scalar_params.amount_extra_consumption = amount_extra_consumption

    # Override control function with fixed trajectories
    config.control_function = control_function

    return config


def run_scenario(config):
    """
    Run integration and return the objective (welfare) value.

    Parameters
    ----------
    config : ModelConfiguration
        Configuration for integration

    Returns
    -------
    float
        Objective value (discounted welfare integral)
    """
    # Run integration without detailed output for speed
    results = integrate_model(config, store_detailed_output=False)

    # Calculate objective (same as in optimization.py)
    rho = config.scalar_params.rho
    t = results['t']
    U = results['U']
    L = results['L']

    t_start = config.integration_params.t_start
    discount_factors = np.exp(-rho * (t - t_start))
    integrand = discount_factors * U

    objective = np.trapezoid(integrand, t)

    return objective


def compute_scc(W_baseline, W_emission, W_consumption,
                emission_amount, consumption_amount):
    """
    Compute SCC from welfare values.

    Parameters
    ----------
    W_baseline : float
        Welfare with no pulse
    W_emission : float
        Welfare with emission pulse
    W_consumption : float
        Welfare with consumption pulse
    emission_amount : float
        Emission pulse size (tCO2)
    consumption_amount : float
        Consumption pulse size ($)

    Returns
    -------
    dict
        Contains: dW_emission, dW_consumption, m_E, m_C, SCC, and sanity checks
    """
    dW_emission = W_emission - W_baseline
    dW_consumption = W_consumption - W_baseline

    m_E = dW_emission / emission_amount  # welfare per tCO2
    m_C = dW_consumption / consumption_amount  # welfare per $

    SCC = -m_E / m_C  # $/tCO2

    # Sanity checks
    emission_check = dW_emission < 0  # Should be negative (damage)
    consumption_check = dW_consumption > 0  # Should be positive (benefit)
    scc_check = SCC > 0  # Should be positive

    return {
        'W_baseline': W_baseline,
        'W_emission': W_emission,
        'W_consumption': W_consumption,
        'dW_emission': dW_emission,
        'dW_consumption': dW_consumption,
        'm_E': m_E,
        'm_C': m_C,
        'SCC': SCC,
        'emission_check': emission_check,
        'consumption_check': consumption_check,
        'scc_check': scc_check,
    }


def run_sensitivity_test(config_path, control_function, pulse_year,
                         emission_amount, consumption_amount):
    """
    Run pulse-size sensitivity test to verify SCC is in the linear (marginal) regime.

    Parameters
    ----------
    config_path : str
        Path to config JSON file
    control_function : callable
        Fixed control function returning (f, s)
    pulse_year : float
        Year to apply pulses
    emission_amount : float
        Base emission pulse amount in tCO2
    consumption_amount : float
        Base consumption pulse amount in $

    Returns
    -------
    list of dict
        Results for each scaling factor
    """
    scaling_factors = [0.001, 1.0, 1000]

    print("\n" + "=" * 80)
    print("PULSE-SIZE SENSITIVITY TEST")
    print("=" * 80)
    print(f"\nPulse year: {pulse_year}")
    print(f"Base emission: {emission_amount:.1e} tCO2, Base consumption: ${consumption_amount:.1e}")
    print(f"Scaling factors: {scaling_factors}")

    # Single baseline (shared across all pulse sizes)
    print("\nRunning baseline integration...")
    config_baseline = create_config_with_pulse(
        config_path, control_function, 0.0, 0.0, 0.0, 0.0
    )
    W_baseline = run_scenario(config_baseline)
    print(f"  W_baseline = {W_baseline:.10e}")

    results = []
    for scale in scaling_factors:
        e_amount = emission_amount * scale
        c_amount = consumption_amount * scale

        print(f"\nScale {scale}x: emission={e_amount:.1e} tCO2, consumption=${c_amount:.1e}")

        config_e = create_config_with_pulse(
            config_path, control_function, pulse_year, e_amount, 0.0, 0.0
        )
        W_e = run_scenario(config_e)

        config_c = create_config_with_pulse(
            config_path, control_function, 0.0, 0.0, pulse_year, c_amount
        )
        W_c = run_scenario(config_c)

        scc_result = compute_scc(W_baseline, W_e, W_c, e_amount, c_amount)
        scc_result['scale'] = scale
        results.append(scc_result)
        print(f"  SCC = ${scc_result['SCC']:.2f}/tCO2")

    # Print summary table
    scc_values = [r['SCC'] for r in results]
    scc_ref = scc_values[2]  # 1.0x is the reference

    print("\n" + "-" * 80)
    print(f"{'Scale':>8} {'Emission':>14} {'Consumption':>14} {'m_E':>14} {'m_C':>14} {'SCC ($/tCO2)':>14} {'vs 1x':>8}")
    print("-" * 80)
    for r in results:
        pct_diff = (r['SCC'] / scc_ref - 1.0) * 100.0
        print(f"{r['scale']:>7.1f}x {r['scale']*emission_amount:>13.1e} {r['scale']*consumption_amount:>13.1e} "
              f"{r['m_E']:>14.6e} {r['m_C']:>14.6e} {r['SCC']:>13.2f} {pct_diff:>+7.2f}%")
    print("-" * 80)

    max_deviation = max(abs(s / scc_ref - 1.0) * 100.0 for s in scc_values)
    if max_deviation < 3.0:
        print(f"Max deviation: {max_deviation:.2f}% -- PASS (in linear regime)")
    elif max_deviation < 10.0:
        print(f"Max deviation: {max_deviation:.2f}% -- WARNING (near edge of linear regime)")
    else:
        print(f"Max deviation: {max_deviation:.2f}% -- FAIL (nonlinear effects or numerical noise)")

    return results


def run_time_shift_test(config_path, control_function,
                        emission_amount, consumption_amount):
    """
    Run time-shift invariance test: SCC at multiple pulse years.

    Parameters
    ----------
    config_path : str
        Path to config JSON file
    control_function : callable
        Fixed control function returning (f, s)
    emission_amount : float
        Emission pulse amount in tCO2
    consumption_amount : float
        Consumption pulse amount in $

    Returns
    -------
    list of dict
        Results for each pulse year
    """
    pulse_years = [2025, 2030, 2050, 2075, 2100]

    print("\n" + "=" * 80)
    print("TIME-SHIFT INVARIANCE TEST")
    print("=" * 80)
    print(f"\nEmission pulse: {emission_amount:.1e} tCO2")
    print(f"Consumption pulse: ${consumption_amount:.1e}")
    print(f"Pulse years: {pulse_years}")

    # Single baseline (shared)
    print("\nRunning baseline integration...")
    config_baseline = create_config_with_pulse(
        config_path, control_function, 0.0, 0.0, 0.0, 0.0
    )
    W_baseline = run_scenario(config_baseline)

    results = []
    for year in pulse_years:
        print(f"\nPulse year {year}...")

        config_e = create_config_with_pulse(
            config_path, control_function, year, emission_amount, 0.0, 0.0
        )
        W_e = run_scenario(config_e)

        config_c = create_config_with_pulse(
            config_path, control_function, 0.0, 0.0, year, consumption_amount
        )
        W_c = run_scenario(config_c)

        scc_result = compute_scc(W_baseline, W_e, W_c, emission_amount, consumption_amount)
        scc_result['pulse_year'] = year
        results.append(scc_result)
        print(f"  SCC = ${scc_result['SCC']:.2f}/tCO2")

    # Print summary table
    print("\n" + "-" * 80)
    print(f"{'Year':>8} {'dW_emission':>14} {'dW_consumption':>14} {'m_E':>14} {'m_C':>14} {'SCC ($/tCO2)':>14}")
    print("-" * 80)
    for r in results:
        print(f"{r['pulse_year']:>8.0f} {r['dW_emission']:>14.6e} {r['dW_consumption']:>14.6e} "
              f"{r['m_E']:>14.6e} {r['m_C']:>14.6e} {r['SCC']:>13.2f}")
    print("-" * 80)

    # Check for discontinuities
    scc_values = [r['SCC'] for r in results]
    for i in range(1, len(scc_values)):
        jump = abs(scc_values[i] / scc_values[i-1] - 1.0) * 100.0
        if jump > 50.0:
            print(f"WARNING: Large jump between {pulse_years[i-1]} and {pulse_years[i]}: {jump:.1f}%")

    print(f"\nSCC range: ${min(scc_values):.2f} - ${max(scc_values):.2f}/tCO2")

    return results


def save_results(all_results, output_path, baseline_path):
    """
    Save all SCC results to CSV in row-based format.

    Parameters
    ----------
    all_results : list of dict
        List of results, each containing test_type, SCC values, and metadata
    output_path : str
        Path to output CSV file
    baseline_path : str
        Path to baseline results directory
    """
    # Define column order for consistent output
    columns = [
        'timestamp',
        'test_type',
        'pulse_year',
        'scale_factor',
        'emission_amount',
        'consumption_amount',
        'W_baseline',
        'W_emission',
        'W_consumption',
        'dW_emission',
        'dW_consumption',
        'm_E',
        'm_C',
        'SCC',
        'emission_check',
        'consumption_check',
        'scc_check',
        'baseline_path',
    ]

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for result in all_results:
            row = {
                'timestamp': timestamp,
                'test_type': result.get('test_type', 'main'),
                'pulse_year': result.get('pulse_year', ''),
                'scale_factor': result.get('scale', 1.0),
                'emission_amount': result.get('emission_amount', ''),
                'consumption_amount': result.get('consumption_amount', ''),
                'W_baseline': result.get('W_baseline', ''),
                'W_emission': result.get('W_emission', ''),
                'W_consumption': result.get('W_consumption', ''),
                'dW_emission': result.get('dW_emission', ''),
                'dW_consumption': result.get('dW_consumption', ''),
                'm_E': result.get('m_E', ''),
                'm_C': result.get('m_C', ''),
                'SCC': result.get('SCC', ''),
                'emission_check': result.get('emission_check', ''),
                'consumption_check': result.get('consumption_check', ''),
                'scc_check': result.get('scc_check', ''),
                'baseline_path': baseline_path,
            }
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Social Cost of Carbon using perturbation method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python calculate_scc.py \\
        --baseline data/output/config_019_f-f-f-f-f_1_100k_mu_up_false_20260205-123456/ \\
        --pulse-year 2025

Pulse amounts default to values in config scalar_parameters:
    - amount_extra_emission (tCO2)
    - amount_extra_consumption ($)
        """
    )

    parser.add_argument('--baseline', required=True,
                        help='Path to baseline results directory')
    parser.add_argument('--pulse-year', type=float, required=True,
                        help='Year to apply pulses')
    parser.add_argument('--emission-amount', type=float, default=None,
                        help='Emission pulse amount in tCO2 (default: from config scalar_parameters)')
    parser.add_argument('--consumption-amount', type=float, default=None,
                        help='Consumption pulse amount in $ (default: from config scalar_parameters)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: scc_results.csv in baseline dir)')
    parser.add_argument('--sensitivity-test', action='store_true',
                        help='Run pulse-size sensitivity and time-shift invariance tests')

    args = parser.parse_args()

    # Load baseline
    print("=" * 80)
    print("SCC CALCULATION")
    print("=" * 80)
    print(f"\nLoading baseline from: {args.baseline}")

    baseline = load_baseline(args.baseline)
    print(f"Config: {baseline['config_path']}")
    print(f"Time range: {baseline['t_values'][0]:.0f} - {baseline['t_values'][-1]:.0f}")
    print(f"Number of time points: {len(baseline['t_values'])}")

    # Load config to get pulse amounts from scalar_parameters
    config = load_configuration(baseline['config_path'])

    # Use command line if specified, otherwise read from config
    emission_amount = args.emission_amount
    if emission_amount is None:
        emission_amount = config.scalar_params.amount_extra_emission

    consumption_amount = args.consumption_amount
    if consumption_amount is None:
        consumption_amount = config.scalar_params.amount_extra_consumption

    # Validate pulse amounts
    if emission_amount <= 0.0:
        raise ValueError(
            f"amount_extra_emission must be > 0, got {emission_amount}. "
            f"Set it in the JSON config or use --emission-amount"
        )
    if consumption_amount <= 0.0:
        raise ValueError(
            f"amount_extra_consumption must be > 0, got {consumption_amount}. "
            f"Set it in the JSON config or use --consumption-amount"
        )

    # Create fixed control function (convert calendar years to relative time)
    t_base = config.scalar_params.t_base
    control_function = create_fixed_control_function(
        baseline['t_values'],
        baseline['f_values'],
        baseline['s_values'],
        t_base
    )

    # Print pulse parameters
    print(f"\nPulse year: {args.pulse_year}")
    print(f"Emission pulse: {emission_amount:.2e} tCO2 ({emission_amount/1e9:.1f} GtCO2)")
    print(f"Consumption pulse: ${consumption_amount:.2e} (${consumption_amount/1e9:.1f}B)")

    # Create configs for 3 scenarios
    print("\n" + "-" * 80)
    print("Running integrations...")
    print("-" * 80)

    # Baseline (no pulse)
    print("\n1. Baseline (no pulse)...")
    config_baseline = create_config_with_pulse(
        baseline['config_path'], control_function,
        year_extra_emission=0.0, amount_extra_emission=0.0,
        year_extra_consumption=0.0, amount_extra_consumption=0.0
    )
    print(f"   DEBUG: year_extra_emission={config_baseline.scalar_params.year_extra_emission}, amount={config_baseline.scalar_params.amount_extra_emission}")
    W_baseline = run_scenario(config_baseline)
    print(f"   W_baseline = {W_baseline:.10e}")

    # Emission pulse
    print(f"\n2. Emission pulse (+{emission_amount/1e9:.1f} GtCO2 at {args.pulse_year})...")
    config_emission = create_config_with_pulse(
        baseline['config_path'], control_function,
        year_extra_emission=args.pulse_year, amount_extra_emission=emission_amount,
        year_extra_consumption=0.0, amount_extra_consumption=0.0
    )
    print(f"   DEBUG: year_extra_emission={config_emission.scalar_params.year_extra_emission}, amount={config_emission.scalar_params.amount_extra_emission}")
    W_emission = run_scenario(config_emission)
    print(f"   W_emission = {W_emission:.10e}")

    # Consumption pulse
    print(f"\n3. Consumption pulse (+${consumption_amount/1e9:.1f}B at {args.pulse_year})...")
    config_consumption = create_config_with_pulse(
        baseline['config_path'], control_function,
        year_extra_emission=0.0, amount_extra_emission=0.0,
        year_extra_consumption=args.pulse_year, amount_extra_consumption=consumption_amount
    )
    print(f"   DEBUG: year_extra_consumption={config_consumption.scalar_params.year_extra_consumption}, amount={config_consumption.scalar_params.amount_extra_consumption}")
    W_consumption = run_scenario(config_consumption)
    print(f"   W_consumption = {W_consumption:.10e}")

    # Compute SCC
    results = compute_scc(
        W_baseline, W_emission, W_consumption,
        emission_amount, consumption_amount
    )

    # Add metadata
    results['test_type'] = 'main'
    results['pulse_year'] = args.pulse_year
    results['emission_amount'] = emission_amount
    results['consumption_amount'] = consumption_amount
    results['baseline_path'] = args.baseline

    # Collect all results for saving
    all_results = [results]

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nWelfare values:")
    print(f"  W_baseline:     {results['W_baseline']:.10e}")
    print(f"  W_emission:     {results['W_emission']:.10e}")
    print(f"  W_consumption:  {results['W_consumption']:.10e}")

    print("\nWelfare changes:")
    check_e = "✓" if results['emission_check'] else "✗ UNEXPECTED"
    check_c = "✓" if results['consumption_check'] else "✗ UNEXPECTED"
    print(f"  ΔW_emission:    {results['dW_emission']:.6e}  {check_e} (should be negative)")
    print(f"  ΔW_consumption: {results['dW_consumption']:.6e}  {check_c} (should be positive)")

    print("\nMarginal values:")
    print(f"  m_E (∂W/∂E):    {results['m_E']:.6e}  welfare/tCO2")
    print(f"  m_C (∂W/∂C):    {results['m_C']:.6e}  welfare/$")

    print("\n" + "-" * 80)
    check_scc = "✓" if results['scc_check'] else "✗ UNEXPECTED"
    print(f"  SCC = ${results['SCC']:.2f}/tCO2  {check_scc}")
    print("-" * 80)

    # Run sensitivity tests if requested and collect results
    if args.sensitivity_test:
        sensitivity_results = run_sensitivity_test(
            baseline['config_path'], control_function,
            args.pulse_year, emission_amount, consumption_amount
        )
        # Add metadata and test_type to sensitivity results
        for r in sensitivity_results:
            r['test_type'] = 'sensitivity'
            r['pulse_year'] = args.pulse_year
            r['emission_amount'] = emission_amount * r['scale']
            r['consumption_amount'] = consumption_amount * r['scale']
        all_results.extend(sensitivity_results)

        time_shift_results = run_time_shift_test(
            baseline['config_path'], control_function,
            emission_amount, consumption_amount
        )
        # Add metadata and test_type to time-shift results
        for r in time_shift_results:
            r['test_type'] = 'time_shift'
            r['emission_amount'] = emission_amount
            r['consumption_amount'] = consumption_amount
        all_results.extend(time_shift_results)

    # Save all results
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.baseline, 'scc_results.csv')

    save_results(all_results, output_path, args.baseline)
    print(f"\nResults saved to: {output_path}")

    # Return SCC for programmatic use
    return results['SCC']


if __name__ == '__main__':
    main()
