#!/usr/bin/env python3
"""
Compare optimization convergence across different max_evaluations settings.

Analyzes control variables (f and s) from optimization results to compare:
- Statistics (mean, std, median) for different iteration counts
- RMS differences relative to best objective value baseline (greatest = least negative)
- Two time periods: [0, 400] and [5, 80]
- Groups results by flag patterns with independent baseline selection per group

Usage:
    ./compare_optimization_convergence.py [directory_patterns...]

Examples:
    ./compare_optimization_convergence.py                           # Use default: data/output/*
    ./compare_optimization_convergence.py data/output/config_013*   # Only config_013 runs
    ./compare_optimization_convergence.py data/output/config_011* data/output/config_012*
"""

import sys
import os
import argparse
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re

def extract_config_info(directory_name):
    """
    Extract flag pattern, max_evaluations, and algorithm from directory name.

    Handles multiple directory name formats:
    - config_NNN_{flags}_{n_points}_{fract_gdp}_{max_eval}_{opt}_[BOBYQA_]TIMESTAMP
    - config_NNN_{flags}_{fract_gdp}_{max_eval}_list[N]_TIMESTAMP

    Returns: (flag_pattern, max_eval, algorithm) or (None, None, None) if cannot parse
    """
    # Pattern 1: config_NNN_{flags}_{n_points}_{fract_gdp}_{max_eval}_{opt}_[BOBYQA_]TIMESTAMP
    pattern_el = r'config_\d+_([tf]-[tf]-[tf]-[tf]-[tf])_\d+_([\d.]+)_(\d+k?).*_el_'
    # Pattern 2: config_NNN_{flags}_{fract_gdp}_{max_eval}_list[N]_TIMESTAMP (no n_points)
    pattern_list = r'config_\d+_([tf]-[tf]-[tf]-[tf]-[tf])_([\d.]+)_(\d+k?)_list'

    match = re.search(pattern_el, directory_name)
    if not match:
        match = re.search(pattern_list, directory_name)

    if match:
        flag_pattern = match.group(1)
        max_eval_str = match.group(3)

        # Convert k suffix to thousands
        if max_eval_str.endswith('k'):
            max_eval = int(max_eval_str[:-1]) * 1000
        else:
            max_eval = int(max_eval_str)

        # Detect algorithm
        algorithm = 'BOBYQA' if '_BOBYQA' in directory_name else 'SBPLX'

        return flag_pattern, max_eval, algorithm

    return None, None, None

def parse_optimal_objective(output_dir):
    """
    Parse the optimal objective value from terminal_output.txt.

    Searches for the last occurrence of 'Objective value:' in the file
    to get the full precision value from iteration output. The iteration
    output shows unscaled values, so we multiply by OBJECTIVE_SCALE to
    get the properly scaled objective value.

    Returns the optimal objective value as a float, or None if not found.
    """
    from src.constants import OBJECTIVE_SCALE

    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    if not terminal_output_path.exists():
        return None

    last_objective = None
    with open(terminal_output_path, 'r') as f:
        for line in f:
            if 'Objective value:' in line:
                value_str = line.split(':')[1].strip()
                last_objective = float(value_str) * OBJECTIVE_SCALE

    return last_objective

def parse_optimization_summary(output_dir):
    """
    Parse optimization summary from terminal_output.txt.

    Returns a dict with:
        - iterations_performed: int
        - total_evaluations: int
        - final_control_points: int
        - total_runtime_seconds: float
    Returns None for any value not found.
    """
    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    if not terminal_output_path.exists():
        return {
            'iterations_performed': None,
            'total_evaluations': None,
            'final_control_points': None,
            'total_runtime_seconds': None
        }

    summary = {
        'iterations_performed': None,
        'total_evaluations': None,
        'final_control_points': None,
        'total_runtime_seconds': None
    }

    with open(terminal_output_path, 'r') as f:
        for line in f:
            if 'Iterations performed:' in line:
                value_str = line.split(':')[1].strip()
                summary['iterations_performed'] = int(value_str)
            elif 'Total evaluations:' in line:
                value_str = line.split(':')[1].strip()
                summary['total_evaluations'] = int(value_str)
            elif 'Final control points:' in line:
                value_str = line.split(':')[1].strip()
                summary['final_control_points'] = int(value_str)
            elif 'Total runtime:' in line:
                # Extract seconds from format: "Total runtime: 8814.53 seconds (146.91 minutes)"
                value_str = line.split(':')[1].strip()
                seconds_str = value_str.split()[0]  # Get "8814.53" before "seconds"
                summary['total_runtime_seconds'] = float(seconds_str)

    return summary

def load_optimization_results(directory_patterns):
    """
    Load all optimization results from directories matching patterns.

    Parameters
    ----------
    directory_patterns : list of str
        List of glob patterns for directories to search (e.g., ['data/output/config_013*'])

    Returns
    -------
    dict
        Dictionary keyed by directory_name, each entry containing:
        - 'flag_pattern': str
        - 'algorithm': str
        - 'max_eval': int
        - 'df': DataFrame with optimization results
        - 'objective': float (optimal objective value)
        - 'summary': dict with iterations_performed, total_evaluations, etc.
    """
    all_runs = {}

    # Expand all patterns and find optimization_results.csv files
    csv_files = []
    for pattern in directory_patterns:
        # Expand the glob pattern to get directories
        matching_dirs = glob.glob(pattern)
        for dir_path in matching_dirs:
            dir_path = Path(dir_path)
            if dir_path.is_dir():
                # Look for optimization_results.csv files in this directory
                csv_files.extend(dir_path.glob('*_optimization_results.csv'))

    print(f"Found {len(csv_files)} optimization results files")

    for csv_file in csv_files:
        directory_name = csv_file.parent.name
        flag_pattern, max_eval, algorithm = extract_config_info(directory_name)

        if flag_pattern is None:
            print(f"Warning: Could not parse {directory_name}")
            continue

        # Read CSV and clean column names
        df = pd.read_csv(csv_file)
        # Column names are in format "variable_name, Description, (units)"
        # Extract just the variable name
        df.columns = [col.split(',')[0].strip() for col in df.columns]

        # Parse optimal objective value and optimization summary
        optimal_obj = parse_optimal_objective(csv_file.parent)
        summary = parse_optimization_summary(csv_file.parent)

        # Skip failed runs (objective = -1e+30 or worse)
        if optimal_obj is not None and optimal_obj < -1e+20:
            print(f"  Skipping failed run: {directory_name} (obj={optimal_obj})")
            continue

        # Detect and convert calendar years to relative time if needed
        # If min(t) > 1000, assume calendar years and convert to relative time
        if df['t'].min() > 1000:
            t_base = 2020.0  # Assume t_base = 2020
            df['t'] = df['t'] - t_base
            print(f"  Converted calendar years to relative time for: {directory_name}")

        # Store with directory_name as unique key
        all_runs[directory_name] = {
            'flag_pattern': flag_pattern,
            'algorithm': algorithm,
            'max_eval': max_eval,
            'df': df,
            'objective': optimal_obj,
            'summary': summary,
        }
        print(f"  Loaded: {directory_name} - {flag_pattern}, {algorithm}, {max_eval} evals, obj={optimal_obj}")

    return all_runs

def compute_statistics(df, time_period, variables=['f', 's']):
    """
    Compute mean, std, median for variables in given time period.

    Parameters
    ----------
    df : DataFrame
        Results with 't', 'f', 's' columns
    time_period : tuple
        (t_min, t_max) for filtering
    variables : list
        Variables to analyze

    Returns
    -------
    dict
        {variable: {'mean': float, 'std': float, 'median': float}}
    """
    t_min, t_max = time_period
    mask = (df['t'] >= t_min) & (df['t'] <= t_max)
    df_filtered = df[mask]

    stats = {}
    for var in variables:
        if var in df_filtered.columns:
            values = df_filtered[var].values
            stats[var] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
            }
        else:
            stats[var] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}

    return stats

def compute_rms_difference(df1, df2, time_period, variables=['f', 's']):
    """
    Compute RMS difference between two DataFrames for given time period.

    Parameters
    ----------
    df1, df2 : DataFrame
        Results to compare (must have same time points)
    time_period : tuple
        (t_min, t_max) for filtering
    variables : list
        Variables to compare

    Returns
    -------
    dict
        {variable: rms_difference}
    """
    t_min, t_max = time_period

    # Filter both DataFrames
    mask1 = (df1['t'] >= t_min) & (df1['t'] <= t_max)
    mask2 = (df2['t'] >= t_min) & (df2['t'] <= t_max)

    df1_filtered = df1[mask1].sort_values('t').reset_index(drop=True)
    df2_filtered = df2[mask2].sort_values('t').reset_index(drop=True)

    # Ensure same time points (interpolate if needed)
    if not np.allclose(df1_filtered['t'].values, df2_filtered['t'].values):
        print(f"Warning: Time points don't match, interpolating...")
        # Use df1 time points as reference
        t_ref = df1_filtered['t'].values
        rms = {}
        for var in variables:
            if var in df1_filtered.columns and var in df2_filtered.columns:
                val1 = df1_filtered[var].values
                val2 = np.interp(t_ref, df2_filtered['t'].values, df2_filtered[var].values)
                rms[var] = np.sqrt(np.mean((val1 - val2)**2))
            else:
                rms[var] = np.nan
        return rms

    rms = {}
    for var in variables:
        if var in df1_filtered.columns and var in df2_filtered.columns:
            diff = df1_filtered[var].values - df2_filtered[var].values
            rms[var] = np.sqrt(np.mean(diff**2))
        else:
            rms[var] = np.nan

    return rms

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare optimization convergence across different runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                                    # Use default: data/output/*
    %(prog)s data/output/config_013*            # Only config_013 runs
    %(prog)s data/output/config_011* data/output/config_012*
    %(prog)s 'data/output/*_f-f-t-f-f_*'        # Quote patterns with special chars
        """
    )
    parser.add_argument(
        'directories',
        nargs='*',
        default=['data/output/*'],
        help='Directory patterns to search (default: data/output/*)'
    )
    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_arguments()

    print("="*80)
    print("OPTIMIZATION CONVERGENCE ANALYSIS")
    print("="*80)
    print(f"Directory patterns: {args.directories}")

    # Load all results (keyed by directory name)
    all_runs = load_optimization_results(args.directories)

    if not all_runs:
        print("ERROR: No results found!")
        sys.exit(1)

    # Time periods to analyze
    time_periods = {
        'Full [0-400]': (0, 400),
        'Early [5-80]': (5, 80),
    }

    # Get all unique flag patterns
    flag_patterns = sorted(set(run['flag_pattern'] for run in all_runs.values()))

    print(f"\nTotal runs loaded: {len(all_runs)}")
    print(f"Flag patterns: {flag_patterns}")

    # ========================================================================
    # Find best baseline for each flag pattern
    # Best = greatest (least negative) objective value
    # ========================================================================
    print("\n" + "="*80)
    print("SELECTING BASELINE FOR EACH FLAG PATTERN")
    print("="*80)

    baselines = {}
    for flag_pattern in flag_patterns:
        best_obj = -np.inf
        best_run_name = None

        for run_name, run_data in all_runs.items():
            if run_data['flag_pattern'] != flag_pattern:
                continue
            obj = run_data['objective']
            if obj is not None and obj > best_obj:
                best_obj = obj
                best_run_name = run_name

        if best_run_name is not None:
            baselines[flag_pattern] = {
                'run_name': best_run_name,
                'objective': best_obj,
                'df': all_runs[best_run_name]['df']
            }
            print(f"\n{flag_pattern}: {best_run_name} (obj = {best_obj:.6e})")
        else:
            print(f"\n{flag_pattern}: WARNING - No valid baseline found!")

    # ========================================================================
    # Summary table - process every run individually (one row per run/period)
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    # Create summary DataFrame - one row per run per period
    summary_rows = []

    for run_name in sorted(all_runs.keys()):
        run_data = all_runs[run_name]
        flag_pattern = run_data['flag_pattern']
        algorithm = run_data['algorithm']
        max_eval = run_data['max_eval']
        df = run_data['df']
        obj_value = run_data['objective']
        summary = run_data['summary']

        for period_name, time_period in time_periods.items():
            stats = compute_statistics(df, time_period, variables=['f', 's'])

            # Get baseline info for this flag pattern
            baseline_info = baselines.get(flag_pattern)
            baseline_obj = baseline_info['objective'] if baseline_info else None
            baseline_df = baseline_info['df'] if baseline_info else None
            baseline_run_name = baseline_info['run_name'] if baseline_info else None

            # Compute objective departure (best - current, positive means worse than best)
            obj_departure = (baseline_obj - obj_value) if (baseline_obj is not None and obj_value is not None) else np.nan

            # Compute RMS vs baseline
            if baseline_df is not None:
                if run_name == baseline_run_name:
                    rms = {'f': 0.0, 's': 0.0}
                else:
                    rms = compute_rms_difference(df, baseline_df, time_period, variables=['f', 's'])
            else:
                rms = {'f': np.nan, 's': np.nan}

            # Build single row with f and s columns
            row = {
                'period': period_name,
                'flags': flag_pattern,
                'algorithm': algorithm,
                'max_eval': max_eval,
                'run_name': run_name,
                'objective': obj_value,
                'obj_departure': obj_departure,
                'iterations_performed': summary['iterations_performed'],
                'total_evaluations': summary['total_evaluations'],
                'final_control_points': summary['final_control_points'],
                'total_runtime_seconds': summary['total_runtime_seconds'],
            }

            # Add f statistics
            if 'f' in stats:
                row['f_mean'] = stats['f']['mean']
                row['f_std'] = stats['f']['std']
                row['f_median'] = stats['f']['median']
                row['f_rms_vs_baseline'] = rms.get('f', np.nan)

            # Add s statistics
            if 's' in stats:
                row['s_mean'] = stats['s']['mean']
                row['s_std'] = stats['s']['std']
                row['s_median'] = stats['s']['median']
                row['s_rms_vs_baseline'] = rms.get('s', np.nan)

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        # Sort by: period (desc for Full before Early), flags (asc), objective (desc)
        summary_df = summary_df.sort_values(
            by=['period', 'flags', 'objective'],
            ascending=[False, True, False]
        ).reset_index(drop=True)

        # Print summary tables
        print("\nFull period [0-400]:")
        print("-" * 80)
        full_df = summary_df[summary_df['period'] == 'Full [0-400]']
        display_cols = ['flags', 'run_name', 'objective', 'obj_departure',
                        'f_mean', 'f_std', 'f_rms_vs_baseline',
                        's_mean', 's_std', 's_rms_vs_baseline']
        print(full_df[display_cols].to_string(index=False))

        print("\n\nEarly period [5-80]:")
        print("-" * 80)
        early_df = summary_df[summary_df['period'] == 'Early [5-80]']
        print(early_df[display_cols].to_string(index=False))

        # Save to CSV
        output_file = 'optimization_convergence_analysis.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n\nSummary saved to: {output_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
