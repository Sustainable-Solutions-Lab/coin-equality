#!/usr/bin/env python3
"""
Create standard comparison plots for config_037 baseline
Uses the STANDARD comparison format from visualization_utils.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comparison_utils import discover_result_directories, load_optimization_summaries, load_results_csvs
from src.visualization_utils import create_comparison_report_pdf

def main():
    print("="*80)
    print("Creating Config 037 Comparison Plots (Standard Format)")
    print("="*80)

    # Define the config_037 result directories
    pattern = "data/output/config_037_*_100k_*_20260328-033040"

    print(f"\nDiscovering result directories matching: {pattern}")
    directories = discover_result_directories([pattern])

    # Sort directories to ensure consistent order:
    # r-uniform f-f-f-f, r-uniform t-f-f-f, u-max f-f-f-f, u-max t-f-f-f, u-uniform f-f-f-f, u-uniform t-f-f-f
    def sort_key(d):
        d_str = str(d)
        if 'r-uniform_f-f-f-f' in d_str:
            return (0, d_str)
        elif 'r-uniform_t-f-f-f' in d_str:
            return (1, d_str)
        elif 'u-max_f-f-f-f' in d_str:
            return (2, d_str)
        elif 'u-max_t-f-f-f' in d_str:
            return (3, d_str)
        elif 'u-uniform_f-f-f-f' in d_str:
            return (4, d_str)
        elif 'u-uniform_t-f-f-f' in d_str:
            return (5, d_str)
        else:
            return (99, d_str)

    directories = sorted(directories, key=sort_key)

    print(f"\nFound {len(directories)} result directories (in order):")
    for d in directories:
        print(f"  - {d.name}")

    # Load optimization summaries
    print("\nLoading optimization summaries...")
    optimization_data = load_optimization_summaries(directories)

    # Load results CSVs
    print("Loading results...")
    results_data = load_results_csvs(directories)

    # Create comparison plots using standard format (2020-2100)
    print("\nGenerating comparison_plots_2020-2100.pdf (standard format)...")
    output_pdf_short = Path('data/output/config_037_comparison_plots_2020-2100.pdf')
    create_comparison_report_pdf(optimization_data, results_data, output_pdf_short,
                                  t_min=2020, t_max=2100, t_offset=0)

    # Create comparison plots using standard format (full horizon)
    print("Generating comparison_plots_2020-2420.pdf (standard format)...")
    output_pdf_full = Path('data/output/config_037_comparison_plots_2020-2420.pdf')
    create_comparison_report_pdf(optimization_data, results_data, output_pdf_full,
                                  t_min=2020, t_max=2420, t_offset=0)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated (using STANDARD format):")
    print(f"  - {output_pdf_short}")
    print(f"  - {output_pdf_full}")
    print("\nFormat:")
    print("  - Page 1: Optimization summary (objective, time, evaluations)")
    print("  - Pages 2+: Results comparison (legend in first panel, 5 plots per page)")
    print("  - Colors: tab10 automatic (blue, orange, green, red, purple, brown)")

if __name__ == "__main__":
    main()
