#!/usr/bin/env python3
"""
Compare optimization results across multiple runs.

Usage:
    python compare_results.py <path1> [path2] [path3] [...]

Arguments:
    path1, path2, ...: Directory paths or glob patterns
                       (e.g., "results/run_*/" or "results/baseline/")
                       Unlimited number of paths supported

Outputs:
    optimization_comparison_summary.xlsx - Optimization metrics by iteration
    results_comparison_summary.xlsx      - Time series results for all variables
    comparison_plots.pdf                 - PDF report with comparative visualizations (full time range)
    comparison_plots_2025-2100.pdf       - PDF report focused on 2025-2100 period

Examples:
    python compare_results.py data/output/config_014_*_1_100k_list
    python compare_results.py "data/output/scenario_*/"
    python compare_results.py data/output/config_011_f-f-f-f-f_* data/output/config_011_t-f-t-f-f_*
"""

import sys
import os
import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comparison_utils import (
    discover_result_directories,
    load_optimization_summaries,
    load_results_csvs,
    create_comparison_xlsx,
    create_results_comparison_xlsx
)
from src.visualization_utils import create_comparison_report_pdf


def create_comparison_directory():
    """Create timestamped output directory for comparison results."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path('data') / 'output' / f'comparison_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """Main entry point for comparison tool."""

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <path1> [path2] [path3] [...]")
        print("\nArguments:")
        print("  path1, path2, ...: Directory paths or glob patterns (unlimited)")
        print("\nExamples:")
        print('  python compare_results.py data/output/config_014_*_1_100k_list')
        print('  python compare_results.py "data/output/scenario_*/"')
        print('  python compare_results.py data/output/config_011_f-f-f-f-f_* data/output/config_011_t-f-t-f-f_*')
        sys.exit(1)

    path_patterns = sys.argv[1:]

    print(f"Discovering result directories from patterns: {path_patterns}")

    # Discover directories
    directories = discover_result_directories(path_patterns)

    print(f"\nFound {len(directories)} result directories:")
    for d in directories:
        print(f"  - {d}")

    # Load optimization summaries
    print("\nLoading optimization_summary.csv files...")
    optimization_data = load_optimization_summaries(directories)

    # Load results CSVs (if available)
    print("Loading results.csv files...")
    results_data = load_results_csvs(directories)

    # Create output directory
    output_dir = create_comparison_directory()
    print(f"\nCreating output directory: {output_dir}")

    # Generate XLSX comparisons
    print("\nGenerating optimization comparison workbook...")
    optimization_xlsx_path = output_dir / 'optimization_comparison_summary.xlsx'
    create_comparison_xlsx(optimization_data, directories, optimization_xlsx_path)

    print("Generating results comparison workbook...")
    results_xlsx_path = output_dir / 'results_comparison_summary.xlsx'
    create_results_comparison_xlsx(results_data, directories, results_xlsx_path)

    # Generate PDF comparison report (full time range)
    print("Generating PDF comparison report (full time range)...")
    pdf_path = output_dir / 'comparison_plots.pdf'
    create_comparison_report_pdf(optimization_data, results_data, pdf_path)

    # Generate PDF comparison report (2025-2100 time range)
    # Results data now uses calendar years, so filter directly on calendar years
    print("Generating PDF comparison report (2025-2100)...")
    pdf_path_short = output_dir / 'comparison_plots_2025-2100.pdf'
    create_comparison_report_pdf(optimization_data, results_data, pdf_path_short,
                                  t_min=2025, t_max=2100, t_offset=0)

    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"  Output directory:      {output_dir.absolute()}")
    print(f"  Optimization workbook: {optimization_xlsx_path.name}")
    print(f"  Results workbook:      {results_xlsx_path.name}")
    print(f"  PDF report:            {pdf_path.name}")
    print(f"  PDF report (short):    {pdf_path_short.name}")
    print("="*80)


if __name__ == '__main__':
    main()
