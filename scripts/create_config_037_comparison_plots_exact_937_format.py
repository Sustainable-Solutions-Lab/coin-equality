#!/usr/bin/env python3
"""
Create comparison plots for config_037 matching EXACT format from results937
- Page 1: Optimization Summary (2 panels)
- Page 2-3: Results Comparison with legend on left (specific color mapping)
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from collections import OrderedDict
import re

# CRITICAL: Exact case order matching results937 color scheme
CASE_ORDER = [
    ('r-uniform', 'f-f-f-f'),
    ('r-uniform', 't-f-f-f'),
    ('u-max', 'f-f-f-f'),
    ('u-max', 't-f-f-f'),
    ('u-uniform', 'f-f-f-f'),
    ('u-uniform', 't-f-f-f')
]

# Color mapping: tab10 colormap C0-C5
CASE_COLORS = {
    ('r-uniform', 'f-f-f-f'): 'C0',   # blue
    ('r-uniform', 't-f-f-f'): 'C1',   # orange
    ('u-max', 'f-f-f-f'): 'C2',       # green
    ('u-max', 't-f-f-f'): 'C3',       # red
    ('u-uniform', 'f-f-f-f'): 'C4',   # purple
    ('u-uniform', 't-f-f-f'): 'C5'    # brown
}

def load_config_037_data():
    """Load all config_037 results and optimization summaries"""
    results = OrderedDict()
    opt_summaries = {}

    for tax_policy, scenario in CASE_ORDER:
        case_key = (tax_policy, scenario)

        # Find result directory
        pattern = f"data/output/config_037_{tax_policy}_{scenario}_100k_*"
        dirs = sorted(glob.glob(pattern))

        if not dirs:
            print(f"WARNING: No results found for {tax_policy} {scenario}")
            continue

        result_dir = dirs[-1]

        # Load results CSV
        csv_file = glob.glob(f"{result_dir}/*_optimization_results.csv")[0]
        df = pd.read_csv(csv_file)
        df.columns = [col.split(',')[0].strip('"') for col in df.columns]
        results[case_key] = df

        # Extract optimization summary from terminal output
        terminal_file = glob.glob(f"{result_dir}/terminal_output.txt")[0]
        with open(terminal_file, 'r') as f:
            text = f.read()

        # Extract objective value and evaluations
        obj_match = re.search(r'Optimal objective:\s+([-\d.e]+)', text)
        eval_match = re.search(r'Function evaluations:\s+(\d+)', text)

        if obj_match and eval_match:
            opt_summaries[case_key] = {
                'objective': float(obj_match.group(1)),
                'evaluations': int(eval_match.group(1))
            }

        print(f"Loaded: {tax_policy} {scenario} ({len(df)} time points)")

    return results, opt_summaries

def create_legend_box(fig, ax_position):
    """Create legend box matching results937 style"""
    legend_lines = []
    legend_labels = []

    for tax_policy, scenario in CASE_ORDER:
        case_key = (tax_policy, scenario)
        color = CASE_COLORS[case_key]

        line = plt.Line2D([0], [0], color=color, linewidth=2)
        label = f"config_037_{tax_policy}_{scenario}"

        legend_lines.append(line)
        legend_labels.append(label)

    # Create legend in specified position
    legend = fig.legend(legend_lines, legend_labels,
                       loc='upper left',
                       bbox_to_anchor=ax_position,
                       fontsize=9,
                       frameon=True,
                       framealpha=0.95)

    return legend

def plot_variable(ax, results, var_name, ylabel, title, xlim, ylim_min=None, log_scale=False):
    """Plot a single variable on given axes"""
    for case_key in CASE_ORDER:
        if case_key not in results:
            continue

        df = results[case_key]

        if var_name not in df.columns:
            continue

        color = CASE_COLORS[case_key]

        ax.plot(df['t'], df[var_name],
               color=color,
               linewidth=1.5,
               alpha=0.9)

    ax.set_xlabel('Time (years)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(xlim)

    if ylim_min is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ylim_min, ymax)

    if log_scale:
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

def create_comparison_plots(results, opt_summaries, output_file, xlim, time_label):
    """Create comprehensive comparison plots matching results937 format"""

    with PdfPages(output_file) as pdf:

        # ===== Page 1: Optimization Summary =====
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Optimization Summary Comparison', fontsize=14, fontweight='bold')

        # Left panel: Objective Improvement by Iteration (scatter plot)
        ax = axes[0]
        for i, case_key in enumerate(CASE_ORDER):
            if case_key not in opt_summaries:
                continue

            obj = opt_summaries[case_key]['objective']
            color = CASE_COLORS[case_key]
            tax_policy, scenario = case_key
            label = f"{tax_policy}_{scenario}"

            # Plot as scatter point
            ax.scatter(i+1, obj, color=color, s=100, alpha=0.8, label=label, zorder=3)

            # Add text label below point
            ax.text(i+1, obj * 1.002, label,
                   ha='center', va='bottom', fontsize=7, rotation=0)

        ax.set_xlabel('Configuration', fontsize=11)
        ax.set_ylabel('Objective Value', fontsize=11)
        ax.set_title('Objective Improvement by Iteration', fontsize=12, fontweight='bold')
        ax.set_xlim(0, len(CASE_ORDER) + 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(1, len(CASE_ORDER) + 1))
        ax.set_xticklabels([])  # Hide x-tick labels since we have text annotations

        # Right panel: Evaluations Comparison (scatter plot)
        ax = axes[1]
        for i, case_key in enumerate(CASE_ORDER):
            if case_key not in opt_summaries:
                continue

            evals = opt_summaries[case_key]['evaluations']
            color = CASE_COLORS[case_key]
            tax_policy, scenario = case_key
            label = f"{tax_policy}_{scenario}"

            # Plot as scatter point
            ax.scatter(i+1, evals, color=color, s=100, alpha=0.8, zorder=3)

            # Add text label below point
            ax.text(i+1, evals * 0.98, label,
                   ha='center', va='top', fontsize=7, rotation=0)

        ax.set_xlabel('Configuration', fontsize=11)
        ax.set_ylabel('Function Evaluations', fontsize=11)
        ax.set_title('Evaluations Comparison', fontsize=12, fontweight='bold')
        ax.set_xlim(0, len(CASE_ORDER) + 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(1, len(CASE_ORDER) + 1))
        ax.set_xticklabels([])

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # ===== Page 2: Results Comparison (Part 1) =====
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Results Comparison{time_label}', fontsize=14, fontweight='bold')

        # Create grid with space for legend on left
        gs = fig.add_gridspec(2, 3, left=0.22, right=0.98, top=0.93, bottom=0.07,
                             hspace=0.3, wspace=0.3)

        # Add legend on left side
        create_legend_box(fig, (0.02, 0.93))

        # Create axes
        axes = []
        for i in range(2):
            for j in range(3):
                if i * 3 + j < 5:  # Only 5 panels on page 2
                    axes.append(fig.add_subplot(gs[i, j]))

        # Panel 0: y_net
        plot_variable(axes[0], results, 'y_net',
                     'Effective Income ($/person/yr)',
                     'y_net: Effective Per-Capita Income', xlim)

        # Panel 1: K
        plot_variable(axes[1], results, 'K',
                     'Capital stock ($)',
                     'K: Capital Stock Over Time', xlim, log_scale=True)

        # Panel 2: Consumption
        plot_variable(axes[2], results, 'Consumption',
                     'Total consumption ($/yr)',
                     'Consumption: Total Consumption', xlim, log_scale=True)

        # Panel 3: Savings
        plot_variable(axes[3], results, 'Savings',
                     'Gross investment ($/yr)',
                     'Savings: Gross Investment', xlim, log_scale=True)

        # Panel 4: s (savings rate)
        plot_variable(axes[4], results, 's',
                     'Savings rate',
                     's: Savings Rate Over Time', xlim, ylim_min=0)

        pdf.savefig(fig, dpi=150)
        plt.close()

        # ===== Page 3: Results Comparison (Part 2) =====
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Results Comparison{time_label}', fontsize=14, fontweight='bold')

        # Create grid
        gs = fig.add_gridspec(2, 3, left=0.08, right=0.98, top=0.93, bottom=0.07,
                             hspace=0.3, wspace=0.3)

        # Create axes
        axes = []
        for i in range(2):
            for j in range(3):
                if i * 3 + j < 5:  # Only 5 panels on page 3
                    axes.append(fig.add_subplot(gs[i, j]))

        # Panel 0: Y_gross
        plot_variable(axes[0], results, 'Y_gross',
                     'Gross production ($/yr)',
                     'Y_gross: Gross Production', xlim, log_scale=True)

        # Panel 1: Y_net
        plot_variable(axes[1], results, 'Y_net',
                     'Net output ($/yr)',
                     'Y_net: Net Output After Damages & Abatement', xlim, log_scale=True)

        # Panel 2: delta_T
        plot_variable(axes[2], results, 'delta_T',
                     'Temperature change (°C)',
                     'delta_T: Temperature Change from Pre-Industrial', xlim, ylim_min=0)

        # Panel 3: E
        plot_variable(axes[3], results, 'E',
                     'CO₂ emissions (tCO2/yr)',
                     'E: CO₂ Emissions Rate', xlim)

        # Panel 4: Ecum
        plot_variable(axes[4], results, 'Ecum',
                     'Cumulative emissions (tCO2)',
                     'Ecum: Cumulative CO₂ Emissions', xlim)

        pdf.savefig(fig, dpi=150)
        plt.close()

        print(f"Created: {output_file}")

def main():
    print("="*80)
    print("Creating Config 037 Comparison Plots (Exact 937 Format)")
    print("="*80)

    print("\nLoading results...")
    results, opt_summaries = load_config_037_data()

    print("\nCreating comparison_plots_2020-2100.pdf...")
    create_comparison_plots(results, opt_summaries,
                          output_file='data/output/config_037_comparison_plots_2020-2100.pdf',
                          xlim=(2020, 2100),
                          time_label=' (2020-2100)')

    print("\nCreating comparison_plots_2020-2420.pdf...")
    create_comparison_plots(results, opt_summaries,
                          output_file='data/output/config_037_comparison_plots_2020-2420.pdf',
                          xlim=(2020, 2420),
                          time_label=' (2020-2420)')

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated:")
    print("  - data/output/config_037_comparison_plots_2020-2100.pdf (exact 937 format)")
    print("  - data/output/config_037_comparison_plots_2020-2420.pdf (exact 937 format)")

if __name__ == "__main__":
    main()
