#!/usr/bin/env python3
"""
Create comprehensive plots_full.pdf for config_037 baseline
Matching the standard 6-panel-per-page layout from individual runs
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from collections import OrderedDict

# Scenario and tax policy order
SCENARIO_ORDER = ['f-f-f-f', 't-f-f-f']
TAX_POLICY_ORDER = ['r-uniform', 'u-max', 'u-uniform']

TAX_POLICY_NAMES = {
    'r-uniform': 'Uniform fractional',
    'u-max': 'Progressive',
    'u-uniform': 'Equal utility loss'
}

SCENARIO_NAMES = {
    'f-f-f-f': 'No income effects',
    't-f-f-f': 'Income-dep damage'
}

TAX_COLORS = {
    'r-uniform': 'C0',
    'u-max': 'C1',
    'u-uniform': 'C2'
}

SCENARIO_LINESTYLES = {
    'f-f-f-f': '-',
    't-f-f-f': '--'
}

def load_config_037_results():
    """Load all config_037 results"""
    results = OrderedDict()

    for tax_policy in TAX_POLICY_ORDER:
        results[tax_policy] = OrderedDict()
        for scenario in SCENARIO_ORDER:
            pattern = f"data/output/config_037_{tax_policy}_{scenario}_100k_*"
            dirs = sorted(glob.glob(pattern))

            if not dirs:
                continue

            result_dir = dirs[-1]
            csv_file = glob.glob(f"{result_dir}/*_optimization_results.csv")[0]
            df = pd.read_csv(csv_file)
            df.columns = [col.split(',')[0].strip('"') for col in df.columns]

            results[tax_policy][scenario] = df

    return results

def plot_variable(ax, results, var_name, ylabel, title, xlim, ylim_min=None, log_scale=False):
    """Plot a single variable on given axes"""
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue

            df = results[tax_policy][scenario]

            if var_name not in df.columns:
                continue

            label = f"{TAX_POLICY_NAMES[tax_policy]} - {SCENARIO_NAMES[scenario]}"

            ax.plot(df['t'], df[var_name],
                   color=TAX_COLORS[tax_policy],
                   linestyle=SCENARIO_LINESTYLES[scenario],
                   linewidth=1.2,
                   label=label)

    ax.set_xlabel('Time (years)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(xlim)
    if ylim_min is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ylim_min, ymax)
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)

def create_plots_full(results, output_file, xlim, time_label):
    """Create comprehensive multi-page PDF with 6 panels per page"""

    with PdfPages(output_file) as pdf:

        # Page 1: Economic variables
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Results Comparison{time_label}', fontsize=14, fontweight='bold')

        plot_variable(axes[0, 0], results, 'y_net',
                     'Effective Income ($/person/yr)',
                     'y_net: Effective Per-Capita Income', xlim)

        plot_variable(axes[0, 1], results, 'K',
                     'Capital stock ($)',
                     'K: Capital Stock Over Time', xlim, log_scale=True)

        plot_variable(axes[0, 2], results, 'Consumption',
                     'Total consumption ($/yr)',
                     'Consumption: Total Consumption', xlim, log_scale=True)

        plot_variable(axes[1, 0], results, 'Savings',
                     'Gross investment ($/yr)',
                     'Savings: Gross Investment', xlim, log_scale=True)

        plot_variable(axes[1, 1], results, 's',
                     'Savings rate',
                     's: Savings Rate Over Time', xlim, ylim_min=0)

        plot_variable(axes[1, 2], results, 'Y_gross',
                     'Gross production ($/yr)',
                     'Y_gross: Gross Production', xlim, log_scale=True)

        # Add legend to first plot
        axes[0, 0].legend(fontsize=7, loc='best', framealpha=0.9)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 2: Climate and control variables
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Results Comparison{time_label}', fontsize=14, fontweight='bold')

        plot_variable(axes[0, 0], results, 'Y_net',
                     'Net output ($/yr)',
                     'Y_net: Net Output After Damages & Abatement', xlim, log_scale=True)

        plot_variable(axes[0, 1], results, 'delta_T',
                     'Temperature change (°C)',
                     'delta_T: Temperature Change from Pre-Industrial', xlim, ylim_min=0)

        plot_variable(axes[0, 2], results, 'E',
                     'CO₂ emissions (tCO2/yr)',
                     'E: CO₂ Emissions Rate', xlim)

        plot_variable(axes[1, 0], results, 'Ecum',
                     'Cumulative emissions (tCO2)',
                     'Ecum: Cumulative CO₂ Emissions', xlim)

        plot_variable(axes[1, 1], results, 'f',
                     'log₁₀(MAC) [log₁₀($/tCO₂)]',
                     'f: Control: log₁₀(Marginal Abatement Cost)', xlim)

        plot_variable(axes[1, 2], results, 'mu',
                     'Abatement fraction',
                     'mu: Emissions Abatement Fraction', xlim, ylim_min=0)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 3: Costs, damages, and welfare
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Results Comparison{time_label}', fontsize=14, fontweight='bold')

        plot_variable(axes[0, 0], results, 'Lambda',
                     'Abatement cost fraction',
                     'Lambda: Abatement Cost (% of Output)', xlim, ylim_min=0)

        plot_variable(axes[0, 1], results, 'AbateCost',
                     'Abatement cost ($/yr)',
                     'AbateCost: Total Abatement Cost', xlim, log_scale=True)

        plot_variable(axes[0, 2], results, 'Omega',
                     'Climate damage fraction',
                     'Omega: Climate Damage (% of Output)', xlim, ylim_min=0)

        plot_variable(axes[1, 0], results, 'Climate_damage',
                     'Climate damage ($/yr)',
                     'Climate_damage: Total Climate Damage', xlim, log_scale=True)

        plot_variable(axes[1, 1], results, 'gini',
                     'Gini before redistribution',
                     'Gini: Background Gini Index', xlim, ylim_min=0)

        plot_variable(axes[1, 2], results, 'U',
                     'Mean utility',
                     'U: Mean Utility Per Capita', xlim)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        print(f"Created: {output_file}")

def main():
    print("="*80)
    print("Creating Config 037 Comprehensive Plots (Standard 6-panel format)")
    print("="*80)

    print("\nLoading results...")
    results = load_config_037_results()

    print("\nCreating plots_full.pdf (all years: 2020-2420)...")
    create_plots_full(results,
                     output_file='data/output/config_037_plots_full.pdf',
                     xlim=(2020, 2420),
                     time_label=' (2020-2420)')

    print("\nCreating plots_full_2020-2100.pdf (near-term: 2020-2100)...")
    create_plots_full(results,
                     output_file='data/output/config_037_plots_full_2020-2100.pdf',
                     xlim=(2020, 2100),
                     time_label=' (2020-2100)')

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated:")
    print("  - data/output/config_037_plots_full.pdf (3 pages, 6 panels each)")
    print("  - data/output/config_037_plots_full_2020-2100.pdf (3 pages, 6 panels each)")

if __name__ == "__main__":
    main()
