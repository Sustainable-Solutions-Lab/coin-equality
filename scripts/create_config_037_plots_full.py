#!/usr/bin/env python3
"""
Create comprehensive plots_full.pdf for config_037 baseline
Similar to individual run outputs but combining all 6 cases
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

def create_plot_page(pdf, results, var_name, ylabel, title, ylim_min=None, xlim=(2020, 2120)):
    """Create a single plot page with all cases"""
    fig, ax = plt.subplots(figsize=(11, 8.5))

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
                   label=label,
                   linewidth=1.8)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(xlim)
    if ylim_min is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ylim_min, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()

def create_plots_full(results, output_file='data/output/config_037_plots_full.pdf', xlim=(2020, 2420), time_label=''):
    """Create comprehensive multi-page PDF"""

    with PdfPages(output_file) as pdf:

        # Page 1: Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, 'Config 037 Baseline Results',
                ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.5, f'Comprehensive Variable Comparison{time_label}',
                ha='center', fontsize=18)
        fig.text(0.5, 0.4, 'Tax Policies: r-uniform, u-max, u-uniform',
                ha='center', fontsize=14)
        fig.text(0.5, 0.35, 'Scenarios: f-f-f-f (no income effects), t-f-f-f (income-dep damage)',
                ha='center', fontsize=14)
        fig.text(0.5, 0.28, f'Time range: {xlim[0]}-{xlim[1]}',
                ha='center', fontsize=12)
        fig.text(0.5, 0.23, f'Generated: March 28, 2026',
                ha='center', fontsize=12, style='italic')

        # Legend
        fig.text(0.5, 0.15, 'Legend:', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.5, 0.12, 'Blue = r-uniform, Orange = u-max, Green = u-uniform',
                ha='center', fontsize=11)
        fig.text(0.5, 0.09, 'Solid line = No income effects, Dashed line = Income-dependent damage',
                ha='center', fontsize=11)

        plt.axis('off')
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Climate variables
        create_plot_page(pdf, results, 'marginal_abatement_cost',
                        'Marginal Abatement Cost ($/tCO₂)',
                        'Marginal Abatement Cost (Social Cost of Carbon)',
                        ylim_min=0, xlim=xlim)

        create_plot_page(pdf, results, 'delta_T',
                        'Temperature Change (°C)',
                        'Global Mean Temperature Change from Pre-Industrial',
                        ylim_min=0, xlim=xlim)

        create_plot_page(pdf, results, 'mu',
                        'Abatement Fraction (μ)',
                        'Emissions Abatement Fraction',
                        ylim_min=0, xlim=xlim)

        create_plot_page(pdf, results, 'E',
                        'CO₂ Emissions (tCO₂/yr)',
                        'Annual CO₂ Emissions',
                        ylim_min=0, xlim=xlim)

        create_plot_page(pdf, results, 'Ecum',
                        'Cumulative Emissions (tCO₂)',
                        'Cumulative CO₂ Emissions',
                        xlim=xlim)

        # Economic variables
        create_plot_page(pdf, results, 'y_net',
                        'Per-Capita Income ($/person/yr)',
                        'Effective Per-Capita Income After Abatement',
                        ylim_min=0, xlim=xlim)

        create_plot_page(pdf, results, 'Y_net',
                        'Net Output ($/yr)',
                        'Total Output Net of Climate Damage and Abatement Cost',
                        ylim_min=0, xlim=xlim)

        create_plot_page(pdf, results, 'Consumption',
                        'Total Consumption ($/yr)',
                        'Total Consumption',
                        ylim_min=0, xlim=xlim)

        create_plot_page(pdf, results, 'K',
                        'Capital Stock ($)',
                        'Total Capital Stock',
                        ylim_min=0, xlim=xlim)

        # Damages and costs (as percentages)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        for tax_policy in TAX_POLICY_ORDER:
            for scenario in SCENARIO_ORDER:
                if scenario not in results[tax_policy]:
                    continue
                df = results[tax_policy][scenario]
                label = f"{TAX_POLICY_NAMES[tax_policy]} - {SCENARIO_NAMES[scenario]}"
                ax.plot(df['t'], df['Lambda'] * 100,
                       color=TAX_COLORS[tax_policy],
                       linestyle=SCENARIO_LINESTYLES[scenario],
                       label=label,
                       linewidth=1.8)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Abatement Cost (% of Output)', fontsize=12)
        ax.set_title('Abatement Cost as Percentage of Output', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(xlim)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best', framealpha=0.95)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(11, 8.5))
        for tax_policy in TAX_POLICY_ORDER:
            for scenario in SCENARIO_ORDER:
                if scenario not in results[tax_policy]:
                    continue
                df = results[tax_policy][scenario]
                ax.plot(df['t'], df['Omega'] * 100,
                       color=TAX_COLORS[tax_policy],
                       linestyle=SCENARIO_LINESTYLES[scenario],
                       linewidth=1.8)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Climate Damage (% of Gross Output)', fontsize=12)
        ax.set_title('Climate Damage as Percentage of Gross Output', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(xlim)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Savings rate
        fig, ax = plt.subplots(figsize=(11, 8.5))
        for tax_policy in TAX_POLICY_ORDER:
            for scenario in SCENARIO_ORDER:
                if scenario not in results[tax_policy]:
                    continue
                df = results[tax_policy][scenario]
                ax.plot(df['t'], df['s'] * 100,
                       color=TAX_COLORS[tax_policy],
                       linestyle=SCENARIO_LINESTYLES[scenario],
                       linewidth=1.8)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Savings Rate (%)', fontsize=12)
        ax.set_title('Optimal Savings Rate', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Control variable f (log10 of MAC)
        create_plot_page(pdf, results, 'f',
                        'f = log₁₀(MAC)',
                        'Control Variable: log₁₀ of Marginal Abatement Cost',
                        xlim=xlim)

        # Utility
        create_plot_page(pdf, results, 'U',
                        'Mean Utility Per Capita',
                        'Mean Utility Per Capita',
                        xlim=xlim)

        # Gini index
        create_plot_page(pdf, results, 'Gini',
                        'Gini Index',
                        'Gini Index (Income Inequality)',
                        ylim_min=0, xlim=xlim)

        # Population and productivity
        create_plot_page(pdf, results, 'L',
                        'Population (persons)',
                        'Total Population',
                        xlim=xlim)

        create_plot_page(pdf, results, 'A',
                        'Total Factor Productivity ($)',
                        'Total Factor Productivity (TFP)',
                        xlim=xlim)

        # Carbon intensity
        create_plot_page(pdf, results, 'sigma',
                        'Carbon Intensity (tCO₂/$)',
                        'Carbon Intensity of GDP',
                        xlim=xlim)

        create_plot_page(pdf, results, 'theta1',
                        'Backstop Cost ($/tCO₂)',
                        'Abatement Cost Parameter θ₁(t)',
                        xlim=xlim)

        print(f"Created comprehensive PDF: {output_file}")

def main():
    print("="*80)
    print("Creating Config 037 Comprehensive Plots")
    print("="*80)

    print("\nLoading results...")
    results = load_config_037_results()

    print("\nCreating plots_full.pdf (all years: 2020-2420)...")
    create_plots_full(results,
                     output_file='data/output/config_037_plots_full.pdf',
                     xlim=(2020, 2420),
                     time_label=' (Full Horizon)')

    print("\nCreating plots_full_2020-2100.pdf (near-term: 2020-2100)...")
    create_plots_full(results,
                     output_file='data/output/config_037_plots_full_2020-2100.pdf',
                     xlim=(2020, 2100),
                     time_label=' (2020-2100)')

    # Also regenerate the comparison plot with fixed title
    print("\nRegenerating comparison plot with fixed title...")
    from plot_config_037_comparison import create_comparison_plots
    create_comparison_plots(results)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated:")
    print("  - data/output/config_037_plots_full.pdf (all years: 2020-2420)")
    print("  - data/output/config_037_plots_full_2020-2100.pdf (near-term: 2020-2100)")
    print("  - data/output/config_037_baseline_comparison.pdf (updated, no title overlap)")

if __name__ == "__main__":
    main()
