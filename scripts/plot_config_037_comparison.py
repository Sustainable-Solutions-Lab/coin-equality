#!/usr/bin/env python3
"""
Create comparison plots for config_037 baseline results
Comparing 3 tax policies × 2 scenarios
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from collections import OrderedDict

# Scenario order (CRITICAL: alphabetical glob order is WRONG)
SCENARIO_ORDER = ['f-f-f-f', 't-f-f-f']
SCENARIO_NAMES = {
    'f-f-f-f': 'S1: No income effects',
    't-f-f-f': 'S2: Income-dep damage'
}

TAX_POLICY_ORDER = ['r-uniform', 'u-max', 'u-uniform']
TAX_POLICY_NAMES = {
    'r-uniform': 'Uniform fractional tax',
    'u-max': 'Progressive (tax richest)',
    'u-uniform': 'Equal utility loss'
}

# Colors: tab10 colormap order
TAX_COLORS = {
    'r-uniform': 'C0',  # blue
    'u-max': 'C1',      # orange
    'u-uniform': 'C2'   # green
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
            # Find directory
            pattern = f"data/output/config_037_{tax_policy}_{scenario}_100k_*"
            dirs = sorted(glob.glob(pattern))

            if not dirs:
                print(f"WARNING: No results found for {tax_policy} {scenario}")
                continue

            # Use most recent
            result_dir = dirs[-1]

            # Load CSV
            csv_file = glob.glob(f"{result_dir}/*_optimization_results.csv")[0]
            df = pd.read_csv(csv_file)

            # Simplify column names
            df.columns = [col.split(',')[0].strip('"') for col in df.columns]

            results[tax_policy][scenario] = df
            print(f"Loaded: {tax_policy} {scenario} ({len(df)} time points)")

    return results

def create_comparison_plots(results, output_file='data/output/config_037_baseline_comparison.pdf'):
    """Create comprehensive comparison plots"""

    fig = plt.figure(figsize=(16, 12))

    # Define subplots: 3 rows × 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35,
                         left=0.08, right=0.96, top=0.94, bottom=0.06)

    # Row 1: Tax policy comparison
    ax1 = fig.add_subplot(gs[0, 0])  # MAC by tax policy
    ax2 = fig.add_subplot(gs[0, 1])  # Temperature by tax policy
    ax3 = fig.add_subplot(gs[0, 2])  # Abatement fraction by tax policy

    # Row 2: Scenario comparison
    ax4 = fig.add_subplot(gs[1, 0])  # MAC by scenario
    ax5 = fig.add_subplot(gs[1, 1])  # Temperature by scenario
    ax6 = fig.add_subplot(gs[1, 2])  # Income by scenario

    # Row 3: Economic variables
    ax7 = fig.add_subplot(gs[2, 0])  # Abatement cost (% GDP)
    ax8 = fig.add_subplot(gs[2, 1])  # Climate damage (% GDP)
    ax9 = fig.add_subplot(gs[2, 2])  # Savings rate

    # === Row 1: Tax Policy Comparison (both scenarios on same plot) ===

    # Plot 1: MAC by tax policy
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax1.plot(df['t'], df['marginal_abatement_cost'],
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    label=f"{TAX_POLICY_NAMES[tax_policy]} ({SCENARIO_NAMES[scenario]})",
                    linewidth=1.5)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Marginal Abatement Cost ($/tCO₂)')
    ax1.set_title('A. MAC Comparison Across Tax Policies', fontweight='bold', loc='left')
    ax1.set_xlim(2020, 2120)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='upper left')

    # Plot 2: Temperature by tax policy
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax2.plot(df['t'], df['delta_T'],
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    linewidth=1.5)

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Temperature Change (°C)')
    ax2.set_title('B. Temperature Trajectories', fontweight='bold', loc='left')
    ax2.set_xlim(2020, 2120)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Abatement fraction by tax policy
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax3.plot(df['t'], df['mu'],
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    linewidth=1.5)

    ax3.set_xlabel('Year')
    ax3.set_ylabel('Abatement Fraction (μ)')
    ax3.set_title('C. Abatement Fraction Trajectories', fontweight='bold', loc='left')
    ax3.set_xlim(2020, 2120)
    ax3.set_ylim(0, None)
    ax3.grid(True, alpha=0.3)

    # === Row 2: Focus on one tax policy (u-uniform) comparing scenarios ===

    # Plot 4: MAC by scenario (u-uniform only)
    for scenario in SCENARIO_ORDER:
        if scenario not in results['u-uniform']:
            continue
        df = results['u-uniform'][scenario]
        ax4.plot(df['t'], df['marginal_abatement_cost'],
                linestyle=SCENARIO_LINESTYLES[scenario],
                label=SCENARIO_NAMES[scenario],
                linewidth=2)

    ax4.set_xlabel('Year')
    ax4.set_ylabel('Marginal Abatement Cost ($/tCO₂)')
    ax4.set_title('D. MAC: Equal Utility Loss Tax Policy', fontweight='bold', loc='left')
    ax4.set_xlim(2020, 2120)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    # Plot 5: Temperature by scenario (all tax policies)
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax5.plot(df['t'], df['delta_T'],
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    linewidth=1.5,
                    alpha=0.8)

    ax5.set_xlabel('Year')
    ax5.set_ylabel('Temperature Change (°C)')
    ax5.set_title('E. Temperature: All Cases', fontweight='bold', loc='left')
    ax5.set_xlim(2020, 2120)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Income by scenario
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax6.plot(df['t'], df['y_net'],
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    linewidth=1.5)

    ax6.set_xlabel('Year')
    ax6.set_ylabel('Per-Capita Income ($/person/yr)')
    ax6.set_title('F. Effective Per-Capita Income', fontweight='bold', loc='left')
    ax6.set_xlim(2020, 2120)
    ax6.grid(True, alpha=0.3)

    # === Row 3: Economic Variables ===

    # Plot 7: Abatement cost (% of output)
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax7.plot(df['t'], df['Lambda'] * 100,
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    linewidth=1.5)

    ax7.set_xlabel('Year')
    ax7.set_ylabel('Abatement Cost (% of GDP)')
    ax7.set_title('G. Abatement Cost as % of Output', fontweight='bold', loc='left')
    ax7.set_xlim(2020, 2120)
    ax7.set_ylim(0, None)
    ax7.grid(True, alpha=0.3)

    # Plot 8: Climate damage (% of output)
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax8.plot(df['t'], df['Omega'] * 100,
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    linewidth=1.5)

    ax8.set_xlabel('Year')
    ax8.set_ylabel('Climate Damage (% of GDP)')
    ax8.set_title('H. Climate Damage as % of Output', fontweight='bold', loc='left')
    ax8.set_xlim(2020, 2120)
    ax8.set_ylim(0, None)
    ax8.grid(True, alpha=0.3)

    # Plot 9: Savings rate
    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax9.plot(df['t'], df['s'] * 100,
                    color=TAX_COLORS[tax_policy],
                    linestyle=SCENARIO_LINESTYLES[scenario],
                    linewidth=1.5)

    ax9.set_xlabel('Year')
    ax9.set_ylabel('Savings Rate (%)')
    ax9.set_title('I. Optimal Savings Rate', fontweight='bold', loc='left')
    ax9.set_xlim(2020, 2120)
    ax9.set_ylim(15, 45)
    ax9.grid(True, alpha=0.3)

    # Overall title with better spacing
    fig.suptitle('Config 037 Baseline Comparison: Tax Policies and Scenarios',
                fontsize=14, fontweight='bold', y=0.985)

    # Subtitle with proper positioning
    fig.text(0.5, 0.965, '(Solid line = No income effects, Dashed line = Income-dependent damage)',
            ha='center', fontsize=11, style='italic')

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

def create_tax_policy_comparison(results, output_file='data/output/config_037_tax_policy_focus.pdf'):
    """Create focused comparison of tax policies for each scenario"""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, scenario in enumerate(SCENARIO_ORDER):
        # MAC
        ax = axes[i, 0]
        for tax_policy in TAX_POLICY_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax.plot(df['t'], df['marginal_abatement_cost'],
                   color=TAX_COLORS[tax_policy],
                   label=TAX_POLICY_NAMES[tax_policy],
                   linewidth=2)

        ax.set_xlabel('Year')
        ax.set_ylabel('MAC ($/tCO₂)')
        ax.set_title(f'{SCENARIO_NAMES[scenario]}: MAC', fontweight='bold')
        ax.set_xlim(2020, 2120)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

        # Temperature
        ax = axes[i, 1]
        for tax_policy in TAX_POLICY_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax.plot(df['t'], df['delta_T'],
                   color=TAX_COLORS[tax_policy],
                   linewidth=2)

        ax.set_xlabel('Year')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{SCENARIO_NAMES[scenario]}: Temperature', fontweight='bold')
        ax.set_xlim(2020, 2120)
        ax.grid(True, alpha=0.3)

        # Abatement fraction
        ax = axes[i, 2]
        for tax_policy in TAX_POLICY_ORDER:
            if scenario not in results[tax_policy]:
                continue
            df = results[tax_policy][scenario]
            ax.plot(df['t'], df['mu'],
                   color=TAX_COLORS[tax_policy],
                   linewidth=2)

        ax.set_xlabel('Year')
        ax.set_ylabel('Abatement Fraction (μ)')
        ax.set_title(f'{SCENARIO_NAMES[scenario]}: Abatement', fontweight='bold')
        ax.set_xlim(2020, 2120)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Config 037 Baseline: Tax Policy Comparison by Scenario',
                fontsize=14, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def create_summary_table(results, output_file='data/output/config_037_summary_table.csv'):
    """Create summary table with key metrics"""

    summary_data = []

    for tax_policy in TAX_POLICY_ORDER:
        for scenario in SCENARIO_ORDER:
            if scenario not in results[tax_policy]:
                continue

            df = results[tax_policy][scenario]

            # Extract key values at specific years
            row = {
                'Tax Policy': TAX_POLICY_NAMES[tax_policy],
                'Scenario': SCENARIO_NAMES[scenario]
            }

            for year in [2020, 2050, 2100]:
                data_year = df[df['t'] == year].iloc[0]
                row[f'MAC_{year}'] = data_year['marginal_abatement_cost']
                row[f'Temp_{year}'] = data_year['delta_T']
                row[f'Mu_{year}'] = data_year['mu']
                row[f'Income_{year}'] = data_year['y_net']

            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Saved: {output_file}")

    return summary_df

def main():
    print("="*80)
    print("Config 037 Baseline Comparison Plots")
    print("="*80)

    # Load results
    print("\nLoading config_037 results...")
    results = load_config_037_results()

    # Create plots
    print("\nCreating comparison plots...")
    create_comparison_plots(results)

    print("\nCreating tax policy focused plots...")
    create_tax_policy_comparison(results)

    print("\nCreating summary table...")
    summary_df = create_summary_table(results)

    print("\n" + "="*80)
    print("Summary Table Preview:")
    print("="*80)
    print(summary_df.to_string(index=False))

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - data/output/config_037_baseline_comparison.pdf")
    print("  - data/output/config_037_tax_policy_focus.pdf")
    print("  - data/output/config_037_summary_table.csv")

if __name__ == "__main__":
    main()
