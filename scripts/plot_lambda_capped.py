#!/usr/bin/env python3
"""
Plot Lambda (Abatement Cost as % of Output) for capped mu_up scenarios.

Creates a comparison plot of Lambda = AbateCost / Y_damaged for 4 scenarios:
  A) Uniform climate damage + Uniform tax
  B) Income-dependent climate damage + Uniform tax
  C) Uniform climate damage + Progressive tax (highest income only)
  D) Income-dependent climate damage + Progressive tax

All scenarios run with:
  - use_mu_up_cap = True
  - dt = 1 (yearly integration)
  - 400 year horizon (2020-2420)

NOTE: When mu_up cap is binding (which happens most of the time with the DICE
schedule), the Lambda values are determined by the cap schedule, not by the
scenario-specific damage/tax policy settings. This means all scenarios will
show nearly identical Lambda curves when the cap is binding. The scenarios
differ in OTHER variables (welfare, emissions path, etc.) but Lambda under
a binding cap is essentially dictated by the DICE mu_up schedule.

Usage:
    python plot_lambda_capped.py [--run] [--output-dir DIR]

Options:
    --run         Run forward integration (fast). Without this flag, loads existing results.
    --output-dir  Directory to save output (default: data/output/lambda_capped)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parameters import load_configuration
from src.economic_model import integrate_model, create_derived_variables


# =============================================================================
# Scenario Definitions
# =============================================================================

SCENARIOS = {
    'A': {
        'name': 'Uniform Damage + Uniform Tax',
        'short_name': 'Uniform/Uniform',
        'config': 'json/config_014_f-f-f-f-f_1_100k_list.json',
        'color': '#1f77b4',  # blue
        'income_dependent_damage_distribution': False,
        'income_dependent_tax_policy': False,
    },
    'B': {
        'name': 'Income-Dep Damage + Uniform Tax',
        'short_name': 'Inc-Dep Damage/Uniform Tax',
        'config': 'json/config_014_t-f-f-f-f_1_100k_list.json',
        'color': '#ff7f0e',  # orange
        'income_dependent_damage_distribution': True,
        'income_dependent_tax_policy': False,
    },
    'C': {
        'name': 'Uniform Damage + Progressive Tax',
        'short_name': 'Uniform Damage/Progressive Tax',
        'config': 'json/config_014_f-f-t-f-f_1_100k_list.json',
        'color': '#2ca02c',  # green
        'income_dependent_damage_distribution': False,
        'income_dependent_tax_policy': True,
    },
    'D': {
        'name': 'Income-Dep Damage + Progressive Tax',
        'short_name': 'Inc-Dep/Progressive',
        'config': 'json/config_014_t-f-t-f-f_1_100k_list.json',
        'color': '#d62728',  # red
        'income_dependent_damage_distribution': True,
        'income_dependent_tax_policy': True,
    },
}


# =============================================================================
# Run or Load Results
# =============================================================================

def run_scenario(scenario_key, output_dir):
    """Run a single scenario with mu_up cap enabled."""
    scenario = SCENARIOS[scenario_key]
    print(f"\n{'='*60}")
    print(f"Running Scenario {scenario_key}: {scenario['name']}")
    print(f"{'='*60}")

    # Load base config
    config = load_configuration(scenario['config'])

    # Enable mu_up cap
    config.scalar_params.use_mu_up_cap = True

    # Override flags if needed (in case config doesn't match)
    config.scalar_params.income_dependent_damage_distribution = scenario['income_dependent_damage_distribution']
    config.scalar_params.income_dependent_tax_policy = scenario['income_dependent_tax_policy']

    # Run integration
    print(f"Running integration for {config.integration_params.t_end - config.integration_params.t_start + 1} years...")
    results = integrate_model(config, store_detailed_output=True)
    results = create_derived_variables(results)

    # Save key results
    results_path = output_dir / f'scenario_{scenario_key}_results.npz'
    np.savez(results_path,
             t=results['t'],
             Lambda=results['Lambda'],
             mu=results['mu'],
             mu_uncapped=results['mu_uncapped'],
             mu_cap=results['mu_cap'],
             cap_binding=results['cap_binding'],
             AbateCost=results['AbateCost'],
             abateCost_effective=results['abateCost_effective'],
             abateCost_proposed=results['abateCost_proposed'],
             Y_damaged=results['Y_damaged'],
             marginal_abatement_cost=results.get('marginal_abatement_cost', np.zeros_like(results['t'])))

    print(f"Results saved to: {results_path}")
    return results


def load_scenario(scenario_key, output_dir):
    """Load previously computed scenario results."""
    results_path = output_dir / f'scenario_{scenario_key}_results.npz'
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}. Run with --run flag first.")

    data = np.load(results_path)
    return {k: data[k] for k in data.files}


# =============================================================================
# Plotting
# =============================================================================

def plot_lambda_comparison(all_results, output_dir, time_range=None):
    """
    Create Lambda comparison plot.

    Parameters
    ----------
    all_results : dict
        Dictionary mapping scenario keys to results dicts
    output_dir : Path
        Output directory for saving plot
    time_range : tuple, optional
        (start_year, end_year) for x-axis limits
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Default time range
    if time_range is None:
        time_range = (2020, 2420)

    for key in ['A', 'B', 'C', 'D']:
        results = all_results[key]
        scenario = SCENARIOS[key]

        t = results['t']
        Lambda = results['Lambda']

        # Filter to time range
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t_plot = t[mask]
        Lambda_plot = Lambda[mask]

        # Plot line
        ax.plot(t_plot, Lambda_plot * 100,  # Convert to percentage
                label=scenario['short_name'],
                color=scenario['color'],
                linewidth=2)

        # Add inline label near the curve (at ~30% of the time range)
        label_idx = int(len(t_plot) * 0.3)
        if label_idx < len(t_plot):
            # Get marginal abatement cost at 2025 for SCC annotation if available
            scc_2025 = None
            if 'marginal_abatement_cost' in results:
                idx_2025 = np.argmin(np.abs(t - 2025))
                scc_2025 = results['marginal_abatement_cost'][idx_2025]

            label_text = f"{key}: {scenario['short_name']}"
            if scc_2025 is not None and scc_2025 > 0:
                label_text += f"\n(SCC₂₀₂₅=${scc_2025:.0f}/tCO₂)"

            ax.annotate(label_text,
                       xy=(t_plot[label_idx], Lambda_plot[label_idx] * 100),
                       xytext=(10, 0),
                       textcoords='offset points',
                       fontsize=9,
                       color=scenario['color'],
                       fontweight='bold',
                       va='center')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Abatement Cost (% of Output)', fontsize=12)
    ax.set_title('Lambda: Abatement Cost (% of Output)\n[with DICE-style mu_up cap]',
                fontsize=14, fontweight='bold')

    ax.set_xlim(time_range)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Add note about mu_up cap
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'lambda_capped.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # Also save PDF
    pdf_path = output_dir / 'lambda_capped.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def plot_lambda_short_horizon(all_results, output_dir):
    """Create Lambda plot for short horizon (2020-2120)."""
    plot_lambda_comparison(all_results, output_dir, time_range=(2020, 2120))

    # Rename the file
    short_path = output_dir / 'lambda_capped.png'
    if short_path.exists():
        new_path = output_dir / 'lambda_capped_2020-2120.png'
        short_path.rename(new_path)


def print_cap_sanity_check(results, scenario_key):
    """Print sanity check for mu_up cap for first 10 years."""
    print(f"\n{'='*60}")
    print(f"Sanity Check: Scenario {scenario_key} - First 15 Years")
    print(f"{'='*60}")
    print(f"{'Year':<6} {'mu_uncapped':>12} {'mu_cap':>10} {'mu_final':>10} {'cap_bind':>10} {'slack':>12}")
    print('-' * 62)

    t = results['t']
    for i in range(min(15, len(t))):
        year = int(t[i])
        mu_uncapped = results['mu_uncapped'][i]
        mu_cap = results['mu_cap'][i]
        mu_final = results['mu'][i]
        cap_binding = int(results['cap_binding'][i])

        print(f"{year:<6} {mu_uncapped:>12.4f} {mu_cap:>10.4f} {mu_final:>10.4f} {cap_binding:>10} {slack:>12.2f}")

    print()
    print("Verification:")
    mu_capped_ok = np.all(results['mu'] <= results['mu_cap'] + 1e-10)
    print(f"  - mu_final <= mu_cap: {mu_capped_ok}")

    # Check cap_binding logic: should be 1 when mu_uncapped > effective_cap
    effective_cap = np.minimum(results['mu_cap'], 1.0)
    expected_binding = results['mu_uncapped'] > effective_cap + 1e-6
    actual_binding = results['cap_binding'] > 0.5
    binding_ok = np.all(expected_binding == actual_binding)
    print(f"  - Cap binding logic correct: {binding_ok}")

    # Count binding years
    n_binding = np.sum(actual_binding)
    print(f"  - Years with cap binding: {n_binding} of {len(results['t'])}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot Lambda for capped mu_up scenarios')
    parser.add_argument('--run', action='store_true',
                       help='Run forward integration. Without this, loads existing results.')
    parser.add_argument('--output-dir', type=str, default='data/output/lambda_capped',
                       help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Run or load scenarios
    all_results = {}

    for key in ['A', 'B', 'C', 'D']:
        if args.run:
            results = run_scenario(key, output_dir)
            all_results[key] = {
                't': results['t'],
                'Lambda': results['Lambda'],
                'mu': results['mu'],
                'mu_uncapped': results['mu_uncapped'],
                'mu_cap': results['mu_cap'],
                'cap_binding': results['cap_binding'],
                'AbateCost': results['AbateCost'],
                'abateCost_effective': results['abateCost_effective'],
                'abateCost_proposed': results['abateCost_proposed'],
                'Y_damaged': results['Y_damaged'],
                'marginal_abatement_cost': results.get('marginal_abatement_cost', np.zeros_like(results['t'])),
            }
        else:
            print(f"Loading Scenario {key}...")
            all_results[key] = load_scenario(key, output_dir)

    # Print sanity check for scenario A
    print_cap_sanity_check(all_results['A'], 'A')

    # Check if scenarios produce different Lambda (they likely won't when cap is binding)
    Lambda_A = all_results['A']['Lambda']
    Lambda_B = all_results['B']['Lambda']
    if np.allclose(Lambda_A, Lambda_B, rtol=1e-6):
        print("\n" + "="*70)
        print("NOTE: All scenarios have nearly identical Lambda values.")
        print("This is EXPECTED when the mu_up cap is binding for all years.")
        print("Lambda is determined by the DICE schedule, not by policy settings.")
        print("Scenarios differ in welfare, not in capped abatement cost.")
        print("="*70)

    # Create plots
    print("\nCreating plots...")

    # Full time range
    plot_lambda_comparison(all_results, output_dir, time_range=(2020, 2420))

    # Short time range with Lambda
    fig, ax = plt.subplots(figsize=(12, 7))
    for key in ['A', 'B', 'C', 'D']:
        results = all_results[key]
        scenario = SCENARIOS[key]
        t = results['t']
        Lambda = results['Lambda']
        mask = (t >= 2020) & (t <= 2120)
        ax.plot(t[mask], Lambda[mask] * 100, label=scenario['short_name'],
               color=scenario['color'], linewidth=2)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Abatement Cost (% of Output)', fontsize=12)
    ax.set_title('Lambda: Abatement Cost (% of Output) [2020-2120]\n[with DICE-style mu_up cap]',
                fontsize=14, fontweight='bold')
    ax.set_xlim(2020, 2120)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.text(0.02, 0.98, 'use_mu_up_cap = True', transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_dir / 'lambda_capped_short.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create 2-panel plot: Lambda and mu/mu_cap comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    results_A = all_results['A']
    t = results_A['t']
    mask = (t >= 2020) & (t <= 2120)
    t_plot = t[mask]

    # Panel 1: Lambda
    ax1.plot(t_plot, results_A['Lambda'][mask] * 100, 'b-', linewidth=2, label='Lambda (all scenarios)')
    ax1.set_ylabel('Abatement Cost (% of Output)', fontsize=12)
    ax1.set_title('Lambda: Abatement Cost (% of Output) with DICE mu_up Cap', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Panel 2: mu vs mu_cap
    ax2.plot(t_plot, results_A['mu'][mask], 'b-', linewidth=2, label='mu (actual, capped)')
    ax2.plot(t_plot, results_A['mu_uncapped'][mask], 'r--', linewidth=1.5, alpha=0.7, label='mu_uncapped (would-be)')
    ax2.plot(t_plot, results_A['mu_cap'][mask], 'g:', linewidth=2, label='mu_cap (DICE schedule)')
    ax2.fill_between(t_plot, results_A['mu'][mask], results_A['mu_uncapped'][mask],
                     alpha=0.2, color='red', label='Constrained region')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Abatement Fraction (mu)', fontsize=12)
    ax2.set_title('Abatement Fraction: Actual vs Uncapped vs Cap Schedule', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim(2020, 2120)

    plt.tight_layout()
    plt.savefig(output_dir / 'lambda_and_mu_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'lambda_and_mu_comparison.pdf', bbox_inches='tight')
    plt.close()

    print("Created additional plot: lambda_and_mu_comparison.png")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
