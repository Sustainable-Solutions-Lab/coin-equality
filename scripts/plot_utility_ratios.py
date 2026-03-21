"""
Generate 2-panel PDF figure showing utility ratios under different climate damage and taxation scenarios.

Panel A: Climate damage impact as function of Gini index and damage distribution exponent
Panel B: Progressive taxation impact as function of Gini index and mean tax rate

Computes utility ratios using CRRA utility, Lorenz curves, and adaptive numerical integration.

Usage (examples):
- PDF only:
    python plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json
- PDF + CSV export for both panels:
    python plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json --csv
- PDF + CSV + XLSX export for both panels (requires pandas):
    python plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json --csv --xlsx

Notes:
- The required positional argument is the JSON configuration file.
- All output files are written to a timestamped directory under data/output/.
- CSV and XLSX files are named {run_name}_panel_a.{csv,xlsx} and {run_name}_panel_b.{csv,xlsx}.
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import quad
from scipy.optimize import root_scalar
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parameters import load_configuration, evaluate_params_at_time
from src.distribution_utilities import (
    L_empirical_lorenz_derivative,
    L_pareto_derivative,
    L_empirical_lorenz,
    L_pareto,
)
from src.constants import EPSILON, EMPIRICAL_LORENZ_BASE_GINI

DELTA_T_REPRESENTATIVE = 2.0

GINI_RANGE_DAMAGE = (0.0, EMPIRICAL_LORENZ_BASE_GINI)
N_GINI_DAMAGE = 50
EXPONENT_RANGE = (0.0, 1.0)
N_EXPONENT = 50

GINI_RANGE_TAX = (0.0, EMPIRICAL_LORENZ_BASE_GINI)
N_GINI_TAX = 50
TAX_RATE_RANGE = (0.0, 0.2)
N_TAX_RATE = 50


def calculate_crra_utility(income, eta):
    """
    Calculate CRRA utility for given income and risk aversion.

    Parameters
    ----------
    income : float or ndarray
        Income level(s)
    eta : float
        Coefficient of relative risk aversion

    Returns
    -------
    float or ndarray
        Utility value(s)

    Notes
    -----
    For η = 1: U(y) = ln(max(y, EPSILON))
    For η ≠ 1: U(y) = max(y, EPSILON)^(1-η) / (1-η)

    Note: Savings rate s is irrelevant for utility ratios since it cancels out.
    """
    income_safe = np.maximum(income, EPSILON)

    if np.abs(eta - 1.0) < EPSILON:
        return np.log(income_safe)
    else:
        return (income_safe ** (1.0 - eta)) / (1.0 - eta)


def compute_aggregate_utility(gini, eta, use_empirical_lorenz):
    """
    Compute aggregate utility over income distribution using adaptive integration.

    Parameters
    ----------
    gini : float
        Gini coefficient
    eta : float
        CRRA parameter
    use_empirical_lorenz : bool
        Whether to use empirical Lorenz curve

    Returns
    -------
    float
        Aggregate utility (population-weighted average)

    Notes
    -----
    Income is normalized to mean = 1.0.
    Savings rate is irrelevant for utility ratios and omitted.
    """
    if use_empirical_lorenz:
        dLdF_func = lambda F: L_empirical_lorenz_derivative(F, gini)
    else:
        dLdF_func = lambda F: L_pareto_derivative(F, gini)

    def utility_integrand(F):
        y_F = dLdF_func(F)
        return calculate_crra_utility(y_F, eta)

    U_aggregate, _ = quad(utility_integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200)
    return U_aggregate


def apply_climate_damage_to_income_and_integrate(
    Omega_base, damage_exponent, gini, eta, use_empirical_lorenz, normalize_aggregate_damage=True
):
    """
    Apply income-dependent climate damage and compute aggregate utility using adaptive integration.

    Parameters
    ----------
    Omega_base : float
        Base damage fraction (from temperature)
    damage_exponent : float
        Income-dependent damage exponent
    gini : float
        Gini coefficient
    eta : float
        CRRA parameter
    use_empirical_lorenz : bool
        Lorenz curve formulation
    normalize_aggregate_damage : bool, optional
        If True, rescales income-dependent damages so aggregate damage equals Omega_base

    Returns
    -------
    float
        Aggregate utility after applying climate damage

    Notes
    -----
    Income is normalized to mean = 1.0.
    Savings rate is irrelevant for utility ratios and omitted.
    """
    if use_empirical_lorenz:
        dLdF_func = lambda F: L_empirical_lorenz_derivative(F, gini)
    else:
        dLdF_func = lambda F: L_pareto_derivative(F, gini)

    if damage_exponent < EPSILON:
        def utility_integrand(F):
            y_lorenz_F = dLdF_func(F)
            omega_F = Omega_base
            y_damaged_F = y_lorenz_F * (1.0 - omega_F)
            return calculate_crra_utility(y_damaged_F, eta)
    else:
        if normalize_aggregate_damage:
            def damage_integrand(F):
                y_lorenz_F = dLdF_func(F)
                omega_F = Omega_base * y_lorenz_F ** (-damage_exponent)
                return y_lorenz_F * omega_F

            damage_unscaled, _ = quad(damage_integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200)
            target_damage = Omega_base
            scale = target_damage / max(damage_unscaled, EPSILON)
        else:
            scale = 1.0

        def utility_integrand(F):
            y_lorenz_F = dLdF_func(F)
            omega_F = Omega_base * y_lorenz_F ** (-damage_exponent)
            omega_F = scale * omega_F
            omega_F = np.clip(omega_F, 0.0, 1.0 - 1e-8)
            y_damaged_F = y_lorenz_F * (1.0 - omega_F)
            return calculate_crra_utility(y_damaged_F, eta)

    U_aggregate, _ = quad(utility_integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200)
    return U_aggregate


def compute_aggregate_utility_uniform_taxation(tax_rate, gini, eta, use_empirical_lorenz):
    """
    Compute aggregate utility with uniform taxation using adaptive integration.

    Parameters
    ----------
    tax_rate : float
        Uniform tax rate
    gini : float
        Gini coefficient
    eta : float
        CRRA parameter
    use_empirical_lorenz : bool
        Lorenz formulation

    Returns
    -------
    float
        Aggregate utility after uniform taxation

    Notes
    -----
    Income is normalized to mean = 1.0.
    Savings rate is irrelevant for utility ratios and omitted.
    """
    if use_empirical_lorenz:
        dLdF_func = lambda F: L_empirical_lorenz_derivative(F, gini)
    else:
        dLdF_func = lambda F: L_pareto_derivative(F, gini)

    def utility_integrand(F):
        y_F = dLdF_func(F) * (1.0 - tax_rate)
        return calculate_crra_utility(y_F, eta)

    U_aggregate, _ = quad(utility_integrand, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200)
    return U_aggregate


def find_Fmax_for_progressive_tax(tax_rate_mean, gini, use_empirical_lorenz):
    """
    Find Fmax such that progressive taxation (cap income above Fmax) yields target revenue.

    Solves: L(Fmax) + (1 - Fmax) * dL/dF(Fmax) = 1 - tax_rate_mean

    Parameters
    ----------
    tax_rate_mean : float
        Mean tax rate (fraction of GDP)
    gini : float
        Gini coefficient
    use_empirical_lorenz : bool
        Lorenz formulation

    Returns
    -------
    float
        Fmax threshold for progressive taxation

    Notes
    -----
    Income is normalized to mean = 1.0.
    """
    if use_empirical_lorenz:
        L_func = lambda F: L_empirical_lorenz(F, gini)
        dLdF_func = lambda F: L_empirical_lorenz_derivative(F, gini)
    else:
        L_func = lambda F: L_pareto(F, gini)
        dLdF_func = lambda F: L_pareto_derivative(F, gini)

    def tax_revenue_residual(Fmax):
        L_Fmax = L_func(Fmax)
        dLdF_Fmax = dLdF_func(Fmax)
        revenue = (1.0 - L_Fmax) - (1.0 - Fmax) * dLdF_Fmax
        return revenue - tax_rate_mean

    bracket_low = EPSILON
    bracket_high = 1.0 - EPSILON

    residual_low = tax_revenue_residual(bracket_low)
    residual_high = tax_revenue_residual(bracket_high)

    if residual_low * residual_high > 0:
        if residual_low < 0:
            Fmax = bracket_low
        else:
            Fmax = bracket_high
    else:
        result = root_scalar(tax_revenue_residual, bracket=[bracket_low, bracket_high], method='brentq')
        Fmax = result.root

    return Fmax


def compute_aggregate_utility_progressive_taxation(tax_rate_mean, gini, eta, use_empirical_lorenz, debug=False):
    """
    Compute aggregate utility with progressive taxation using adaptive integration.

    Progressive taxation caps income at y_Fmax for all F >= Fmax.

    Parameters
    ----------
    tax_rate_mean : float
        Mean tax rate (fraction of GDP)
    gini : float
        Gini coefficient
    eta : float
        CRRA parameter
    use_empirical_lorenz : bool
        Lorenz formulation
    debug : bool, optional
        If True, print diagnostic information

    Returns
    -------
    float
        Aggregate utility after progressive taxation

    Notes
    -----
    Income is normalized to mean = 1.0.
    Savings rate is irrelevant for utility ratios and omitted.
    When Fmax ≈ 0 (can't tax progressively), falls back to uniform taxation with uniform income.
    """
    Fmax = find_Fmax_for_progressive_tax(tax_rate_mean, gini, use_empirical_lorenz)

    if debug:
        print(f'    Progressive tax: Fmax={Fmax:.10f}, fraction taxed={(1-Fmax):.10f}')

    if Fmax < 1e-8:
        if debug:
            print(f'    Fmax ≈ 0: Falling back to uniform taxation with uniform income')
        y_mean_after_tax = 1.0 - tax_rate_mean
        U_aggregate = calculate_crra_utility(y_mean_after_tax, eta)
        return U_aggregate

    if use_empirical_lorenz:
        dLdF_func = lambda F: L_empirical_lorenz_derivative(F, gini)
    else:
        dLdF_func = lambda F: L_pareto_derivative(F, gini)

    y_Fmax = dLdF_func(Fmax)
    u_Fmax = calculate_crra_utility(y_Fmax, eta)

    def utility_integrand_below_Fmax(F):
        y_F = dLdF_func(F)
        return calculate_crra_utility(y_F, eta)

    U_below_Fmax, _ = quad(utility_integrand_below_Fmax, 0.0, Fmax, epsabs=1e-10, epsrel=1e-10, limit=200)
    U_above_Fmax = (1.0 - Fmax) * u_Fmax

    U_aggregate = U_below_Fmax + U_above_Fmax
    return U_aggregate


def compute_damage_ratio_grid(gini_mesh, exponent_mesh, base_params):
    """
    Compute utility ratios for climate damage sweep.

    For each (Gini, exponent) point:
    - Baseline: no damage
    - Reference: uniform damage (exponent=0) at current Gini
    - Test: income-dependent damage (current exponent) at current Gini
    - Ratio = ΔU(Gini, exp) / ΔU(Gini, exp=0)
            = (U_test - U_baseline) / (U_reference - U_baseline)

    This measures the additional harm from income-dependent damage
    (beyond what uniform damage would cause) at a given inequality level.

    Parameters
    ----------
    gini_mesh : ndarray
        2D array of Gini values
    exponent_mesh : ndarray
        2D array of damage exponent values
    base_params : dict
        Base economic parameters

    Returns
    -------
    ndarray
        2D array of utility ratios
    """
    eta = base_params['eta']
    Omega_base = base_params['Omega_base']
    use_empirical_lorenz = base_params['use_empirical_lorenz']

    ratio_grid = np.zeros_like(gini_mesh)

    n_rows, n_cols = gini_mesh.shape

    for i in range(n_rows):
        for j in range(n_cols):
            gini = max(gini_mesh[i, j], EPSILON)
            exponent = exponent_mesh[i, j]

            U_no_damage = compute_aggregate_utility(gini, eta, use_empirical_lorenz)

            U_uniform_damage = apply_climate_damage_to_income_and_integrate(
                Omega_base, 0.0, gini, eta, use_empirical_lorenz
            )

            U_income_dependent_damage = apply_climate_damage_to_income_and_integrate(
                Omega_base, exponent, gini, eta, use_empirical_lorenz
            )

            ratio_grid[i, j] = (U_income_dependent_damage - U_no_damage) / (U_uniform_damage - U_no_damage)

    return ratio_grid


def compute_tax_ratio_grid(gini_mesh, tax_rate_mesh, base_params):
    """
    Compute utility ratios for progressive taxation sweep.

    Parameters
    ----------
    gini_mesh : ndarray
        2D array of Gini values
    tax_rate_mesh : ndarray
        2D array of mean tax rate values
    base_params : dict
        Base economic parameters

    Returns
    -------
    ndarray
        2D array of utility ratios
    """
    eta = base_params['eta']
    use_empirical_lorenz = base_params['use_empirical_lorenz']

    ratio_grid = np.zeros_like(gini_mesh)

    n_rows, n_cols = gini_mesh.shape

    for i in range(n_rows):
        for j in range(n_cols):
            gini = max(gini_mesh[i, j], 10.0 * 1e-8)
            tax_rate = max(tax_rate_mesh[i, j], 10.0 * 1e-8)

            U_no_tax = compute_aggregate_utility(gini, eta, use_empirical_lorenz)

            U_uniform_tax = compute_aggregate_utility_uniform_taxation(
                tax_rate, gini, eta, use_empirical_lorenz
            )

            U_progressive_tax = compute_aggregate_utility_progressive_taxation(
                tax_rate, gini, eta, use_empirical_lorenz, debug=False
            )

            ratio_grid[i, j] = (U_uniform_tax - U_no_tax) / (U_progressive_tax - U_no_tax)

    return ratio_grid


def create_contour_plot(ax, X, Y, Z, xlabel, ylabel, title, colorbar_label, vmin=None, vmax=None, xlim=None, cmap='viridis_r', contour_step=None, colorbar_ticks=None):
    """
    Create a single contour plot on given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    X : ndarray
        2D meshgrid for x-axis
    Y : ndarray
        2D meshgrid for y-axis
    Z : ndarray
        2D array of values to contour
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    colorbar_label : str
        Label for colorbar
    vmin : float, optional
        Minimum value for color scale (default: 1.0)
    vmax : float, optional
        Maximum value for color scale (default: 10.0)
    xlim : tuple, optional
        X-axis limits (min, max)
    cmap : str, optional
        Colormap name (default: 'viridis_r')
    contour_step : float, optional
        Step size for contour lines (default: 1.0 for integer steps starting at 2)
    colorbar_ticks : array-like, optional
        Custom tick positions for colorbar (default: integers 1-10)

    Notes
    -----
    Colorbar shows a wedge indicating values exceeding vmax.
    Contour lines are drawn at regular intervals based on contour_step.
    """
    z_min = vmin if vmin is not None else 1.0
    z_max_colorbar = vmax if vmax is not None else 10.0

    z_max_data = np.max(Z)

    levels_fill = np.linspace(z_min, z_max_colorbar, 50)

    if z_max_data > z_max_colorbar:
        contour = ax.contourf(X, Y, Z, levels=levels_fill, cmap=cmap, extend='max')
    else:
        contour = ax.contourf(X, Y, Z, levels=levels_fill, cmap=cmap, extend='neither')

    if contour_step is not None:
        contour_levels = np.arange(z_min + contour_step, np.floor(z_max_data / contour_step) * contour_step + contour_step / 2, contour_step)
    else:
        contour_levels = np.arange(2, int(np.floor(z_max_data)) + 1, 1)

    if len(contour_levels) > 0:
        ax.contour(X, Y, Z, levels=contour_levels, colors='black', alpha=0.4, linewidths=0.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if xlim is not None:
        ax.set_xlim(xlim)

    cbar = plt.colorbar(contour, ax=ax, extend='max' if z_max_data > z_max_colorbar else 'neither')
    cbar.set_label(colorbar_label, fontsize=10)

    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
        cbar.set_ticklabels([f'{t:.1f}' for t in colorbar_ticks])
    else:
        cbar.set_ticks(np.arange(1, 11, 1))
        cbar.set_ticklabels([str(i) for i in range(1, 11)])

    ax.grid(True, alpha=0.2, linestyle='--')


def create_two_panel_figure(damage_data, tax_data, output_pdf):
    """
    Create 2-panel PDF figure with both ratio plots.

    Parameters
    ----------
    damage_data : dict
        Contains 'gini_mesh', 'exponent_mesh', 'ratio_grid' for Panel A
    tax_data : dict
        Contains 'gini_mesh', 'tax_rate_mesh', 'ratio_grid' for Panel B
    output_pdf : Path
        Output PDF file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    create_contour_plot(
        axes[0],
        damage_data['gini_mesh'],
        damage_data['exponent_mesh'],
        damage_data['ratio_grid'],
        'Gini Index',
        'Climate Damage Distribution Exponent',
        '(a) Multiplier considering increased utility losses\nfrom income-dependent damage distribution',
        'Utility Ratio',
        vmin=1.0,
        vmax=3.0,
        xlim=(0.0, EMPIRICAL_LORENZ_BASE_GINI),
        cmap='plasma_r',
        contour_step=0.2,
        colorbar_ticks=np.arange(1.0, 3.2, 0.2)
    )

    create_contour_plot(
        axes[1],
        tax_data['gini_mesh'],
        tax_data['tax_rate_mesh'],
        tax_data['ratio_grid'],
        'Gini Index',
        'Mean Tax Rate',
        '(b) Multiplier considering decreased utility losses\nfrom taxing only highest-income earners',
        'Utility Ratio',
        xlim=(0.0, EMPIRICAL_LORENZ_BASE_GINI)
    )

    plt.tight_layout()

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)

    plt.close(fig)


def write_panel_a_csv(path, gini_mesh, exponent_mesh, ratio_grid):
    """Write Panel A grid with Gini as columns and damage exponents as rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gini_vals = gini_mesh[0, :]
    exponent_vals = exponent_mesh[:, 0]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['damage_exponent'] + list(gini_vals)
        writer.writerow(header)
        for exp_idx, exp_val in enumerate(exponent_vals):
            row = [exp_val] + list(ratio_grid[exp_idx, :])
            writer.writerow(row)


def write_panel_a_xlsx(path, gini_mesh, exponent_mesh, ratio_grid):
    try:
        import pandas as pd
    except ImportError:
        print('pandas not installed; skipping Panel A XLSX export')
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gini_vals = gini_mesh[0, :]
    exponent_vals = exponent_mesh[:, 0]

    data = {
        'damage_exponent': exponent_vals,
    }
    for col_idx, g_val in enumerate(gini_vals):
        data[g_val] = ratio_grid[:, col_idx]

    df = pd.DataFrame(data)
    df.to_excel(path, index=False)


def write_panel_b_csv(path, gini_mesh, tax_rate_mesh, ratio_grid):
    """Write Panel B grid with Gini as columns and tax rates as rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gini_vals = gini_mesh[0, :]
    tax_rate_vals = tax_rate_mesh[:, 0]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['tax_rate'] + list(gini_vals)
        writer.writerow(header)
        for tax_idx, tax_val in enumerate(tax_rate_vals):
            row = [tax_val] + list(ratio_grid[tax_idx, :])
            writer.writerow(row)


def write_panel_b_xlsx(path, gini_mesh, tax_rate_mesh, ratio_grid):
    try:
        import pandas as pd
    except ImportError:
        print('pandas not installed; skipping Panel B XLSX export')
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gini_vals = gini_mesh[0, :]
    tax_rate_vals = tax_rate_mesh[:, 0]

    data = {
        'tax_rate': tax_rate_vals,
    }
    for col_idx, g_val in enumerate(gini_vals):
        data[g_val] = ratio_grid[:, col_idx]

    df = pd.DataFrame(data)
    df.to_excel(path, index=False)


def main(config_path, output_pdf, csv_export, xlsx_export):
    """
    Main execution function.

    Parameters
    ----------
    config_path : str
        Path to JSON configuration file
    output_pdf : str or None
        Output PDF filename (if None, uses default in timestamped directory)
    csv_export : bool
        Whether to export Panel A and Panel B as CSV files
    xlsx_export : bool
        Whether to export Panel A and Panel B as XLSX files
    """
    config = load_configuration(config_path)

    run_name = config.run_name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path('data') / 'output' / f'utility_ratio_plots_{run_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_out_path(maybe_path, default_name):
        if maybe_path is None:
            return output_dir / default_name
        candidate = Path(maybe_path)
        return candidate if candidate.is_absolute() else output_dir / candidate

    output_pdf_path = resolve_out_path(output_pdf, f'{run_name}_utility_ratios.pdf')

    params_t0 = evaluate_params_at_time(0.0, config)

    eta = params_t0['eta']
    use_empirical_lorenz = config.scalar_params.use_empirical_lorenz

    psi1 = config.scalar_params.psi1
    psi2 = config.scalar_params.psi2
    Omega_base = psi1 * DELTA_T_REPRESENTATIVE + psi2 * (DELTA_T_REPRESENTATIVE ** 2)

    base_params_damage = {
        'eta': eta,
        'Omega_base': Omega_base,
        'use_empirical_lorenz': use_empirical_lorenz,
    }

    base_params_tax = {
        'eta': eta,
        'use_empirical_lorenz': use_empirical_lorenz,
    }

    print(f'Representative economic state:')
    print(f'  Omega_base = {Omega_base:.6f} (at ΔT = {DELTA_T_REPRESENTATIVE}°C)')
    print(f'  eta = {eta:.3f}')
    print()

    print('Computing Panel A: Climate damage utility ratios...')
    gini_vals_damage = np.linspace(GINI_RANGE_DAMAGE[0], GINI_RANGE_DAMAGE[1], N_GINI_DAMAGE)
    exponent_vals = np.linspace(EXPONENT_RANGE[0], EXPONENT_RANGE[1], N_EXPONENT)
    gini_mesh_damage, exponent_mesh = np.meshgrid(gini_vals_damage, exponent_vals)

    ratio_grid_damage = compute_damage_ratio_grid(gini_mesh_damage, exponent_mesh, base_params_damage)

    print(f'  Panel A ratio range: [{np.min(ratio_grid_damage):.4f}, {np.max(ratio_grid_damage):.4f}]')
    print(f'  Panel A ratio at exponent=0, Gini=0: {ratio_grid_damage[0, 0]:.4f}')
    print(f'  Panel A ratio at exponent=0, Gini=max: {ratio_grid_damage[0, -1]:.4f}')

    min_idx = np.unravel_index(np.argmin(ratio_grid_damage), ratio_grid_damage.shape)
    min_gini = gini_vals_damage[min_idx[1]]
    min_exp = exponent_vals[min_idx[0]]
    print(f'  Panel A minimum at Gini={min_gini:.3f}, exponent={min_exp:.3f}')

    print('Computing Panel B: Progressive taxation utility ratios...')
    gini_vals_tax = np.linspace(GINI_RANGE_TAX[0], GINI_RANGE_TAX[1], N_GINI_TAX)
    tax_rate_vals = np.linspace(TAX_RATE_RANGE[0], TAX_RATE_RANGE[1], N_TAX_RATE)
    gini_mesh_tax, tax_rate_mesh = np.meshgrid(gini_vals_tax, tax_rate_vals)

    ratio_grid_tax = compute_tax_ratio_grid(gini_mesh_tax, tax_rate_mesh, base_params_tax)

    print(f'  Panel B ratio range: [{np.min(ratio_grid_tax):.4f}, {np.max(ratio_grid_tax):.4f}]')
    print(f'  Panel B ratio at tax_rate=0, Gini=0: {ratio_grid_tax[0, 0]:.4f}')
    print(f'  Panel B ratio at tax_rate=0, Gini=max: {ratio_grid_tax[0, -1]:.4f}')

    damage_data = {
        'gini_mesh': gini_mesh_damage,
        'exponent_mesh': exponent_mesh,
        'ratio_grid': ratio_grid_damage,
    }

    tax_data = {
        'gini_mesh': gini_mesh_tax,
        'tax_rate_mesh': tax_rate_mesh,
        'ratio_grid': ratio_grid_tax,
    }

    if csv_export:
        panel_a_csv_path = output_dir / f'{run_name}_panel_a.csv'
        panel_b_csv_path = output_dir / f'{run_name}_panel_b.csv'
        write_panel_a_csv(panel_a_csv_path, gini_mesh_damage, exponent_mesh, ratio_grid_damage)
        write_panel_b_csv(panel_b_csv_path, gini_mesh_tax, tax_rate_mesh, ratio_grid_tax)
        print(f'Exported CSV files: {panel_a_csv_path}, {panel_b_csv_path}')

    if xlsx_export:
        panel_a_xlsx_path = output_dir / f'{run_name}_panel_a.xlsx'
        panel_b_xlsx_path = output_dir / f'{run_name}_panel_b.xlsx'
        write_panel_a_xlsx(panel_a_xlsx_path, gini_mesh_damage, exponent_mesh, ratio_grid_damage)
        write_panel_b_xlsx(panel_b_xlsx_path, gini_mesh_tax, tax_rate_mesh, ratio_grid_tax)
        print(f'Exported XLSX files: {panel_a_xlsx_path}, {panel_b_xlsx_path}')

    print(f'Creating 2-panel PDF figure: {output_pdf_path}')
    create_two_panel_figure(damage_data, tax_data, output_pdf_path)

    print(f'Output saved to: {output_dir}')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 2-panel PDF of utility ratios under climate damage and taxation scenarios'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output PDF filename (default: {run_name}_utility_ratios.pdf in timestamped directory)'
    )

    parser.add_argument(
        '--csv',
        action='store_true',
        help='Export both Panel A and Panel B data as CSV files'
    )
    parser.add_argument(
        '--xlsx',
        action='store_true',
        help='Export both Panel A and Panel B data as XLSX files (requires pandas)'
    )

    args = parser.parse_args()
    main(args.config, args.output, args.csv, args.xlsx)
