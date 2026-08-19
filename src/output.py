"""
Output functions for COIN_equality model.

Creates CSV files and PDF plots of model results in timestamped directories.
"""

import os
import csv
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


# Variable metadata for better plot labels and descriptions
VARIABLE_METADATA = {
    't': {'description': 'Time', 'units': 'yr', 'group': 'time'},
    'A': {'description': 'Total Factor Productivity', 'units': '', 'group': 'economic'},
    'e': {'description': 'CO₂ Emissions', 'units': 'tCO₂/yr', 'group': 'climate'},
    'Ecum': {'description': 'Cumulative Emissions', 'units': 'tCO₂', 'group': 'climate'},
    'K': {'description': 'Capital Stock', 'units': '$', 'group': 'economic'},
    'L': {'description': 'Population', 'units': 'people', 'group': 'economic'},
    'lambda_abate': {'description': 'Abatement Cost Fraction', 'units': '', 'group': 'abatement'},
    'Omega': {'description': 'Climate Damage Fraction', 'units': '', 'group': 'climate'},
    'Omega_base': {'description': 'Base Climate Damage Fraction', 'units': '', 'group': 'climate'},
    'Omega_calc': {'description': 'Lagged Climate Damage Fraction', 'units': '', 'group': 'climate'},
    'U': {'description': 'Mean Population Utility', 'units': '', 'group': 'inequality'},
    'y_gross': {'description': 'Per-Capita Gross Income', 'units': '$/person', 'group': 'economic'},
    'y_damaged': {'description': 'Per-Capita Income After Climate Damage', 'units': '$/person', 'group': 'economic'},
    'y_net': {'description': 'Per-Capita Net Income', 'units': '$/person', 'group': 'economic'},
    'abateCost_amount': {'description': 'Abatement Expenditure Per Capita', 'units': '$/person', 'group': 'abatement'},
    'dK_dt': {'description': 'Capital Growth Rate', 'units': '$/yr', 'group': 'economic'},
    'delta_T': {'description': 'Temperature Change', 'units': '°C', 'group': 'climate'},
    'max_average_tax_rate': {'description': 'Maximum Average Tax Rate', 'units': '', 'group': 'inequality'},
    'max_marginal_tax_rate': {'description': 'Maximum Marginal Tax Rate', 'units': '', 'group': 'inequality'},
    'f': {'description': 'log₁₀(MAC)', 'units': 'log₁₀($/tCO₂)', 'group': 'policy'},
    'mu': {'description': 'Emissions Abatement Fraction', 'units': '', 'group': 'abatement'},
    'sigma': {'description': 'Carbon Intensity of GDP', 'units': 'tCO₂/$', 'group': 'climate'},
    'theta1': {'description': 'Marginal Abatement Cost at mu=1', 'units': '$/tCO₂', 'group': 'abatement'},
    's': {'description': 'Savings Rate', 'units': '', 'group': 'policy'},
    'gini': {'description': 'Background Gini Index', 'units': '', 'group': 'inequality'},
    'Gini': {'description': 'Gini Index', 'units': '', 'group': 'inequality'},
    'tax_amount': {'description': 'Per-Capita Tax Amount', 'units': '$/person', 'group': 'policy'},
    'climate_damage': {'description': 'Per-Capita Climate Damage', 'units': '$/person', 'group': 'climate'},
    'aggregate_utility': {'description': 'Aggregate Utility', 'units': '', 'group': 'inequality'},
    'log_deltaU_norm': {'description': 'Log Normalized Utility-Loss (log ΔU)', 'units': '', 'group': 'economic'},
    'emission_ratio': {'description': 'CO2e to CO2 Ratio', 'units': '', 'group': 'climate'},
}

# Variable grouping for organized layout with combined charts
# Ordered by: dimensionless ratios, dollar variables, everything else, specified functions
VARIABLE_GROUPS = {
    'dimensionless_ratios': [
        {'type': 'combined', 'title': 'Control Variables', 'variables': ['f', 's'], 'units': 'see legend'},
        {'type': 'single', 'variables': ['mu']},
        {'type': 'combined', 'title': 'Economic Impact Fractions', 'variables': ['Omega', 'lambda_abate'], 'units': 'fraction'},
        {'type': 'single', 'variables': ['Gini']},
        {'type': 'combined', 'title': 'Max Tax Rates', 'variables': ['max_average_tax_rate', 'max_marginal_tax_rate'], 'units': 'dimensionless'},
        {'type': 'single', 'variables': ['U']},
    ],
    'dollar_variables': [
        {'type': 'combined', 'title': 'Per-Capita Income', 'variables': ['y_gross', 'y_damaged', 'y_net'], 'units': '$/person'},
        {'type': 'single', 'variables': ['K']},
        {'type': 'single', 'variables': ['abateCost_amount']},
        {'type': 'single', 'variables': ['dK_dt']},
        {'type': 'single', 'variables': ['tax_amount']},
        {'type': 'single', 'variables': ['climate_damage']}
    ],
    'physical_variables': [
        {'type': 'single', 'variables': ['e']},
        {'type': 'single', 'variables': ['emission_ratio']},
        {'type': 'single', 'variables': ['Ecum']},
        {'type': 'single', 'variables': ['delta_T']}
    ],
    'specified_functions': [
        {'type': 'single', 'variables': ['L']},
        {'type': 'single', 'variables': ['A']},
        {'type': 'single', 'variables': ['sigma']},
        {'type': 'single', 'variables': ['theta1']}
    ]
}


def format_scientific_notation(x, pos=None):
    """Custom formatter for scientific notation with proper spacing."""
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exp)
    if abs(exp) <= 2:
        return f'{x:.3g}'
    else:
        return f'{mantissa:.1f}×10^{exp}'


def create_output_directory(run_name, prod_category):
    """
    Create timestamped output directory under data/output/prod/{prod_category}/.

    Parameters
    ----------
    run_name : str
        Name of the model run
    prod_category : str
        Production category path, e.g. 'discount/prod_discount_exact'
        or '9cases'. Determines the subdirectory under data/output/prod/.

    Returns
    -------
    str
        Path to created output directory

    Notes
    -----
    Directory format: ./data/output/prod/{prod_category}/{run_name}_YYYYMMDD-HHMMSS
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join('data', 'output', 'prod', prod_category, f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def copy_config_file(config_path, output_dir):
    """
    Copy configuration JSON file to output directory.

    Parameters
    ----------
    config_path : str
        Path to input configuration file
    output_dir : str
        Directory to copy file to

    Returns
    -------
    str
        Path to copied configuration file

    Notes
    -----
    Preserves the original filename and formatting.
    Ensures reproducibility by keeping exact copy of configuration used.
    """
    filename = os.path.basename(config_path)
    output_path = os.path.join(output_dir, filename)
    shutil.copy2(config_path, output_path)
    return output_path


def write_optimization_summary(opt_results, sensitivity_results, output_dir, run_name, filename='optimization_summary.csv', t_base=0.0):
    """
    Write optimization summary statistics to CSV file.

    Parameters
    ----------
    opt_results : dict
        Optimization results from UtilityOptimizer
    sensitivity_results : dict or None
        Sensitivity analysis results (optional)
    output_dir : str
        Directory to write CSV file
    run_name : str
        Name of the model run to prepend to filename
    filename : str
        Name of CSV file
    t_base : float
        Base year for time functions (added to relative times for output)

    Returns
    -------
    str
        Path to created CSV file

    Notes
    -----
    Creates a CSV file with optimization statistics including:
    - Optimal control point values
    - Objective function value
    - Number of function evaluations
    - Convergence status
    - Iteration-by-iteration results (for iterative refinement mode)
    - Sensitivity analysis statistics (if provided)
    """
    csv_path = os.path.join(output_dir, f"{run_name}_{filename}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Optimization Summary'])
        writer.writerow([])

        writer.writerow(['Parameter', 'Value'])

        # Check if this is dual optimization (has s values)
        is_dual = 's_optimal_values' in opt_results and opt_results['s_optimal_values'] is not None

        # Write optimal f values
        for i, val in enumerate(opt_results['optimal_values']):
            writer.writerow([f'Optimal f value at control point {i}', f"{val:.6f}"])

        # Write optimal s values if dual optimization
        if is_dual:
            for i, val in enumerate(opt_results['s_optimal_values']):
                writer.writerow([f'Optimal s value at control point {i}', f"{val:.6f}"])

        writer.writerow(['Optimal objective', f"{opt_results['optimal_objective']:.12e}"])
        writer.writerow(['Function evaluations', opt_results['n_evaluations']])
        writer.writerow(['Status', opt_results['status']])

        if 'algorithm' in opt_results:
            writer.writerow(['Algorithm', opt_results['algorithm']])
        if 'n_iterations' in opt_results:
            writer.writerow(['Number of iterations', opt_results['n_iterations']])
        if 'termination_name' in opt_results:
            writer.writerow(['Termination reason', opt_results['termination_name']])
        if 'termination_code' in opt_results and opt_results['termination_code'] is not None:
            writer.writerow(['Termination code', opt_results['termination_code']])

        writer.writerow([])

        # Write final control points for f (times converted to calendar years)
        writer.writerow(['Final f(t) Control Points'])
        writer.writerow(['Time', 'f Value'])
        for t_rel, value in opt_results['control_points']:
            writer.writerow([f"{t_rel + t_base:.2f}", f"{value:.6f}"])

        # Write final control points for s if dual optimization
        if is_dual and 's_control_points' in opt_results:
            writer.writerow([])
            writer.writerow(['Final s(t) Control Points'])
            writer.writerow(['Time', 's Value'])
            for t_rel, value in opt_results['s_control_points']:
                writer.writerow([f"{t_rel + t_base:.2f}", f"{value:.6f}"])

        if 'iteration_history' in opt_results:
            writer.writerow([])
            writer.writerow(['Iterative Refinement - Iteration History'])
            writer.writerow(['Iteration', 'Control_Points', 'Objective', 'Evaluations', 'Status'])
            for iter_result in opt_results['iteration_history']:
                writer.writerow([
                    iter_result['iteration'],
                    iter_result['n_control_points'],
                    f"{iter_result['optimal_objective']:.12e}",
                    iter_result['n_evaluations'],
                    iter_result['termination_name']
                ])

            writer.writerow([])
            writer.writerow(['Iterative Refinement - Control Values by Iteration'])

            max_points = max(iter_result['n_control_points'] for iter_result in opt_results['iteration_history'])
            header = ['Iteration', 'Control_Points'] + [f'f_{i}' for i in range(max_points)]
            writer.writerow(header)

            for iter_result in opt_results['iteration_history']:
                row = [
                    iter_result['iteration'],
                    iter_result['n_control_points']
                ]
                for val in iter_result['optimal_values']:
                    row.append(f"{val:.6f}")
                for _ in range(max_points - len(iter_result['optimal_values'])):
                    row.append('')
                writer.writerow(row)

            writer.writerow([])
            writer.writerow(['Iterative Refinement - Control Times by Iteration'])
            header = ['Iteration', 'Control_Points'] + [f't_{i}' for i in range(max_points)]
            writer.writerow(header)

            for i, iter_result in enumerate(opt_results['iteration_history']):
                control_times = opt_results['iteration_control_grids'][i]
                row = [
                    iter_result['iteration'],
                    iter_result['n_control_points']
                ]
                for t_rel in control_times:
                    row.append(f"{t_rel + t_base:.2f}")
                for _ in range(max_points - len(control_times)):
                    row.append('')
                writer.writerow(row)

        if sensitivity_results:
            writer.writerow([])
            writer.writerow(['Sensitivity Analysis'])
            writer.writerow(['f_Value', 'Objective'])
            for f_val, obj in zip(sensitivity_results['f_values'], sensitivity_results['objectives']):
                writer.writerow([f"{f_val:.6f}", f"{obj:.6e}"])

    return csv_path


def write_results_csv(results, output_dir, run_name='', filename='results.csv'):
    """
    Write results dictionary to CSV file.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    output_dir : str
        Directory to write CSV file
    run_name : str
        Name of the model run (prepended to filename)
    filename : str
        Name of CSV file

    Returns
    -------
    str
        Path to created CSV file

    Notes
    -----
    Each column is a variable, each row is a time point.
    First row contains variable names (header).
    """
    if run_name:
        csv_path = os.path.join(output_dir, f"{run_name}_{filename}")
    else:
        csv_path = os.path.join(output_dir, filename)

    # Define column order as specified
    ordered_columns = [
        't',  # Time
        # Time-dependent driving parameters
        'A',  # Total factor productivity
        'L',  # Population
        'theta1',  # Marginal abatement cost at mu=1
        'sigma',  # Carbon intensity of GDP
        # Decision variables
        'f',  # Control: log10(marginal_abatement_cost)
        's',  # Savings rate
        # State variables
        'K',  # Capital stock
        'gini',  # Background Gini index
        'Gini',  # Gini index
        'Ecum',  # Cumulative emissions
        'delta_T',  # Global mean temperature change
        # Per capita variables
        'y_gross',  # Per-capita gross income
        'y_damaged',  # Per-capita income after climate damage
        'y_net',  # Per-capita net income after abatement
        'climate_damage',  # Per-capita climate damage
        'tax_amount',  # Per-capita tax amount
        # Dimensionless variables
        'mu',  # Abatement fraction
        'Omega',  # Climate damage as fraction of gross output
        'Omega_base',  # Base damage from temperature
        'Omega_calc',  # Climate damage from previous timestep (lagged)
        'lambda_abate',  # Abatement cost as fraction of damaged output
        'abateCost_amount',  # Abatement expenditure per capita
        'U',  # Mean utility per capita
        'aggregate_utility',  # Aggregate utility from integration
        # Emissions
        'e',  # CO2 emissions
        'emission_ratio',  # CO2e to CO2 ratio
        'E_extra',  # Extra emission pulse
        'consumption_extra',  # Extra consumption pulse
        # Other
        'dK_dt',  # Net capital accumulation
        'log_deltaU_norm',  # Log normalized utility-loss (log deltaU)
        'max_average_tax_rate',  # Maximum average tax rate across income bins
        'max_marginal_tax_rate',  # Maximum marginal tax rate across income bins
    ]

    # Add any remaining variables not in the ordered list
    # Exclude 2D distribution arrays (those go in xlsx file) and quadrature info
    exclude_from_csv = {'y_net_yi', 'omega_yi', 'utility_yi', 'xi', 'wi', 'xi_edges', 'Fi', 'Fwi', 'Fi_edges', 'average_tax_rate_yi', 'marginal_tax_rate_yi', 'params_list'}
    remaining_vars = sorted([k for k in results.keys()
                            if k not in ordered_columns and k not in exclude_from_csv])
    var_names = ordered_columns + remaining_vars

    # Define variable descriptions and units
    var_info = {
        't': ('Time', 'yr'),
        'A': ('Total factor productivity', '$'),
        'L': ('Population', 'persons'),
        'theta1': ('Marginal abatement cost at mu=1', '$/tCO2'),
        'sigma': ('Carbon intensity of GDP', 'tCO2/$'),
        'f': ('Control: log10(marginal_abatement_cost)', 'log10($/tCO2)'),
        's': ('Savings rate', 'dimensionless'),
        'K': ('Capital stock', '$'),
        'gini': ('Background Gini index', 'dimensionless'),
        'Gini': ('Gini index', 'dimensionless'),
        'Ecum': ('Cumulative CO2 emissions', 'tCO2'),
        'delta_T': ('Global mean temperature change', '°C'),
        'y_gross': ('Per-capita gross income', '$/person/yr'),
        'y_damaged': ('Per-capita income after climate damage', '$/person/yr'),
        'y_net': ('Per-capita net income after abatement', '$/person/yr'),
        'climate_damage': ('Per-capita climate damage', '$/person/yr'),
        'tax_amount': ('Per-capita tax amount', '$/person/yr'),
        'mu': ('Abatement fraction', 'dimensionless'),
        'Omega': ('Climate damage as fraction of gross output', 'dimensionless'),
        'Omega_base': ('Base damage from temperature before income adjustment', 'dimensionless'),
        'Omega_calc': ('Climate damage from previous timestep (lagged)', 'dimensionless'),
        'lambda_abate': ('Abatement cost as fraction of damaged output', 'dimensionless'),
        'abateCost_amount': ('Abatement expenditure per capita', '$/person/yr'),
        'U': ('Mean utility per capita', 'dimensionless'),
        'aggregate_utility': ('Aggregate utility from integration', 'dimensionless'),
        'e': ('CO2 emissions', 'tCO2/yr'),
        'emission_ratio': ('Ratio of CO2e to industrial CO2', 'dimensionless'),
        'E_extra': ('Extra emission pulse', 'tCO2/yr'),
        'consumption_extra': ('Extra consumption pulse', '$/person/yr'),
        'dK_dt': ('Net capital accumulation', '$/yr'),
        'log_deltaU_norm': ('Log normalized utility-loss (log deltaU)', 'dimensionless'),
        'max_average_tax_rate': ('Maximum average tax rate across income bins', 'dimensionless'),
        'max_marginal_tax_rate': ('Maximum marginal tax rate across income bins', 'dimensionless'),
    }

    # Create headers with format: "variable, description, (units)"
    headers = []
    for var in var_names:
        if var in var_info:
            desc, units = var_info[var]
            headers.append(f"{var}, {desc}, ({units})")
        else:
            # Fallback for any variables not in the dictionary
            headers.append(var)

    # Open CSV file and write
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(headers)

        # Get number of time points
        n_points = len(results['t'])

        # Write data rows
        for i in range(n_points):
            row = [results[var][i] for var in var_names]
            writer.writerow(row)

    return csv_path


def write_distribution_xlsx(results, output_dir, run_name, filename='distributions.xlsx'):
    """
    Write income distribution data to Excel file with multiple sheets.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    output_dir : str
        Directory to write Excel file
    run_name : str
        Name of the model run (added to filename)
    filename : str
        Name of Excel file (default: 'distributions.xlsx')

    Returns
    -------
    str
        Path to created Excel file

    Notes
    -----
    Creates Excel file with sheets:
    - 'Quadrature_Info': xi, wi, xi_edges, Fi, Fwi, Fi_edges
    - 'y_net_yi': Per capita net income distribution over time
    - 'omega_yi': Climate damage fractions (dimensionless) at quadrature points over time
    - 'utility_yi': Utility distribution over time

    Each distribution sheet has time in the first column and income bins in remaining columns.
    """
    # Check if distribution data is available
    if 'y_net_yi' not in results:
        print("Warning: Distribution data not available, skipping xlsx output")
        return None

    xlsx_path = os.path.join(output_dir, f"{run_name}_{filename}")

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Sheet 1: Quadrature information
        quad_df = pd.DataFrame({
            'xi': results['xi'],
            'wi': results['wi'],
            'Fi': results['Fi'],
            'Fwi': results['Fwi'],
        })
        # Add edges separately since they have different length
        max_len = max(len(results['xi']), len(results['xi_edges']))
        quad_df_full = pd.DataFrame(index=range(max_len))
        quad_df_full['xi'] = pd.Series(results['xi'])
        quad_df_full['wi'] = pd.Series(results['wi'])
        quad_df_full['xi_edges'] = pd.Series(results['xi_edges'])
        quad_df_full['Fi'] = pd.Series(results['Fi'])
        quad_df_full['Fwi'] = pd.Series(results['Fwi'])
        quad_df_full['Fi_edges'] = pd.Series(results['Fi_edges'])
        quad_df_full.to_excel(writer, sheet_name='Quadrature_Info', index=False)

        # Sheet 2: y_net_yi (net income distribution)
        t = results['t']
        n_bins = results['y_net_yi'].shape[1]
        bin_names = [f'bin_{i}' for i in range(n_bins)]
        y_net_df = pd.DataFrame(results['y_net_yi'], columns=bin_names)
        y_net_df.insert(0, 't', t)
        y_net_df.to_excel(writer, sheet_name='y_net_yi', index=False)

        # Sheet 3: omega_yi (damage fractions at quadrature points)
        omega_yi_df = pd.DataFrame(results['omega_yi'], columns=bin_names)
        omega_yi_df.insert(0, 't', t)
        omega_yi_df.to_excel(writer, sheet_name='omega_yi', index=False)

        # Sheet 4: utility_yi (utility distribution)
        utility_df = pd.DataFrame(results['utility_yi'], columns=bin_names)
        utility_df.insert(0, 't', t)
        utility_df.to_excel(writer, sheet_name='utility_yi', index=False)

        # Sheet 5: average_tax_rate_yi (average tax rate at quadrature points)
        if 'average_tax_rate_yi' in results:
            avg_tax_df = pd.DataFrame(results['average_tax_rate_yi'], columns=bin_names)
            avg_tax_df.insert(0, 't', t)
            avg_tax_df.to_excel(writer, sheet_name='average_tax_rate_yi', index=False)

        # Sheet 6: marginal_tax_rate_yi (marginal tax rate at quadrature points)
        if 'marginal_tax_rate_yi' in results:
            marg_tax_df = pd.DataFrame(results['marginal_tax_rate_yi'], columns=bin_names)
            marg_tax_df.insert(0, 't', t)
            marg_tax_df.to_excel(writer, sheet_name='marginal_tax_rate_yi', index=False)

    print(f"Distribution data saved to: {xlsx_path}")
    return xlsx_path


def plot_results_pdf(results, output_dir, run_name, filename='plots.pdf', config_filename=None, use_first_90_percent_for_ylim=False):
    """
    Create PDF with time series plots of all variables, organized by topic with combined charts.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    output_dir : str
        Directory to write PDF file
    run_name : str
        Name of the model run to display in header
    filename : str
        Name of PDF file
    config_filename : str, optional
        Name of configuration file to display on each page
    use_first_90_percent_for_ylim : bool, optional
        If True, calculate y-axis limits based on first 90% of time values (for full PDF)

    Returns
    -------
    str
        Path to created PDF file

    Notes
    -----
    Creates multi-page PDF organized by variable groups (economic, climate, etc.).
    Supports both single-variable and multi-variable combined charts.
    """
    pdf_path = os.path.join(output_dir, f"{run_name}_{filename}")

    # Get time array
    t = results['t']

    # Create PDF with organized plots by groups
    with PdfPages(pdf_path) as pdf:

        # Plot each group on separate pages
        for group_name, chart_specs in VARIABLE_GROUPS.items():
            # Filter chart specs to only include those with available variables
            available_charts = []
            for spec in chart_specs:
                available_vars = [var for var in spec['variables'] if var in results]
                if available_vars:
                    spec_copy = spec.copy()
                    spec_copy['variables'] = available_vars
                    available_charts.append(spec_copy)

            if not available_charts:
                continue

            # Determine optimal subplot layout
            n_charts = len(available_charts)
            if n_charts <= 4:
                rows, cols = 2, 2
                figsize = (12, 8)
            elif n_charts <= 6:
                rows, cols = 2, 3
                figsize = (15, 8)
            elif n_charts <= 9:
                rows, cols = 3, 3
                figsize = (15, 12)
            else:
                # Split large groups across multiple pages
                charts_per_page = 6
                for page_start in range(0, n_charts, charts_per_page):
                    page_charts = available_charts[page_start:page_start + charts_per_page]
                    _create_plot_page_new(t, results, page_charts, group_name, run_name, pdf, page_start//charts_per_page + 1, config_filename=config_filename, use_first_90_percent_for_ylim=use_first_90_percent_for_ylim)
                continue

            # Create single page for this group
            _create_plot_page_new(t, results, available_charts, group_name, run_name, pdf, layout=(rows, cols), figsize=figsize, config_filename=config_filename, use_first_90_percent_for_ylim=use_first_90_percent_for_ylim)

    return pdf_path


def _create_plot_page_new(t, results, chart_specs, group_name, run_name, pdf, page_num=None, layout=None, figsize=None, config_filename=None, use_first_90_percent_for_ylim=False):
    """
    Create a single page of plots for a variable group with support for combined charts.

    Parameters
    ----------
    t : array
        Time array
    results : dict
        Results dictionary
    chart_specs : list
        List of chart specifications (single or combined)
    group_name : str
        Name of the variable group
    run_name : str
        Name of the model run to display in header
    pdf : PdfPages
        PDF object to add page to
    page_num : int, optional
        Page number for multi-page groups
    layout : tuple, optional
        (rows, cols) layout. If None, defaults to (2, 3)
    figsize : tuple, optional
        Figure size. If None, defaults to (15, 10)
    config_filename : str, optional
        Name of configuration file to display on page
    use_first_90_percent_for_ylim : bool, optional
        If True, calculate y-axis limits based on first 90% of time values
    """
    if layout is None:
        layout = (2, 3)
    if figsize is None:
        figsize = (15, 10)

    rows, cols = layout
    n_charts = len(chart_specs)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Create title with run_name
    title = f'{run_name} - COIN_equality Model Results - {group_name.title()} Variables'
    if page_num is not None:
        title += f' (Page {page_num})'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    # Handle single subplot case
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Define colors for multi-line plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Variables that should use logarithmic y-axis
    log_scale_vars = {'y_net', 'y_damaged', 'y_gross', 'K', 'A'}

    # Plot each chart
    for i, chart_spec in enumerate(chart_specs):
        if i >= len(axes):
            break

        ax = axes[i]
        chart_type = chart_spec['type']
        var_list = chart_spec['variables']

        if chart_type == 'single':
            # Single variable plot
            var_name = var_list[0]
            meta = VARIABLE_METADATA.get(var_name, {})
            description = meta.get('description', var_name)
            units = meta.get('units', '')

            # Plot the time series
            ax.plot(t, results[var_name], linewidth=2, color=colors[0], alpha=0.8)

            # Set labels
            ax.set_xlabel('Time (yr)', fontsize=11)
            if units:
                ylabel = f'{description}\n({units})'
            else:
                ylabel = description
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{var_name}: {description}', fontsize=12, fontweight='bold', pad=10)

        elif chart_type == 'combined':
            # Combined variables plot
            chart_title = chart_spec.get('title', 'Combined Variables')
            chart_units = chart_spec.get('units', '')

            # Plot each variable with different colors
            for j, var_name in enumerate(var_list):
                meta = VARIABLE_METADATA.get(var_name, {})
                description = meta.get('description', var_name)

                color = colors[j % len(colors)]
                ax.plot(t, results[var_name], linewidth=2, color=color, alpha=0.8,
                       label=description)

            # Set labels
            ax.set_xlabel('Time (yr)', fontsize=11)
            if chart_units:
                ylabel = f'{chart_title}\n({chart_units})'
            else:
                ylabel = chart_title
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(chart_title, fontsize=12, fontweight='bold', pad=10)

            # Add legend
            ax.legend(fontsize=9, loc='best', framealpha=0.8)

        # Apply logarithmic scale if any variable in this chart uses it
        if any(var in log_scale_vars for var in var_list):
            ax.set_yscale('log')

        # Improve grid and formatting
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Use custom scientific notation formatting (skip for log scale - already handles it)
        if not any(var in log_scale_vars for var in var_list):
            all_data = np.concatenate([results[var] for var in var_list])
            max_val = np.max(np.abs(all_data))

            if max_val == 0:
                pass  # No formatting needed for zero data
            elif max_val > 1e4 or max_val < 1e-2:
                ax.yaxis.set_major_formatter(FuncFormatter(format_scientific_notation))

        # Set background color
        ax.set_facecolor('#fafafa')

    # Hide unused subplots
    for i in range(n_charts, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Apply zero-bound expansion AFTER layout adjustment (to prevent being overridden)
    for i, chart_spec in enumerate(chart_specs):
        if i >= len(axes):
            break
        ax = axes[i]
        var_list = chart_spec['variables']

        # Skip log scale plots
        if not any(var in log_scale_vars for var in var_list):
            # Determine which time indices to use for calculating y-axis limits
            if use_first_90_percent_for_ylim:
                # Use first 90% of time points for y-axis scaling (excluding first time step)
                n_points = len(t)
                n_points_for_ylim = int(0.9 * n_points)
                all_data = np.concatenate([results[var][1:n_points_for_ylim] for var in var_list])
            else:
                # Use all data points (excluding first time step)
                all_data = np.concatenate([results[var][1:] for var in var_list])

            finite_data = all_data[np.isfinite(all_data)]
            data_min = np.min(finite_data) if len(finite_data) > 0 else -1.0
            data_max = np.max(finite_data) if len(finite_data) > 0 else 1.0

            # Add padding to the data range (5% on each side)
            data_range = data_max - data_min
            padding = 0.05 * data_range if data_range > 0 else 0.05 * abs(data_max)
            ymin_default = data_min - padding
            ymax_default = data_max + padding

            # Apply zero-bound expansion if data doesn't cross zero
            if data_min * data_max > 0:  # Same sign
                abs_data_min = abs(data_min)
                abs_data_max = abs(data_max)
                # If the smaller DATA bound is less than half the larger, replace it with zero
                if min(abs_data_min, abs_data_max) < 0.5 * max(abs_data_min, abs_data_max):
                    if abs_data_min < abs_data_max:
                        # Data starts closer to zero - set lower bound to zero
                        ymin_default = 0
                    else:
                        # Data ends closer to zero - set upper bound to zero
                        ymax_default = 0

            # Set the y-axis limits
            ax.set_ylim(ymin_default, ymax_default)

    # Add config filename at bottom of page if provided
    if config_filename:
        fig.text(0.99, 0.01, f'Config: {config_filename}',
                ha='right', va='bottom', fontsize=6, color='gray', style='italic')

    # Save to PDF
    pdf.savefig(fig, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_results(results, run_name, prod_category, plot_short_horizon=None, output_dir=None, config_filename=None):
    """
    Save model results to CSV and PDF in timestamped directory.

    Parameters
    ----------
    results : dict
        Results dictionary from integrate_model()
    run_name : str
        Name of the model run
    prod_category : str
        Production category path, e.g. 'discount/prod_discount_exact'
        or '9cases'. Used when output_dir is None to route outputs under
        data/output/prod/{prod_category}/.
    plot_short_horizon : float or None
        DEPRECATED - ignored. The 2025-2100 plot is always created.
    output_dir : str or None
        If provided, uses this directory for output. If None, creates new timestamped directory.
    config_filename : str, optional
        Name of configuration file to display on each PDF page

    Returns
    -------
    dict
        Dictionary with paths:
        - 'output_dir': path to output directory
        - 'csv_file': path to CSV file
        - 'pdf_file': path to full PDF file
        - 'pdf_file_2025_2100': path to 2025-2100 PDF file (if data covers this range)

    Notes
    -----
    Creates directory: ./data/output/prod/{prod_category}/{run_name}_YYYYMMDD-HHMMSS (if output_dir not provided)
    Writes files:
    - results.csv: all variables in tabular format
    - plots.pdf: time series plots for entire integration period
    - plots_2025-2100.pdf: time series plots for calendar years 2025-2100 (if available)
    """
    if output_dir is None:
        output_dir = create_output_directory(run_name, prod_category)
    csv_file = write_results_csv(results, output_dir, run_name)
    xlsx_file = write_distribution_xlsx(results, output_dir, run_name)

    output_dict = {
        'output_dir': output_dir,
        'csv_file': csv_file,
        'xlsx_file': xlsx_file,
    }

    # Exclude quadrature arrays and distribution data from plots (these are in XLSX only)
    exclude_from_plots = {'y_net_yi', 'omega_yi', 'utility_yi', 'xi', 'wi', 'xi_edges', 'Fi', 'Fwi', 'Fi_edges', 'average_tax_rate_yi', 'marginal_tax_rate_yi', 'params_list'}
    results_for_plots = {k: v for k, v in results.items() if k not in exclude_from_plots}

    # Always create full PDF
    pdf_file = plot_results_pdf(results_for_plots, output_dir, run_name, filename='plots.pdf', config_filename=config_filename, use_first_90_percent_for_ylim=True)
    output_dict['pdf_file'] = pdf_file

    # Always create 2025-2100 filtered PDF (using calendar years)
    t = results['t']
    mask = (t >= 2025) & (t <= 2100)
    if np.any(mask):
        n_time = len(t)
        results_2025_2100 = {}
        for key, val in results_for_plots.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 1 and len(val) == n_time:
                    results_2025_2100[key] = val[mask]
                elif val.ndim == 2 and val.shape[0] == n_time:
                    results_2025_2100[key] = val[mask, :]
                else:
                    results_2025_2100[key] = val
            else:
                results_2025_2100[key] = val

        pdf_file_2025_2100 = plot_results_pdf(results_2025_2100, output_dir, run_name, filename='plots_2025-2100.pdf', config_filename=config_filename)
        output_dict['pdf_file_2025_2100'] = pdf_file_2025_2100

    return output_dict
