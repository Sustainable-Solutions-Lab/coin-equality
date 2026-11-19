"""
Output files of COIN_equality runs.

Each run writes, into a timestamped directory: the results CSV (one row per time step),
the income distributions workbook, the PDF reports (see visualization_utils), the
optimization summary (optimizations only) and the configuration actually run.
"""

import os
import csv
import json
from datetime import datetime
import pandas as pd
from src.visualization_utils import (
    VARIABLE_LABELS,
    YLIM_TIME_FRACTION,
    plot_grouped_results_pdf,
    plot_results_report_pdf,
)


# Results entries that are not time series: the quadrature grid, and the per-quantile
# distributions (time x quadrature node), which go to the workbook
QUADRATURE_KEYS = ['xi', 'wi', 'xi_edges', 'Fi', 'Fwi', 'Fi_edges']
DISTRIBUTION_KEYS = ['y_net_yi', 'omega_yi', 'utility_yi', 'average_tax_rate_yi', 'marginal_tax_rate_yi']

# Calendar years of the short-horizon grouped report
SHORT_REPORT_YEARS = (2025, 2100)


def create_output_directory(run_name, prod_category):
    """
    Create the timestamped run directory data/output/prod/{prod_category}/{run_name}_YYYYMMDD-HHMMSS.

    Parameters
    ----------
    run_name : str
        Name of the model run
    prod_category : str
        Production category path, e.g. '9cases' or 'discount/prod_discount_exact'

    Returns
    -------
    str
        Path to the created directory (relative to the working directory)
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join('data', 'output', 'prod', prod_category, f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def write_config_file(config_dict, path):
    """
    Write the configuration actually run (including command-line overrides) as JSON.

    Uses the layout of the input files (2-space indent), so without overrides the written
    file is byte-identical to the input configuration.
    """
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def write_optimization_summary(opt_results, path, t_base):
    """
    Write the optimization summary CSV.

    Sections: final control values and objective (discounted utility), final f(t) and s(t)
    control points in calendar years, and per refinement iteration the objective,
    evaluations and termination reason, the f values, and the f control times.

    Parameters
    ----------
    opt_results : dict
        Results of optimization.optimize
    path : str
        Output CSV path
    t_base : float
        Base year (added to the relative control times)
    """
    controls = opt_results['controls']
    history = opt_results['iteration_history']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Optimization Summary'])
        writer.writerow([])

        writer.writerow(['Parameter', 'Value'])
        for name, (_, values) in controls.items():
            for i, val in enumerate(values):
                writer.writerow([f'Optimal {name} value at control point {i}', f"{val:.6f}"])
        writer.writerow(['Optimal objective', f"{opt_results['objective']:.12e}"])
        writer.writerow(['Function evaluations', opt_results['n_evaluations']])
        writer.writerow(['Status', 'success'])
        writer.writerow(['Algorithm', opt_results['algorithm']])
        writer.writerow(['Number of iterations', opt_results['n_iterations']])

        for name, (times, values) in controls.items():
            writer.writerow([])
            writer.writerow([f'Final {name}(t) Control Points'])
            writer.writerow(['Time', f'{name} Value'])
            for t_rel, value in zip(times, values):
                writer.writerow([f"{t_rel + t_base:.2f}", f"{value:.6f}"])

        writer.writerow([])
        writer.writerow(['Iterative Refinement - Iteration History'])
        writer.writerow(['Iteration', 'Control_Points', 'Objective', 'Evaluations', 'Status'])
        for iter_result in history:
            writer.writerow([
                iter_result['iteration'],
                len(iter_result['controls']['f'][0]),
                f"{iter_result['objective']:.12e}",
                iter_result['n_evaluations'],
                iter_result['termination_name'],
            ])

        max_points = max(len(iter_result['controls']['f'][0]) for iter_result in history)
        for title, column_prefix, column_index, formatter in [
                ('Control Values by Iteration', 'f', 1, lambda value: f"{value:.6f}"),
                ('Control Times by Iteration', 't', 0, lambda t_rel: f"{t_rel + t_base:.2f}")]:
            writer.writerow([])
            writer.writerow([f'Iterative Refinement - {title}'])
            writer.writerow(['Iteration', 'Control_Points'] + [f'{column_prefix}_{i}' for i in range(max_points)])
            for iter_result in history:
                entries = iter_result['controls']['f'][column_index]
                writer.writerow([iter_result['iteration'], len(entries)]
                                + [formatter(entry) for entry in entries]
                                + [''] * (max_points - len(entries)))


def time_series(results):
    """The results entries with one value per time step."""
    return {name: values for name, values in results.items()
            if name not in QUADRATURE_KEYS and name not in DISTRIBUTION_KEYS}


def write_results_csv(results, path):
    """
    Write the time series to CSV, one column per variable and one row per time step.

    Columns: the VARIABLE_LABELS variables in that order, with header
    "name, description, (units)", then the remaining time series alphabetically with
    their bare name as header.
    """
    series = time_series(results)
    names = list(VARIABLE_LABELS) + sorted(name for name in series if name not in VARIABLE_LABELS)
    headers = [f"{name}, {VARIABLE_LABELS[name][0]}, ({VARIABLE_LABELS[name][1]})" if name in VARIABLE_LABELS
               else name for name in names]
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(zip(*[series[name] for name in names]))


def write_distribution_xlsx(results, path):
    """
    Write the income distributions to an Excel workbook.

    Sheets: 'Quadrature_Info' (xi, wi, xi_edges, Fi, Fwi, Fi_edges) and one sheet per
    DISTRIBUTION_KEYS entry with time in the first column and one column per quadrature
    node (bin_0, bin_1, ...).
    """
    bin_names = [f'bin_{i}' for i in range(len(results['xi']))]
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Edges have one more entry than nodes, so build columns as Series of unequal length
        pd.DataFrame({name: pd.Series(results[name]) for name in QUADRATURE_KEYS}).to_excel(
            writer, sheet_name='Quadrature_Info', index=False)
        for name in DISTRIBUTION_KEYS:
            sheet = pd.DataFrame(results[name], columns=bin_names)
            sheet.insert(0, 't', results['t'])
            sheet.to_excel(writer, sheet_name=name, index=False)


def save_results(results, run_name, output_dir, config_filename):
    """
    Write the results CSV, the distributions workbook and the PDF reports into output_dir.

    Parameters
    ----------
    results : dict
        Results of integrate_model (detailed) and create_derived_variables
    run_name : str
        Prefix of the file names, shown in the plots
    output_dir : str
        Existing output directory
    config_filename : str
        Name of the configuration file, shown on each PDF page

    Returns
    -------
    dict
        Paths of the written files: 'csv_file', 'xlsx_file', 'pdf_file' (grouped report),
        'pdf_file_2025_2100' (grouped report for SHORT_REPORT_YEARS), 'pdf_file_full'
        (full report)
    """
    paths = {
        'csv_file': os.path.join(output_dir, f'{run_name}_results.csv'),
        'xlsx_file': os.path.join(output_dir, f'{run_name}_distributions.xlsx'),
        'pdf_file': os.path.join(output_dir, f'{run_name}_plots.pdf'),
        'pdf_file_2025_2100': os.path.join(output_dir, f'{run_name}_plots_{SHORT_REPORT_YEARS[0]}-{SHORT_REPORT_YEARS[1]}.pdf'),
        'pdf_file_full': os.path.join(output_dir, 'plots_full.pdf'),
    }
    write_results_csv(results, paths['csv_file'])
    write_distribution_xlsx(results, paths['xlsx_file'])

    series = time_series(results)
    plot_grouped_results_pdf(series, paths['pdf_file'], run_name, config_filename, YLIM_TIME_FRACTION)
    short_horizon = (series['t'] >= SHORT_REPORT_YEARS[0]) & (series['t'] <= SHORT_REPORT_YEARS[1])
    plot_grouped_results_pdf({name: values[short_horizon] for name, values in series.items()},
                             paths['pdf_file_2025_2100'], run_name, config_filename, 1.0)
    plot_results_report_pdf(series, paths['pdf_file_full'], run_name, config_filename)
    return paths
