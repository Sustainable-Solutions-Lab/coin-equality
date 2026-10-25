"""
PDF plots of model results.

Two reports of the time series in the results dictionary:
- a grouped report (one page per topic, with combined charts), for the whole run and
  for 2025-2100
- a full report (one variable per panel, 6 panels per page), including derived variables
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


# Description and units of the reported variables, in results-CSV column order (also
# the CSV header "name, description, (units)")
VARIABLE_LABELS = {
    't': ('Time', 'yr'),
    'A': ('Total factor productivity', '$'),
    'L': ('Population', 'persons'),
    'theta1': ('Marginal abatement cost at mu=1', '$/tCO2'),
    'sigma': ('Carbon intensity of GDP', 'tCO2/$'),
    'f': ('Control: log10(marginal_abatement_cost)', 'log10($/tCO2)'),
    's': ('Savings rate', 'dimensionless'),
    'K': ('Capital stock', '$'),
    'gini': ('Background Gini index', 'dimensionless'),
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

# Grouped report: one page per group; a chart is one variable or several on shared axes
VARIABLE_GROUPS = {
    'dimensionless_ratios': [
        {'variables': ['f', 's'], 'title': 'Control Variables', 'units': 'see legend'},
        {'variables': ['mu']},
        {'variables': ['Omega', 'lambda_abate'], 'title': 'Economic Impact Fractions', 'units': 'fraction'},
        {'variables': ['gini']},
        {'variables': ['max_average_tax_rate', 'max_marginal_tax_rate'], 'title': 'Max Tax Rates', 'units': 'dimensionless'},
        {'variables': ['U']},
    ],
    'dollar_variables': [
        {'variables': ['y_gross', 'y_damaged', 'y_net'], 'title': 'Per-Capita Income', 'units': '$/person'},
        {'variables': ['K']},
        {'variables': ['abateCost_amount']},
        {'variables': ['dK_dt']},
        {'variables': ['tax_amount']},
        {'variables': ['climate_damage']},
    ],
    'physical_variables': [
        {'variables': ['e']},
        {'variables': ['emission_ratio']},
        {'variables': ['Ecum']},
        {'variables': ['delta_T']},
    ],
    'specified_functions': [
        {'variables': ['L']},
        {'variables': ['A']},
        {'variables': ['sigma']},
        {'variables': ['theta1']},
    ],
}

# Full report panels: (variable, y-axis label, title)
REPORT_PANELS = [
    # Economic variables
    ('y_net', 'Effective income ($/person/yr)', 'Effective Per-Capita Income'),
    ('K', 'Capital stock ($)', 'Capital Stock Over Time'),
    ('Consumption', 'Total consumption ($/yr)', 'Total Consumption'),
    ('Savings', 'Gross investment ($/yr)', 'Gross Investment'),
    ('s', 'Savings rate', 'Savings Rate Over Time'),
    ('Y_gross', 'Gross production ($/yr)', 'Gross Production'),
    ('Y_net', 'Net output ($/yr)', 'Net Output After Damages & Abatement'),
    # Climate variables
    ('delta_T', 'Temperature change (°C)', 'Temperature Change from Pre-Industrial'),
    ('E', 'CO₂ emissions (tCO2/yr)', 'CO₂ Emissions Rate'),
    ('Ecum', 'Cumulative emissions (tCO2)', 'Cumulative CO₂ Emissions'),
    # Abatement and damage
    ('f', 'log₁₀(MAC) (log₁₀($/tCO₂))', 'Control: log₁₀(Marginal Abatement Cost)'),
    ('mu', 'Abatement fraction', 'Emissions Abatement Fraction'),
    ('lambda_abate', 'Abatement cost fraction', 'Abatement Cost (% of Output)'),
    ('AbateCost', 'Abatement cost ($/yr)', 'Total Abatement Cost'),
    ('Omega', 'Climate damage fraction', 'Climate Damage (% of Output)'),
    ('Climate_damage', 'Climate damage ($/yr)', 'Total Climate Damage'),
    # Inequality, utility and discounting
    ('gini', 'Background Gini index', 'Background Gini Index'),
    ('U', 'Mean utility', 'Mean Utility Per Capita'),
    ('marginal_abatement_cost', 'Marginal abatement cost ($/tCO2)', 'Marginal Abatement Cost'),
    ('r_consumption', 'Effective discount rate', 'Effective Discount Rate on Consumption'),
    # Exogenous drivers
    ('A', 'Total factor productivity ($)', 'Total Factor Productivity'),
    ('L', 'Population (persons)', 'Population'),
    ('sigma', 'Carbon intensity (tCO2/$)', 'Carbon Intensity of GDP'),
    ('theta1', 'Marginal cost at mu=1 ($/tCO2)', 'Abatement Cost Parameter (theta1)'),
]
REPORT_PANELS_PER_PAGE = 6

# Variables plotted on a logarithmic y-axis
LOG_SCALE_VARIABLES = {'y_gross', 'y_damaged', 'y_net', 'K', 'A', 'Consumption', 'Savings', 'Y_gross', 'Y_net',
                       'AbateCost', 'marginal_abatement_cost', 'Climate_damage'}

# Y-axis limits: padding as a fraction of the data range; a same-sign axis is extended to
# zero when its smaller bound is below this fraction of the larger
YLIM_PADDING_FRACTION = 0.05
YLIM_ZERO_EXTENSION_RATIO = 0.5
# Fraction of the time series (from the start) used for y-axis limits, ignoring late-time outliers
YLIM_TIME_FRACTION = 0.9


def format_scientific_notation(x, pos):
    """Tick formatter: plain for 1e-2 to 1e3, else mantissa x 10^exponent."""
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exp)
    if abs(exp) <= 2:
        return f'{x:.3g}'
    return f'{mantissa:.1f}×10^{exp}'


def set_zero_bound_ylim(ax, data):
    """
    Y-axis limits around the finite data with padding, extended to zero when the data keep
    one sign and their smaller magnitude is under YLIM_ZERO_EXTENSION_RATIO of the larger.
    """
    finite_data = data[np.isfinite(data)]
    data_min = np.min(finite_data) if len(finite_data) > 0 else -1.0
    data_max = np.max(finite_data) if len(finite_data) > 0 else 1.0

    data_range = data_max - data_min
    padding = YLIM_PADDING_FRACTION * (data_range if data_range > 0 else abs(data_max))
    ymin = data_min - padding
    ymax = data_max + padding

    if data_min * data_max > 0 and (min(abs(data_min), abs(data_max))
                                     < YLIM_ZERO_EXTENSION_RATIO * max(abs(data_min), abs(data_max))):
        if abs(data_min) < abs(data_max):
            ymin = 0
        else:
            ymax = 0
    ax.set_ylim(ymin, ymax)


def _add_config_footnote(fig, config_filename):
    fig.text(0.99, 0.01, f'Config: {config_filename}', ha='right', va='bottom', fontsize=6, color='gray',
             style='italic')


def plot_grouped_results_pdf(results, pdf_path, run_name, config_filename, ylim_time_fraction):
    """
    Grouped report: one page per VARIABLE_GROUPS entry.

    Parameters
    ----------
    results : dict
        Time series from integrate_model / create_derived_variables
    pdf_path : str
        Output PDF path
    run_name : str
        Shown in the page titles
    config_filename : str
        Shown as a footnote on each page
    ylim_time_fraction : float
        Fraction of the time series (from the start, excluding the first step) used for
        the y-axis limits of linear axes
    """
    t = results['t']
    n_ylim = int(ylim_time_fraction * len(t))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    with PdfPages(pdf_path) as pdf:
        for group_name, charts in VARIABLE_GROUPS.items():
            cols = 2 if len(charts) <= 4 else 3
            fig, axes = plt.subplots(2, cols, figsize=(12 if cols == 2 else 15, 8))
            fig.suptitle(f'{run_name} - COIN_equality Model Results - {group_name.title()} Variables',
                         fontsize=16, fontweight='bold', y=0.99)
            axes = axes.flatten()

            for ax, chart in zip(axes, charts):
                variables = chart['variables']
                if len(variables) == 1:
                    description, units = VARIABLE_LABELS[variables[0]]
                    ax.plot(t, results[variables[0]], linewidth=2, color=colors[0], alpha=0.8)
                    ax.set_ylabel(f'{description}\n({units})', fontsize=11)
                    ax.set_title(f'{variables[0]}: {description}', fontsize=12, fontweight='bold', pad=10)
                else:
                    for color, name in zip(colors, variables):
                        ax.plot(t, results[name], linewidth=2, color=color, alpha=0.8, label=VARIABLE_LABELS[name][0])
                    ax.set_ylabel(f"{chart['title']}\n({chart['units']})", fontsize=11)
                    ax.set_title(chart['title'], fontsize=12, fontweight='bold', pad=10)
                    ax.legend(fontsize=9, loc='best', framealpha=0.8)
                ax.set_xlabel('Time (yr)', fontsize=11)

                if any(name in LOG_SCALE_VARIABLES for name in variables):
                    ax.set_yscale('log')
                else:
                    max_abs = np.max(np.abs(np.concatenate([results[name] for name in variables])))
                    if max_abs > 1e4 or 0 < max_abs < 1e-2:
                        ax.yaxis.set_major_formatter(FuncFormatter(format_scientific_notation))
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.set_facecolor('#fafafa')

            for ax in axes[len(charts):]:
                ax.set_visible(False)

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            # Y-limits after the layout, which would otherwise override them
            for ax, chart in zip(axes, charts):
                if not any(name in LOG_SCALE_VARIABLES for name in chart['variables']):
                    set_zero_bound_ylim(ax, np.concatenate([results[name][1:n_ylim] for name in chart['variables']]))

            _add_config_footnote(fig, config_filename)
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)


def plot_results_report_pdf(results, pdf_path, run_name, config_filename):
    """
    Full report: one variable of REPORT_PANELS per panel, REPORT_PANELS_PER_PAGE per page
    (16:9 landscape).
    """
    t = results['t']
    n_ylim = int(YLIM_TIME_FRACTION * len(t))

    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(REPORT_PANELS), REPORT_PANELS_PER_PAGE):
            page_panels = REPORT_PANELS[page_start:page_start + REPORT_PANELS_PER_PAGE]
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig.suptitle(f'{run_name} - Results', fontsize=14, fontweight='bold')
            axes = axes.flatten()

            for ax, (name, ylabel, title) in zip(axes, page_panels):
                ax.plot(t, results[name], linewidth=2, color=plt.cm.tab10(0))
                ax.set_xlabel('Time (years)', fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_title(f'{name}: {title}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                if name in LOG_SCALE_VARIABLES:
                    ax.set_yscale('log')
            for ax in axes[len(page_panels):]:
                ax.axis('off')

            plt.tight_layout()
            # Y-limits after the layout, which would otherwise override them
            for ax, (name, _, _) in zip(axes, page_panels):
                if name not in LOG_SCALE_VARIABLES:
                    set_zero_bound_ylim(ax, results[name][1:n_ylim])

            fig.text(0.01, 0.005, f'Config: {config_filename}', fontsize=7, color='gray', ha='left', va='bottom')
            pdf.savefig(fig, orientation='landscape')
            plt.close(fig)
