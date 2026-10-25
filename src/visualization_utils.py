"""
Visualization utilities for COIN_equality optimization results.

Provides the plotting functions for the run report PDF (time series plots of results.csv).

All plotting functions use a unified interface that works for both single
and multiple cases, ensuring consistent formatting across all reports.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import numpy as np


def clean_column_names(df):
    """
    Extract simple variable names from CSV headers with format 'var_name, Description, (units)'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with full column names from results.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with simplified column names (just the variable name)
    """
    new_columns = []
    for col in df.columns:
        # Extract variable name (everything before first comma)
        var_name = col.split(',')[0].strip()
        new_columns.append(var_name)
    df.columns = new_columns
    return df


def plot_timeseries(ax, case_data, variable, ylabel, title, show_legend=False,
                    t_min=None, t_max=None, t_offset=0):
    """
    Create time series plot on given axes (unified function for single/multi-run).

    Works for both single-run and multi-run cases with consistent formatting.
    For single case: plots one line without legend.
    For multiple cases: plots multiple lines (legend controlled by show_legend).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    case_data : dict
        Dictionary mapping case names to DataFrames containing results
        Single case: {'Case': df}
        Multiple cases: {'Case1': df1, 'Case2': df2, ...}
    variable : str
        Column name in DataFrame to plot (e.g., 'T_atm', 'y', 'K')
    ylabel : str
        Label for y-axis
    title : str
        Plot title
    show_legend : bool, optional
        If True, show legend on this plot (default: False)
    t_min : float, optional
        Minimum time value to include (before offset). If None, use all data.
    t_max : float, optional
        Maximum time value to include (before offset). If None, use all data.
    t_offset : float, optional
        Offset to add to time values for display (default: 0)

    Notes
    -----
    - Uses tab10 colormap for consistent colors across plots
    - Legend display controlled by show_legend parameter
    - Gracefully handles missing variables (skips if not in DataFrame)
    """
    colors = plt.cm.tab10(np.arange(10))

    for idx, (case_name, df) in enumerate(case_data.items()):
        if variable in df.columns:
            # Apply time filtering
            df_filtered = df
            if t_min is not None:
                df_filtered = df_filtered[df_filtered['t'] >= t_min]
            if t_max is not None:
                df_filtered = df_filtered[df_filtered['t'] <= t_max]

            ax.plot(
                df_filtered['t'] + t_offset,
                df_filtered[variable],
                label=case_name,
                linewidth=2,
                color=colors[idx % 10]
            )

    ax.set_xlabel('Time (years)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'{variable}: {title}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set custom x-axis ticks when time offset is applied (e.g., for 2025-2100 plots)
    if t_offset > 0 and t_min is not None and t_max is not None:
        # Set x-axis limits
        ax.set_xlim(t_min + t_offset, t_max + t_offset)
        # Major ticks at 25-year intervals (2025, 2050, 2075, 2100)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        # Minor ticks every 5 years
        ax.xaxis.set_minor_locator(MultipleLocator(5))

    # Apply logarithmic scale for specific variables
    log_scale_vars = {'y_net', 'K', 'Consumption', 'Savings', 'Y_gross', 'Y_net', 'A', 'redistribution', 'Redistribution_amount', 'AbateCost', 'marginal_abatement_cost', 'Climate_damage'}
    if variable in log_scale_vars:
        ax.set_yscale('log')

    if show_legend and len(case_data) > 1:
        ax.legend(loc='best', fontsize=8)


def create_legend_panel(ax, case_data):
    """
    Create a dedicated legend panel showing all case names and colors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to use for legend
    case_data : dict
        Dictionary mapping case names to DataFrames
    """
    colors = plt.cm.tab10(np.arange(10))

    # Turn off axis decorations
    ax.axis('off')

    # Create dummy lines for legend
    handles = []
    labels = []
    for idx, case_name in enumerate(case_data.keys()):
        line = plt.Line2D([0], [0], color=colors[idx % 10], linewidth=3)
        handles.append(line)
        labels.append(case_name)

    # Create legend in center of panel
    ax.legend(handles, labels, loc='center', fontsize=12, frameon=True,
              title='Cases', title_fontsize=14)
    ax.set_title('Legend', fontsize=11, fontweight='bold', pad=10)


def create_results_report_pdf(case_data, output_pdf, config_filename=None):
    """
    Create PDF report with all time series plots (6 per page, 16:9 landscape).

    Generates comprehensive visualization of all key variables from results.csv.
    Uses unified plotting function for consistent formatting across single and
    multi-run cases.

    Parameters
    ----------
    case_data : dict
        Dictionary mapping case names to results DataFrames
        {case_name: results_df, ...}
    output_pdf : Path or str
        Output PDF file path
    config_filename : str, optional
        Name of configuration file to display as footnote on each page

    Notes
    -----
    Layout:
    - 16:9 aspect ratio optimized for screen viewing
    - 6 plots per page in 2×3 grid
    - ~4-5 pages total for all 25 variables
    - Handles missing variables gracefully (shows "not available" message)
    """
    # Clean column names from 'var, Description, (units)' format to just 'var'
    case_data = {name: clean_column_names(df.copy()) for name, df in case_data.items()}

    # Define all variables to plot (in display order, grouped by category)
    variable_specs = [
        # Economic Variables
        ('y_net', 'Effective income ($/person/yr)', 'Effective Per-Capita Income'),
        ('K', 'Capital stock ($)', 'Capital Stock Over Time'),
        ('Consumption', 'Total consumption ($/yr)', 'Total Consumption'),
        ('Savings', 'Gross investment ($/yr)', 'Gross Investment'),
        ('s', 'Savings rate', 'Savings Rate Over Time'),
        ('Y_gross', 'Gross production ($/yr)', 'Gross Production'),
        ('Y_net', 'Net output ($/yr)', 'Net Output After Damages & Abatement'),

        # Climate Variables
        ('delta_T', 'Temperature change (°C)', 'Temperature Change from Pre-Industrial'),
        ('E', 'CO₂ emissions (tCO2/yr)', 'CO₂ Emissions Rate'),
        ('Ecum', 'Cumulative emissions (tCO2)', 'Cumulative CO₂ Emissions'),

        # Abatement and Damage
        ('f', 'log₁₀(MAC) (log₁₀($/tCO₂))', 'Control: log₁₀(Marginal Abatement Cost)'),
        ('mu', 'Abatement fraction', 'Emissions Abatement Fraction'),
        ('lambda_abate', 'Abatement cost fraction', 'Abatement Cost (% of Output)'),
        ('AbateCost', 'Abatement cost ($/yr)', 'Total Abatement Cost'),
        ('Omega', 'Climate damage fraction', 'Climate Damage (% of Output)'),
        ('Climate_damage', 'Climate damage ($/yr)', 'Total Climate Damage'),

        # Inequality and Utility
        ('gini', 'Background Gini index', 'Background Gini Index'),
        ('U', 'Mean utility', 'Mean Utility Per Capita'),
        ('redistribution', 'Per-capita redistribution ($/person/yr)', 'Per-Capita Redistribution'),
        ('Redistribution_amount', 'Total redistribution ($/yr)', 'Total Redistribution'),
        # Abatement Economics
        ('marginal_abatement_cost', 'Marginal abatement cost ($/tCO2)', 'Marginal Abatement Cost'),
        ('r_consumption', 'Effective discount rate', 'Effective Discount Rate on Consumption'),

        # Exogenous Drivers
        ('A', 'Total factor productivity ($)', 'Total Factor Productivity'),
        ('L', 'Population (persons)', 'Population'),
        ('sigma', 'Carbon intensity (tCO2/$)', 'Carbon Intensity of GDP'),
        ('theta1', 'Marginal cost at mu=1 ($/tCO2)', 'Abatement Cost Parameter (theta1)'),
    ]

    # Determine plots per page and starting index
    is_multi_case = len(case_data) > 1
    plots_per_page = 5 if is_multi_case else 6

    with PdfPages(output_pdf) as pdf:
        # Process plots in groups
        for page_start in range(0, len(variable_specs), plots_per_page):
            page_vars = variable_specs[page_start:page_start + plots_per_page]

            # Create 2×3 subplot grid in 16:9 landscape orientation
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig.suptitle('Results Comparison', fontsize=14, fontweight='bold')
            axes = axes.flatten()

            # For multi-case, use first position for legend
            if is_multi_case:
                create_legend_panel(axes[0], case_data)
                axes_offset = 1
            else:
                axes_offset = 0

            for idx, (var_name, ylabel, title) in enumerate(page_vars):
                ax_idx = idx + axes_offset
                # Check if variable exists in at least one dataset
                if any(var_name in df.columns for df in case_data.values()):
                    plot_timeseries(axes[ax_idx], case_data, var_name, ylabel, title)
                else:
                    # Variable not found - show message
                    axes[ax_idx].text(0.5, 0.5, f'{var_name}\nnot available',
                                     ha='center', va='center',
                                     transform=axes[ax_idx].transAxes,
                                     fontsize=11, color='gray')
                    axes[ax_idx].set_title(title, fontsize=11)

            # Hide any unused subplots on last page
            for idx in range(len(page_vars) + axes_offset, 6):
                axes[idx].axis('off')

            plt.tight_layout()

            # Apply zero-bound expansion AFTER layout (to prevent being overridden)
            log_scale_vars = {'y_net', 'K', 'Consumption', 'Savings', 'Y_gross', 'Y_net', 'A', 'redistribution', 'Redistribution_amount', 'AbateCost', 'marginal_abatement_cost', 'Climate_damage'}
            for idx, (var_name, ylabel, title) in enumerate(page_vars):
                ax_idx = idx + axes_offset
                ax = axes[ax_idx]

                # Skip if variable not available or uses log scale
                if not any(var_name in df.columns for df in case_data.values()):
                    continue
                if var_name in log_scale_vars:
                    continue

                # Check actual data range using first 90% of time points for y-axis scaling
                all_data = []
                for df in case_data.values():
                    if var_name in df.columns:
                        # Use first 90% of time points to avoid late-time outliers (skip first time step)
                        n_points = len(df)
                        n_points_for_ylim = int(0.9 * n_points)
                        all_data.extend(df[var_name].values[1:n_points_for_ylim])

                if not all_data:
                    continue

                data_min = np.min(all_data)
                data_max = np.max(all_data)

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

            # Add config filename as footnote if provided
            if config_filename:
                fig.text(0.01, 0.005, f'Config: {config_filename}',
                        fontsize=7, color='gray', ha='left', va='bottom')

            pdf.savefig(fig, orientation='landscape')
            plt.close(fig)

    print(f"Results report saved to: {output_pdf}")
