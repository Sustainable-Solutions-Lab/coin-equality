"""
Production Figure S (Supplementary): OPC2030 vs DT2100 scatter plot with parameter coloring.

Uses the same data source as Fig. 3: master_results_contour.csv from the 2D sweep.
Grid: 41 tax_equity x 41 y_damage_distribution_exponent = 1681 runs.

Visualization:
- X-axis: Temperature Change 2100 (°C)
- Y-axis: Optimal Carbon Price 2030 ($/tCO₂)
- Color: Tax Equity Parameter (τ)
- Lines: Connect constant damage distribution exponent (β) values

Output: data/output/prod/figures/prod_figS_dt2100_vs_opc2030.pdf/png
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod')
MASTER_CSV = os.path.join(PROD_OUTPUT, 'master_results_contour.csv')
FIGURES_DIR = os.path.join(PROD_OUTPUT, 'figures')


def load_contour_data():
    """Load contour data from master results CSV.

    Only validated points are used. Returns rows and a dictionary keyed by grid_id.
    """
    rows = []
    with open(MASTER_CSV, 'r') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    n_total = len(rows)
    n_validated = sum(1 for r in rows if r['validated'] == 'True')

    source_counts = {}
    for r in rows:
        src = r['selected_source']
        if src:
            source_counts[src] = source_counts.get(src, 0) + 1

    print(f'Loaded {n_total} rows from {os.path.basename(MASTER_CSV)}')
    print(f'  Validated: {n_validated}')
    for src in sorted(source_counts):
        print(f'  {src}: {source_counts[src]}')

    results = {}
    for r in rows:
        if r['validated'] != 'True':
            continue
        gid = int(r['grid_id'])
        results[gid] = {
            'tax_equity': float(r['tax_equity']),
            'y_damage_distribution_exponent': float(r['y_damage_distribution_exponent']),
            'mac_2030': float(r['mac_selected']),
            'delta_t_2100': float(r['dt_selected']),
        }

    n_excluded = n_total - len(results)
    print(f'  Excluded (outliers + missing): {n_excluded}')
    print(f'  Using {len(results)} validated points')

    return rows, results


def extract_plot_data(results):
    """Extract x, y, color data from validated results.

    Returns:
        dt2100: array of temperature values (x-axis)
        opc2030: array of carbon price values (y-axis)
        tax_equity_vals: array of tax equity values (color)
        y_damage_vals: array of damage exponent values (for line grouping)
    """
    data_list = []
    for gid, row in results.items():
        data_list.append({
            'dt2100': row['delta_t_2100'],
            'opc2030': row['mac_2030'],
            'tax_equity': row['tax_equity'],
            'y_damage': row['y_damage_distribution_exponent'],
        })

    # Sort by y_damage then tax_equity for line plotting
    data_list = sorted(data_list, key=lambda x: (x['y_damage'], x['tax_equity']))

    dt2100 = np.array([d['dt2100'] for d in data_list])
    opc2030 = np.array([d['opc2030'] for d in data_list])
    tax_equity = np.array([d['tax_equity'] for d in data_list])
    y_damage = np.array([d['y_damage'] for d in data_list])

    return dt2100, opc2030, tax_equity, y_damage, data_list


def plot_relationship(dt2100, opc2030, tax_equity, y_damage, data_list):
    """Create scatter plot with colored points and connecting lines for constant beta.

    Points colored by tax_equity (τ).
    Lines connect points with same y_damage_distribution_exponent (β).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get unique beta values and color map for tax_equity
    unique_beta = sorted(set(y_damage))
    cmap = plt.cm.viridis

    # Normalize tax_equity for colormapping (0 to 1)
    te_min, te_max = tax_equity.min(), tax_equity.max()
    te_norm = (tax_equity - te_min) / (te_max - te_min) if te_max > te_min else np.zeros_like(tax_equity)
    colors = cmap(te_norm)

    # Plot lines for constant beta
    for beta_val in unique_beta:
        mask = np.abs(y_damage - beta_val) < 1e-10
        if np.sum(mask) > 1:
            # Sort by tax_equity for this beta value
            indices = np.where(mask)[0]
            sorted_indices = indices[np.argsort(tax_equity[indices])]
            ax.plot(dt2100[sorted_indices], opc2030[sorted_indices],
                   color='gray', linewidth=0.5, alpha=0.3, zorder=1)

    # Plot scatter points colored by tax_equity
    scatter = ax.scatter(dt2100, opc2030, c=te_norm, cmap='viridis',
                        s=30, alpha=0.7, edgecolors='none', zorder=2)

    # Colorbar for tax_equity
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Tax Equity Parameter (τ)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Labels and title
    ax.set_xlabel('Temperature Change 2100 (°C)', fontsize=12)
    ax.set_ylabel('Optimal Carbon Price 2030 ($/tCO₂)', fontsize=12)
    ax.set_title('Relationship between 2100 Temperature and 2030 Optimal Carbon Price', fontsize=13)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    return fig


def main():
    master_rows, results = load_contour_data()

    # Extract data for plotting
    dt2100, opc2030, tax_equity, y_damage, data_list = extract_plot_data(results)

    # Print summary statistics
    print(f'\nData summary:')
    print(f'  Grid points plotted: {len(dt2100)}')
    print(f'  ΔT2100 range: {dt2100.min():.4f} to {dt2100.max():.4f} °C')
    print(f'  OPC2030 range: {opc2030.min():.2f} to {opc2030.max():.2f} $/tCO₂')
    print(f'  Tax Equity (τ) range: {tax_equity.min():.4f} to {tax_equity.max():.4f}')
    print(f'  Damage Exponent (β) range: {y_damage.min():.4f} to {y_damage.max():.4f}')

    # Create the plot
    fig = plot_relationship(dt2100, opc2030, tax_equity, y_damage, data_list)

    # Save figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pdf_path = os.path.join(FIGURES_DIR, 'prod_figS_dt2100_vs_opc2030.pdf')
    png_path = os.path.join(FIGURES_DIR, 'prod_figS_dt2100_vs_opc2030.png')

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f'\nSaved {pdf_path}')
    print(f'Saved {png_path}')

    print(f'\nData source: {os.path.basename(MASTER_CSV)}')
    print(f'  Column mapping:')
    print(f'    ΔT2100 (x-axis): dt_selected')
    print(f'    OPC2030 (y-axis): mac_selected')
    print(f'    τ (color): tax_equity')
    print(f'    β (line grouping): y_damage_distribution_exponent')


if __name__ == '__main__':
    main()
