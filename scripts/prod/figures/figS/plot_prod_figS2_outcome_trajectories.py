"""
Production Figure S2: Outcome-space plot with selected β trajectories.

Uses the same data source as Fig. 3: master_results_contour.csv from the 2D sweep.

Visualization:
- X-axis: Temperature Change 2100 (°C)
- Y-axis: Optimal Carbon Price 2030 ($/tCO₂)
- Color: Tax Equity Parameter (τ)
- Lines: Trajectories for selected β values (0, 0.25, 0.5, 0.75, 1.0)

Each line represents how outcomes vary with τ for a fixed β value.
Shows the family of outcome relationships without visual clutter.

Output: data/output/prod/figures/prod_figS2_outcome_trajectories.pdf/png
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'output', 'prod')
MASTER_CSV = os.path.join(PROD_OUTPUT, 'master_results_contour.csv')
FIGURES_DIR = os.path.join(PROD_OUTPUT, 'figures')


def load_contour_data():
    """Load contour data from master results CSV.

    Only validated points are used.
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

    return results


def extract_trajectories(results, selected_betas):
    """Extract trajectory data for selected β values.

    For each β value, collect all (τ, ΔT2100, OPC2030) tuples and sort by τ.

    Returns dict: {beta_val: {'tau': array, 'dt2100': array, 'opc2030': array}}
    """
    trajectories = {beta: {'tau': [], 'dt2100': [], 'opc2030': []} for beta in selected_betas}

    for gid, row in results.items():
        beta = row['y_damage_distribution_exponent']
        # Find closest selected beta (within tolerance)
        closest_beta = min(selected_betas, key=lambda b: abs(b - beta))
        if abs(beta - closest_beta) < 1e-6:  # Exact match or very close
            trajectories[closest_beta]['tau'].append(row['tax_equity'])
            trajectories[closest_beta]['dt2100'].append(row['delta_t_2100'])
            trajectories[closest_beta]['opc2030'].append(row['mac_2030'])

    # Convert to arrays and sort by tau
    for beta in selected_betas:
        if trajectories[beta]['tau']:
            tau_arr = np.array(trajectories[beta]['tau'])
            dt_arr = np.array(trajectories[beta]['dt2100'])
            opc_arr = np.array(trajectories[beta]['opc2030'])

            # Sort by tau
            sort_idx = np.argsort(tau_arr)
            trajectories[beta]['tau'] = tau_arr[sort_idx]
            trajectories[beta]['dt2100'] = dt_arr[sort_idx]
            trajectories[beta]['opc2030'] = opc_arr[sort_idx]

            print(f'β = {beta:.2f}: {len(tau_arr)} points (τ: {tau_arr.min():.4f}–{tau_arr.max():.4f})')

    return trajectories


def plot_outcome_trajectories(trajectories, selected_betas):
    """Create outcome-space plot with colored trajectories for selected β values.

    X-axis: ΔT2100
    Y-axis: OPC2030
    Color: τ (normalized to [0, 1])
    Lines: One per β value
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.viridis
    line_styles = ['-', '--', '-.', ':', '-']  # Different styles for each β
    line_widths = [2.0, 2.0, 2.0, 2.0, 2.0]

    # Global min/max for tau normalization (across all trajectories)
    all_taus = []
    for beta in selected_betas:
        all_taus.extend(trajectories[beta]['tau'])
    tau_min, tau_max = min(all_taus), max(all_taus)
    tau_range = tau_max - tau_min if tau_max > tau_min else 1.0

    # Plot each trajectory
    for idx, beta in enumerate(selected_betas):
        traj = trajectories[beta]
        if len(traj['tau']) < 2:
            print(f'Warning: β = {beta:.2f} has < 2 points, skipping')
            continue

        tau_norm = (traj['tau'] - tau_min) / tau_range
        colors = cmap(tau_norm)

        # Plot line segments between consecutive points with color-coded segments
        for i in range(len(traj['tau']) - 1):
            ax.plot(traj['dt2100'][i:i+2], traj['opc2030'][i:i+2],
                   color=colors[i], linewidth=line_widths[idx],
                   linestyle=line_styles[idx], alpha=0.9, zorder=2)

        # Add marker at start and end for clarity
        ax.plot(traj['dt2100'][0], traj['opc2030'][0], 'o', color='black',
               markersize=4, zorder=3, alpha=0.6)
        ax.plot(traj['dt2100'][-1], traj['opc2030'][-1], 's', color='black',
               markersize=4, zorder=3, alpha=0.6)

    # Create colorbar for tau
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=tau_min, vmax=tau_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Tax Equity Parameter (τ)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Labels and title
    ax.set_xlabel('Temperature Change 2100 (°C)', fontsize=12)
    ax.set_ylabel('Optimal Carbon Price 2030 ($/tCO₂)', fontsize=12)
    ax.set_title('Outcome-Space Trajectories: OPC2030 vs ΔT2100 for Selected β Values',
                fontsize=13, pad=15)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=10)

    # Create custom legend for β values
    from matplotlib.lines import Line2D
    legend_elements = []
    for idx, beta in enumerate(selected_betas):
        legend_elements.append(Line2D([0], [0], color='gray', linewidth=2,
                                     linestyle=line_styles[idx],
                                     label=f'β = {beta:.2f}'))
    ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.95)

    plt.tight_layout()

    return fig


def main():
    selected_betas = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = load_contour_data()

    print(f'\nExtracting trajectories for β = {selected_betas}')
    trajectories = extract_trajectories(results, selected_betas)

    print(f'\nCreating outcome-space plot...')
    fig = plot_outcome_trajectories(trajectories, selected_betas)

    # Save figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pdf_path = os.path.join(FIGURES_DIR, 'prod_figS2_outcome_trajectories.pdf')
    png_path = os.path.join(FIGURES_DIR, 'prod_figS2_outcome_trajectories.png')

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f'\nSaved {pdf_path}')
    print(f'Saved {png_path}')

    print(f'\nData source: {os.path.basename(MASTER_CSV)}')
    print(f'Column mapping:')
    print(f'  X-axis (ΔT2100): dt_selected')
    print(f'  Y-axis (OPC2030): mac_selected')
    print(f'  Color (τ): tax_equity')
    print(f'  Lines (β): y_damage_distribution_exponent (selected values)')

    # Print outcome ranges
    all_dt = []
    all_opc = []
    for traj in trajectories.values():
        all_dt.extend(traj['dt2100'])
        all_opc.extend(traj['opc2030'])
    if all_dt:
        print(f'\nOutcome ranges in plot:')
        print(f'  ΔT2100: {min(all_dt):.4f} to {max(all_dt):.4f} °C')
        print(f'  OPC2030: {min(all_opc):.2f} to {max(all_opc):.2f} $/tCO₂')


if __name__ == '__main__':
    main()
