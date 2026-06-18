"""Update fig6_3 with latest available data from all sources.

Loads best available results (original, corrections, iterations) and regenerates fig6_3.
Call this after any new results become available.

Usage:
    python scripts/update_fig6_3_live.py          # Generate and save
    python scripts/update_fig6_3_live.py --interactive  # Interactive label editing
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_utils import place_labels, enable_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXACT_BASE = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'prod_discount_exact'
EXACT_GRID_CSV = PROJECT_ROOT / 'json' / 'prod_discount_exact_grid.csv'
ORIG_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'prod_discount'
ORIG_GRID_CSV = PROJECT_ROOT / 'json' / 'prod_discount_grid.csv'
OUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'prod' / 'figures'

os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [3.0, 4.0, 5.0, 6.0, 7.0]
OCP_YEAR, DT_YEAR, R_YEAR = 2030, 2100, 2030
LINE_WIDTH = 2.0
SCATTER_SIZE = 50
MARKER_SIZE = 7

TARGET_COLORS = {
    3.0: '#e41a1c',   # Red
    4.0: '#377eb8',   # Blue
    5.0: '#4daf4a',   # Green
    6.0: '#ff7f00',   # Orange
    7.0: '#984ea3',   # Purple
}

BN_DISCOUNT_RATES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
BN_SCC_2025 = np.array([571, 207, 102, 58, 37])

from fig_style import TAX_EQUITY_COLORS, TAX_EQUITY_LABELS, TAX_EQUITY_ORDER

# Label positions for interactive editing
LABEL_POSITIONS = {
    ('panel_b', 'flat_tax'): {'x': 1.9100, 'y': 2.5783, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'prog_tax'): {'x': 0.8055, 'y': 1.8528, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_b', 'hprog_tax'): {'x': 0.2865, 'y': 1.4114, 'fontsize': 9, 'rotation': 0, 'ha': 'left', 'va': 'center'},
}

LABEL_TEXTS = {
    'flat_tax': 'Flat tax',
    'prog_tax': 'Progressive tax',
    'hprog_tax': 'Highly progressive tax',
}

LABEL_COLORS = {
    'flat_tax': '#2C3E50',
    'prog_tax': '#d62728',
    'hprog_tax': '#1f77b4',
}

MARKERS = {

}
ARROWS = {

}


def find_best_result(gid):
    """Find best available result for a gid across all directories."""
    dirs = []

    # Collect all possible directories, checking priority order
    sources = [
        # Warmstart results (highest priority - most recent re-optimizations)
        (sorted(EXACT_BASE.glob(f'warmstart_gid_{gid}_*')), 'warmstart'),
        # Original
        (sorted(EXACT_BASE.glob(f'*_gid-{gid}_*')), 'original'),
    ]

    # Low-target tests (1-2% targets)
    for base_name in ['prod_discount_exact_flat_low_targets',
                      'prod_discount_exact_progressive_low_targets',
                      'prod_discount_exact_hiprog_low_targets']:
        base = PROJECT_ROOT / 'data' / 'output' / 'prod' / base_name
        if base.exists():
            sources.append((sorted(base.glob(f'*_gid-{gid}_*')), base_name))

    # Corrections and iterations
    for base_name in ['prod_discount_exact_r6', 'prod_discount_exact_r6_refine',
                      'prod_discount_exact_iter1', 'prod_discount_exact_iter2', 'prod_discount_exact_iter3']:
        base = PROJECT_ROOT / 'data' / 'output' / 'prod' / base_name
        if base.exists():
            sources.append((sorted(base.glob(f'*_gid-{gid}_*')), base_name))

    # Return latest successful result, checking sources in priority order
    for source_dirs, source_name in sources:
        for d in reversed(source_dirs):
            status_file = d / 'run_status.csv'
            if status_file.exists():
                with open(status_file, 'r') as f:
                    row = list(csv.DictReader(f))[0]
                    if row['success'] == 'True':
                        return d

    return None


def load_prod_discount_data():
    """Load original prod_discount sweep."""
    grid = pd.read_csv(ORIG_GRID_CSV)
    discount_dir = ORIG_DIR

    dir_index = {}
    for entry in sorted(discount_dir.iterdir()):
        status_path = entry / 'run_status.csv'
        if not status_path.exists():
            continue
        with open(status_path, 'r') as f:
            for row in csv.DictReader(f):
                if row['success'] == 'True':
                    dir_index[int(row['grid_id'])] = entry.name

    results = {}
    for _, grow in grid.iterrows():
        grid_id = int(grow['grid_id'])
        tax_equity = float(grow['tax_equity'])
        ydmg = float(grow['y_damage_distribution_exponent'])
        rho = float(grow['rho'])

        dir_name = dir_index.get(grid_id)
        if dir_name is None:
            continue

        run_dir = discount_dir / dir_name
        results_files = sorted(run_dir.glob('*_results.csv'))
        df = pd.read_csv(results_files[-1])

        t_col = [c for c in df.columns if c.startswith('t,')][0]
        years = df[t_col].values
        idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
        idx_dt = np.argmin(np.abs(years - DT_YEAR))
        idx_r = np.argmin(np.abs(years - R_YEAR))

        mac_col = [c for c in df.columns if c.startswith('marginal_abatement_cost')][0]
        delta_t_col = [c for c in df.columns if c.startswith('delta_T,')][0]

        entry = {
            'rho': rho,
            'ocp': df[mac_col].values[idx_ocp],
            'dt': df[delta_t_col].values[idx_dt],
            'r': df['r_consumption'].values[idx_r] * 100,
        }

        key = (tax_equity, ydmg)
        results.setdefault(key, []).append(entry)

    data = {}
    for key in sorted(results):
        entries = sorted(results[key], key=lambda e: e['rho'])
        data[key] = {
            'rho': np.array([e['rho'] for e in entries]),
            'ocp': np.array([e['ocp'] for e in entries]),
            'dt': np.array([e['dt'] for e in entries]),
            'r': np.array([e['r'] for e in entries]),
        }

    return data


def load_exact_data():
    """Load best available exact target-rate data from all sources."""
    data = {}

    # Load from main prod_discount_exact grid
    exact_grid = pd.read_csv(EXACT_GRID_CSV)

    for _, grow in exact_grid.iterrows():
        gid = int(grow['grid_id'])
        tax_equity = float(grow['tax_equity'])
        ydmg = float(grow['y_damage_distribution_exponent'])
        rho = float(grow['rho'])

        run_dir = find_best_result(gid)
        if run_dir is None:
            continue

        status_file = run_dir / 'run_status.csv'
        with open(status_file, 'r') as f:
            status_row = list(csv.DictReader(f))[0]

        if status_row['success'] != 'True':
            continue

        # Try to load from results.csv, fall back to run_status.csv for warmstart results
        results_files = sorted(run_dir.glob('*_results.csv'))

        if results_files:
            # Load from results.csv (full data)
            df = pd.read_csv(results_files[-1])

            t_col = [c for c in df.columns if c.startswith('t,')][0]
            years = df[t_col].values
            idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
            idx_dt = np.argmin(np.abs(years - DT_YEAR))
            idx_r = np.argmin(np.abs(years - R_YEAR))

            mac_col = [c for c in df.columns if c.startswith('marginal_abatement_cost')][0]
            delta_t_col = [c for c in df.columns if c.startswith('delta_T,')][0]

            r_actual = df['r_consumption'].values[idx_r] * 100
            ocp = df[mac_col].values[idx_ocp]
            dt = df[delta_t_col].values[idx_dt]
        else:
            # Use data from run_status.csv (warmstart results)
            try:
                ocp = float(status_row['mac_2030'])
                dt = float(status_row['delta_t_2100'])
                # r_actual not available from status, use target_r as estimate
                r_actual = rho * 100
            except (ValueError, KeyError):
                continue

        target_idx = (gid - 1) % 5
        target_r = TARGETS[target_idx]

        entry = {
            'gid': gid,
            'rho': rho,
            'ocp': ocp,
            'dt': dt,
            'r': r_actual,
            'r_target': target_r,
        }

        key = (tax_equity, ydmg)
        data.setdefault(key, []).append(entry)

    # Load from low-target grids (1-2% targets for flat, progressive, hiprog)
    low_target_configs = [
        ('prod_discount_exact_flat_low_targets', 0.0),           # tax_equity for flat
        ('prod_discount_exact_progressive_low_targets', 0.5),    # tax_equity for progressive
        ('prod_discount_exact_hiprog_low_targets', 0.998),       # tax_equity for hiprog
    ]

    for config_name, tax_equity in low_target_configs:
        grid_file = PROJECT_ROOT / 'json' / f'{config_name}_grid.csv'
        if grid_file.exists():
            grid = pd.read_csv(grid_file)
            for _, grow in grid.iterrows():
                gid = int(grow['grid_id'])
                ydmg = float(grow['y_damage_distribution_exponent'])
                rho = float(grow['rho'])

                # Find best result in this config
                base = PROJECT_ROOT / 'data' / 'output' / 'prod' / config_name
                run_dirs = sorted(base.glob(f'{config_name}_gid-{gid}_*'))

                if not run_dirs:
                    continue

                run_dir = None
                for d in reversed(run_dirs):
                    status_file = d / 'run_status.csv'
                    if status_file.exists():
                        with open(status_file, 'r') as f:
                            row = list(csv.DictReader(f))[0]
                            if row['success'] == 'True':
                                run_dir = d
                                break

                if run_dir is None:
                    continue

                results_files = sorted(run_dir.glob('*_results.csv'))
                if not results_files:
                    continue

                df = pd.read_csv(results_files[-1])

                t_col = [c for c in df.columns if c.startswith('t,')][0]
                years = df[t_col].values
                idx_ocp = np.argmin(np.abs(years - OCP_YEAR))
                idx_dt = np.argmin(np.abs(years - DT_YEAR))
                idx_r = np.argmin(np.abs(years - R_YEAR))

                mac_col = [c for c in df.columns if c.startswith('marginal_abatement_cost')][0]
                delta_t_col = [c for c in df.columns if c.startswith('delta_T,')][0]

                r_actual = df['r_consumption'].values[idx_r] * 100
                # Target mapping for low-targets: gid 1-4 map to targets [1.0, 2.0, 1.0, 2.0]
                target_map = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}
                target_r = target_map.get(gid, 2.0)

                entry = {
                    'gid': gid,
                    'rho': rho,
                    'ocp': df[mac_col].values[idx_ocp],
                    'dt': df[delta_t_col].values[idx_dt],
                    'r': r_actual,
                    'r_target': target_r,
                }

                key = (tax_equity, ydmg)
                data.setdefault(key, []).append(entry)

    for key in data:
        data[key] = sorted(data[key], key=lambda e: e['r_target'])

    return data


def load_equilibrium_points():
    """Load the 7 completed low-target equilibrium test points."""
    points = []

    configs = [
        ('prod_discount_exact_flat_low_targets', 0.0, [51, 52, 53, 54]),
        ('prod_discount_exact_progressive_low_targets', 0.5, [1]),
        ('prod_discount_exact_hiprog_low_targets', 0.998, [1, 2]),
    ]

    for config_name, tax_equity, gids in configs:
        for gid in gids:
            base = PROJECT_ROOT / 'data' / 'output' / 'prod' / config_name
            dirs = sorted(base.glob(f'*_gid-{gid}_*'))

            if not dirs:
                continue

            latest_dir = dirs[-1]
            results_file = latest_dir / f'{config_name}_gid-{gid}_results.csv'

            if not results_file.exists():
                continue

            try:
                df = pd.read_csv(results_file)

                # Get year 2030 data
                year_col = df.columns[0]
                year_2030_idx = df[year_col].astype(float).sub(2030).abs().idxmin()
                year_2030 = df.iloc[year_2030_idx]

                # Extract r_consumption
                r_cons_col = [c for c in df.columns if 'r_consumption' in c.lower()]
                if not r_cons_col:
                    continue

                r_consumption = float(year_2030[r_cons_col[0]]) * 100

                # Extract temperature at 2100
                temp_col = [c for c in df.columns if 'delta_t' in c.lower()]
                year_2100_idx = df[year_col].astype(float).sub(2100).abs().idxmin()
                year_2100 = df.iloc[year_2100_idx]
                temp_2100 = float(year_2100[temp_col[0]]) if temp_col else None

                # Extract MAC (proxy for OCP)
                mac_col = [c for c in df.columns if 'marginal_abatement_cost' in c.lower()]
                mac = float(year_2030[mac_col[0]]) if mac_col else None

                points.append({
                    'tax_equity': tax_equity,
                    'gid': gid,
                    'r_consumption': r_consumption,
                    'mac': mac,
                    'temp_2100': temp_2100,
                })
            except Exception as e:
                print(f"Warning: Could not load {config_name} gid-{gid}: {e}")
                continue

    return points


def plot_figure(interactive_mode=False):
    """Create updated fig6_3."""
    print('Loading data...')
    orig_data = load_prod_discount_data()
    exact_data = load_exact_data()
    equilibrium_points = load_equilibrium_points()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel A: OCP vs r_consumption
    # First: fill areas between β=0 and β=0.5 for each tax_equity
    for tax_equity in TAX_EQUITY_ORDER:
        key_b0 = (tax_equity, 0.0)
        key_b05 = (tax_equity, 0.5)
        if key_b0 in orig_data and key_b05 in orig_data:
            d_b0 = orig_data[key_b0]
            d_b05 = orig_data[key_b05]
            # Use the full range for x-axis to ensure fill covers all points
            r_min = min(d_b0['r'].min(), d_b05['r'].min())
            r_max = max(d_b0['r'].max(), d_b05['r'].max())
            r_range = np.linspace(r_min, r_max, 1000)
            ocp_b0_interp = np.interp(r_range, d_b0['r'], d_b0['ocp'])
            ocp_b05_interp = np.interp(r_range, d_b05['r'], d_b05['ocp'])
            ax_a.fill_between(r_range, ocp_b0_interp, ocp_b05_interp,
                            color=TAX_EQUITY_COLORS[tax_equity], alpha=0.15, zorder=1)

    # Then: plot lines
    for key in sorted(orig_data.keys()):
        tax_equity = key[0]
        ydmg = key[1]
        d = orig_data[key]
        linewidth = 2.5 if ydmg == 0.0 else 1.2  # Thick for β=0, thin for β=0.5
        ax_a.plot(d['r'], d['ocp'], linestyle='-', color=TAX_EQUITY_COLORS[tax_equity],
                  linewidth=linewidth, zorder=2)

    ax_a.plot(BN_DISCOUNT_RATES, BN_SCC_2025, 'o-',
              color='#228B22', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, zorder=4)


    ax_a.set_yscale('log')
    ax_a.set_ylim(20, 1000)
    ax_a.set_xlim(0, 8)
    ax_a.set_yticks([20, 30, 50, 100, 200, 300, 500, 1000])
    ax_a.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax_a.yaxis.set_tick_params(which='minor', size=0)

    ax_a.set_xlabel(f'Discount rate in {R_YEAR} (%/year)', fontsize=11)
    ax_a.set_ylabel(f'Optimal carbon price in {OCP_YEAR} ($/tCO₂)', fontsize=11)
    ax_a.set_title('(A)  Optimal Carbon Price', fontsize=12, fontweight='bold')
    ax_a.grid(axis='y', alpha=0.3, which='major')
    ax_a.set_axisbelow(True)

    # Panel B: ΔT vs r_consumption
    # First: fill areas between β=0 and β=0.5 for each tax_equity
    for tax_equity in TAX_EQUITY_ORDER:
        key_b0 = (tax_equity, 0.0)
        key_b05 = (tax_equity, 0.5)
        if key_b0 in orig_data and key_b05 in orig_data:
            d_b0 = orig_data[key_b0]
            d_b05 = orig_data[key_b05]
            # Use the full range for x-axis to ensure fill covers all points
            r_min = min(d_b0['r'].min(), d_b05['r'].min())
            r_max = max(d_b0['r'].max(), d_b05['r'].max())
            r_range = np.linspace(r_min, r_max, 1000)
            dt_b0_interp = np.interp(r_range, d_b0['r'], d_b0['dt'])
            dt_b05_interp = np.interp(r_range, d_b05['r'], d_b05['dt'])
            ax_b.fill_between(r_range, dt_b0_interp, dt_b05_interp,
                            color=TAX_EQUITY_COLORS[tax_equity], alpha=0.15, zorder=1)

    # Then: plot lines
    for key in sorted(orig_data.keys()):
        tax_equity = key[0]
        ydmg = key[1]
        d = orig_data[key]
        linewidth = 2.5 if ydmg == 0.0 else 1.2  # Thick for β=0, thin for β=0.5
        ax_b.plot(d['r'], d['dt'], linestyle='-', color=TAX_EQUITY_COLORS[tax_equity],
                  linewidth=linewidth, zorder=2)


    # Overlay exact target points with convergence tolerance checking
    TOLERANCE_PP = 0.10  # Tolerance in percentage points (0.10 = ±0.10%)

    for key in sorted(exact_data.keys()):
        tax_equity = key[0]
        ydmg = key[1]
        entries = exact_data[key]

        for entry in entries:
            error_pp = abs(entry['r'] - entry['r_target'])  # Error in percentage points

            if error_pp <= TOLERANCE_PP:
                # Valid convergence: plot at target r_consumption (small markers)
                ax_a.plot(entry['r_target'], entry['ocp'], 'o',
                         color=TAX_EQUITY_COLORS[tax_equity], markersize=3.5, zorder=3)
                ax_b.plot(entry['r_target'], entry['dt'], 'o',
                         color=TAX_EQUITY_COLORS[tax_equity], markersize=3.5, zorder=3)
            else:
                # Failed convergence: plot at ACTUAL r_consumption with X marker
                ax_a.plot(entry['r'], entry['ocp'], 'x', markersize=8, linewidth=2,
                         color=TAX_EQUITY_COLORS[tax_equity], alpha=0.4, zorder=2)
                ax_b.plot(entry['r'], entry['dt'], 'x', markersize=8, linewidth=2,
                         color=TAX_EQUITY_COLORS[tax_equity], alpha=0.4, zorder=2)

    # Overlay equilibrium points (large stars at actual r_consumption)
    for pt in equilibrium_points:
        tax_equity = pt['tax_equity']
        color = TAX_EQUITY_COLORS[tax_equity]
        ax_a.plot(pt['r_consumption'], pt['mac'], '*', markersize=20, color=color,
                 markeredgecolor='black', markeredgewidth=0.8, zorder=5)
        ax_b.plot(pt['r_consumption'], pt['temp_2100'], '*', markersize=20, color=color,
                 markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    ax_b.set_xlim(0, 8)
    ax_b.set_ylim(1.2, 3.8)
    ax_b.set_xlabel(f'Discount rate in {R_YEAR} (%/year)', fontsize=11)
    ax_b.set_ylabel(f'Temperature change in {DT_YEAR} (°C)', fontsize=11)
    ax_b.set_title('(B)  Temperature Change', fontsize=12, fontweight='bold')
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.set_axisbelow(True)

    # Add tax equity case labels on right panel (Panel B) using label_utils
    panel_b_labels = place_labels(ax_b, 'panel_b', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)

    # Convert to full panel_labels format for enable_interactive
    panel_labels = {('panel_b', k): v for k, v in panel_b_labels.items()}

    fig.tight_layout(w_pad=3)

    if interactive_mode:
        # Enable interactive mode for label editing
        enable_interactive(fig, {'panel_b': ax_b}, panel_labels,
                          os.path.abspath(__file__), {}, {})
        plt.show()
        return

    # Save
    out_base = os.path.join(OUT_DIR, 'prod_fig6_3_discount_exact')
    for ext in ('.png', '.pdf', '.svg'):
        fig.savefig(out_base + ext, bbox_inches='tight',
                    dpi=300 if ext == '.png' else None)
        print(f'Updated: {out_base}{ext}')

    plt.close(fig)


if __name__ == '__main__':
    interactive_mode = '--interactive' in sys.argv
    plot_figure(interactive_mode=interactive_mode)
