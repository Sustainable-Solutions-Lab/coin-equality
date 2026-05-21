"""
Diagnostic / Classification / Resubmission gate for sensitivity sweeps.

Classifies every point in a config sweep, detects outliers and wrong-basin
convergence, generates resubmission plans, validates warm-start results.

Usage:
    python scripts/neighbor_utils.py diagnose config_060 rho
    python scripts/neighbor_utils.py diagnose config_060 rho --tax 0.5 --ydmg 0.5
    python scripts/neighbor_utils.py plan config_060 rho
    python scripts/neighbor_utils.py validate config_060_warmstart rho
    python scripts/neighbor_utils.py neighbor config_060 rho 0.004 --tax 0.5 --ydmg 0.5
"""

import argparse
import csv
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAX_EQUITY_VALUES = [0.0, 0.5, 0.998]
YDMG_VALUES = [0.0, 0.5, 1.0]

# Years at which to extract time-series metrics
DIAGNOSTIC_YEARS = [2030, 2050, 2070, 2080, 2100, 2120, 2150]
MU_YEARS = [2030, 2050, 2070, 2100]
DT_YEARS = [2100, 2150]

# Path-flag thresholds
MAC_COLLAPSE_RATIO = 0.3      # flag if mac_min_2070_2100 < 0.3 * mac_max_2030_2050
MU_COLLAPSE_THRESHOLD = 0.02  # flag if mu drops below this in 2070-2100
MU_ACTIVE_THRESHOLD = 0.1     # only flag mu collapse if mu_2030 was above this
POST2100_SPIKE_RATIO = 2.0    # flag if mac_2120 > 2.0 * max(mac_2050, mac_2100)

# Smoothness thresholds (fraction of scenario range)
SMOOTHNESS_NOISY_THRESHOLD = 0.03   # >3% = ACCEPT_NOISY
SMOOTHNESS_OUTLIER_THRESHOLD = 0.10  # >10% = REVIEW_OUTLIER or REVIEW_REGIME_CHANGE

# Classification labels
ACCEPT = 'ACCEPT'
ACCEPT_NOISY = 'ACCEPT_NOISY'
REVIEW_REGIME_CHANGE = 'REVIEW_REGIME_CHANGE'
REVIEW_OUTLIER = 'REVIEW_OUTLIER'
REJECT_WRONG_BASIN = 'REJECT_WRONG_BASIN'
REJECT_EMPTY = 'REJECT_EMPTY'
AMBIGUOUS_MULTI_BASIN = 'AMBIGUOUS_MULTI_BASIN'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def find_data_dirs(config_name):
    """Find base + warmstart directories for a config.

    E.g. config_060 -> [data/output/config_060, data/output/config_060_warmstart]
    """
    output_dir = PROJECT_ROOT / 'data' / 'output'
    dirs = [output_dir / config_name]
    warmstart_dir = output_dir / f'{config_name}_warmstart'
    if warmstart_dir.exists():
        dirs.append(warmstart_dir)
    return [d for d in dirs if d.exists()]


def find_col(columns, prefix):
    """Find column name by prefix (handles multi-part headers like 't, Time, (yr)')."""
    matches = [c for c in columns if c.startswith(prefix)]
    return matches[0]


def extract_diagnostics_from_timepath(results_csv_path):
    """Extract MAC, mu, dT at key years + compute path flags from full time series.

    Returns dict with all diagnostic fields, or None if file can't be read.
    """
    df = pd.read_csv(results_csv_path)
    columns = df.columns.tolist()

    t_col = find_col(columns, 't,')
    mu_col = find_col(columns, 'mu,')
    dt_col = find_col(columns, 'delta_T,')

    times = df[t_col].values
    mac = df['marginal_abatement_cost'].values
    mu = df[mu_col].values
    delta_t = df[dt_col].values

    def val_at_year(arr, year):
        idx = np.argmin(np.abs(times - year))
        return arr[idx]

    result = {}

    # Extract at specific years
    for yr in DIAGNOSTIC_YEARS:
        result[f'mac_{yr}'] = val_at_year(mac, yr)
    for yr in MU_YEARS:
        result[f'mu_{yr}'] = val_at_year(mu, yr)
    for yr in DT_YEARS:
        result[f'dt_{yr}'] = val_at_year(delta_t, yr)

    # Window aggregates for path flags
    mask_2030_2050 = (times >= 2030) & (times <= 2050)
    mask_2070_2100 = (times >= 2070) & (times <= 2100)

    mac_max_2030_2050 = np.max(mac[mask_2030_2050])
    mac_min_2070_2100 = np.min(mac[mask_2070_2100])
    mu_min_2070_2100 = np.min(mu[mask_2070_2100])

    result['mac_max_2030_2050'] = mac_max_2030_2050
    result['mac_min_2070_2100'] = mac_min_2070_2100

    # Path flags
    result['flag_mac_collapse'] = mac_min_2070_2100 < MAC_COLLAPSE_RATIO * mac_max_2030_2050
    result['flag_mu_collapse'] = (
        mu_min_2070_2100 < MU_COLLAPSE_THRESHOLD
        and result['mu_2030'] > MU_ACTIVE_THRESHOLD
    )
    result['flag_post2100_spike'] = (
        result['mac_2120'] > POST2100_SPIKE_RATIO * max(result['mac_2050'], result['mac_2100'])
    )

    return result


def load_all_runs(data_dirs, config_prefix, param_name):
    """Load all runs from multiple directories, deduplicating (latest timestamp wins).

    Returns list of dicts, one per unique (tax_equity, ydmg, param_value) combination.
    """
    all_runs = {}

    for data_dir in data_dirs:
        pattern = str(data_dir / f'{config_prefix}_*')
        for run_dir in sorted(glob(pattern)):
            status_file = os.path.join(run_dir, 'run_status.csv')
            if not os.path.exists(status_file):
                # Directory exists but no status = timed out / empty
                dirname = os.path.basename(run_dir)
                param_token = f'_{param_name}-'
                if param_token not in dirname:
                    continue
                param_str = dirname.split(param_token)[1].split('_')[0]
                tax_str = dirname.split('_tax-')[1].split('_')[0]
                ydmg_str = dirname.split('_ydmg-')[1].split('_')[0]

                key = (round(float(tax_str), 4), round(float(ydmg_str), 4), round(float(param_str), 4))
                # Only record empty if no prior record exists
                if key not in all_runs:
                    all_runs[key] = {
                        'tax_equity': key[0],
                        'ydmg': key[1],
                        'param_value': key[2],
                        'success': False,
                        'optimizer_status': '',
                        'objective': np.nan,
                        'dir': run_dir,
                        'has_results': False,
                    }
                continue

            with open(status_file, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)

            success = row.get('success', 'False') == 'True'

            # Extract param value from directory name
            dirname = os.path.basename(run_dir)
            param_token = f'_{param_name}-'
            param_str = dirname.split(param_token)[1].split('_')[0]
            param_val = round(float(param_str), 4)
            tax_eq = round(float(row['tax_equity']), 4)
            ydmg = round(float(row['y_damage_distribution_exponent']), 4)

            key = (tax_eq, ydmg, param_val)

            entry = {
                'tax_equity': tax_eq,
                'ydmg': ydmg,
                'param_value': param_val,
                'success': success,
                'optimizer_status': row.get('optimizer_status', ''),
                'objective': float(row['objective_value']) if row.get('objective_value') else np.nan,
                'dir': run_dir,
                'has_results': False,
            }

            # Latest timestamp wins
            if key not in all_runs or run_dir > all_runs[key]['dir']:
                all_runs[key] = entry

    return list(all_runs.values())


def generate_diagnostics(data_dirs, config_prefix, param_name, tax_filter, ydmg_filter):
    """Generate diagnostics for all (or filtered) scenarios.

    Returns a list of dicts with all diagnostic fields.
    """
    runs = load_all_runs(data_dirs, config_prefix, param_name)
    print(f'Loaded {len(runs)} unique parameter points')

    # Apply scenario filter
    if tax_filter is not None:
        runs = [r for r in runs if abs(r['tax_equity'] - tax_filter) < 1e-6]
    if ydmg_filter is not None:
        runs = [r for r in runs if abs(r['ydmg'] - ydmg_filter) < 1e-6]

    print(f'After filtering: {len(runs)} points')

    diagnostics = []
    n_with_path = 0
    n_empty = 0

    for run in runs:
        diag = {
            'tax_equity': run['tax_equity'],
            'ydmg': run['ydmg'],
            'param_value': run['param_value'],
            'success': run['success'],
            'optimizer_status': run['optimizer_status'],
            'objective': run['objective'],
            'dir': run['dir'],
        }

        # Initialize all diagnostic fields to NaN
        for yr in DIAGNOSTIC_YEARS:
            diag[f'mac_{yr}'] = np.nan
        for yr in MU_YEARS:
            diag[f'mu_{yr}'] = np.nan
        for yr in DT_YEARS:
            diag[f'dt_{yr}'] = np.nan
        diag['mac_max_2030_2050'] = np.nan
        diag['mac_min_2070_2100'] = np.nan
        diag['flag_mac_collapse'] = False
        diag['flag_mu_collapse'] = False
        diag['flag_post2100_spike'] = False

        if run['success']:
            results_files = sorted(glob(os.path.join(run['dir'], '*_optimization_results.csv')))
            if results_files:
                path_diag = extract_diagnostics_from_timepath(results_files[-1])
                diag.update(path_diag)
                diag['has_results'] = True
                n_with_path += 1
            else:
                n_empty += 1
        else:
            n_empty += 1

        diagnostics.append(diag)

    print(f'Extracted time paths: {n_with_path}, empty/failed: {n_empty}')
    return diagnostics


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_points(diagnostics):
    """Two-pass classification of all diagnostic points.

    Pass 1: Path-level (REJECT_EMPTY, REJECT_WRONG_BASIN)
    Pass 2: Smoothness (ACCEPT, ACCEPT_NOISY, REVIEW_OUTLIER, REVIEW_REGIME_CHANGE)

    Adds 'label', 'needs_rerun', 'rerun_reason', 'usable_as_anchor' to each dict.
    """
    # --- Pass 1: path-level classification ---
    for d in diagnostics:
        reasons = []

        if not d['success'] or np.isnan(d.get('mac_2030', np.nan)):
            d['label'] = REJECT_EMPTY
            reasons.append('no_data_or_failed')
        elif d['flag_mac_collapse'] or d['flag_mu_collapse'] or d['flag_post2100_spike']:
            d['label'] = REJECT_WRONG_BASIN
            if d['flag_mac_collapse']:
                reasons.append(f"mac_collapse({d['mac_max_2030_2050']:.0f}->{d['mac_min_2070_2100']:.0f})")
            if d['flag_mu_collapse']:
                reasons.append(f"mu_collapse(mu_2030={d['mu_2030']:.3f},min_2070_2100<{MU_COLLAPSE_THRESHOLD})")
            if d['flag_post2100_spike']:
                reasons.append(f"post2100_spike(mac_2120={d['mac_2120']:.0f})")
        else:
            d['label'] = 'PENDING'

        d['_pass1_reasons'] = reasons

    # --- Pass 2: smoothness classification per scenario ---
    # Group by (tax_equity, ydmg)
    scenarios = {}
    for d in diagnostics:
        key = (d['tax_equity'], d['ydmg'])
        scenarios.setdefault(key, []).append(d)

    for scenario_key, points in scenarios.items():
        _classify_scenario_smoothness(points)

    # --- Set derived columns ---
    for d in diagnostics:
        all_reasons = d.pop('_pass1_reasons')
        all_reasons.extend(d.pop('_pass2_reasons', []))
        d['rerun_reason'] = ';'.join(all_reasons)
        d['needs_rerun'] = d['label'] in (REJECT_WRONG_BASIN, REJECT_EMPTY, REVIEW_OUTLIER)
        d['usable_as_anchor'] = d['label'] in (ACCEPT, ACCEPT_NOISY)

    return diagnostics


def _classify_scenario_smoothness(points):
    """Pass 2: smoothness classification for points within one scenario.

    Only operates on PENDING points. Uses only non-rejected points as references.
    """
    # Separate clean (PENDING) from rejected
    clean = [p for p in points if p['label'] == 'PENDING']
    if not clean:
        return

    clean.sort(key=lambda p: p['param_value'])

    # Compute scenario ranges from clean points only
    mac_values = np.array([p['mac_2030'] for p in clean])
    dt_values = np.array([p['dt_2100'] for p in clean])
    mac_range = np.ptp(mac_values) if len(mac_values) > 1 else 1.0
    dt_range = np.ptp(dt_values) if len(dt_values) > 1 else 1.0

    # Avoid division by zero for very flat scenarios
    mac_range = max(mac_range, 1.0)
    dt_range = max(dt_range, 0.01)

    param_values = np.array([p['param_value'] for p in clean])
    n = len(clean)

    for i, p in enumerate(clean):
        p['_pass2_reasons'] = []

        if n < 3 or i == 0 or i == n - 1:
            # Boundary points: can't compute interpolation, accept by default
            p['label'] = ACCEPT
            continue

        # Interpolate from nearest clean neighbors (already sorted, all clean)
        left = clean[i - 1]
        right = clean[i + 1]

        frac = (param_values[i] - param_values[i - 1]) / (param_values[i + 1] - param_values[i - 1])

        # MAC@2030 deviation
        mac_expected = left['mac_2030'] + (right['mac_2030'] - left['mac_2030']) * frac
        mac_dev = abs(p['mac_2030'] - mac_expected)
        mac_dev_frac = mac_dev / mac_range

        # dT@2100 deviation
        dt_expected = left['dt_2100'] + (right['dt_2100'] - left['dt_2100']) * frac
        dt_dev = abs(p['dt_2100'] - dt_expected)
        dt_dev_frac = dt_dev / dt_range

        max_dev_frac = max(mac_dev_frac, dt_dev_frac)

        if max_dev_frac > SMOOTHNESS_OUTLIER_THRESHOLD:
            # Check if this is a regime transition
            is_regime = _check_regime_transition(clean, i, mac_range, dt_range)
            if is_regime:
                p['label'] = REVIEW_REGIME_CHANGE
                p['_pass2_reasons'].append(
                    f"mac_2030_dev={mac_dev_frac:.0%};dt_2100_dev={dt_dev_frac:.0%};"
                    f"monotonic_jump;both_sides_smooth"
                )
            else:
                p['label'] = REVIEW_OUTLIER
                p['_pass2_reasons'].append(
                    f"mac_2030_dev={mac_dev_frac:.0%};dt_2100_dev={dt_dev_frac:.0%}"
                )
        elif max_dev_frac > SMOOTHNESS_NOISY_THRESHOLD:
            p['label'] = ACCEPT_NOISY
            p['_pass2_reasons'].append(f"mac_2030_dev={mac_dev_frac:.1%};dt_2100_dev={dt_dev_frac:.1%}")
        else:
            p['label'] = ACCEPT

    # Also label non-PENDING points with empty pass2 reasons
    for p in points:
        if '_pass2_reasons' not in p:
            p['_pass2_reasons'] = []


def _check_regime_transition(clean_points, idx, mac_range, dt_range):
    """Check if point at idx is a regime transition rather than an outlier.

    Criteria:
    1. Jump is monotonic (point sits between left and right sequences)
    2. Points on both sides form smooth sequences
    """
    n = len(clean_points)
    param_values = [p['param_value'] for p in clean_points]
    mac_values = [p['mac_2030'] for p in clean_points]

    # Need at least 2 points on each side
    if idx < 2 or idx > n - 3:
        return False

    # Check monotonicity: is the point between left and right values?
    left_val = mac_values[idx - 1]
    right_val = mac_values[idx + 1]
    point_val = mac_values[idx]

    is_monotonic = (min(left_val, right_val) <= point_val <= max(left_val, right_val))

    # Check smoothness on both sides (2 points each)
    def side_smooth(i1, i2, i3):
        """Check if three consecutive points are smooth."""
        if i3 >= n or i1 < 0:
            return True  # not enough points, assume smooth
        frac = (param_values[i2] - param_values[i1]) / (param_values[i3] - param_values[i1])
        expected = mac_values[i1] + (mac_values[i3] - mac_values[i1]) * frac
        dev = abs(mac_values[i2] - expected) / mac_range
        return dev < SMOOTHNESS_OUTLIER_THRESHOLD

    left_smooth = side_smooth(idx - 2, idx - 1, idx)
    right_smooth = side_smooth(idx, idx + 1, idx + 2)

    return is_monotonic and left_smooth and right_smooth


# ---------------------------------------------------------------------------
# Neighbor selection
# ---------------------------------------------------------------------------
def find_valid_neighbor(classification, target_param, tax_equity, ydmg):
    """Find nearest usable_as_anchor=True neighbor(s) for a target parameter value.

    Returns list of (param_value, dir) tuples.
    - If one clear nearest: returns [(param, dir)]
    - If two at similar distance: returns [(param_below, dir_below), (param_above, dir_above)]
    """
    scenario = [
        c for c in classification
        if abs(c['tax_equity'] - tax_equity) < 1e-6
        and abs(c['ydmg'] - ydmg) < 1e-6
        and c['usable_as_anchor']
        and abs(c['param_value'] - target_param) > 1e-8  # exclude self
    ]

    if not scenario:
        raise ValueError(
            f"No usable anchors for target={target_param}, "
            f"tax_equity={tax_equity}, ydmg={ydmg}"
        )

    # Find nearest below and above
    below = [c for c in scenario if c['param_value'] < target_param]
    above = [c for c in scenario if c['param_value'] > target_param]

    nearest_below = max(below, key=lambda c: c['param_value']) if below else None
    nearest_above = min(above, key=lambda c: c['param_value']) if above else None

    if nearest_below is None:
        return [(nearest_above['param_value'], nearest_above['dir'])]
    if nearest_above is None:
        return [(nearest_below['param_value'], nearest_below['dir'])]

    dist_below = target_param - nearest_below['param_value']
    dist_above = nearest_above['param_value'] - target_param

    # If both are at similar distance (within 3x), return both for competing warm-starts
    ratio = max(dist_below, dist_above) / min(dist_below, dist_above)
    if ratio < 3.0:
        return [
            (nearest_below['param_value'], nearest_below['dir']),
            (nearest_above['param_value'], nearest_above['dir']),
        ]

    # Otherwise return the closer one
    if dist_below <= dist_above:
        return [(nearest_below['param_value'], nearest_below['dir'])]
    return [(nearest_above['param_value'], nearest_above['dir'])]


# ---------------------------------------------------------------------------
# Resubmission plan generation
# ---------------------------------------------------------------------------
def generate_resubmission_plan(classification, diagnostics, param_name):
    """Generate markdown resubmission plan for points that need reruns.

    Returns string with markdown content.
    """
    # Index diagnostics by key
    diag_by_key = {}
    for d in diagnostics:
        key = (d['tax_equity'], d['ydmg'], d['param_value'])
        diag_by_key[key] = d

    # Find points that need rerun
    rerun_points = [c for c in classification if c['needs_rerun']]

    if not rerun_points:
        return f"# Resubmission Plan\n\nNo points need rerun. All points are ACCEPT or ACCEPT_NOISY.\n"

    # Group by scenario
    scenarios = {}
    for p in rerun_points:
        key = (p['tax_equity'], p['ydmg'])
        scenarios.setdefault(key, []).append(p)

    lines = [f"# Resubmission Plan for {param_name} sweep\n"]
    lines.append(f"Total points needing rerun: {len(rerun_points)}\n")

    for (tax_eq, ydmg), points in sorted(scenarios.items()):
        lines.append(f"\n## Scenario: tax_equity={tax_eq}, ydmg={ydmg}\n")

        for p in sorted(points, key=lambda x: x['param_value']):
            pv = p['param_value']
            key = (tax_eq, ydmg, pv)
            d = diag_by_key.get(key, {})

            lines.append(f"### {param_name}={pv}\n")
            lines.append(f"**Classification**: {p['label']}")
            lines.append(f"**Reasons**: {p['rerun_reason']}\n")

            # Current values
            mac30 = d.get('mac_2030', np.nan)
            mac100 = d.get('mac_2100', np.nan)
            dt100 = d.get('dt_2100', np.nan)
            mu100 = d.get('mu_2100', np.nan)
            obj = d.get('objective', np.nan)

            lines.append(f"**Current values**: MAC@2030={mac30:.1f}, MAC@2100={mac100:.1f}, "
                         f"dT@2100={dt100:.4f}, mu@2100={mu100:.4f}")
            lines.append(f"**Objective**: {obj:.6e}\n")

            # Find proposed anchors
            anchors = find_valid_neighbor(classification, pv, tax_eq, ydmg)
            lines.append("**Proposed anchors**:")
            for i, (anchor_param, anchor_dir) in enumerate(anchors):
                anchor_key = (tax_eq, ydmg, anchor_param)
                anchor_diag = diag_by_key.get(anchor_key, {})
                anchor_class = next(
                    (c for c in classification
                     if abs(c['param_value'] - anchor_param) < 1e-8
                     and abs(c['tax_equity'] - tax_eq) < 1e-6
                     and abs(c['ydmg'] - ydmg) < 1e-6),
                    {}
                )
                direction = 'below' if anchor_param < pv else 'above'
                a_mac = anchor_diag.get('mac_2030', np.nan)
                a_dt = anchor_diag.get('dt_2100', np.nan)
                a_label = anchor_class.get('label', '?')
                lines.append(
                    f"- Anchor {chr(65+i)}: {param_name}={anchor_param} ({direction}) "
                    f"-- {a_label}, MAC@2030={a_mac:.1f}, dT@2100={a_dt:.4f}"
                )

            # Excluded anchors
            excluded = [
                c for c in classification
                if abs(c['tax_equity'] - tax_eq) < 1e-6
                and abs(c['ydmg'] - ydmg) < 1e-6
                and not c['usable_as_anchor']
                and abs(c['param_value'] - pv) > 1e-8
                and abs(c['param_value'] - pv) < 0.01  # only nearby exclusions
            ]
            if excluded:
                lines.append("\n**Excluded anchors**:")
                for ex in sorted(excluded, key=lambda x: x['param_value']):
                    lines.append(
                        f"- {param_name}={ex['param_value']}: {ex['label']} ({ex['rerun_reason']})"
                    )

            # Expected ranges from anchor interpolation
            if len(anchors) == 2:
                a1 = diag_by_key.get((tax_eq, ydmg, anchors[0][0]), {})
                a2 = diag_by_key.get((tax_eq, ydmg, anchors[1][0]), {})
                lines.append("\n**Expected acceptable ranges** (from anchor interpolation):")
                for metric in ['mac_2030', 'mac_2100', 'dt_2100', 'mu_2100']:
                    v1 = a1.get(metric, np.nan)
                    v2 = a2.get(metric, np.nan)
                    if not np.isnan(v1) and not np.isnan(v2):
                        lo, hi = min(v1, v2), max(v1, v2)
                        fmt = '.1f' if 'mac' in metric else ('.4f' if 'dt' in metric else '.3f')
                        lines.append(f"- {metric}: {lo:{fmt}} -- {hi:{fmt}}")
            elif len(anchors) == 1:
                a1 = diag_by_key.get((tax_eq, ydmg, anchors[0][0]), {})
                lines.append("\n**Expected acceptable ranges** (single anchor, approximate):")
                for metric in ['mac_2030', 'mac_2100', 'dt_2100', 'mu_2100']:
                    v1 = a1.get(metric, np.nan)
                    if not np.isnan(v1):
                        margin = abs(v1) * 0.15
                        fmt = '.1f' if 'mac' in metric else ('.4f' if 'dt' in metric else '.3f')
                        lines.append(f"- {metric}: {v1 - margin:{fmt}} -- {v1 + margin:{fmt}}")

            lines.append("\n**Post-run acceptance criteria**:")
            lines.append("- No path flags (mac_collapse, mu_collapse, post2100_spike)")
            lines.append("- MAC@2030 within expected range")
            lines.append("- dT@2100 within expected range")
            lines.append("- Objective <= old objective (more negative = better)\n")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Post-run validation
# ---------------------------------------------------------------------------
def validate_warmstart_runs(warmstart_dirs, base_classification, base_diagnostics, param_name):
    """Validate warm-start results against base classification and expected ranges.

    Returns list of dicts with validation verdicts.
    """
    # Load new runs
    new_diagnostics = []
    for ws_dir in warmstart_dirs:
        config_prefix = os.path.basename(str(ws_dir)).replace('_warmstart', '')
        new_diags = generate_diagnostics([ws_dir], config_prefix, param_name, None, None)
        new_diagnostics.extend(new_diags)

    # Classify new runs independently (path flags)
    for d in new_diagnostics:
        reasons = []
        if not d['success'] or np.isnan(d.get('mac_2030', np.nan)):
            d['verdict'] = 'MERGE_BLOCKED'
            d['verdict_reasons'] = 'no_data_or_failed'
            continue

        if d['flag_mac_collapse']:
            reasons.append('mac_collapse')
        if d['flag_mu_collapse']:
            reasons.append('mu_collapse')
        if d['flag_post2100_spike']:
            reasons.append('post2100_spike')

        if reasons:
            d['verdict'] = 'MERGE_BLOCKED'
            d['verdict_reasons'] = ';'.join(reasons)
        else:
            d['verdict'] = 'MERGE_READY'
            d['verdict_reasons'] = ''

    return new_diagnostics


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_diagnostics_csv(diagnostics, output_path):
    """Write diagnostics to CSV."""
    if not diagnostics:
        return

    # Define column order
    base_cols = ['tax_equity', 'ydmg', 'param_value', 'success', 'optimizer_status', 'objective']
    mac_cols = [f'mac_{yr}' for yr in DIAGNOSTIC_YEARS]
    mu_cols = [f'mu_{yr}' for yr in MU_YEARS]
    dt_cols = [f'dt_{yr}' for yr in DT_YEARS]
    agg_cols = ['mac_max_2030_2050', 'mac_min_2070_2100']
    flag_cols = ['flag_mac_collapse', 'flag_mu_collapse', 'flag_post2100_spike']
    meta_cols = ['dir']

    columns = base_cols + mac_cols + mu_cols + dt_cols + agg_cols + flag_cols + meta_cols

    df = pd.DataFrame(diagnostics)
    # Only include columns that exist
    columns = [c for c in columns if c in df.columns]
    df = df[columns].sort_values(['tax_equity', 'ydmg', 'param_value'])
    df.to_csv(output_path, index=False)
    print(f'Wrote diagnostics: {output_path}')


def write_classification_csv(classification, output_path):
    """Write classification to CSV."""
    if not classification:
        return

    columns = ['tax_equity', 'ydmg', 'param_value', 'label', 'needs_rerun',
               'rerun_reason', 'usable_as_anchor']

    df = pd.DataFrame(classification)
    columns = [c for c in columns if c in df.columns]
    df = df[columns].sort_values(['tax_equity', 'ydmg', 'param_value'])
    df.to_csv(output_path, index=False)
    print(f'Wrote classification: {output_path}')


def print_classification_summary(classification, param_name):
    """Print summary of classification results."""
    from collections import Counter

    label_counts = Counter(c['label'] for c in classification)
    total = len(classification)

    print(f"\n{'='*60}")
    print(f"CLASSIFICATION SUMMARY ({total} points)")
    print(f"{'='*60}")
    for label in [ACCEPT, ACCEPT_NOISY, REVIEW_REGIME_CHANGE, REVIEW_OUTLIER,
                  REJECT_WRONG_BASIN, REJECT_EMPTY]:
        count = label_counts.get(label, 0)
        if count > 0:
            print(f"  {label:<25s}: {count:>4d} ({100*count/total:.1f}%)")

    needs_rerun = sum(1 for c in classification if c['needs_rerun'])
    usable = sum(1 for c in classification if c['usable_as_anchor'])
    print(f"\n  needs_rerun:    {needs_rerun}")
    print(f"  usable_as_anchor: {usable}")

    # Per-scenario breakdown for scenarios with issues
    scenarios = {}
    for c in classification:
        key = (c['tax_equity'], c['ydmg'])
        scenarios.setdefault(key, []).append(c)

    problem_scenarios = {
        k: v for k, v in scenarios.items()
        if any(c['label'] not in (ACCEPT, ACCEPT_NOISY) for c in v)
    }

    if problem_scenarios:
        print(f"\n{'='*60}")
        print("SCENARIOS WITH ISSUES")
        print(f"{'='*60}")
        for (tax_eq, ydmg), points in sorted(problem_scenarios.items()):
            problems = [c for c in points if c['label'] not in (ACCEPT, ACCEPT_NOISY)]
            print(f"\n  tax_equity={tax_eq}, ydmg={ydmg}: {len(problems)} issue(s)")
            for p in sorted(problems, key=lambda x: x['param_value']):
                print(f"    {param_name}={p['param_value']:.4f}: {p['label']} "
                      f"({p['rerun_reason']})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_diagnose(args):
    """Run diagnostic + classification."""
    config_name = args.config
    param_name = args.param
    tax_filter = args.tax
    ydmg_filter = args.ydmg

    data_dirs = find_data_dirs(config_name)
    config_prefix = config_name

    print(f"Diagnosing {config_name} ({param_name} sweep)")
    print(f"Data directories: {[str(d) for d in data_dirs]}")

    diagnostics = generate_diagnostics(data_dirs, config_prefix, param_name, tax_filter, ydmg_filter)
    classification = classify_points(diagnostics)

    output_dir = PROJECT_ROOT / 'data' / 'output'
    diag_path = output_dir / f'diagnostics_{config_name}.csv'
    class_path = output_dir / f'classification_{config_name}.csv'

    write_diagnostics_csv(diagnostics, diag_path)
    write_classification_csv(classification, class_path)
    print_classification_summary(classification, param_name)


def cmd_plan(args):
    """Generate resubmission plan from existing classification."""
    config_name = args.config
    param_name = args.param

    output_dir = PROJECT_ROOT / 'data' / 'output'
    class_path = output_dir / f'classification_{config_name}.csv'
    diag_path = output_dir / f'diagnostics_{config_name}.csv'

    class_df = pd.read_csv(class_path)
    diag_df = pd.read_csv(diag_path)

    classification = class_df.to_dict('records')
    diagnostics = diag_df.to_dict('records')

    plan_md = generate_resubmission_plan(classification, diagnostics, param_name)

    plan_path = output_dir / f'resubmission_plan_{config_name}.md'
    with open(plan_path, 'w') as f:
        f.write(plan_md)
    print(f'Wrote resubmission plan: {plan_path}')


def cmd_validate(args):
    """Validate warm-start results."""
    config_name = args.config  # e.g. config_060_warmstart
    param_name = args.param

    # Load base classification
    base_config = config_name.replace('_warmstart', '')
    output_dir = PROJECT_ROOT / 'data' / 'output'
    class_path = output_dir / f'classification_{base_config}.csv'
    diag_path = output_dir / f'diagnostics_{base_config}.csv'

    base_class = pd.read_csv(class_path).to_dict('records')
    base_diag = pd.read_csv(diag_path).to_dict('records')

    warmstart_dir = output_dir / config_name
    results = validate_warmstart_runs([warmstart_dir], base_class, base_diag, param_name)

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS ({len(results)} runs)")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x['param_value']):
        print(f"  {param_name}={r['param_value']:.4f}: {r['verdict']} "
              f"MAC@2030={r.get('mac_2030', np.nan):.1f} dT@2100={r.get('dt_2100', np.nan):.4f} "
              f"{r.get('verdict_reasons', '')}")

    report_path = output_dir / f'validation_report_{config_name}.md'
    with open(report_path, 'w') as f:
        f.write(f"# Validation Report: {config_name}\n\n")
        for r in sorted(results, key=lambda x: x['param_value']):
            f.write(f"## {param_name}={r['param_value']:.4f}\n")
            f.write(f"**Verdict**: {r['verdict']}\n")
            if r.get('verdict_reasons'):
                f.write(f"**Reasons**: {r['verdict_reasons']}\n")
            f.write(f"- MAC@2030: {r.get('mac_2030', np.nan):.2f}\n")
            f.write(f"- MAC@2100: {r.get('mac_2100', np.nan):.2f}\n")
            f.write(f"- dT@2100: {r.get('dt_2100', np.nan):.4f}\n")
            f.write(f"- mu@2100: {r.get('mu_2100', np.nan):.4f}\n")
            f.write(f"- Objective: {r.get('objective', np.nan):.6e}\n\n")
    print(f'Wrote validation report: {report_path}')


def cmd_neighbor(args):
    """Find valid neighbor for a specific point."""
    config_name = args.config
    param_name = args.param
    target = args.target
    tax_eq = args.tax
    ydmg = args.ydmg

    output_dir = PROJECT_ROOT / 'data' / 'output'
    class_path = output_dir / f'classification_{config_name}.csv'

    class_df = pd.read_csv(class_path)
    classification = class_df.to_dict('records')

    neighbors = find_valid_neighbor(classification, target, tax_eq, ydmg)

    print(f"Target: {param_name}={target} (tax_equity={tax_eq}, ydmg={ydmg})")
    print(f"Valid neighbor(s):")
    for param, dir_path in neighbors:
        print(f"  {param_name}={param}: {dir_path}")


def main():
    parser = argparse.ArgumentParser(description='Diagnostic / Classification / Resubmission gate')
    subparsers = parser.add_subparsers(dest='command')

    # diagnose
    p_diag = subparsers.add_parser('diagnose', help='Run diagnostic + classification')
    p_diag.add_argument('config', help='Config name (e.g. config_060)')
    p_diag.add_argument('param', help='Parameter name (rho or eta)')
    p_diag.add_argument('--tax', type=float, default=None, help='Filter by tax_equity')
    p_diag.add_argument('--ydmg', type=float, default=None, help='Filter by ydmg')

    # plan
    p_plan = subparsers.add_parser('plan', help='Generate resubmission plan')
    p_plan.add_argument('config', help='Config name')
    p_plan.add_argument('param', help='Parameter name')

    # validate
    p_val = subparsers.add_parser('validate', help='Validate warm-start results')
    p_val.add_argument('config', help='Config name (e.g. config_060_warmstart)')
    p_val.add_argument('param', help='Parameter name')

    # neighbor
    p_nbr = subparsers.add_parser('neighbor', help='Find valid neighbor')
    p_nbr.add_argument('config', help='Config name')
    p_nbr.add_argument('param', help='Parameter name')
    p_nbr.add_argument('target', type=float, help='Target parameter value')
    p_nbr.add_argument('--tax', type=float, required=True, help='tax_equity value')
    p_nbr.add_argument('--ydmg', type=float, required=True, help='ydmg value')

    args = parser.parse_args()

    if args.command == 'diagnose':
        cmd_diagnose(args)
    elif args.command == 'plan':
        cmd_plan(args)
    elif args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'neighbor':
        cmd_neighbor(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
