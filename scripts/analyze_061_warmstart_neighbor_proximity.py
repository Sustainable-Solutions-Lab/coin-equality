"""Analyze whether config_061 warmstart_loose results are closer to neighbors.

For each candidate grid_id, compares:
  - Original result's distance to left/right neighbors (MAC and delta_T)
  - Warmstart result's distance to left/right neighbors
  - Whether warmstart is closer to neighbors even when objective is not better

Usage:
    python scripts/analyze_061_warmstart_neighbor_proximity.py
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_CSV = PROJECT_ROOT / 'json' / 'config_061_grid.csv'
CANDIDATES_CSV = PROJECT_ROOT / 'json' / 'config_061_warmstart_loose_candidates.csv'

ORIG_DIRS = [
    PROJECT_ROOT / 'data' / 'output' / 'config_061',
]

WARMSTART_DIR = PROJECT_ROOT / 'data' / 'output' / 'config_061_warmstart_loose'


def load_grid():
    rows = {}
    with open(GRID_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row['grid_id'])] = {
                'grid_id': int(row['grid_id']),
                'sweep': row['sweep'],
                'sample_id': int(row['sample_id']),
                'tax_equity': float(row['tax_equity']),
                'y_damage_distribution_exponent': float(row['y_damage_distribution_exponent']),
                'eta': float(row['eta']),
                'rho': float(row['rho']),
            }
    return rows


def load_results_from_dirs(dirs):
    """Load best result per grid_id from given directories."""
    results = {}
    for base_dir in dirs:
        if not base_dir.exists():
            continue
        for entry in sorted(base_dir.iterdir()):
            if not entry.is_dir():
                continue
            status_path = entry / 'run_status.csv'
            if not status_path.exists():
                continue
            with open(status_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['success'] != 'True':
                        continue
                    gid = int(row['grid_id'])
                    obj = float(row['objective_value'])
                    if gid not in results or obj < results[gid]['objective_value']:
                        results[gid] = {
                            'grid_id': gid,
                            'objective_value': obj,
                            'mac_2030': float(row['mac_2030']),
                            'delta_t_2100': float(row['delta_t_2100']),
                        }
    return results


def load_warmstart_results():
    """Load ALL warmstart_loose results per grid_id (best objective among warmstart runs)."""
    results = {}
    base_dir = WARMSTART_DIR
    if not base_dir.exists():
        return results
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        status_path = entry / 'run_status.csv'
        if not status_path.exists():
            continue
        with open(status_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['success'] != 'True':
                    continue
                gid = int(row['grid_id'])
                obj = float(row['objective_value'])
                if gid not in results or obj < results[gid]['objective_value']:
                    results[gid] = {
                        'grid_id': gid,
                        'objective_value': obj,
                        'mac_2030': float(row['mac_2030']),
                        'delta_t_2100': float(row['delta_t_2100']),
                    }
    return results


def load_candidates():
    candidates = set()
    with open(CANDIDATES_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.add(int(row['grid_id']))
    return candidates


def build_neighbor_map(grid):
    """For each grid_id, find left and right neighbor grid_ids (same sweep, tax_equity, sample_id ± 1)."""
    groups = defaultdict(dict)
    for gid, g in grid.items():
        key = (g['sweep'], g['tax_equity'])
        groups[key][g['sample_id']] = gid

    neighbors = {}
    for gid, g in grid.items():
        key = (g['sweep'], g['tax_equity'])
        sid = g['sample_id']
        group = groups[key]
        left_gid = group.get(sid - 1)
        right_gid = group.get(sid + 1)
        neighbors[gid] = (left_gid, right_gid)
    return neighbors


def main():
    grid = load_grid()
    print(f"Loaded {len(grid)} grid points")

    original = load_results_from_dirs(ORIG_DIRS)
    print(f"Loaded {len(original)} original (cold-start) results")

    warmstart = load_warmstart_results()
    print(f"Loaded {len(warmstart)} warmstart_loose results")

    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidate grid_ids")

    neighbors = build_neighbor_map(grid)

    # For each candidate, compute distance to neighbors for both original and warmstart
    mac_dist_orig = []
    mac_dist_ws = []
    dt_dist_orig = []
    dt_dist_ws = []
    obj_diff = []  # positive = warmstart worse (higher obj since we minimize negative)
    n_ws_closer_mac = 0
    n_ws_closer_dt = 0
    n_ws_better_obj = 0
    n_ws_worse_obj = 0
    n_ws_same_obj = 0
    n_analyzed = 0

    for gid in sorted(candidates):
        if gid not in original or gid not in warmstart:
            continue

        left_gid, right_gid = neighbors[gid]

        # Get neighbor MACs/DTs from original results (neighbors are not candidates,
        # so their original values are the ground truth)
        neighbor_macs = []
        neighbor_dts = []
        for ngid in [left_gid, right_gid]:
            if ngid is not None and ngid in original:
                neighbor_macs.append(original[ngid]['mac_2030'])
                neighbor_dts.append(original[ngid]['delta_t_2100'])

        if len(neighbor_macs) == 0:
            continue

        n_analyzed += 1
        neighbor_mac_mean = np.mean(neighbor_macs)
        neighbor_dt_mean = np.mean(neighbor_dts)

        # Distance from original to neighbor mean
        d_mac_orig = abs(original[gid]['mac_2030'] - neighbor_mac_mean)
        d_dt_orig = abs(original[gid]['delta_t_2100'] - neighbor_dt_mean)

        # Distance from warmstart to neighbor mean
        d_mac_ws = abs(warmstart[gid]['mac_2030'] - neighbor_mac_mean)
        d_dt_ws = abs(warmstart[gid]['delta_t_2100'] - neighbor_dt_mean)

        mac_dist_orig.append(d_mac_orig)
        mac_dist_ws.append(d_mac_ws)
        dt_dist_orig.append(d_dt_orig)
        dt_dist_ws.append(d_dt_ws)

        if d_mac_ws < d_mac_orig:
            n_ws_closer_mac += 1
        if d_dt_ws < d_dt_orig:
            n_ws_closer_dt += 1

        # Objective comparison (both are negative, lower = better)
        obj_o = original[gid]['objective_value']
        obj_w = warmstart[gid]['objective_value']
        obj_diff.append(obj_w - obj_o)  # positive means warmstart is worse

        if obj_w < obj_o - 1e-15:
            n_ws_better_obj += 1
        elif obj_w > obj_o + 1e-15:
            n_ws_worse_obj += 1
        else:
            n_ws_same_obj += 1

    mac_dist_orig = np.array(mac_dist_orig)
    mac_dist_ws = np.array(mac_dist_ws)
    dt_dist_orig = np.array(dt_dist_orig)
    dt_dist_ws = np.array(dt_dist_ws)
    obj_diff = np.array(obj_diff)

    print(f"\n{'=' * 70}")
    print(f"NEIGHBOR PROXIMITY ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Candidates analyzed:  {n_analyzed}")

    print(f"\n--- Objective comparison ---")
    print(f"  Warmstart better obj:  {n_ws_better_obj} ({100*n_ws_better_obj/n_analyzed:.1f}%)")
    print(f"  Warmstart same obj:    {n_ws_same_obj} ({100*n_ws_same_obj/n_analyzed:.1f}%)")
    print(f"  Warmstart worse obj:   {n_ws_worse_obj} ({100*n_ws_worse_obj/n_analyzed:.1f}%)")
    print(f"  Mean obj difference:   {np.mean(obj_diff):.4e}")
    print(f"  Max obj difference:    {np.max(np.abs(obj_diff)):.4e}")

    # Relative objective difference
    obj_orig_vals = np.array([original[gid]['objective_value'] for gid in sorted(candidates)
                              if gid in original and gid in warmstart
                              and neighbors[gid][0] is not None or neighbors[gid][1] is not None][:n_analyzed])
    rel_obj_diff = np.abs(obj_diff) / np.abs(obj_orig_vals)
    print(f"  Mean |relative obj diff|: {np.mean(rel_obj_diff):.4e}")
    print(f"  Max |relative obj diff|:  {np.max(rel_obj_diff):.4e}")

    print(f"\n--- MAC@2030 distance to neighbors ---")
    print(f"  Warmstart closer:   {n_ws_closer_mac} / {n_analyzed} ({100*n_ws_closer_mac/n_analyzed:.1f}%)")
    print(f"  Mean dist original: {np.mean(mac_dist_orig):.4f} $/tCO2")
    print(f"  Mean dist warmstart:{np.mean(mac_dist_ws):.4f} $/tCO2")
    print(f"  Median dist orig:   {np.median(mac_dist_orig):.4f} $/tCO2")
    print(f"  Median dist ws:     {np.median(mac_dist_ws):.4f} $/tCO2")
    print(f"  Mean reduction:     {np.mean(mac_dist_orig - mac_dist_ws):.4f} $/tCO2 ({100*np.mean((mac_dist_orig - mac_dist_ws) / np.maximum(mac_dist_orig, 1e-10)):.1f}%)")

    print(f"\n--- DeltaT@2100 distance to neighbors ---")
    print(f"  Warmstart closer:   {n_ws_closer_dt} / {n_analyzed} ({100*n_ws_closer_dt/n_analyzed:.1f}%)")
    print(f"  Mean dist original: {np.mean(dt_dist_orig):.6f} C")
    print(f"  Mean dist warmstart:{np.mean(dt_dist_ws):.6f} C")
    print(f"  Median dist orig:   {np.median(dt_dist_orig):.6f} C")
    print(f"  Median dist ws:     {np.median(dt_dist_ws):.6f} C")
    print(f"  Mean reduction:     {np.mean(dt_dist_orig - dt_dist_ws):.6f} C ({100*np.mean((dt_dist_orig - dt_dist_ws) / np.maximum(dt_dist_orig, 1e-10)):.1f}%)")

    # Break down by sweep
    print(f"\n{'=' * 70}")
    print(f"BREAKDOWN BY SWEEP")
    print(f"{'=' * 70}")

    for sweep in ['eta', 'rho', 'beta']:
        sweep_gids = [gid for gid in sorted(candidates)
                      if gid in original and gid in warmstart and grid[gid]['sweep'] == sweep]

        if not sweep_gids:
            continue

        sweep_mac_orig = []
        sweep_mac_ws = []
        sweep_dt_orig = []
        sweep_dt_ws = []
        sweep_obj_better = 0
        sweep_obj_same = 0
        sweep_obj_worse = 0
        sweep_mac_closer = 0

        for gid in sweep_gids:
            left_gid, right_gid = neighbors[gid]
            neighbor_macs = []
            neighbor_dts = []
            for ngid in [left_gid, right_gid]:
                if ngid is not None and ngid in original:
                    neighbor_macs.append(original[ngid]['mac_2030'])
                    neighbor_dts.append(original[ngid]['delta_t_2100'])
            if len(neighbor_macs) == 0:
                continue

            nmac = np.mean(neighbor_macs)
            ndt = np.mean(neighbor_dts)

            d_mac_o = abs(original[gid]['mac_2030'] - nmac)
            d_mac_w = abs(warmstart[gid]['mac_2030'] - nmac)
            d_dt_o = abs(original[gid]['delta_t_2100'] - ndt)
            d_dt_w = abs(warmstart[gid]['delta_t_2100'] - ndt)

            sweep_mac_orig.append(d_mac_o)
            sweep_mac_ws.append(d_mac_w)
            sweep_dt_orig.append(d_dt_o)
            sweep_dt_ws.append(d_dt_w)

            if d_mac_w < d_mac_o:
                sweep_mac_closer += 1

            obj_o = original[gid]['objective_value']
            obj_w = warmstart[gid]['objective_value']
            if obj_w < obj_o - 1e-15:
                sweep_obj_better += 1
            elif obj_w > obj_o + 1e-15:
                sweep_obj_worse += 1
            else:
                sweep_obj_same += 1

        n_s = len(sweep_mac_orig)
        print(f"\n  {sweep} sweep ({n_s} candidates):")
        print(f"    Obj better/same/worse: {sweep_obj_better}/{sweep_obj_same}/{sweep_obj_worse}")
        print(f"    MAC closer to neighbors: {sweep_mac_closer}/{n_s} ({100*sweep_mac_closer/n_s:.1f}%)")
        print(f"    Mean MAC dist: orig={np.mean(sweep_mac_orig):.3f}, ws={np.mean(sweep_mac_ws):.3f} $/tCO2")
        print(f"    Mean DT dist:  orig={np.mean(sweep_dt_orig):.5f}, ws={np.mean(sweep_dt_ws):.5f} C")

    # Conditional analysis: among those where warmstart does NOT improve objective
    print(f"\n{'=' * 70}")
    print(f"CONDITIONAL: Warmstart NOT better objective, but closer to neighbors?")
    print(f"{'=' * 70}")

    n_not_better = 0
    n_not_better_but_closer_mac = 0
    n_not_better_but_closer_dt = 0
    not_better_mac_reduction = []

    for i, gid in enumerate(sorted(candidates)):
        if gid not in original or gid not in warmstart:
            continue
        left_gid, right_gid = neighbors[gid]
        neighbor_macs = []
        neighbor_dts = []
        for ngid in [left_gid, right_gid]:
            if ngid is not None and ngid in original:
                neighbor_macs.append(original[ngid]['mac_2030'])
                neighbor_dts.append(original[ngid]['delta_t_2100'])
        if len(neighbor_macs) == 0:
            continue

        obj_o = original[gid]['objective_value']
        obj_w = warmstart[gid]['objective_value']

        # Not better = same or worse objective
        if obj_w >= obj_o - 1e-15:
            n_not_better += 1
            nmac = np.mean(neighbor_macs)
            ndt = np.mean(neighbor_dts)
            d_mac_o = abs(original[gid]['mac_2030'] - nmac)
            d_mac_w = abs(warmstart[gid]['mac_2030'] - nmac)
            d_dt_o = abs(original[gid]['delta_t_2100'] - ndt)
            d_dt_w = abs(warmstart[gid]['delta_t_2100'] - ndt)

            if d_mac_w < d_mac_o:
                n_not_better_but_closer_mac += 1
                not_better_mac_reduction.append(d_mac_o - d_mac_w)
            if d_dt_w < d_dt_o:
                n_not_better_but_closer_dt += 1

    print(f"  Cases where ws obj >= orig obj: {n_not_better}")
    if n_not_better > 0:
        print(f"  Of those, ws closer in MAC:     {n_not_better_but_closer_mac} ({100*n_not_better_but_closer_mac/n_not_better:.1f}%)")
        print(f"  Of those, ws closer in DT:      {n_not_better_but_closer_dt} ({100*n_not_better_but_closer_dt/n_not_better:.1f}%)")
        if not_better_mac_reduction:
            print(f"  Mean MAC reduction (when closer): {np.mean(not_better_mac_reduction):.3f} $/tCO2")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
