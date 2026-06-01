"""Check config_061 grid_ids with multiple runs and compare latest vs best objective."""

import csv
import os
from collections import defaultdict

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "output", "config_061"
)


def extract_sweep_type(dirname):
    """Extract sweep type (eta, rho, beta) from directory name."""
    # Format: config_061_{sweep}_tax-...
    parts = dirname.split("_")
    return parts[2]


def load_all_runs():
    """Load all successful runs from config_061 directories."""
    records = []
    dirs = sorted(os.listdir(DATA_DIR))
    n_dirs = len(dirs)
    n_failed = 0
    n_missing = 0

    for dirname in dirs:
        csv_path = os.path.join(DATA_DIR, dirname, "run_status.csv")
        if not os.path.exists(csv_path):
            n_missing += 1
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["success"] != "True":
                    n_failed += 1
                    continue
                records.append({
                    "grid_id": int(row["grid_id"]),
                    "objective_value": float(row["objective_value"]),
                    "mac_2030": float(row["mac_2030"]),
                    "sweep": row["sweep"],
                    "dirname": dirname,
                })

    print(f"Total directories: {n_dirs}")
    print(f"Missing run_status.csv: {n_missing}")
    print(f"Failed runs (skipped): {n_failed}")
    print(f"Successful runs loaded: {len(records)}")
    return records


def analyze_duplicates(records):
    """Group by grid_id and analyze latest vs best."""
    by_grid = defaultdict(list)
    for r in records:
        by_grid[r["grid_id"]].append(r)

    n_single = 0
    n_multi = 0
    n_latest_is_best = 0
    n_latest_not_best = 0

    # Track by sweep type
    by_sweep = defaultdict(lambda: {
        "multi": 0,
        "latest_is_best": 0,
        "latest_not_best": 0,
        "obj_diffs": [],
        "mac_diffs": [],
    })

    latest_not_best_details = []

    for grid_id in sorted(by_grid.keys()):
        runs = by_grid[grid_id]
        if len(runs) == 1:
            n_single += 1
            continue

        n_multi += 1

        # Sort by dirname alphabetically -> last = latest timestamp
        runs_sorted = sorted(runs, key=lambda r: r["dirname"])
        latest_run = runs_sorted[-1]
        best_run = max(runs_sorted, key=lambda r: r["objective_value"])
        sweep = latest_run["sweep"]

        by_sweep[sweep]["multi"] += 1

        if latest_run["dirname"] == best_run["dirname"]:
            n_latest_is_best += 1
            by_sweep[sweep]["latest_is_best"] += 1
        else:
            n_latest_not_best += 1
            by_sweep[sweep]["latest_not_best"] += 1
            obj_diff = best_run["objective_value"] - latest_run["objective_value"]
            mac_diff = best_run["mac_2030"] - latest_run["mac_2030"]
            by_sweep[sweep]["obj_diffs"].append(obj_diff)
            by_sweep[sweep]["mac_diffs"].append(mac_diff)
            latest_not_best_details.append({
                "grid_id": grid_id,
                "sweep": sweep,
                "n_runs": len(runs),
                "latest_obj": latest_run["objective_value"],
                "best_obj": best_run["objective_value"],
                "obj_diff": obj_diff,
                "latest_mac": latest_run["mac_2030"],
                "best_mac": best_run["mac_2030"],
                "mac_diff": mac_diff,
                "latest_dir": latest_run["dirname"],
                "best_dir": best_run["dirname"],
            })

    print(f"\n{'='*70}")
    print("GRID_ID DUPLICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Unique grid_ids:                {len(by_grid)}")
    print(f"  Single-run grid_ids:          {n_single}")
    print(f"  Multi-run grid_ids:           {n_multi}")
    print(f"    Latest = best:              {n_latest_is_best}")
    print(f"    Latest != best:             {n_latest_not_best}")

    print(f"\n{'='*70}")
    print("BREAKDOWN BY SWEEP TYPE")
    print(f"{'='*70}")
    for sweep in sorted(by_sweep.keys()):
        s = by_sweep[sweep]
        print(f"\n--- {sweep} ---")
        print(f"  Multi-run grid_ids:           {s['multi']}")
        print(f"    Latest = best:              {s['latest_is_best']}")
        print(f"    Latest != best:             {s['latest_not_best']}")
        if s["obj_diffs"]:
            print(f"    Objective diff (best-latest):")
            print(f"      min:    {min(s['obj_diffs']):.10e}")
            print(f"      max:    {max(s['obj_diffs']):.10e}")
            print(f"      mean:   {sum(s['obj_diffs'])/len(s['obj_diffs']):.10e}")
            print(f"    MAC_2030 diff (best-latest):")
            print(f"      min:    {min(s['mac_diffs']):.4f}")
            print(f"      max:    {max(s['mac_diffs']):.4f}")
            print(f"      mean:   {sum(s['mac_diffs'])/len(s['mac_diffs']):.4f}")

    if latest_not_best_details:
        print(f"\n{'='*70}")
        print("DETAILS: GRID_IDS WHERE LATEST RUN IS NOT THE BEST")
        print(f"{'='*70}")
        for d in sorted(latest_not_best_details, key=lambda x: abs(x["obj_diff"]), reverse=True):
            print(f"\n  grid_id {d['grid_id']} ({d['sweep']}, {d['n_runs']} runs):")
            print(f"    Best  obj: {d['best_obj']:.10e}  MAC_2030: {d['best_mac']:.4f}")
            print(f"      dir: {d['best_dir']}")
            print(f"    Latest obj: {d['latest_obj']:.10e}  MAC_2030: {d['latest_mac']:.4f}")
            print(f"      dir: {d['latest_dir']}")
            print(f"    Diff obj: {d['obj_diff']:.10e}  MAC: {d['mac_diff']:.4f}")


if __name__ == "__main__":
    records = load_all_runs()
    analyze_duplicates(records)
