"""Universal merged loader for config_061 results.

Loads results from config_061, config_061_warmstart, and
config_061_warmstart_loose. Picks best objective per grid_id.

Usage as module:
    from scripts.load_061_merged import load_results_061_merged
    results = load_results_061_merged()  # dict: grid_id -> row dict
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIRS = [
    PROJECT_ROOT / 'data' / 'output' / 'config_061',
    PROJECT_ROOT / 'data' / 'output' / 'config_061_warmstart',
    PROJECT_ROOT / 'data' / 'output' / 'config_061_warmstart_loose',
]


def load_results_061_merged():
    """Load config_061 + config_061_warmstart + config_061_warmstart_loose.

    Pick best objective per grid_id. Returns dict: grid_id -> row dict.
    Row dict has string values matching run_status.csv columns, plus
    'source' indicating which directory the best result came from.
    """
    results = {}

    for base_dir in SOURCE_DIRS:
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
                    if gid not in results or obj < float(results[gid]['objective_value']):
                        row_copy = dict(row)
                        row_copy['source'] = base_dir.name
                        results[gid] = row_copy

    return results


if __name__ == '__main__':
    results = load_results_061_merged()
    print(f"Loaded {len(results)} unique grid_ids")

    # Count by source
    source_counts = {}
    for r in results.values():
        src = r['source']
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nBy source:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")
