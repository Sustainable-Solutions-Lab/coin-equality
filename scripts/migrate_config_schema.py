"""
One-off migration of configuration files to the cleaned-up schema (October 2026).

- optimization_parameters: the per-control settings move under "controls" -> "f" / "s"
  (initial_guess, bounds, n_points_initial, n_points_final and their comments); the NLopt
  tolerances ftol_rel, ftol_abs and xtol_rel become explicit (0.0 = disabled, NLopt's
  default); max_evaluations_time_points, optimize_time_points and looser_epsilon are removed
  (time-point adjustment and gradient-based optimization were removed)
- integration_parameters: the unused plot_short_horizon is removed

Usage (from the repository root):
    python scripts/migrate_config_schema.py json/prod_s data/output/prod_s/9cases_s tests/cases
"""

import json
import sys
from pathlib import Path

# Old optimization_parameters key -> (control, new key)
CONTROL_KEYS = {
    f'{prefix}{field}_{name}': (name, f'{prefix}{field}')
    for name in ['f', 's']
    for field in ['initial_guess', 'bounds', 'n_points_initial', 'n_points_final']
    for prefix in ['', '_']
}
CONTROL_KEYS['_bounds_s_note'] = ('s', '_bounds_note')
CONTROL_KEY_ORDER = ['initial_guess', '_initial_guess', 'bounds', '_bounds', 'n_points_initial',
                     '_n_points_initial', 'n_points_final', '_n_points_final', '_bounds_note']
REMOVED_OPTIMIZATION_KEYS = {'max_evaluations_time_points', 'optimize_time_points', 'looser_epsilon'}
OPTIMIZATION_KEY_ORDER = ['max_evaluations_initial', 'max_evaluations_final', 'optimization_iterations',
                          'algorithm', 'ftol_rel', 'ftol_abs', 'xtol_rel', 'xtol_abs']


def migrate_optimization_parameters(old):
    controls = {}
    remaining = {}
    for key, value in old.items():
        if key in CONTROL_KEYS:
            name, new_key = CONTROL_KEYS[key]
            controls.setdefault(name, {})[new_key] = value
        elif key not in REMOVED_OPTIMIZATION_KEYS:
            remaining[key] = value
    for tolerance in ['ftol_rel', 'ftol_abs', 'xtol_rel']:
        remaining.setdefault(tolerance, 0.0)
    unexpected = set(remaining) - set(OPTIMIZATION_KEY_ORDER)
    if unexpected:
        raise ValueError(f'Unexpected optimization_parameters keys {sorted(unexpected)}')
    new = {key: remaining[key] for key in OPTIMIZATION_KEY_ORDER}
    new['controls'] = {name: {key: spec[key] for key in CONTROL_KEY_ORDER if key in spec}
                       for name, spec in sorted(controls.items())}
    return new


def migrate(config):
    config['integration_parameters'].pop('plot_short_horizon')
    config['optimization_parameters'] = migrate_optimization_parameters(config['optimization_parameters'])
    return config


def main():
    paths = [path for root in sys.argv[1:] for path in sorted(Path(root).rglob('*.json'))]
    for path in paths:
        text = path.read_text()
        migrated = json.dumps(migrate(json.loads(text)), indent=2)
        path.write_text(migrated + ('\n' if text.endswith('\n') else ''))
    print(f'Migrated {len(paths)} configuration files')


if __name__ == '__main__':
    main()
