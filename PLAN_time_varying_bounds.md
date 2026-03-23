# Implementation Plan: Time-Varying Upper Bound on f

## Goal
Allow the upper bound on `f` (log₁₀ of marginal abatement cost) to vary over time by defining time-value pairs in the JSON config, interpolated to control point times.

## Current State
- `bounds_f` in JSON is `[min, max]` - constant bounds for all times
- In `optimization.py:822-824`, upper bounds are set uniformly:
  ```python
  upper_bounds = np.concatenate([np.full(n_f_points, bounds_f[1]), np.full(n_s_points, bounds_s[1])])
  ```
- `initial_guess_f` is an absolute value applied uniformly

## Design Change: Initial Guess as Fraction of Feasible Region
Reinterpret `initial_guess_f` and `initial_guess_s` as fractions of the allowable decision space:
```python
actual_initial = lower_bound + initial_guess * (upper_bound - lower_bound)
```

Benefits:
- No magic numbers in code
- Naturally respects time-varying bounds
- When bounds are [0, 1], behavior unchanged (e.g., 0.04 * 1.0 = 0.04)
- When upper bound varies, initial guess scales appropriately

## Proposed JSON Config Format
```json
"optimization_parameters": {
    "bounds_f": [0.0, 1.0],
    "bounds_f_upper_points": [[2020, 0.9], [2100, 0.5], [2200, 0.3], [2420, 0.1]],
    "_bounds_f_upper_points": "Time-varying upper bound on f. List of [year, value] pairs. Linear interpolation between points.",
    "initial_guess_f": 0.04,
    "_initial_guess_f": "Fraction of feasible region [lower_bound, upper_bound] for initial guess"
}
```

## Files to Modify

### 1. `parameters.py` (~line 539 in OptimizationParameters dataclass)
Add new field:
```python
bounds_f_upper_points: list = None  # Optional: [[t1, v1], [t2, v2], ...] for time-varying upper bound
```

### 2. `optimization.py` - `optimize_with_iterative_refinement()` (~lines 1380-1382)

Current code:
```python
if iteration == 1:
    f_initial_guess = np.full(n_points_f, initial_guess_scalar)
    f_initial_guess[-1] = 0.0
```

New code:
```python
if iteration == 1:
    # Compute bounds at control point times
    lower_f = bounds_f[0]
    if bounds_f_upper_points is not None:
        t_base = self.base_config.scalar_params.t_base
        bound_times = np.array([p[0] for p in bounds_f_upper_points]) - t_base
        bound_values = np.array([p[1] for p in bounds_f_upper_points])
        upper_f = np.interp(f_control_times, bound_times, bound_values)
    else:
        upper_f = np.full(n_points_f, bounds_f[1])

    # Initial guess as fraction of feasible region
    f_initial_guess = lower_f + initial_guess_scalar * (upper_f - lower_f)
    f_initial_guess[-1] = lower_f  # Last point at lower bound
```

Same pattern for `s_initial_guess` (~lines 1414-1416).

### 3. `optimization.py` - `optimize_control_points_f_and_s()` (~lines 818-824)

Current code:
```python
lower_bounds = np.concatenate([np.full(n_f_points, bounds_f[0]), np.full(n_s_points, bounds_s[0])])
upper_bounds = np.concatenate([np.full(n_f_points, bounds_f[1]), np.full(n_s_points, bounds_s[1])])
```

New code:
```python
# Lower bounds (constant)
lower_bounds_f = np.full(n_f_points, bounds_f[0])
lower_bounds_s = np.full(n_s_points, bounds_s[0])

# Upper bounds for f - time-varying or constant
bounds_f_upper_points = self.base_config.optimization_params.bounds_f_upper_points
if bounds_f_upper_points is not None:
    t_base = self.base_config.scalar_params.t_base
    bound_times = np.array([p[0] for p in bounds_f_upper_points]) - t_base
    bound_values = np.array([p[1] for p in bounds_f_upper_points])
    upper_bounds_f = np.interp(f_control_times, bound_times, bound_values)
else:
    upper_bounds_f = np.full(n_f_points, bounds_f[1])

upper_bounds_s = np.full(n_s_points, bounds_s[1])

lower_bounds = np.concatenate([lower_bounds_f, lower_bounds_s])
upper_bounds = np.concatenate([upper_bounds_f, upper_bounds_s])
```

### 4. `optimization.py` - `optimize_control_points()` (~lines 636-642)
Same pattern for single-variable optimization (if used).

## Verification
1. Create a test config with `bounds_f_upper_points` that decreases over time
2. Run optimization and verify:
   - Optimized `f` values respect time-varying upper bounds
   - Initial guesses are scaled appropriately (e.g., with upper_bound=0.5 and initial_guess_f=0.04, actual initial = 0.02)
3. Compare results with and without `bounds_f_upper_points` when bounds are [0, 1] - should be identical

## Notes
- Uses simple `np.interp()` for linear interpolation (no new dependencies)
- Times in JSON are calendar years (like control points), converted to relative time internally
- If `bounds_f_upper_points` is not specified, behavior is unchanged
- Lower bounds remain constant (from `bounds_f[0]`)
