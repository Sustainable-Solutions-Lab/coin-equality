# The 2026 cleanup of the COIN_equality code

This document explains the cleanup done in October 2026 on branch `refactor-cleanup`. It covers what changed, how we made sure results did not change, and what to know when working with the cleaned-up code.

The starting point was the code of the public release (`Sustainable-Solutions-Lab/coin-equality`, commit `fb5965c3` in this repo). It had grown organically:
- three near-identical copies of the NLopt optimizer setup;
- separate code paths for optimizing f alone and f with s;
- output-variable lists maintained by hand in three or four places;
- command-line parsing duplicated in both run scripts;
- about 15 functions that nothing called;
- defaults and backward-compatibility branches that contradict the fail-fast conventions in `CLAUDE.md`.

The goal was cleaner, shorter, more readable code that gives the same numbers.

## Results are unchanged

Before changing anything, we built a regression suite (`tests/`). Its golden baseline was generated from the untouched code, and every change was checked against it before being committed.

- **Forward integrations are bit-identical** to the original code in every test case. The one exception is a diagnostic in the η = 1 case, which differs by 3.5e-16 (one unit of round-off).
- **The published results are reproduced.** Replaying the nine reference runs in `data/output/prod_s/9cases_s/` with their stored optimal controls reproduces every column of their published `*_results.csv` to 3.4e-13. The exception is `r_consumption`, whose formula was changed after those runs were made.
- **Optimizer behavior is unchanged.** The short optimization test cases follow exactly the same optimizer path, evaluation by evaluation.
- **Full re-optimization matches the manuscript.** Three of the nine reference cases were re-optimized from scratch with the new code (5 refinement iterations, 21,000–31,000 evaluations each).

  Relative differences from the published runs:

  | Case | Objective | Carbon price 2030 | Carbon price 2100 | Temperature 2100 |
  |---|---|---|---|---|
  | gid-1 (τ = 0, β = 0) | −2.8e-14 | −9.0e-7 | −8.1e-8 | +3.0e-8 |
  | gid-5 (τ = 0.5, β = 0.5) | −3.8e-15 | +1.5e-6 | +6.8e-7 | +2.4e-8 |
  | gid-9 (τ = 0.998, β = 1) | +8.1e-14 | −8.0e-8 | −3.3e-7 | −5.8e-8 |

  For example, the gid-5 carbon price in 2030 is $102.1068/tCO₂ against the published $102.1067. Over 2020–2200, log₁₀ of the carbon price agrees within 2.6e-6 and the savings rate within 4.2e-7 in all three cases.

  The objective of every refinement iteration agrees to all 13 printed digits, except the last digit of gid-9's final iteration. The number of evaluations per iteration differs slightly from the published runs, because the two machines round differently and the optimizer takes a slightly different path. It converges to the same answer.

## Size and speed

| File | Lines before | Lines after |
|---|---:|---:|
| `src/optimization.py` | 1,548 | 362 |
| `src/parameters.py` | 1,002 | 490 |
| `src/economic_model.py` | 976 | 575 |
| `src/output.py` | 880 | 213 |
| `src/visualization_utils.py` | 715 | 274 |
| `src/distribution_utilities.py` | 661 | 385 |
| `src/constants.py` | 99 | 109 |
| `src/config_cli.py` (new) | — | 108 |
| `scripts/run_optimization.py` | 677 | 178 |
| `scripts/run_integration.py` | 461 | 140 |
| **Total** | **7,019** | **2,834** |

One evaluation of the objective (a 400-year integration) now takes:
- **τ = 0:** 36 ms instead of 237 ms;
- **τ > 0:** 194 ms instead of 429 ms, where the tax root solve dominates.

Both were measured on the same machine. The main reason is that the old code built a new PCHIP interpolant of f(t) and s(t) at every time step.

Whole re-optimizations on this machine took 14, 68 and 89 minutes for gid-1, gid-5 and gid-9. The published runs took 92, 148 and 232 minutes, on different hardware.

## What changed

### Removed

- **Dead code:**
  - `gini_from_distribution`, `stepwise_integrate`, `_phi`, `L_pareto`, and the `*_and_derivative` Lorenz variants;
  - `calculate_initial_capital`, `sensitivity_analysis`, `compare_scenarios`, and a plotting function that referred to a variable that no longer exists;
  - the multi-run comparison report;
  - per-call timing instrumentation;
  - the welfare-adjusted marginal abatement cost diagnostics, which were equal to the marginal abatement cost by construction.
- **Features no published run used (decided for this cleanup):**
  - the time-point adjustment phase of the optimizer;
  - gradient-based (`LD_*`) optimization with finite-difference gradients.

  A gradient-based algorithm now raises an error instead of silently getting a zero gradient.
- **Unused configuration:**
  - the `control_function` section, whose value was always replaced;
  - `plot_short_horizon`;
  - the obsolete optimizer keys `optimize_time_points`, `max_evaluations_time_points` and `looser_epsilon`.

All other options were kept and now have test cases, even though no published run uses them:
- optimizing f with a prescribed savings rate;
- emission and consumption pulses;
- the Pareto-Lorenz income distribution;
- income-dependent aggregate damage;
- the logistic, piecewise-linear and constant time functions.

### Restructured

- **Optimizer (`src/optimization.py`):**
  - `optimize(config)` runs iterative refinement over the configured controls, with one code path whether s is optimized or prescribed.
  - One `_run_nlopt` helper replaces three copies of the NLopt setup.
  - `refinement_schedule` gives both the number of control points and the evaluation budget for each iteration.
  - `control_trajectory` is the single PCHIP interpolation of control points; it replaces two equivalent functions.
- **Model interface:** `integrate_model(config, f_values, s_values, store_detailed_output)` takes the controls as arrays on the time grid. The configuration no longer carries a control function, and the scripts no longer copy configurations by hand to swap one in.
- **Core model (`src/economic_model.py`, `src/distribution_utilities.py`):**
  - The Gauss–Legendre quadrature grid is built once per integration (`QuadratureGrid`).
  - The exogenous time functions are evaluated once over the whole time grid.
  - `calculate_tendencies` returns its tendencies and diagnostics, and `integrate_model` stores whatever diagnostics it returns. This replaces three hand-maintained lists of about 40 variable names.
  - The tax root solve works on the normalized pre-tax incomes the model already computes, instead of recomputing them.
  - A per-step lookup of the previous step's damage turned out to be exactly the identity: every quadrature node lies strictly inside its own interval, which was checked for 2 to 512 nodes. It was removed.
- **Configuration (`src/parameters.py`, `src/config_cli.py`):**
  - Time-function types are dispatched through a table of factories.
  - `configuration_from_dict` builds the configuration from a parsed dictionary, so the scripts no longer write overridden configurations to temporary files.
  - Command-line overrides are shared by both scripts.
- **Outputs (`src/output.py`, `src/visualization_utils.py`):**
  - Data files (CSV, workbook, summary, configuration) and plotting are now separate modules.
  - The two PDF report styles share their y-axis, log-scale and label logic.
  - One label table defines both the CSV headers and the plot labels.
- **Constants:** all numerical constants used by the simulation live in `src/constants.py`. These include the optimizer's control-point grid settings, the tax solver's bracket and tolerances, the temperature impulse response, and the 2020 climate state.

### Configuration schema

Every field is now required: there are no defaults or fallbacks. The settings of each optimized control moved under `optimization_parameters.controls`. The savings rate is optimized if and only if `controls` has an `s` entry; otherwise it follows `time_functions.s`.

```json
"optimization_parameters": {
  "max_evaluations_initial": 2000,
  "max_evaluations_final": 100000,
  "optimization_iterations": 5,
  "algorithm": "LN_SBPLX",
  "ftol_rel": 0.0,
  "ftol_abs": 0.0,
  "xtol_rel": 0.0,
  "xtol_abs": 1e-08,
  "controls": {
    "f": {"initial_guess": 2.5, "bounds": [-1.0, 4.0], "n_points_initial": 4, "n_points_final": 24},
    "s": {"initial_guess": 0.256, "bounds": [0.0, 1.0], "n_points_initial": 4, "n_points_final": 24}
  }
}
```

| Old key | New location |
|---|---|
| `initial_guess_f`, `bounds_f`, `n_points_initial_f`, `n_points_final_f` | `controls.f.initial_guess`, `.bounds`, `.n_points_initial`, `.n_points_final` |
| `initial_guess_s`, `bounds_s`, `n_points_initial_s`, `n_points_final_s` | `controls.s.…` (same names) |
| `ftol_rel`, `ftol_abs`, `xtol_rel` (previously optional) | Required; `0.0` disables a criterion (NLopt's default) |
| `optimize_time_points`, `max_evaluations_time_points`, `looser_epsilon`, `plot_short_horizon` | Removed |
| Impulse-response coefficients, initial cumulative emissions and heat-uptake temperatures (never set in a configuration) | Constants in `src/constants.py` |

All 14,801 configurations in `json/prod_s/`, the 9 copies in the reference run directories and the test cases were converted by a one-off script. The script is preserved in commit `6b4a7854` and was then deleted. Every converted configuration loads, and the reference cases give identical results. Command-line overrides follow the new paths, for example `--optimization_parameters.controls.f.n_points_final 12`.

## Changes in behavior

These were deliberate:

1. **Bug fix in `run_integration.py`.** In the released code, replaying an optimization output directory built its interpolants on calendar years but evaluated them at years since 2020. Every value fell before the data and was clipped to the lower bound, so f = −1 at all times and the replay was wrong. It now reproduces the optimization run to 8e-15.
2. **The configuration saved with a run includes command-line overrides.** Before, the original file was copied, so a run with overrides could not be reproduced from its own output directory.
3. **The optimization summary reports the objective unscaled** (−6.49e9 instead of −6.49e-4), as the terminal log already did.
4. **Three redundant columns are no longer in the results CSV:**
   - `Gini`, a copy of `gini`;
   - `Lambda`, the same as `lambda_abate`;
   - `dEcum_dt`, the same as `E`.

   All other columns keep their names, order and values. No figure script reads the removed ones.
5. **`run_integration.py` also writes `plots_full.pdf`**, as `run_optimization.py` does. In its configuration-file mode, a prescribed savings rate now works.

Otherwise, the results CSV and the optimization summary are byte-identical to what the old code wrote, and the distributions workbook is identical cell by cell.

## Out of scope

- **Manuscript figure scripts** (`scripts/prod-s/`) and **`parameter_estimation/`**. These were deliberately left unchanged as manuscript artifacts, and they have their own duplication that a later pass could address. They only read published outputs, plus `fig1`'s use of the Lorenz and tax functions, which are unchanged.
- **Numerical safeguards** were not altered, because the aim was identical behavior: the economic-collapse branch, the clipping of damage fractions, and the floor on cumulative emissions.

## Findings worth knowing

- **The Pareto-Lorenz option is integrated inaccurately.** Its income density is singular at the top of the distribution, so with 32 quadrature nodes the quadrature misses 2% (Gini 0.38) to 10% (Gini 0.57) of total income. The empirical Lorenz curve used in every manuscript run is integrated to within 6e-7. This needs fixing before the Pareto option is used for results.
- **Gini check inside the optimizer.** A Gini coefficient above the empirical curve's maximum raises an error inside the model. The optimizer treats such errors as infeasible trajectories, so a bad Gini path would make every evaluation infeasible instead of stopping the run. Checking the Gini path when the configuration is loaded, and giving infeasibility its own exception type, would make this fail fast.
- **How precisely re-optimizations reproduce.** Forward runs are deterministic, and reproduce across machines to about 1e-12. Re-optimizations on another machine take slightly different optimizer paths but converge to the same answer:
  - full 400-year horizon: objective to about 1e-14, controls to about 1e-6;
  - short 50-year horizon: the objective is flat near the end, so the controls agree only to about 1e-3 even though the objective agrees to 1e-10.
- **Missing batch driver.** The stored reference runs were produced by a batch driver that is not in this repository (it also wrote `run_status.csv`).

## Working with the code from here

- Run the regression suite before every commit: `.venv/bin/python -m pytest` (about a minute). The terminal summary lists the largest relative difference for each case.
- Tolerances are scaled to each variable's magnitude:
  - 1e-12 for cases without iterative solves;
  - 1e-9 for cases through the tax root solve (that solver's own tolerance);
  - for the short converged optimization, ten times the spread measured when the starting guess is perturbed.

  A pure refactoring should show differences of 0 or near round-off.
- Regenerate the baseline (`python -m tests.generate_baseline`) only for a deliberate, approved change in the numbers, and say so in the commit message. When an output variable is removed on purpose, add it to `REMOVED_OUTPUT_KEYS` in `tests/regression_cases.py`.
- `CLAUDE.md` (Testing section) and `README.md` (Configuration, Output Files, Testing) describe the details.

## Commits

| Commit | Change |
|---|---|
| `d787a7a8` | Regression test harness and golden baseline (before any change) |
| `0e881362` | Remove dead code |
| `e07aa719` | Fix `run_integration` replay; end-to-end test of both scripts |
| `e9961bf4` | Remove time-point adjustment and gradient-based optimization |
| `d51ba3af` | Shared configuration loading and command-line overrides |
| `eb331e04` | Consolidate the optimizer; controls passed to the model as arrays |
| `b2a397d1` | Core model: one quadrature grid, vectorized inputs, single output list |
| `6b4a7854` | Configuration schema without defaults; all configurations migrated |
| `026f7ac4` | Drop redundant output columns; remove the migration script |
| `f82bb129` | Separate data output from plotting |
| `97ebb762` | Slim the run scripts |
| `06d5ac0b` | Documentation (README, CLAUDE.md) |
