# Parameter Estimation Utilities

This directory holds offline data-preparation and curve-fitting scripts that produce numerical inputs used by the COIN_equality simulation. None of these scripts are imported by `src/` — they are run by hand, and their outputs are pasted (or vendored) into `src/constants.py` and the run configuration JSON files that the simulation reads.

The goal is **provenance**: every empirical constant in the model should be traceable to a script + input data file inside this repository, so the derivation is reproducible without cloning a second repo.

## Layout

```
parameter_estimation/
├── empirical_distribution/                # Upstream pipeline
│   ├── empirical_global_distribution.py   # builds the global Lorenz coordinates
│   ├── data/
│   │   └── pip_2025-12-28.xlsx            # per-country decile data (PIP / World Bank)
│   └── output/                            # generated at runtime
│
├── lorenz_fit/                            # Polynomial Lorenz fit
│   ├── fit_polynomial_lorenz.py           # fits a convex-combination polynomial
│   ├── data/
│   │   └── empirical_distribution_global.csv  # frozen snapshot of upstream output
│   └── output/                            # generated at runtime
│
├── global_gini/                           # SSP-based global Gini time series
│   ├── global_gini.py
│   ├── data/                              # drop SSP input CSVs here (see below)
│   └── output/                            # generated at runtime
│
└── dice_timefunctions/                    # Gompertz / exponential fits to
    ├── fit_dice_timefunctions.py          # Barrage & Nordhaus 2023 DICE-2023
    └── output/                            # time series for A, L, sigma
                                           # (emission_ratio, Eland to follow)
```

Each subdirectory is one self-contained utility. Scripts use script-relative paths (`Path(__file__)`) so they can be run from any working directory.

## What each utility produces, and where its output ends up in the model

| Utility | Output of the script | Where it lands in the model |
|---|---|---|
| `empirical_distribution/empirical_global_distribution.py` | `output/empirical_distribution_global.csv` (global Lorenz curve coordinates aggregated across all countries × deciles, plus a printed empirical global Gini) | Frozen as the input to `lorenz_fit/`. Not consumed directly by the simulation. |
| `lorenz_fit/fit_polynomial_lorenz.py` | `output/polynomial_fit_results_<method>_<timestamp>.csv` (best-fit weights w_k and powers p_k for polynomials of various degrees, plus goodness-of-fit metrics) | The degree-4 convex-combination row becomes `EMPIRICAL_LORENZ_P0..P3` and `EMPIRICAL_LORENZ_W1..W3` in `src/constants.py`. |
| `global_gini/global_gini.py` | `output/global_gini_table.csv` (global Gini by year × SSP) and `output/global_gini_trends.png` | The SSP2 trajectory is fit (separately, by hand or future script) to an exponential form; the resulting `exponential_scaling`, `growth_rate`, `additive_constant` populate the `gini` time-function block in the run configuration JSON files. |
| `dice_timefunctions/fit_dice_timefunctions.py` | `output/dice_timefunctions_fits.json` (JSON-ready Gompertz blocks) + `output/dice_timefunctions_report.txt` (per-step B&N-vs-fit table with R², RMSE, max relative error per variable) | The three Gompertz blocks `initial_value` / `final_value` / `adjustment_coefficient` for `A`, `L`, `sigma` are pasted into the `time_functions` section of the run configuration JSON files. Same scaffolding will host the exponential fits for `emission_ratio` and `Eland` when those are added. |

## Data flow

```
empirical_distribution/data/pip_2025-12-28.xlsx
      │
      │  empirical_global_distribution.py
      ▼
empirical_distribution/output/empirical_distribution_global.csv
      │
      │  (manually copy snapshot →)
      ▼
lorenz_fit/data/empirical_distribution_global.csv
      │
      │  fit_polynomial_lorenz.py --use-convex-combination --min-degree 4 --max-degree 4
      ▼
lorenz_fit/output/polynomial_fit_results_convex_combination_<timestamp>.csv
      │
      │  (manually transcribe degree-4 row →)
      ▼
src/constants.py :: EMPIRICAL_LORENZ_P{0..3}, EMPIRICAL_LORENZ_W{1..3}


global_gini/data/{All_GDP_percapita, pop_ssp_database, ISO_level_projections_PC_model_projections}.csv
      │
      │  global_gini.py
      ▼
global_gini/output/global_gini_table.csv
      │
      │  (separately fit SSP2 row to exponential_growth form →)
      ▼
run configuration JSON :: time_functions.gini.{exponential_scaling, growth_rate, additive_constant}
```

The "manual copy" and "manual transcribe" arrows are deliberate. Treating them as explicit steps preserves a **frozen snapshot** of the inputs that produced the published constants, so reviewers can re-run the fit on the exact same data even if the upstream pipeline is later refreshed.

## How to run

```bash
# 1. Build the empirical global distribution from per-country decile data
python parameter_estimation/empirical_distribution/empirical_global_distribution.py

# 2. Fit the polynomial Lorenz curve to the (frozen) empirical distribution
python parameter_estimation/lorenz_fit/fit_polynomial_lorenz.py \
    --use-convex-combination --min-degree 4 --max-degree 4

# 3. Compute the global Gini time series from SSP projections
#    (first drop the three SSP CSVs into global_gini/data/ — see below)
python parameter_estimation/global_gini/global_gini.py
```

To refresh the polynomial Lorenz constants from new data:

```bash
# regenerate the upstream distribution
python parameter_estimation/empirical_distribution/empirical_global_distribution.py
# overwrite the frozen snapshot with the new output
cp parameter_estimation/empirical_distribution/output/empirical_distribution_global.csv \
   parameter_estimation/lorenz_fit/data/empirical_distribution_global.csv
# rerun the fit
python parameter_estimation/lorenz_fit/fit_polynomial_lorenz.py \
    --use-convex-combination --min-degree 4 --max-degree 4
# then transcribe the new (p_k, w_k) into src/constants.py
```

## Required input files (not all checked into git)

`global_gini/data/` must contain (case-sensitive filenames):

```
All_GDP_percapita.csv
pop_ssp_database.csv
ISO_level_projections_PC_model_projections.csv
```

`empirical_distribution/data/pip_2025-12-28.xlsx` and `lorenz_fit/data/empirical_distribution_global.csv` are vendored and present.

## Provenance of vendored scripts

| Script | Vendored from | Notes on changes during vendoring |
|---|---|---|
| `empirical_global_distribution.py` | `~/global-lorenz/empirical_global_distribution.py` | Inlined the two helpers (`read_country_data`, `filter_most_recent_complete`) that the original imported from the `global_lorenz.country_fitting` package, so no upstream package needs to come along. Switched hard-coded paths to script-relative. Replaced deprecated `np.trapz` with `np.trapezoid`. |
| `fit_polynomial_lorenz.py` | `~/global-lorenz/fit_polynomial_lorenz.py` | Self-contained in upstream (no `global_lorenz` package imports). Only change: argparse defaults for `--input` and `--output-dir` now use script-relative paths via `Path(__file__)` so the script works from any CWD. The script already used `np.trapezoid`. |
| `global_gini.py` | User-supplied standalone script | Switched `/mnt/data/...` paths to script-relative. Replaced deprecated `np.trapz` with `np.trapezoid`. |

The upstream `~/global-lorenz` repository contains the full Lorenz-fitting workflow (country-level fits, multiple Lorenz functional forms, summary reports). Only the polynomial-Lorenz fit and its immediate upstream are vendored here, because that is what produces the constants the paper uses. The rest of the upstream workflow is not in the paper's critical path and is left in place.
