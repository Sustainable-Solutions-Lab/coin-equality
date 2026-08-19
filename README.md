# COIN_equality

**Income inequality increases optimal carbon price**

A stylized climate-economy model exploring how income inequality and progressive taxation affect optimal carbon pricing.

This repository accompanies the manuscript:

> Papargyri, L., & Caldeira, K. *Income inequality increases optimal carbon price.* Submitted to *Proceedings of the National Academy of Sciences* (2026). <!-- DOI to be added upon publication -->

## Table of Contents

- [Overview](#overview)
- [Installation and Requirements](#installation-and-requirements)
- [Quick Start](#quick-start)
- [Analyzing Results](#analyzing-results)
   - [Reproducing the Figures](#reproducing-the-figures)
   - [SI Data Files](#si-data-files)
   - [Data Availability](#data-availability)
- [Model Overview](#model-overview)
   - [Objective Function](#objective-function)
   - [Core Components](#core-components)
   - [Key Insights](#key-insights)
- [Configuration](#configuration)
   - [Configuration Files](#configuration-files)
   - [Policy Configuration](#policy-configuration)
   - [Tax Policy: Continuous Equity Knob](#tax-policy-continuous-equity-knob)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [License](#license)
- [Authors](#authors)

## Overview

This project develops a stylized climate-economy model that incorporates income inequality and progressive taxation into optimal carbon pricing. The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1088/1748-9326/acf949) by adding:

1. **Income distribution** - a continuous income distribution (empirical Lorenz curve) with time-varying inequality (Gini coefficient)
2. **Progressive taxation** - a continuous equal-utility-loss tax schedule parameterized by a single Tax Progressivity Parameter τ (code name: `tax_equity`)
3. **Income-dependent climate damage** - climate damage distributed across incomes via a power-law controlled by the Climate Damage Vulnerability Exponent β (code name: `y_damage_distribution_exponent`; positive values shift damage toward lower incomes; zero gives uniform damage)

The model optimizes time trajectories to maximize discounted aggregate utility, jointly optimizing the carbon pricing trajectory f(t) and the savings rate s(t) as independent control variables (free savings rate, `bounds_s = [0.0, 1.0]`). All manuscript results use this configuration.

## Installation and Requirements

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installing Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

Required packages include:
- numpy - numerical computations
- scipy - scientific computing and optimization
- pandas - data manipulation and analysis
- matplotlib - plotting and visualization
- nlopt - nonlinear optimization
- mpmath - hypergeometric functions in climate damage calculations
- openpyxl - Excel file generation

## Quick Start

### Running an Optimization

Run an optimization with a configuration file:

```bash
python scripts/run_optimization.py data/output/prod_s/9cases_s/9cases_s_gid-1_20260709_231226/9cases_s_gid-1.json 9cases_s
```

The first argument is the configuration file (stored inside each run directory alongside the results; the example above is the base case τ=0, β=0). The second argument names the output subfolder: results are written to `data/output/prod/{category}/{run_name}_YYYYMMDD-HHMMSS/`.

This finds the optimal time trajectories of f(t) = log₁₀(optimal carbon price) and s(t) = savings rate that maximize the discounted time-integral of aggregate utility.

### Running a Forward Integration

Run a forward integration using control trajectories from an optimization result:

```bash
python scripts/run_integration.py data/output/prod/{category}/{run_name}_YYYYMMDD-HHMMSS/
```

## Analyzing Results

Simulation results are in **`data/output/prod_s/`** and the manuscript figure scripts are in `scripts/prod-s/figures/` (one folder or script per figure). The final figure PDFs are included in `data/output/prod_s/figures/`.

### Reproducing the figures

| Figure | Script | Data source |
|--------|--------|-------------|
| Fig 1 (conceptual) | `scripts/prod-s/figures/fig1/plot_fig1_conceptual.py` | No simulation data (illustrative diagram) |
| Fig 2 (6 reference cases) | `scripts/prod-s/figures/fig2/plot_fig2_allcases.py` | `data/output/prod_s/9cases_s/` (included) |
| Fig 3 (contours) | `scripts/prod-s/figures/fig3/plot_fig3_contours.py` | 1,681 contour grid runs (external archive; grid values in `data/si_datafiles/fig3_data.xlsx`) |
| Fig 4 (ridgeline) + SI Fig 1 (scatter) | `scripts/prod-s/figures/fig4/plot_fig4_ridgeline.py` | Monte Carlo and parameter sweeps (external archive; run values in `data/si_datafiles/fig4_data.xlsx`) |
| Fig 5 (cloud) | `scripts/prod-s/figures/fig5/plot_fig5_cloud.py` | `outputs/audits/points_cloud_1_6_balanced.csv` (included) |
| SI Fig 2 (growth contour) | `scripts/prod-s/figures/figS/plot_si2_growth_contour.py` | `data/output/prod_s/9cases_s/` (included) |
| SI Fig 3 (DICE reference) | `scripts/prod-s/figures/plot_dice_reference_si.py` | Cached run extractions in `outputs/audits/` (included) |

Figures 1, 2, 5, SI 2, and SI 3 regenerate directly from this repository. Figures 3 and 4 require the full simulation outputs (see Data Availability). `json/prod_s/` contains the exact configuration file of **every run whose result appears in a figure** (14,801 configs), so every simulation behind the manuscript can be re-run with `scripts/run_optimization.py`.

### SI data files

`data/si_datafiles/` contains one xlsx per figure with exactly the data plotted:

| File | Contents |
|------|----------|
| `fig2_data.xlsx` | Carbon price and temperature trajectories (2025–2100) for the 6 reference cases (3 τ × 2 β) |
| `fig3_data.xlsx` | 41 × 41 grid over (τ, β): optimal carbon price 2030 and temperature change 2100 |
| `fig4_data.xlsx` | Sensitivity runs behind the ridgelines (and SI Fig 1 scatter): MC joint, ρ sweep, η sweep |
| `fig5_data.xlsx` | The plotted Monte Carlo cells (β, τ, η, ρ, r_μ, carbon price, temperature) |
| `figs2_data.xlsx` | Consumption growth rate matrix (year × income rank), 2030 cross-section, ḡ and g_μ |
| `figs3_data.xlsx` | Plotted values of the three curves (DICE-2023, COIN Gini=0, COIN empirical Gini) |

### Data availability

Included in this repository:

| Directory | Contents |
|-----------|----------|
| `data/output/prod_s/9cases_s/` | 9 reference-case runs (complete outputs: results, optimization summary, distributions xlsx, config) |
| `data/output/prod_s/figures/` | Final manuscript figure PDFs |
| `data/si_datafiles/` | Per-figure data files (see above) |
| `json/prod_s/` | Configuration files for all 14,801 figure-used runs |
| `outputs/audits/` | Fig 5 dataset and cached extractions for SI Fig 3 |

The full Monte Carlo, parameter sweep, and contour grid outputs (~22 GB) are not included. <!-- Zenodo/archive DOI to be added -->

## Model Overview

### Objective Function

The model optimizes the time-integral of aggregate utility by choosing two control trajectories:

```
max∫₀^∞ e^(-ρt) · U(t) · L(t) dt
```

where:
- ρ = pure rate of time preference
- U(t) = mean utility of the population at time t
- L(t) = population at time t

Control variables:
- f(t) = log₁₀(optimal carbon price) in $/tCO₂
- s(t) = savings rate (fraction of output invested in capital) — optimized jointly with f(t) in the free savings rate variant (`prod_s`), or prescribed exogenously in the fixed savings rate variant (`prod`)

### Core Components

The model combines three subsystems:

1. **Economic Model (Solow-Swan Growth)**
   - Cobb-Douglas production function
   - Capital accumulation with depreciation
   - Climate damage reducing output
   - Income distribution (Pareto-Lorenz or Empirical Lorenz)

2. **Climate Model**
   - Temperature proportional to cumulative emissions
   - Industrial emissions from economic activity (scaled by emission_ratio for non-CO2 GHGs)
   - Exogenous land-use emissions (Eland)
   - Abatement reducing emissions

3. **Utility and Inequality**
   - CRRA (isoelastic) utility function
   - Income-dependent climate damage distribution
   - Progressive taxation via continuous equal-utility-loss schedule

#### Empirical Lorenz Formulation

The model supports an empirical Lorenz curve formulation as an alternative to the Pareto-Lorenz distribution. The base empirical Lorenz curve is defined as:

```
L_base(F) = w₀·F^p₀ + w₁·F^p₁ + w₂·F^p₂ + w₃·F^p₃
```

where w₀ = (1 - w₁ - w₂ - w₃), and the parameters are:

| Parameter | Value |
|-----------|-------|
| p₀ | 1.500036 |
| w₁ | 0.3776187268483524 |
| p₁ | 4.367440 |
| w₂ | 0.3671247620949191 |
| p₂ | 14.072005 |
| w₃ | 0.09538538350961864 |
| p₃ | 135.059674 |

The base Gini coefficient is computed as:

```
Gini_base = 1 - 2·[w₀/(p₀+1) + w₁/(p₁+1) + w₂/(p₂+1) + w₃/(p₃+1)]
```

To construct a Lorenz curve for an arbitrary Gini coefficient G, we use linear interpolation between perfect equality and the base curve:

```
L(F) = (1 - G/Gini_base)·F + (G/Gini_base)·L_base(F)
```

This formulation is controlled by the `use_empirical_lorenz` boolean parameter in the configuration.

#### Consumption Discount Rate (Welfare-Weighted Ramsey Rule)

The model reports an effective consumption discount rate `r_consumption(t)` as a derived diagnostic. The standard Ramsey rule

```
r = ρ + η·g
```

assumes a single representative agent whose consumption grows at one rate `g`. Because the model tracks a full income distribution (via Gauss-Legendre quadrature) whose shape evolves over time, different income groups grow at different rates and there is no unique `g`. The correct social discount rate for a marginal consumption increment distributed across society — under the utilitarian welfare function `W = Σᵢ ωᵢ·u(cᵢ)` used throughout the model — is the **marginal-utility-weighted** Ramsey rule:

```
r = ρ − d/dt ln( Σᵢ ωᵢ·u'(cᵢ) ),   with CRRA  u'(c) = c^(−η)
  = ρ + η·( Σᵢ ωᵢ·cᵢ^(−η)·gᵢ ) / ( Σᵢ ωᵢ·cᵢ^(−η) )
```

where the welfare weights `ωᵢ` are the quadrature population shares and `gᵢ = ċᵢ/cᵢ` is the growth rate of income group `i`. The relevant growth rate is thus a marginal-utility-weighted average that gives more weight to lower-income (high marginal utility) groups.

Writing `cᵢ = c̄·xᵢ` (mean consumption × relative consumption), the correction relative to the representative-agent rule reduces exactly to

```
r − r_representative = − d/dt ln S,   where  S = Σᵢ ωᵢ·xᵢ^(−η) ≥ 1
```

is a pure consumption-inequality index. The correction is therefore **zero when the distribution is static** (regardless of growth) and is driven entirely by how fast inequality changes: falling inequality (the poor catching up) raises `r_consumption`, while rising inequality lowers it. For configurations with a meaningfully time-varying Gini this is a first-order effect on the discount rate (on the order of ~1 percentage point per year in early decades for a strongly evolving distribution).

### Key Insights

1. **Progressive taxation increases optimal carbon prices**: When climate damages fall disproportionately on the poor and taxation is progressive, the optimal carbon price rises because mitigating climate damage has higher marginal utility value.

2. **Inequality aversion amplifies the effect**: Higher values of η (elasticity of marginal utility of consumption) increase the welfare weight on lower-income individuals, making income-dependent damages more costly and driving higher optimal abatement.

3. **Time preference dominates carbon pricing level**: The discount rate ρ has the largest effect on the optimal carbon price, while tax progressivity (`tax_equity`) and damage distribution (`y_damage_distribution_exponent`) determine how much inequality considerations shift the optimum relative to the flat-tax baseline.

## Configuration

### Configuration Files

All model parameters are specified in JSON configuration files. See `data/output/prod_s/9cases_s/9cases_s_gid-1_20260709_231226/9cases_s_gid-1.json` for a current example (base case: τ=0, β=0).

Key configuration sections:
- `scalar_parameters` - Time-invariant constants (α, δ, η, ρ, etc.)
- `time_functions` - Time-dependent functions (A(t), L(t), σ(t), θ₁(t), gini(t), emission_ratio(t), Eland(t), etc.)
- `control_function` - Carbon pricing policy f(t) = log₁₀(optimal carbon price)
- `integration_parameters` - Time span and step size
- `optimization_parameters` - Optimization settings

#### Time Functions

The model uses several time-dependent exogenous functions specified in the `time_functions` section:

| Function | Description | Typical Type |
|----------|-------------|--------------|
| `A` | Total factor productivity | gompertz_growth |
| `L` | Population | gompertz_growth |
| `sigma` | Carbon intensity of GDP (tCO2/$) | gompertz_growth |
| `theta1` | Abatement cost coefficient ($/tCO2) | double_exponential_growth |
| `gini` | Background Gini index | exponential_growth |
| `emission_ratio` | CO2-equivalent to CO2 ratio (accounts for non-CO2 GHGs) | exponential_growth |
| `Eland` | Total land emissions (tCO2/yr) | exponential_growth |

**emission_ratio**: Converts industrial CO2 emissions to CO2-equivalent emissions by accounting for non-CO2 greenhouse gases (e.g., methane, N2O). Uses exponential decline from 1.40 in 2020 to 1.21 in 2100 (Barrage & Nordhaus 2023).

**Eland**: Exogenous land-use emissions that decline exponentially over time. At t_base=2020, Eland ≈ 4.5 GtCO2/yr with a decline rate of -2.107%/yr.

### Policy Configuration

The model's key policy parameters are (manuscript name first, code name in parentheses):

- **Tax Progressivity Parameter τ** (`tax_equity`, float, [0, 1)) — progressivity of the tax schedule (see below)
- **Climate Damage Vulnerability Exponent β** (`y_damage_distribution_exponent`, float) — how strongly climate damage falls on lower incomes. Named cases: β = 0 (uniform vulnerability: every person loses the same fraction of income), β = 0.5 (increased low-income vulnerability), β = 1 (equal absolute damage: every person loses the same dollar amount).

- **`income_dependent_aggregate_damage`** (boolean)
   - When true: aggregate damage is computed directly from the per-quantile income-weighted sum of damage fractions; richer economies experience a lower aggregate damage fraction.
   - When false: the per-quantile damage array is rescaled so the aggregate damage fraction matches `Omega_base` (temperature-only DICE-like scaling).
- **`eta`** (float) — elasticity of marginal utility of consumption
- **`rho`** (float) — pure rate of time preference

#### Tax policy: continuous equity knob

The tax schedule is the equal-utility-loss formula with a progressivity-adjusted exponent:

```
η_eff = 1 + (tax_equity / (1 − tax_equity)) · (η − 1)
```

where `tax_equity ∈ [0, 1)` and `η` is the elasticity of marginal utility of consumption.

| `tax_equity` | `η_eff`   | Resulting schedule                          | Behavior                              |
|:------------:|:---------:|:--------------------------------------------|:--------------------------------------|
| 0            | 1         | `c = y · exp(−K)`                           | Flat fractional rate (DICE-like)      |
| 0.5          | η         | Equal-utility-loss formula                  | Same utility loss for all earners     |
| → 1          | → ∞       | Steep progressive schedule                  | Tax concentrated on highest incomes   |

At `η = 1`, `η_eff = 1` for any `tax_equity` and the knob is degenerate — the tax is always a flat fractional rate.

**Tax schedule** (closed form):

- For `η_eff ≠ 1`: `c(F) = [y(F)^(1−η_eff) − (1−η_eff)K]^(1/(1−η_eff))`
- For `η_eff = 1`: `c(F) = y(F) · exp(−K)`

`K` is solved numerically via Brent's method in log-K space, in normalized income units so the solver is stable at large `η_eff` and realistic income scales. The revenue constraint is `∫₀¹ [y(F) − c(F)] dF = tax_amount`.

**Derivation:** Starting from `U = c^(1−η)/(1−η)` (so `dc/du = c^η` for equal utility loss), take a marginal tax rate `r = r0·c^(η_tax)`. Tax amount per person is `tax = r·c = r0·c^(η_tax+1)`. Iterating that infinitesimally to a finite revenue yields the ODE `dc/dt = −r0·c^(η_tax+1)`, whose closed form is the schedule above with `η_eff = η_tax + 1`. Monotonicity `dc/dy > 0` holds automatically, so no explicit iteration loop is needed.

**Usage:**

```json
"scalar_parameters": {
    "tax_equity": 0.5,
    "eta": 2.0,
    "income_dependent_aggregate_damage": false
}
```

### Output Files

Each run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD_HHMMSS/
├── {run_name}.json                      # Configuration used for the run
├── {run_name}_results.csv               # Complete time series data
├── {run_name}_optimization_summary.csv  # Optimizer convergence summary
├── {run_name}_distributions.xlsx        # Income/consumption distribution output
├── run_status.csv                       # Run status metadata
└── terminal_output.txt                  # Console output
```

## Project Structure

```
coin-equality/
├── README.md                          # This file
├── LICENSE                            # MIT (code); data files CC-BY-4.0
├── requirements.txt                   # Python dependencies
├── src/                               # Core library modules
│   ├── constants.py                   # Numerical constants and tolerances
│   ├── parameters.py                  # Parameter definitions and configuration
│   ├── distribution_utilities.py      # Income distribution and utility integration
│   ├── economic_model.py              # Economic production and tendencies
│   ├── optimization.py                # Optimization framework
│   ├── output.py                      # Output generation (CSV and PDF)
│   └── visualization_utils.py         # Unified visualization functions
├── scripts/
│   ├── run_optimization.py            # Main optimization script
│   ├── run_integration.py             # Forward integration from optimization results
│   └── prod-s/                        # Manuscript figure pipeline
│       ├── label_utils.py             # Interactive label positioning for figures
│       └── figures/                   # Publication figure scripts
│           ├── fig1/                  # Fig 1: conceptual diagram
│           ├── fig2/                  # Fig 2: 6 reference cases
│           ├── fig3/                  # Fig 3: contour plots (τ × β)
│           ├── fig4/                  # Fig 4: ridgeline + SI Fig 1 scatter
│           ├── fig5/                  # Fig 5: cloud scatter
│           ├── figS/                  # SI Fig 2: growth contour
│           └── plot_dice_reference_si.py  # SI Fig 3: DICE-2023 reference
├── json/prod_s/                       # Configs for all 14,801 figure-used runs
│   ├── 9cases_s/  prod_contour_s/  mc/  single_param/  dice_comparison/
├── data/
│   ├── output/prod_s/
│   │   ├── 9cases_s/                  # 9 reference-case runs (complete outputs)
│   │   └── figures/                   # Final manuscript figure PDFs
│   └── si_datafiles/                  # Per-figure data files (xlsx)
├── outputs/audits/                    # Fig 5 dataset + SI Fig 3 cached extractions
└── parameter_estimation/              # Offline data-fitting for model constants
```

## License

MIT License

Copyright (c) 2026 Lamprini Papargyri and Ken Caldeira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Code is licensed under the MIT License (above; see also [LICENSE](LICENSE)). Data files (`data/`, `outputs/`) are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) — please cite the paper when reusing them.

## Authors

Lamprini Papargyri and Ken Caldeira
