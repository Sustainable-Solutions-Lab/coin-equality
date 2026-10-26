# COIN_equality: detailed methods

This document specifies the methods behind this repository precisely enough to reimplement it from scratch. The aim is to reproduce its numbers to within about two orders of magnitude of floating-point round-off, or to within the convergence tolerance of an iterative step where that tolerance is looser. [README.md](../README.md) covers usage, and [REFACTOR.md](../REFACTOR.md) covers how the code reached its current form. Here, every formula, constant, branch condition and numerical procedure that affects a result is written out, along with how to check a reimplementation.

The code is the reference. Each section names the function that implements it as `file:function`.

## Contents

1. [What can be reproduced, and how closely](#1-what-can-be-reproduced-and-how-closely)
2. [Software environment](#2-software-environment)
3. [Conventions](#3-conventions)
4. [Inputs](#4-inputs)
5. [Quadrature over population rank](#5-quadrature-over-population-rank)
6. [Lorenz curves](#6-lorenz-curves)
7. [One time step](#7-one-time-step)
8. [Time integration](#8-time-integration)
9. [Derived output variables](#9-derived-output-variables)
10. [Objective](#10-objective)
11. [Control trajectories](#11-control-trajectories)
12. [Optimization](#12-optimization)
13. [Manuscript run families](#13-manuscript-run-families)
14. [Output files](#14-output-files)
15. [Parameter estimation](#15-parameter-estimation)
16. [Figures and SI data](#16-figures-and-si-data)
17. [Verifying a reimplementation](#17-verifying-a-reimplementation)
18. [Numerical caveats](#18-numerical-caveats)

---

## 1. What can be reproduced, and how closely

The pipeline has three stages, and each can be reproduced to a different precision.

| Stage | Deterministic? | Attainable agreement |
|---|---|---|
| **Forward integration** with given control arrays f(t), s(t) (Sections 4–9) | Yes | Bit-identical if the formulas are evaluated as written (see "Evaluation order" below), including branch conditions and clipping. A clean-room reimplementation from this document achieved that (Section 17.1). Other algebraically equivalent orderings give 1e-16 to 3e-12. Steps with a progressive tax (τ > 0) go through a Brent root solve with tolerance 1e-9 in log ΔU. Using the same algorithm (Section 7.8) gives round-off agreement; a different root finder can differ by up to about 1e-9. |
| **Objective** (Section 10) | Yes | Same as the forward integration. |
| **Optimization** (Sections 11–12) | Deterministic for fixed inputs, but chaotic | SBPLX is deterministic, but the path it takes depends on every bit of every objective value. A reimplementation that differs only by round-off takes a different path, so it ends at a slightly different point. Re-optimizing manuscript runs with the 2026 code, whose evaluations differ from the original runs only by round-off, gave:<br>• objectives within 1e-13 relative;<br>• controls within about 3e-6 in f = log10 MAC;<br>• MAC in 2030 and 2100, and ΔT in 2100, within 2e-6 relative (REFACTOR.md). |

**Evaluation order.** Formulas are written in the order the code evaluates them:
- each product, sum or quotient is evaluated left to right, with parenthesized sub-expressions first;
- population means are `np.sum(Fwi * x)` (NumPy pairwise summation), not `np.dot`;
- powers are `**` unless stated otherwise.

Re-associating a formula changes results at the level of round-off. In the clean-room test, the following each cost 1e-16 to 3e-12 relative: computing σ·(y_gross·emission_ratio) instead of (σ·y_gross)·emission_ratio; computing y_net as y_damaged − abateCost; computing ΔT² as ΔT·ΔT; using `np.dot` for means. The tax-rate diagnostics are the most sensitive outputs, and the state variables stay near 1e-14.

The published optimal controls are stored at full precision in the results CSVs (Section 14). The reproducible core of every published result is therefore:

1. replay those controls through the forward model, which reproduces every column of the published time series to 3.4e-13 relative or better (`r_consumption` with the formula given in Section 9);
2. check the optimum by re-optimizing, which reproduces it within the convergence spread above.

Some tooling is not in the repository, nor in its development history (branch `dev-history`):
- the batch driver that launched the production runs, wrote `run_status.csv` and supplied warm starts;
- the generators of the parameter samples;
- the scripts that built the curated tables in `outputs/audits/` and the SI data files in `data/si_datafiles/`.

Sections 13 and 16 reconstruct what those tools did from their outputs. Figures 3 and 4 need run outputs from the external archive, which holds about 22 GB; only the 9 reference runs are in the repository.

## 2. Software environment

The reference results and the regression baseline (`tests/baseline/baseline_meta.json`) were produced with:

| Package | Version | Used for |
|---|---|---|
| Python | 3.10.12 | |
| NumPy | 2.2.6 | all array arithmetic; `np.trapezoid` (NumPy ≥ 2.0) |
| SciPy | 1.15.3 | `scipy.special.roots_legendre`, `scipy.optimize.brentq` and `root_scalar(method='brentq')`, `scipy.interpolate.PchipInterpolator` |
| NLopt (Python bindings) | 2.11.0 in the test environment; the version used for the 2026 production runs is not recorded | `LN_SBPLX` optimizer |
| pandas, openpyxl, matplotlib | 2.3.3, 3.1.5, 3.10.9 | output files and plots only |

All arithmetic is IEEE-754 double precision. Per-step scalars are NumPy `float64` values taken from arrays, and per-quantile quantities are `float64` arrays of length `n_quad`.

## 3. Conventions

- **Time.** The integration time `t` is the calendar year, a float on the grid `t_start, t_start + dt, …, t_end` (Section 8). Time functions and control times use relative time `t_rel = t − t_base`, with `t_base = 2020` in every configuration.
- **Per-capita vs. total.** Lowercase names are per capita (`y_gross`, `y_net`, `e`, `abateCost_amount`, `tax_amount`), and uppercase names are totals (`Y_gross = y_gross·L`, `E = e·L`).
- **Units.**
  - Money: 2019 US dollars (DICE-2023 convention).
  - Income: $/person/yr.
  - Emissions: tCO2 (cumulative) and tCO2/yr (rates).
  - Temperature: °C above preindustrial.
  - The control `f` is log10 of the marginal abatement cost in $/tCO2.
- **Population rank.** `F ∈ [0, 1]` is the population rank ordered by income (F = 0 poorest).
- **`ε`** denotes `EPSILON = 1e-12` throughout.

## 4. Inputs

A run is fully specified by one JSON configuration file plus the fixed constants in `src/constants.py`. Every configuration field is required: there are no defaults. Keys starting with `_` are comments and are ignored. The parser is `src/parameters.py:configuration_from_dict`.

### 4.1 Scalar parameters (`scalar_parameters`)

The manuscript runs all use the following values. Only η, ρ, τ and β vary between runs (Section 13).

| Key | Symbol | Manuscript value | Meaning |
|---|---|---|---|
| `alpha` | α | 0.3 | capital share (Cobb-Douglas) |
| `delta` | δ | 0.1 yr⁻¹ | capital depreciation |
| `psi1` | ψ₁ | 0 °C⁻¹ | linear damage coefficient |
| `psi2` | ψ₂ | 0.003467 °C⁻² | quadratic damage coefficient (DICE-2023) |
| `eta` | η | 1.5 (varied) | elasticity of marginal utility (CRRA) |
| `rho` | ρ | 0.025 yr⁻¹ (varied) | pure rate of time preference |
| `theta2` | θ₂ | 2.6 | abatement cost exponent |
| `mu_max` | μ_max | 1e20 | cap on the abatement fraction (effectively none; μ > 1 means net-negative emissions) |
| `K_initial` | K₀ | 2.95e14 $ | capital in t_start |
| `y_net_reference` | y_ref | 17210.79 $/person/yr | reference income of the damage power law |
| `y_damage_distribution_exponent` | β | 0, 0.5, 1 (varied) | damage vulnerability exponent |
| `tax_equity` | τ | 0, 0.5, 0.998 (varied) | tax progressivity, must be in [0, 1) (checked at load) |
| `income_dependent_aggregate_damage` | | false | see Section 7.11 |
| `use_empirical_lorenz` | | true | empirical (true) or Pareto (false) Lorenz curve |
| `t_base` | | 2020 | time origin of the time functions |
| `year_extra_emission`, `amount_extra_emission` | | 0, 0 | emission pulse (tCO2) in the step containing that year |
| `year_extra_consumption`, `amount_extra_consumption` | | 0, 0 | consumption pulse ($) in the step containing that year |
| `loose_epsilon` | | 1e-9 | `xtol` of the tax root solve (log ΔU units) |

### 4.2 Time functions (`time_functions`)

Each entry has a `type` and that type's arguments, and is evaluated at `t_rel = t − t_base`. They are evaluated once, as NumPy arrays over the whole time grid (`src/parameters.py:evaluate_time_functions`); per-step values are elements of those arrays. The six types, with the arithmetic written in evaluation order:

| `type` | Arguments | Formula |
|---|---|---|
| `constant` | `value` | `value` |
| `exponential_growth` | `exponential_scaling`, `growth_rate`, `additive_constant` | `exponential_scaling * exp(growth_rate * t) + additive_constant` |
| `logistic_growth` | `L0`, `L_inf`, `growth_rate` | `L_inf / (1 + ((L_inf / L0) - 1) * exp(-growth_rate * t))` |
| `piecewise_linear` | `time_points`, `values` | `np.interp(t, time_points, values)` (constant beyond the ends) |
| `double_exponential_growth` | `initial_value`, `growth_rate_1`, `growth_rate_2`, `fract_1` | `initial_value * (fract_1 * exp(growth_rate_1 * t) + (1 - fract_1) * exp(growth_rate_2 * t))` |
| `gompertz_growth` | `initial_value`, `final_value`, `adjustment_coefficient` | `final_value * exp(log(initial_value / final_value) * exp(adjustment_coefficient * t))` |

The model requires these functions: `A`, `L`, `sigma`, `theta1`, `gini`, `emission_ratio`, `Eland`, and also `s` when the savings rate is prescribed rather than optimized. Manuscript values (identical in all production configurations, except `gini` in the DICE-comparison family):

| Name | Meaning (units) | Type | Arguments |
|---|---|---|---|
| `A` | total factor productivity (units such that A·(K/L)^α is in $/person/yr) | gompertz | initial 739.619, final 4906815.811, c = −0.0015 |
| `L` | population (persons) | gompertz | initial 7.752e9, final 1.0825e10, c = −0.03133 |
| `sigma` | CO2 intensity of output (tCO2/$) | gompertz | initial 0.000291355, final 4.4681e-05, c = −0.008164 |
| `theta1` | backstop price, the MAC at μ = 1 ($/tCO2) | double exponential | initial 6035.6, g₁ = −0.224, g₂ = −0.00112, fract₁ = 0.906 |
| `gini` | background Gini of pre-damage income | exponential | scaling 0.19517, rate −0.01058, constant 0.37796 (G(2020) = 0.57313, G(∞) = 0.37796); DICE-comparison runs: scaling 0, constant 0 (G ≡ 0) |
| `emission_ratio` | CO2-equivalent / industrial CO2 | exponential | scaling 1.4, rate −0.002085, constant 0 |
| `Eland` | land-use emissions (tCO2/yr) | exponential | scaling 5.9e9, rate −0.02107, constant 0 |

Section 15 gives the provenance of these values.

### 4.3 Integration parameters (`integration_parameters`)

| Key | Manuscript value |
|---|---|
| `t_start`, `t_end` | 2020.0, 2420.0 (calendar years) |
| `dt` | 1.0 yr |
| `n_quad` | 32 |

### 4.4 Optimization parameters (`optimization_parameters`)

| Key | Manuscript value |
|---|---|
| `max_evaluations_initial`, `max_evaluations_final` | 2000, 100000 |
| `optimization_iterations` | 5 |
| `algorithm` | `LN_SBPLX` |
| `ftol_rel`, `ftol_abs`, `xtol_rel` | 0.0 (disabled) |
| `xtol_abs` | 1e-8 |
| `controls.f` | `initial_guess` 2.5, `bounds` [−1, 4], `n_points_initial` 4, `n_points_final` 24 |
| `controls.s` | `initial_guess` 0.256, `bounds` [0, 1] ([0.2, 0.4] in the 2 runs of `mc_infill_lt1_ws3`), `n_points_initial` 4, `n_points_final` 24 |

`n_points_final` may instead be a list of calendar years, which are used as the control times. This requires `optimization_iterations = 1` (checked at load); the list is sorted at load. With no `controls.s` entry, the savings rate follows `time_functions.s`.

### 4.5 Fixed constants (`src/constants.py`)

| Constant | Value | Use |
|---|---|---|
| `EPSILON` (ε) | 1e-12 | thresholds and clips (Section 7) |
| `NEG_BIGNUM` | −1e30 | utility of a collapsed economy; objective of an infeasible trajectory |
| `OBJECTIVE_SCALE` | 1e-13 | objective scaling at the NLopt interface |
| `EMPIRICAL_LORENZ_P0..P3` | 1.500036, 4.367440, 14.072005, 135.059674 | Lorenz exponents |
| `EMPIRICAL_LORENZ_W1..W3` | 3.776187268483524e-01, 3.671247620949191e-01, 9.538538350961864e-02 | Lorenz weights |
| `EMPIRICAL_LORENZ_W0` | `1.0 - W1 - W2 - W3` = 0.15987112754710983 | |
| `EMPIRICAL_LORENZ_BASE_GINI` (G_b) | `1.0 - 2.0*(W0/(P0+1.0) + W1/(P1+1.0) + W2/(P2+1.0) + W3/(P3+1.0))` = 0.681279513926459 | |
| `OMEGA_BASE_MAX` | 0.8 | cap on the temperature damage fraction |
| `IMPULSE_RESPONSE_A1, A2, A3` | −0.629e-12, 0.203e-12, −0.0521e-12 °C/tCO2 | temperature impulse response (Ricke & Caldeira 2014, median) |
| `IMPULSE_RESPONSE_TAU1, TAU2, TAU3` | 2.241, 35.750, 97.180 yr | |
| R_∞ | `-(A1 + A2 + A3)` = 4.781000000000001e-13 °C/tCO2 | equilibrium warming per unit cumulative emission |
| `DELTA_T_HIST_EQUILIBRIUM` | 1.1126 °C | |
| `ECUM_INITIAL` | `1.1126 / -(A1 + A2 + A3)` = 2327128215854.4233 tCO2 | cumulative emissions in t_start |
| `H1_INITIAL, H2_INITIAL, H3_INITIAL` | −0.07022, 0.21036, −0.08308 °C | heat-uptake mode temperatures in t_start |
| `WIND_DOWN_EFOLDING_TIMES` | 4.0 | control-time grid (Section 11.1) |
| `MAX_POINTS_WITH_ONE_WIND_DOWN_POINT` | 10 | |
| `GEOMETRIC_GROWTH_RATE_BRACKET` | (−0.5, 100.0) | |
| `CONTROL_TIME_DUPLICATE_TOLERANCE` | 1e-10 | PCHIP (Section 11.2) |
| `TAX_LOG_DELTAU_BRACKET_MARGIN` | 50.0 | tax root solve (Section 7.8) |
| `TAX_DELTAU_MAX` | 1e10 | |
| `TAX_MONOTONICITY_TOLERANCE` | `EPSILON * 1e3` = 1e-9 | |

**The 2020 climate state** comes from the manuscript's historical warming fit,

ΔT_hist(t) = 1.1126 − 0.07022·e^((2020−t)/τ₁) + 0.21036·e^((2020−t)/τ₂) − 0.08308·e^((2020−t)/τ₃).

Its constant term is R_∞·Ecum(2020), which fixes `ECUM_INITIAL`. The other three terms are the mode temperatures in 2020, which give `H1..H3_INITIAL`. They were obtained by convolving an exponential emission history (633.5 GtC cumulative to 2020, DICE-2023) with the impulse response. Initial warming: ΔT(2020) = 1.1126 − 0.07022 + 0.21036 − 0.08308 = 1.16966 °C.

## 5. Quadrature over population rank

`src/distribution_utilities.py:quadrature_grid(n_quad)`.

- `xi, wi = scipy.special.roots_legendre(n_quad)`: Gauss-Legendre nodes and weights on [−1, 1], weights summing to 2.
- `Fi = (xi + 1.0) / 2.0`: nodes on [0, 1]. `Fwi = wi / 2.0`: weights on [0, 1], summing to 1.
- `xi_edges = concatenate(([-1.0], -1.0 + cumsum(wi)))` and `Fi_edges = (xi_edges + 1.0) / 2.0`. These are the edges of the population interval that node i represents (its width is its weight). They are output only.

Population means are `Σᵢ Fwi[i]·x[i]`, computed as `np.sum(Fwi * x)`. For n_quad = 32, `Fi[0]` = 0.001368069075259215, `Fi[-1]` = 0.9986319309247408, and `Fwi[0]` = 0.00350930500473721.

By the Chebyshev–Markov–Stieltjes separation theorem, each node lies strictly inside its interval [Fi_edges[i], Fi_edges[i+1]]. Node values therefore double as stepwise values on the intervals. This is why the lagged per-quantile damage (Section 7.2) is simply carried node by node.

## 6. Lorenz curves

The income share of the poorest fraction F of the population is L(F). Income at rank F is proportional to dL/dF, which has mean 1 over F.

### 6.1 Empirical Lorenz curve (all manuscript runs)

Base curve and derivative (`L_empirical_lorenz_base`, `L_empirical_lorenz_base_derivative`):

L_b(F) = Σₖ wₖ·F^pₖ,  dL_b/dF = Σₖ wₖ·pₖ·F^(pₖ−1),  k = 0..3

Each is evaluated as the explicit sum of the four terms in the order k = 0, 1, 2, 3. The derivative is computed as `W_k * P_k * (F ** (P_k - 1.0))`, with F clipped to [ε, 1].

For a Gini coefficient G, the curve is mixed linearly with perfect equality (`L_empirical_lorenz_derivative`):

a = G / G_b,  dL/dF = (1.0 − a) + a·dL_b/dF

This has Gini exactly G and is valid for 0 ≤ G ≤ G_b = 0.681279513926459. A per-step check raises `ValueError` if G > G_b while `use_empirical_lorenz` is true. With G = 0 (the DICE-comparison runs), dL/dF ≡ 1.

The quadrature error of the income mean at n_quad = 32 is Σ Fwi·dL/dF(Fi) − 1 = 3.2e-7 at G = 0.3, 6.05e-7 at G = 0.573 and 7.2e-7 at G = 0.6812. The integrand has a steep top tail (p₃ = 135), which causes the error. The model does not renormalize: this error is part of the published numbers.

### 6.2 Pareto-Lorenz curve (`use_empirical_lorenz = false`; not used for any result)

a = (1 + 1/G)/2, which requires 0 < G < 1. Then dL/dF = (1 − 1/a)·(1 − F_c)^(−1/a), with F_c = clip(F, ε, 1 − ε) (`L_pareto_derivative`).

The density is singular at F → 1, and at n_quad = 32 the quadrature loses 2–10% of income. See Section 18.

## 7. One time step

`src/economic_model.py:calculate_tendencies`. The inputs are:
- the state: K, Ecum, H1, H2, H3;
- the scalar parameters;
- this step's values of t, A, L, σ, θ₁, G, emission_ratio, Eland, f and s;
- the damage ratios carried from the previous step: the per-quantile array qᵢ (`omega_yi_Omega_base_ratio_prev`) and the aggregate ratio Q (`Omega_Omega_base_ratio_prev`).

The operations below are listed in the order in which the code performs them.

### 7.1 Temperature

ΔT = R_∞·Ecum + H1 + H2 + H3, with R_∞ = −(a₁ + a₂ + a₃) computed in that order.

### 7.2 Temperature damage, lagged distribution

- Ω_base = min(0.8, ψ₁·ΔT + ψ₂·ΔT²). `ΔT ** 2` is computed as a power.
- Ω_calc = clip(Q·Ω_base, 0, 1 − ε).
- ω_calc,i = clip(qᵢ·Ω_base, 0, 1 − ε).

The temperature dependence of damage is current, but its distribution over income is the previous step's.

### 7.3 Production

- y_gross = A·(K/L)^α if K > 0 and L > 0, else 0.
- y_damaged = y_gross·(1 − Ω_calc).
- climate_damage = Ω_calc·y_gross.

### 7.4 Collapse branch

If y_gross ≤ ε or y_damaged ≤ ε, the economy has collapsed:
- abateCost = tax = 0, log ΔU = −∞;
- aggregate_utility = U = NEG_BIGNUM. Note that U is *not* multiplied by L in this branch;
- Ω = λ = y_net = μ = e = E_extra = consumption_extra = 0;
- dK/dt = −δK;
- all per-quantile arrays (ω, y_net,i, utility, tax rates) are 0.

The step then continues at Section 7.13. This branch is not reached in the nine reference runs.

### 7.5 Abatement

- e_land = Eland / L.
- e_pot = σ·y_gross·emission_ratio + e_land (tCO2/person/yr, land emissions included).
- MAC = 10.0^f.
- If e_pot > ε and MAC > ε:
  - μ = (MAC/θ₁)^(1/(θ₂ − 1)), then μ = min(μ_max, μ);
  - abateCost = (θ₁/θ₂)·e_pot·μ^θ₂;
  - abateCost = min(abateCost, (1 − ε)·y_damaged).
- Otherwise μ = 0 and abateCost = 0.
- λ = abateCost / y_damaged (the guard `y_damaged > ε` is always true here).
- y_net = (1 − λ)·y_damaged.

This is the DICE cost curve, with marginal cost θ₁·μ^(θ₂−1) = MAC; the control sets the marginal cost directly. Abatement is a per-capita cost paid from damaged output, and land emissions are abated at the same fraction μ.

### 7.6 Pulses

A pulse dated `year` falls in the step [t, t + dt) when `year >= t and year < t + dt` (`pulse_in_step`).

- **Consumption pulse:** consumption_extra = amount_extra_consumption / (L·dt) in that step, else 0.
- **Tax:** tax_amount = abateCost − consumption_extra (per capita). The consumption pulse thus reduces the tax.
- **Emission pulse:** E_extra = amount_extra_emission / dt in that step, else 0. When `amount_extra_emission == 0`, E_extra = 0 without the year test.

### 7.7 Pre-tax income distribution

- dL/dF(Fᵢ) from Section 6, using G of this step.
- y_pretax,i = y_gross·dL/dF(Fᵢ)·(1 − ω_calc,i).
- The tax is solved in units of mean damaged income: ŷᵢ = y_pretax,i / y_damaged.

The mean of y_pretax,i equals y_damaged only approximately. Two effects cause the difference: the quadrature error of Section 6.1, and the fact that the lagged per-quantile damage ω_calc,i was rescaled with post-tax weights in the previous step (Section 7.11) while Ω_calc is applied as an aggregate. The model uses these quantities as they are.

### 7.8 Progressive tax (equal-utility-loss family)

**Tax exponent.** η_eff = 1 + (τ/(1 − τ))·(η − 1). It is computed as `1.0 + (tax_equity / (1.0 - tax_equity)) * (eta - 1.0)`. τ = 0 gives η_eff = 1 (flat rate), and τ = 0.5 gives η_eff = η (equal utility loss). For η = 1.5 and τ = 0.998, η_eff = 250.5.

**Schedule.** For a parameter ΔU > 0 (passed as log ΔU), post-tax normalized income c(ŷ) comes from `compute_post_tax_income_equal_utility`:

- log ΔU = −∞ (no tax): c = ŷ.
- |η_eff − 1| < ε: c = ŷ·exp(−exp(log ΔU)).
- η_eff > 1, with a = η_eff − 1, computed in log space:
  - log_y = log ŷ;
  - s = log(a) + log ΔU + a·log_y;
  - log c = log_y − logaddexp(0, s)/a;
  - c = exp(log c) where ŷ > 0, else 0.

  This is algebraically c = [ŷ^(1−η_eff) + a·ΔU]^(−1/a).
- η_eff < 1:
  - ΔU = exp(log ΔU);
  - inner = max(ŷ^(1−η_eff) − (1 − η_eff)·ΔU, 0);
  - c = inner^(1/(1−η_eff)).

**Revenue root solve.** `find_log_deltaU_equal_utility(ŷ, T̂, η_eff, Fwi, tol = loose_epsilon)` takes the normalized target T̂ = tax_amount / y_damaged. It is called only when tax_amount > ε; otherwise log ΔU = −∞.

- **Flat tax**, when |η_eff − 1| < ε: r = min(T̂, 1 − ε), ΔU = −log(1 − r), and the result is log ΔU = log(ΔU). This closed form assumes Σ Fwi·ŷ = 1. The revenue actually raised is r·Σ Fwi·ŷ, which differs from the target by the relative error of Section 7.7.
- **Otherwise:**
  1. Revenue residual: g(x) = Σ Fwi·(ŷ − c(ŷ; x)) − T̂.
  2. Bracket, with a = |η_eff − 1|:
     - lower bound: x_lo = −|a·log(max ŷ)| − 50;
     - upper bound: x_hi = log(1e10).
  3. If g(x_lo) ≥ 0, return −∞ (no tax).
  4. If g(x_hi) < 0, raise `ValueError`: the revenue cannot be raised.
  5. Solve `scipy.optimize.root_scalar(g, bracket=[x_lo, x_hi], method='brentq', xtol=tol)`. SciPy's defaults apply: rtol = 4·2⁻⁵² ≈ 8.88e-16 and maxiter = 100. Non-convergence raises `RuntimeError`.
  6. Post-solve checks raise `RuntimeError` if c is non-finite, or if any `np.diff(c) < −1e-9`.

  For τ = 0.998, log ΔU is about −670 to −460 (Section 17.2). Working in log space avoids the exp() underflow that would otherwise silently switch the tax off.

**Post-tax income:** y_net,i = c(ŷᵢ; log ΔU)·y_damaged.

### 7.9 Tax-rate diagnostics

These are output only.

- c/y = y_net,i / max(y_pretax,i, ε) where y_pretax,i > ε, else 1.
- Average rate: 1 − c/y.
- Marginal rate: 1 − max(c/y, ε)^η_eff where c/y > ε, else 1.
- `max_average_tax_rate` and `max_marginal_tax_rate` are the maxima over nodes.

### 7.10 Utility

- consumption_i = y_net,i·(1 − s).
- If `eta == 1` (exact comparison): uᵢ = log(max(consumption_i, ε)).
- Otherwise: uᵢ = max(consumption_i, ε)^(1−η) / (1 − η).
- aggregate_utility = Σ Fwi·uᵢ is the mean utility per person. It is reported as the output column `aggregate_utility`.

Savings are a common fraction s of every person's post-tax income.

### 7.11 Damage distribution for the next step

1. Per-quantile damage fraction:
   - if |β| < ε: ωᵢ = Ω_base for all i;
   - otherwise: ωᵢ = Ω_base·(y_net,i / y_ref)^(−β) where y_net,i > ε, else 1 − ε.
2. Clip: ωᵢ = clip(ωᵢ, 0, 1 − ε).
3. Totals: total_income = Σ Fwi·y_net,i and total_damage = Σ Fwi·y_net,i·ωᵢ.
4. If `income_dependent_aggregate_damage` is false (all manuscript runs), and total_damage > ε and total_income > ε:
   - scale = (Ω_base·total_income) / total_damage;
   - ωᵢ = clip(ωᵢ·scale, 0, 1 − ε);
   - total_damage is recomputed.

   This makes the income-weighted aggregate damage equal Ω_base, so β redistributes damage without changing its total. y_ref then cancels, except through the clips.
5. Ω = total_damage / total_income if total_income > ε, else 0.

The output `omega_yi` (and the distributions sheet of that name) is this final ωᵢ, after step 4. It is not the lagged ω_calc,i that was used in this step's budget.

β = 1 gives equal absolute damage per person: ωᵢ·y_net,i is the same for everyone.

### 7.12 Emissions, capital and heat uptake

- e = (1 − μ)·e_pot (tCO2/person/yr). It is negative when μ > 1.
- dK/dt = s·y_net·L − δ·K. This uses aggregate y_net, not Σ Fwi·y_net,i.
- U = aggregate_utility·L.

### 7.13 Tendencies and the ratios carried forward

These are computed in both branches:

- dEcum/dt = e·L + E_extra.
- dH_k/dt = a_k·dEcum/dt − H_k/τ_k, for k = 1, 2, 3.
- If Ω_base > ε: qᵢ ← ωᵢ / Ω_base and Q ← Ω / Ω_base. Otherwise qᵢ ← 0 and Q ← 0.

The heat-uptake equations are the state-space form of the impulse response: warming from a unit pulse emitted at t₀ is R_∞ + Σ a_k·exp(−(t − t₀)/τ_k). That is zero at t = t₀, because a₁ + a₂ + a₃ = −R_∞.

## 8. Time integration

`src/economic_model.py:integrate_model(config, f_values, s_values, store_detailed_output)`.

- **Grid:** `t = np.arange(t_start, t_end + dt, dt)`. For 2020–2420 with dt = 1 that is 401 points, exact integers stored as floats. `f_values` and `s_values` are arrays on this grid (Section 11.3).
- **Initial state:** K = K₀, Ecum = `ECUM_INITIAL`, H_k = `H_k_INITIAL`. The first step's damage ratios are qᵢ = 1 (uniform damage) and Q = 1.
- **Loop:** for each grid index i = 0 … N−1, including the last point:
  1. compute the tendencies at tᵢ (Section 7) and record U[i] together with the state and diagnostics at tᵢ;
  2. update with forward Euler:
     - K ← K + dt·dK/dt;
     - Ecum ← max(0, Ecum + dt·dEcum/dt);
     - H_k ← H_k + dt·dH_k/dt;
  3. carry forward qᵢ and Q.

  The last step's update is computed but not used.
- **Ecum clamp:** cumulative emissions are kept non-negative, so net-negative emissions cannot cool below zero cumulative emissions. No other state is constrained.

All recorded quantities satisfy the Section 7 relations exactly at each output time. The state at tᵢ is the value *before* that step's update.

## 9. Derived output variables

`src/economic_model.py:create_derived_variables`. These are computed after integration, for output only.

- **Totals:**
  - Y_gross, Y_damaged, Y_net = per-capita value × L;
  - AbateCost = abateCost_amount·L;
  - E = e·L;
  - Climate_damage = climate_damage·L.
- **Consumption and savings:**
  - Consumption = (1 − s)·Y_net and consumption = (1 − s)·y_net;
  - Savings = s·Y_net and savings = s·y_net.
- **`marginal_abatement_cost`** = θ₁·μ^(θ₂ − 1). This equals 10^f except where μ was capped or set to 0.
- **`r_consumption`:**
  - c_ij = max((1 − s_i)·y_net,ij, ε);
  - M_i = Σ_j Fwi_j·c_ij^(−η);
  - r_i = exp(ρ·dt)·(M_i / M_{i+1}) − 1, with the ratio formed first, and r_N = r_{N−1} for the last step;
  - dt = t[1] − t[0].

  This is the welfare-weighted Ramsey rule (README). The baseline computed c^(−η) as `np.power(c, -eta)`. The current code writes `c ** (-eta)`, for which NumPy uses a reciprocal fast path when η = 1; that differs by 1 ulp (3.5e-16 in test case B3), and is the only non-zero difference in the regression suite.

  **The stored `r_consumption` columns of the published runs come from the earlier representative-agent rule.** That rule is r_i = exp(ρ)·(consumption_{i+1}/consumption_i)^η − 1, with the same last-step rule, where consumption = (1 − s)·y_net is mean consumption per person. It reproduces the stored columns of all nine reference runs to 9e-16 relative. The current formula differs from them by 1–2% typically, and by up to about 50% at single points.

## 10. Objective

`src/optimization.py:discounted_utility`:

J = `np.trapezoid(exp(−ρ·(t − t_start))·U(t), t)`

U(t) = L·Σ Fwi·uᵢ, and the integral uses the trapezoidal rule on the integration grid. Typical magnitude: J ≈ −6.5e9 for the manuscript configuration.

## 11. Control trajectories

### 11.1 Control times

`src/optimization.py:calculate_control_times(n, t_start, t_end, dt, δ)`. Times are relative to t_base: t_start = 0 and t_end = 400 for the manuscript runs.

1. **Wind-down points:** n_end = 1 if n ≤ 10, else 2. Also t_join = t_end − 4/δ (360 for δ = 0.1).
2. **Points before the wind-down:** n₁ = n − n_end, and n₁ ≥ 3 is required (else `ValueError`). Let m = n₁ − 1 intervals and D = t_join − t_start.
3. **Spacing growth rate:** solve Σ_{i=0}^{m−1} dt·(1 + r₀)^i = D for r₀ with `scipy.optimize.brentq` on the bracket (−0.5, 100), using SciPy's defaults (xtol = 2e-12, rtol ≈ 8.88e-16, maxiter = 100). The sum is a sequential Python `sum` over i = 0 … m−1. If brentq raises `ValueError` (no sign change), use r₀ = D/(m·dt) − 1 instead.
4. **Times:**
   - times₁[0] = t_start;
   - times₁[i+1] = times₁[i] + dt·(1 + r₀)^i, accumulated sequentially;
   - times₁[−1] is then overwritten with t_join.
5. **Wind-down times:** [t_end] if n_end = 1, else [t_end − 1/δ, t_end]. The result is times₁ followed by the wind-down times.

The manuscript grids for n = 4, 6, 9, 15, 24 (calendar years, printed to 6 decimals):

```
 4  r0=179 (fallback)        2020 2021 2380 2420
 6  r0=5.74431128567404      2020 2021 2027.744311 2073.230046 2380 2420
 9  r0=1.44399567581887      2020 2021 2023.443996 2029.417111 2044.015377 2079.693479 2166.890604 2380 2420
15  r0=0.555777043030043     2020 2021 2022.555777 2024.976219 2028.741888 2034.600428 2043.715011 2057.89527
                             2079.95659 2114.279087 2167.677239 2250.752859 2380 2410 2420
24  r0=0.236343243831626     2020 2021 2022.236343 2023.764888 2025.654694 2027.991142 2030.879795 2034.451161
                             2038.866595 2044.325587 2051.074776 2059.419089 2069.735524 2082.490179 2098.259311
                             2117.75537 2141.859191 2171.659788 2208.503554 2254.055096 2310.372436 2380 2410 2420
```

For n = 4 the fallback value of r₀ is not a solution, but it does not matter. The second point is t_start + dt regardless of r₀, and the third is overwritten with t_join.

For explicit control years (a list in `n_points_final`), the times are those years minus t_base, in every iteration.

### 11.2 PCHIP trajectory

`src/optimization.py:control_trajectory(times, values, t)`:

1. Sort the points by time with `np.argsort`.
2. Keep the first point, and every later point whose time exceeds the time of the point just before it in the sorted original array (kept or not) by more than 1e-10: `keep = [True] + (np.diff(times) > 1e-10)`.
3. With a single remaining point, the trajectory is constant.
4. Otherwise evaluate `PchipInterpolator(times, values, extrapolate=False)(t)` where t ≤ times[−1]. For t beyond the last time, use the last value.

Times before the first control time evaluate to NaN. NaN controls make the objective non-finite, which is treated as infeasible (Section 12.3). All grids start at t_start, so this does not arise in practice. No clipping is applied: bounds are enforced only on the optimizer's variables.

For reference, SciPy 1.15's PCHIP (`PchipInterpolator._find_derivatives`) works as follows, with hₖ = x_{k+1} − x_k and mₖ = (y_{k+1} − y_k)/hₖ:

- **Interior derivatives:**
  - d_k = 0 if sign(m_k) ≠ sign(m_{k−1}), m_k = 0 or m_{k−1} = 0;
  - otherwise d_k = 1 / [(w₁/m_{k−1} + w₂/m_k)/(w₁ + w₂)], with w₁ = 2h_k + h_{k−1} and w₂ = h_k + 2h_{k−1}.
- **End derivatives:** d₀ = ((2h₀ + h₁)m₀ − h₀m₁)/(h₀ + h₁).
  - Set d₀ = 0 if sign(d₀) ≠ sign(m₀).
  - Otherwise set d₀ = 3m₀ if sign(m₀) ≠ sign(m₁) and |d₀| > 3|m₀|.
  - The same rule applies at the right end, using the last two intervals.
  - With exactly two points, both derivatives equal the slope.
- **Piecewise cubic:** on [x_k, x_{k+1}], with u = x − x_k, slope = m_k and T = (d_k + d_{k+1} − 2·slope)/h_k, the coefficients are:
  - c₀ = T/h_k;
  - c₁ = (slope − d_k)/h_k − T;
  - c₂ = d_k;
  - c₃ = y_k.

  The cubic is evaluated as c₀u³ + c₁u² + c₂u + c₃ (Horner, `PPoly`).

### 11.3 Control arrays on the integration grid

`control_arrays(config, controls)`:
- f = `control_trajectory(f_times, f_values, t − t_base)`.
- s is either `control_trajectory(s_times, s_values, t − t_base)` when s is optimized, or `time_functions.s(t − t_base)` otherwise.

## 12. Optimization

`src/optimization.py:optimize`.

### 12.1 Refinement schedule

The schedule applies both to the number of points of each control and to the evaluation budget (`refinement_schedule(initial, final, K)`):

- for K = 1: [final];
- otherwise, with base = ((final − 1)/(initial − 1))^(1/(K − 1)), iteration k = 1 … K gets `round(1 + (initial − 1)·base^(k−1))`. Python `round` rounds half to even.

For the manuscript (K = 5):
- control points: [4, 6, 9, 15, 24] for both f and s;
- evaluation budgets: [2000, 5317, 14140, 37602, 100000].

### 12.2 Iterations

For iteration k, each optimized control (f, then s, in the configuration's order) has its Section 11.1 grid for that iteration's point count.

- **Initial values:**
  - Iteration 1: every point is set to `initial_guess`, except the last control point, which is set to **0.0**. For generated grids the last point is t_end, giving f = 0 (MAC $1/tCO2) and s = 0 there. For explicit years it is the latest listed year, whether or not that is t_end.
  - Iteration k > 1: the previous iteration's optimal values are PCHIP-interpolated (Section 11.2) onto the new times.
- **Optimizer variables:** the concatenation [f values, s values]. Bounds are each control's `bounds`, repeated for every point. The initial point is clipped to the bounds.

NLopt setup (`_run_nlopt`):

```python
opt = nlopt.opt(nlopt.LN_SBPLX, n)
opt.set_lower_bounds(lb); opt.set_upper_bounds(ub)
opt.set_max_objective(objective)
opt.set_maxeval(max_evaluations_k)
opt.set_ftol_rel(0.0); opt.set_ftol_abs(0.0); opt.set_xtol_rel(0.0); opt.set_xtol_abs(1e-8)
x_opt = opt.optimize(clip(x0, lb, ub))
```

Everything not set, including the initial step size, uses NLopt's defaults. Derivative-based algorithms (second letter `D`) are rejected. The iteration's objective is `opt.last_optimum_value() / OBJECTIVE_SCALE`, and its termination code is `opt.last_optimize_result()`. Because of the scaling and unscaling, the reported objective can differ in the last digit from J recomputed with the same controls. "Evaluations" counts calls of the objective. In the nine reference runs, the final iteration always ends with XTOL_REACHED; earlier iterations end with XTOL_REACHED or MAXEVAL_REACHED.

### 12.3 Objective seen by NLopt

For a trial vector x:
1. Split x into the control values and build the control arrays (Section 11.3).
2. Compute J (Section 10). If the evaluation raises `ValueError` (the tax cannot be raised, or G > G_b) or `RuntimeError` (tax solve failure), set J = NEG_BIGNUM.
3. If J is not finite (NaN or ±inf), print a warning and return NEG_BIGNUM = −1e30. Note that this value is not scaled.
4. Otherwise return J·1e-13. A caught exception therefore returns −1e30·1e-13 = −1e17.

The result of the last iteration is the optimum: control times (relative to t_base) and values, objective, and total evaluations.

## 13. Manuscript run families

The configurations in `json/prod_s/` are the authoritative inputs: 14,801 files plus the 9 copies stored with the reference runs. They all share the Section 4 template, except that η, ρ, τ and β vary, plus the noted exceptions. `y_damage_distribution_exponent` is β and `tax_equity` is τ.

All families have η = 1.5 and ρ = 0.025 unless the table says otherwise. The 3 τ values are always {0, 0.5, 0.998}.

| Family (`json/prod_s/…`) | Configs | Design |
|---|---|---|
| `9cases_s` | 9 | τ ∈ {0, 0.5, 0.998} × β ∈ {0, 0.5, 1}; gid = 3·i_β + i_τ + 1 (gid-1: τ = 0, β = 0; gid-2: τ = 0.5, β = 0; gid-4: τ = 0, β = 0.5; …) |
| `prod_contour_s` | 1681 | τ ∈ {k/40 : k = 0…39} ∪ {0.998} (the grid point 1.0 replaced by 0.998) × β ∈ {k/40 : k = 0…40}; gid = 41·i_τ + i_β + 1 |
| `mc/mc_beta0_s`, `mc/mc_beta0.5_s` | 1500 each | 500 (η, ρ) pairs × 3 τ; gid = 3k + i_τ + 1. Latin hypercube, η ∈ [1, 2], ρ ∈ [0, 0.05], rounded to 6 decimals. The two β families use different samples. |
| `single_param/single_param_beta{0,0_5}_s_eta` | 1500 each | 500 η values in [1, 2], one per stratum of width 0.002, 8 decimals, in shuffled gid order, × 3 τ. The same η set for both β. |
| `single_param/single_param_beta{0,0_5}_s_rho` | 1666 (β = 0), 1500 (β = 0.5) | 500 ρ values in [0, 0.05], one per stratum of width 1e-4, 8 decimals, shuffled, × 3 τ. The same set for both β. In the β = 0 folder, 166 τ = 0 cells were run twice (two timestamps). |
| `single_param/single_param_beta0_s_rho_lowrmu` | 340 | τ = 0, β = 0; ρ = linspace(−0.029, 0.004, 340), rounded to 6 decimals. gids run 1…1018 in steps of 3: the τ = 0 members of a 3-τ sweep. |
| `dice_comparison/dice_reference_r2_2030` | 20 | G ≡ 0 (gini scaling and constant 0), τ = 0, β = 0; ρ = linspace(−0.015579, 0.024421, 20) |
| `dice_comparison/dice_reference_r3_2030` | 2 | as r2, ρ ∈ {0.03, 0.03616} |
| `dice_comparison/dice_reference_r4_2030` | 9 | as r2. ρ was chosen so that r_μ(2030) ≈ 0.8, 1.0 (twice), 1.2, 2, 3, 4, 4.7 and 5 %/yr; how those ρ were found is not recorded. |
| `mc/mc_low_rmu_*`, `mc/mc_infill_*`, `mc/mc_wrongbasin_*` | 2–484 each | Infill samples for Fig 5: (η, ρ) targeted at bands of r_μ, with η ∈ [1, 2] and ρ down to −0.041; τ ∈ {0.5, 0.998}, or 0.5 only (`mc_infill_2_6_*`, `mc_infill_5_6_*`). Only the cells that were plotted were kept; gid gaps show the original sweeps were larger. `mc_infill_lt1_ws3` (2 runs) has s bounds [0.2, 0.4]. |

**Why the MC design is a Latin hypercube.**
- All 500 η strata of width 0.002 are occupied.
- 497 of the 500 ρ strata of width 1e-4 are occupied. Each of the 3 empty ones is next to a value lying exactly on its lower boundary, as rounding to 6 decimals would produce. iid uniform draws would occupy only about 316 strata.
- The η–ρ correlation is −5e-5 (β = 0) and 0.05 (β = 0.5).

The generator is not in the repository. `dev-history` has only an older iid generator (`scripts/generate_mc_draws.py`: seed 42, different ranges), which did not produce these samples. **The configs, which carry the sampled values, are the record of the design.**

**Warm-started families.** The "ws" families (`mc_infill_lt1_ws*`, `mc_wrongbasin_ws*`) were re-runs warm-started with controls supplied by the batch driver, to move runs out of a wrong local optimum ("wrong basin"). Their configs carry the standard initial guesses, so re-running them with `scripts/run_optimization.py` performs a fresh optimization, not the warm-started one.

**Run directories.** Each run's output is in `data/output/prod_s/<family>/<cell>/`. `<cell>` = `<family>_gid-<gid>_<YYYYMMDD_HHMMSS>`, which is also the stem of the run's config in `json/prod_s/<family>/`.
- Inside a config, `run_name` is `<family>_A_gid-<g>` for gids 1–1000 and `<family>_B_gid-<g − 1000>` above that. The 9cases runs have no A/B.
- Files are named `<run_name>_results.csv`, `<run_name>_optimization_summary.csv`, `<run_name>_distributions.xlsx` and `<run_name>.json`, plus `run_status.csv` and `terminal_output.txt` (Section 14). The batch driver wrote `{run_name}_results.csv` rather than `{run_name}_optimization_results.csv`.

`run_status.csv` holds a header and one data row:
- `run_name`, `grid_id`, τ, β, η, ρ;
- `success`, `optimizer_status`, `objective_value` (scaled by 1e-13), `n_iterations`, `n_evaluations`, `runtime_seconds`;
- `mac_2030`, `mac_2100` (= 10^f in that year);
- `delta_t_2100`, `mu_2030`, `mu_2100`, `Omega_2100`, `y_net_2100`;
- `aggregate_utility` (the unscaled objective J);
- `error_message`.

In the nine reference runs, `mac_2030`, `mu_2030` and `delta_t_2100` equal the results-CSV values at t = 2030 and 2100 bit for bit.

## 14. Output files

`src/output.py`. A run directory holds:

- **`{prefix}_results.csv`**: one row per time step. Column headers have the form `"name, description, (units)"` for the labelled variables, in this order:

  `t, A, L, theta1, sigma, f, s, K, gini, Ecum, delta_T, y_gross, y_damaged, y_net, climate_damage, tax_amount, mu, Omega, Omega_base, Omega_calc, lambda_abate, abateCost_amount, U, aggregate_utility, e, emission_ratio, E_extra, consumption_extra, dK_dt, log_deltaU_norm, max_average_tax_rate, max_marginal_tax_rate`

  The remaining time series follow, sorted by name, with the bare name as header:

  `AbateCost, Climate_damage, Consumption, E, H1, H2, H3, Savings, Y_damaged, Y_gross, Y_net, consumption, marginal_abatement_cost, r_consumption, savings`

  Values are written with Python's `csv` module, which writes each float as its shortest round-trip decimal string, so the values read back exactly. The published files also contain the columns removed in 2026: `Gini`, `Lambda`, `dEcum_dt`, `mu_dollar_numeraire`, `mu_dollar_cost`, `welfare_adjusted_mac`, `mac_over_wmac`.

  The labels of `U` ("Mean utility per capita") and `aggregate_utility` ("Aggregate utility from integration") are misleading. `aggregate_utility` is the mean utility per person, and `U` is that times L.
- **`{prefix}_distributions.xlsx`**:
  - sheet `Quadrature_Info` holds xi, wi, xi_edges, Fi, Fwi, Fi_edges;
  - sheets `y_net_yi`, `omega_yi`, `utility_yi`, `average_tax_rate_yi` and `marginal_tax_rate_yi` each hold a column `t` and then `bin_0 … bin_{n−1}`.
- **`{run_name}_optimization_summary.csv`** (optimizations only), in four sections:
  1. final control values (6 decimals), objective (`%.12e`), evaluations, algorithm and iteration count;
  2. final control points per control (calendar year to 2 decimals, value to 6 decimals);
  3. per-iteration objective, evaluations and termination;
  4. per-iteration f values and f times.

  The published summaries report the objective scaled by 1e-13 (−6.487938701599e-04 for gid-1). The current code reports it unscaled.
- **PDF reports and `terminal_output.txt`**: these do not affect any number.
- **The configuration actually run**, including command-line overrides `--key.subkey value`. Each override is converted to the type of the value it replaces (`src/config_cli.py`).

**`scripts/run_integration.py`** replays a run in one of two modes:
- **From an output directory.** It reads the config and results CSV there, and builds f and s on the grid with `PchipInterpolator(t_csv, column, extrapolate=True)`, clipped to the control bounds. On the same grid this reproduces the stored arrays exactly.
- **From a config.** The optimized controls are held constant at their (clipped) initial guesses.

## 15. Parameter estimation

The scripts in `parameter_estimation/` run offline; nothing in `src/` imports them. Their results were copied by hand into `src/constants.py` and the configuration files. With the Section 2 library versions, three of the four scripts reproduce their committed outputs byte for byte. The fourth (15.1) reproduces its frozen snapshot to round-off.

| Model value | Produced by | Copied as fitted? |
|---|---|---|
| `EMPIRICAL_LORENZ_P0..P3`, `W1..W3` | 15.1 → 15.2 | Weights in full; powers rounded to 6 decimals |
| `time_functions.gini` | 15.3 (SSP table) plus an exponential fit that has no script | Yes, rounded to 5 decimals |
| `time_functions.L`, `sigma` | 15.4 | Yes, rounded to 4–6 significant figures |
| `time_functions.A` | not the 15.4 fit (see there) | No |
| `theta1`, `emission_ratio`, `Eland`, `y_net_reference`, scalar parameters | no script (15.5) | |

### 15.1 Empirical global income distribution

Script: `empirical_distribution/empirical_global_distribution.py`.

**Input.** `empirical_distribution/data/pip_2025-12-28.xlsx`, from the World Bank Poverty and Inequality Platform (downloaded 28 Dec 2025). `pd.read_excel` reads its first sheet, `pip`: 2556 rows, 172 countries, reporting years 1963–2024.

The columns used are `country_name`, `reporting_year`, `decile1..decile10` (share of welfare by decile), `reporting_pop` and `reporting_gdp` (per-capita GDP). The docstrings call `reporting_gdp` "PPP $". Its magnitudes (US 2023: 65,505; India 2022: 2,065) look like constant US$ at market exchange rates; check the PIP documentation before stating units. Income and consumption surveys are mixed, and so are national, urban and rural reporting levels; nothing filters on either.

**Method.**
1. Keep rows with no NaN in the deciles, population or GDP. All 2556 rows pass.
2. Keep one row per country: `sort_values('reporting_year', ascending=False)` then `groupby('country_name').first()`, giving 172 rows with years 1992–2024.

   The default sort is not stable. Four countries have several rows in their latest year, and the rows that were selected (and are in the snapshot) are:
   - China 2022 national, consumption;
   - Haiti 2012, income;
   - Philippines 2023, income;
   - St. Lucia 2015, consumption.

   A stable sort would choose consumption for Haiti and the Philippines, and the Gini would become 0.680706.
3. For each country and decile: income per person = `reporting_gdp × decile_share × 10`, and population = `0.1 × reporting_pop`. This gives 1720 points.
4. Sort the points by income per person, ascending (there are no ties).
5. Population fraction = cumsum(pop)/Σpop; income fraction = cumsum(y·pop)/Σ(y·pop).
6. Gini = 1 − 2·`np.trapezoid(L, p)`, after prepending (0, 0). The result is 0.680860260809561, with total population 7,719,425,883 and mean income 11,627.2.

**Snapshot.** The output (`output/` is not committed) was frozen as `lorenz_fit/data/empirical_distribution_global.csv`, with 1720 rows in the same order. A fresh regeneration matches it to ≤ 9.1e-13 relative, but not bit for bit: some snapshot values carry fewer digits. The snapshot came from an upstream copy of this code (`~/global-lorenz`). The Lorenz fit must be run on the snapshot (see 15.2).

### 15.2 Lorenz curve fit

Script: `lorenz_fit/fit_polynomial_lorenz.py --use-convex-combination --min-degree 4 --max-degree 4`.

**Data.** x = `population_fraction` and y = `income_fraction` from the snapshot, with (0, 0) prepended, so N = 1721.

**Model.** L(x) = Σ_{k=0}^{3} w_k·x^{p_k}, with w_k ≥ 0 and Σ w_k = 1.

**Inner problem (weights for fixed powers).** Minimize Σᵢ (yᵢ − L(xᵢ))², unweighted, with `scipy.optimize.minimize(method='SLSQP')`:
- x0 = [0.25]·4;
- constraints Σw = 1 (equality) and w ≥ 0;
- `options={'maxiter': 2000, 'ftol': 1e-15, 'eps': 1e-10}`, finite-difference gradients.

**Outer problem (powers).** Minimize the inner optimum over (p₀..p₃) with `method='L-BFGS-B'`:
- x0 = `np.linspace(1.1, 20, 4)`;
- bounds (1.01, 1000) on each power;
- `options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}`.

It converges in 49 iterations and 340 evaluations, to a sum of squared residuals of 0.0018113192234209881. The weights are then re-solved at the optimal powers. Neither pruning (w < 1e-8) nor renormalization triggers.

**Results.** These reproduce bit for bit from the snapshot:
- w = [0.15987112754710997, 0.3776187268483524, 0.36712476209491907, 0.09538538350961864];
- p = [1.5000363406623263, 4.3674401252364285, 14.072004843000872, 135.05967442274672].

Fit quality: R² = 0.99997863, maximum absolute error 0.0048.

How the results were copied into `src/constants.py`:
- the powers were rounded to 6 decimals;
- W1–W3 were copied in full;
- W0 = 1 − W1 − W2 − W3 differs from the fitted w₀ by 1.4e-16;
- the weights were not re-fitted at the rounded powers.

G_b with the unrounded powers would be 0.6812795341337843, against the model's 0.681279513926459.

**Sensitivity.** The optimum is flat. Other starting points move the powers by up to 1.1e-4 relative and the weights by about 1e-5, at the same residual to 2e-11. Fitting a fresh regeneration of 15.1 instead of the snapshot gives p = [1.500004, 4.367257, 14.071497, 135.052615]. **The constants therefore reproduce only from the snapshot CSV, the starting point above and SciPy 1.15.**

### 15.3 Global Gini trajectory

Script: `global_gini/global_gini.py`, run with `MPLBACKEND=Agg`. It executes at import and calls `plt.show()`. It is followed by an exponential fit that has no script.

**Inputs** (`global_gini/data/`). The source is not documented in this repo. The dev-history commit that introduced G(t) cites doi:10.1088/1748-9326/acbdb0.
- `All_GDP_percapita.csv`: GDP per capita by iso × SSP, annual 2010–2100.
- `pop_ssp_database.csv`: population (millions) by SSP × iso, 5-yearly 2015–2100.
- `ISO_level_projections_PC_model_projections.csv`: projected decile shares `pred_shares` for 170 countries × 2015–2100 × 5 SSPs × deciles `d1..d10`. The shares are not renormalized; their sums range from 0.99975 to 1.01619.

**Method.**
1. Reshape each file to long format, lower-case the iso codes, and keep population years 2015–2100 and GDP years 2010–2100.
2. Left-merge the shares with population, then with GDP, on (iso, sce, year), and drop NaN. Because population is 5-yearly, 18 years survive. 165 countries remain per cell (164 in SSP4 2100).
3. Decile quantities:
   - decile population = 0.1·pop;
   - decile income = gdp_pc·pop·pred_shares;
   - decile income per person = decile income / decile population.
4. For each (SSP, year): sort the country-deciles by income per person; take cumulative population and income fractions; compute Gini = 1 − 2·`np.trapezoid(y, x)`. Unlike 15.1, no (0, 0) point is prepended; the effect is about x₁y₁/2.
5. Write `output/global_gini_table.csv` (year × SSP). Re-running reproduces it byte for byte. SSP2 values: 0.596178 (2015), 0.565096 (2020), 0.524365 (2050), 0.464670 (2100).

**Exponential fit (no script; reconstructed and confirmed).**
- `scipy.optimize.curve_fit(lambda t, a, b, c: a*np.exp(b*t) + c, t, G, p0=[0.2, -0.01, 0.4])`, unweighted;
- uses all 18 SSP2 values, with t = year − 2020, so 2015 enters as t = −5;
- gives a = 0.1951688, b = −0.01058413, c = 0.37795849, which round to the configured 0.19517, −0.01058, 0.37796.

Dropping 2015, or using t = year − 2015, does not reproduce the configured values. The fit's G(2020) = 0.57313 is 0.008 above the table's 0.565096.

**Data error.** In both `pop_ssp_database.csv` and `All_GDP_percapita.csv`, the TWN rows are identical to the CHN rows (e.g. SSP2 2020 population 1379.229 million). China's population and income are therefore counted twice, the second time with Taiwan's decile shares. Without TWN, the SSP2 Gini is 0.6063 in 2020 and 0.4844 in 2100, and the fit becomes a = 0.29188, b = −0.0075665, c = 0.31927. The published runs use the values with the duplicate.

### 15.4 DICE-2023 Gompertz fits

Script: `dice_timefunctions/fit_dice_timefunctions.py`. Run it from the repository root (it imports `src.parameters`). It writes `output/dice_timefunctions_fits.json` and `output/dice_timefunctions_report.txt`, both committed and reproduced byte for byte.

**Series.** The script hard-codes the DICE-2023 recurrences (Barrage & Nordhaus 2023, `DICE2023-b-4-3-10.gms`; the GAMS file is on the `dev-history` branch under `papers/barrage_nordhaus_2023/`). They run on 5-year steps k = 1..81, t = 5(k − 1), for 2020–2420:
- **A:** AL1 = 5.84; gA(k) = 0.066·exp(−5·0.0015·(k−1)); aL(k+1) = aL(k)/(1 − gA(k)). Converted to model units with × 739.619/5.84.
- **L:** pop1 = 7752.9 million; L(k+1) = L(k)·(10825/L(k))^0.145. Converted with × 1e6.
- **σ:** sig1 = 0.29135; gsig(k) = min(−0.015·0.96^(k−1), −0.005); σ(k+1) = σ(k)·exp(5·gsig(k)). Converted with × 1e-3 to tCO2/$.

**Fits.** The Gompertz form of Section 4.2 is fitted with `scipy.optimize.curve_fit` (`maxfev=20000`), with the initial value fixed to the first value of the series:
- A: log-space residuals, all 81 steps, final value and c free, p0 = (4906815.811, −0.0015);
- L: linear residuals, all steps, final value fixed at 1.0825e10, c free;
- σ: log-space residuals, steps 1–28 only (to 2155, where the −0.005 floor starts to bind), final value and c free.

L is exactly Gompertz, with c = ln(1 − 0.145)/5 = −0.0313307620. σ is exactly Gompertz before the floor, with c = ln(0.96)/5 = −0.0081643989 and final value σ₀·e^(−1.875).

| | Fit: initial / final / c | Model config |
|---|---|---|
| A | 739.619 / 5383526.77859792 / −0.0015406238862464446 | 739.619 / 4906815.811 / −0.0015 |
| L | 7.7529e9 / 1.0825e10 / −0.03133076200907536 | 7.752e9 / 1.0825e10 / −0.03133 |
| σ | 2.9135e-4 / 4.467996959027015e-05 / −0.008164398904051053 | 2.91355e-4 / 4.4681e-05 / −0.008164 |

**A(t) in the model is not the script's fit.** The configured values follow the continuous small-growth approximation: c = −delA, and final = A₀·exp(gA1/(5·delA)) = A₀·e^8.8, evaluated with A₀ = 739.6194361. That A₀ equals q1/(K₀^0.3·pop1^0.7) = 135.7e12/((295e12)^0.3·(7.7529e9)^0.7), i.e. it is calibrated to DICE's 2020 output. This derivation was inferred numerically; no script contains it. Against the DICE series, the model's A(t) is −1.45% in 2050, −3.57% in 2100 and −11.4% in 2420, whereas the fit stays within 0.04%. `parameter_estimation/README.md` says the fitted A block was pasted into the configs; that is not so.

### 15.5 Inputs with no fitting script

- **Eland:** 5.9e9 tCO2/yr × exp(−0.02107·t). This is DICE's eland0 = 5.9 GtCO2/yr with deland = 0.1 per 5-year period: ln(0.9)/5 = −0.0210721, rounded.
- **emission_ratio:** 1.4·exp(−0.002085·t), which gives 1.185 in 2100. DICE's `emissrat` instead declines linearly from 1.40 (2020) to 1.21 (2100) and is constant after that. The source of −0.002085 was not found.
- **theta1:** 6035.6·(0.906·e^(−0.224t) + 0.094·e^(−0.00112t)), described in its dev-history commit (dbbbdc46, 2026-03-24) as a "best 2 double exponential fit to Barrage & Nordhaus 2023". No script or data for the fit is in the repo.
- **y_net_reference = 17210.79:** no derivation found; introduced in dev-history commit d7d40e79. With `income_dependent_aggregate_damage = false`, the rescaling of Section 7.11 cancels y_ref, so it affects results only through the clips at 0 and 1 − ε.
- **Scalar parameters:** α = 0.3, δ = 0.1, ψ₂ = 0.003467, θ₂ = 2.6 and K₀ = 2.95e14 are DICE-2023 values. ρ = 0.025 is labelled "DICE-2023 risk-adjusted rate", but DICE-2023's rartp = exp(0.001 + 0.5·0.05) − 1 = 0.02634.
- **Temperature response:** the impulse-response constants are from Ricke & Caldeira (2014), doi:10.1088/1748-9326/9/12/124002, median model.

## 16. Figures and SI data

The figure scripts are in `scripts/prod-s/figures/`; they are unchanged since the release commit. They write PDFs to `data/output/prod_s/figures/`. `fig2` uses paths relative to the working directory, so run it from the repository root. `scripts/prod-s/label_utils.py` only places labels, markers and arrows at coordinates stored in each script; it computes nothing that is plotted.

**Common definitions.**
- **MAC2030 ("OPC2030")** is the results-CSV column `marginal_abatement_cost` at the row with `t == 2030.0`, in $/tCO2. The grid is annual, so no interpolation is involved. It equals 10^f(2030) to about 1e-15.
- **ΔT2100** is the column `delta_T` at `t == 2100.0`.
- Scripts find CSV columns by the prefix of their `"name, description, (unit)"` headers.
- Where τ or β is matched to a target value, the tolerance is 0.01.

| Item | Script | Reads |
|---|---|---|
| Fig 1 | `fig1/plot_fig1_conceptual.py` | nothing (analytic) |
| Fig 2 | `fig2/plot_fig2_allcases.py` | `9cases_s` results CSVs, 6 runs |
| Fig 3 | `fig3/plot_fig3_contours.py` | `prod_contour_s` results CSVs and configs (archive) |
| Fig 4, SI Fig 1 | `fig4/plot_fig4_ridgeline.py` | `run_status.csv` of `mc/mc_beta{0,0.5}_s` and `single_param/single_param_beta{0,0_5}_s_{rho,eta}` (archive) |
| Fig 5 | `fig5/plot_fig5_cloud.py` | `outputs/audits/points_cloud_1_6_balanced.csv` |
| SI Fig 2 | `figS/plot_si2_growth_contour.py` | gid-1 distributions xlsx, results CSV, config |
| SI Fig 3 | `plot_dice_reference_si.py` | `outputs/audits/dice_si_gini0_runs_2030.csv`, `dice_si_rho_sweep_2030.csv` |

### 16.1 Fig 1: damage and tax shares (analytic)

**Setup.** G = 0.57, η = 1.5, average tax rate 0.20, β ∈ {0, 0.5, 1}, τ ∈ {0, 0.5, 0.998}.
- Grid: F = linspace(1e-6, 1 − 1e-6, 2000).
- Income: y(F) = dL/dF from the empirical Lorenz curve (Section 6.1).

**Damage share:** D(F) = cumsum(y^(1−β))·ΔF, normalized by its last element. This is a left-inclusive cumulative sum, not the trapezoid rule.

**Tax share:**
1. η_eff from Section 7.8.
2. Solve for log ΔU with `brentq` on [−1000, 100], `xtol = 1e-12`, so that Σ(y − c)·ΔF = 0.20, with c from the Section 7.8 schedule.
3. T(F) = cumsum(y − c)·ΔF, normalized.

**Check values** at index 1000 (F = 0.50025):

| Quantity | Value |
|---|---|
| L | 0.144393 |
| D, β = 0 | 0.143929 |
| D, β = 0.5 | 0.311011 |
| D, β = 1 | 0.5005 |
| T, τ = 0 | 0.143929 |
| T, τ = 0.5 | 0.056244 |
| T, τ = 0.998 | 4.9e-19 |

Σ y·ΔF = 1.00427. As a result, the script's printed check "D(0.5) == L(0.5) within 1e-6" prints False, off by 4.6e-4; the plot is not affected.

### 16.2 Fig 2: six reference cases

MAC(t) and ΔT(t) for 2025 ≤ t ≤ 2100, from 9cases gid-1, -2, -3 (β = 0; τ = 0, 0.5, 0.998) and gid-4, -5, -6 (β = 0.5). For each τ, the band between β = 0 and β = 0.5 is filled. Nothing is computed beyond selecting columns.

The 2100 end labels (3.1, 2.9, 2.8, 2.6, 2.2, 1.9) and the marker heights (3.11, 2.89, 2.83, 2.57, 2.18, 1.89) are hard-coded. They agree with the stored ΔT2100 values: 3.1075, 2.8868, 2.8294, 2.5735, 2.1810, 1.8887.

### 16.3 Fig 3: τ × β contours

**Data.** Every `prod_contour_s_gid-*` run directory: τ and β from the config, MAC2030 and ΔT2100 from the results CSV.
- Runs are keyed by gid; a later directory with the same gid overwrites an earlier one.
- `run_status.csv` is not read, so there is **no convergence filter**.
- Values go onto the 41 × 41 grid of sorted unique τ and β, nearest cell by argmin. Empty cells would be filled with `scipy.interpolate.griddata(method='cubic')` and then `method='nearest'`; with the complete archive there are none.

**Plot.**
- Interpolation to 400 × 400 points with `griddata(method='cubic')` (Clough–Tocher on a Delaunay triangulation) on linspace(0, 0.998, 400) × linspace(0, 1, 400).
- MAC2030: filled levels 50:10:350 $/tCO2, with data clipped to [50, 350]; lines every 20.
- ΔT2100: filled levels 1.5:0.1:3.5 °C; lines every 0.1.
- The strip τ ∈ [0.998, 1] is greyed.
- `fig3_contours_smooth.pdf` adds `gaussian_filter(sigma=3)` and is not a manuscript figure.

The fine-grid values depend on how Qhull triangulates the regular grid, but the 41 × 41 values do not. `data/si_datafiles/fig3_data.xlsx` holds those 41 × 41 values. Its (0, 0) cell is 42.623754 and its (0.998, 0) cell is 144.462438, against 42.623837 (gid-1) and 144.462679 (gid-3), separate optimizations of the same parameters (2e-6 relative).

### 16.4 Fig 4 (ridgelines) and SI Fig 1 (scatter)

**Data.** From each cell's `run_status.csv`, fields are read by position: τ [2], β [3], η [4], ρ [5], success [6], optimizer_status [7], n_evaluations [10], mac_2030 [12], delta_t_2100 [14]. Rows with an empty MAC or ΔT are skipped.

**Selection.**
1. A run is converged if `success == 'True'` and the status is XTOL_REACHED or FTOL_REACHED.
2. Duplicates are keyed by (round(β, 2), round(τ, 3), round(η, 6), round(ρ, 6)). The first run read is kept, unless a later one is converged and the kept one is not, or both are converged and the later one used more evaluations. Directories are read in `os.listdir` order, so exact ties depend on directory order. This rule resolves the 166 duplicated ρ-sweep cells.
3. Non-converged runs are dropped, and the rest are grouped by τ ∈ {0, 0.5, 0.998} (|Δτ| < 0.01).

`fig4_data.xlsx` has 500 rows per (row, β, τ), so every cell had a converged run.

**Central case.** The run minimizing hypot(η − 1.5, log10(ρ + 1e-12) − log10(0.025)). For MC with β = 0, this is η = 1.509987, ρ = 0.024683, with MAC2030 = 42.79, 71.11 and 144.96 for τ = 0, 0.5, 0.998.

**Ridgelines.**
- x-ranges are shared across rows. MAC: [0, 1.05·max]. ΔT: [min − 0.1R, max + 0.1R], with R = max − min, floored and ceiled to multiples of 0.25.
- Each group with n ≥ 5 and std ≥ 1e-9 gets `scipy.stats.gaussian_kde(values, bw_method='silverman')`, evaluated at 500 evenly spaced points. Each density integrates to 1.
- A dashed line marks the central case, labelled `$%.0f` or `%.1f°C`.

**SI Fig 1** plots MAC2030 against ΔT2100: x = ΔT2100, y = MAC2030; the script's docstring states the axes the other way round. Limits are the data range ± 3%, and the MAC lower limit is 0 when it would be positive and below half the upper limit.

### 16.5 Fig 5: carbon price against r_μ

**Data.** Only `outputs/audits/points_cloud_1_6_balanced.csv`: 5254 rows, with columns `sweep, cell, beta, tau, eta, rho, mac, dt, converged, chosen_from, was_dup, d_score, alt_mac, alt_dt, r_mu_pct`.

**Selection and panels.**
- Keep 1 ≤ `r_mu_pct` ≤ 6 (all rows pass), and select groups by |β − β₀| < 0.01 and |τ − τ₀| < 0.01. The `converged` column is ignored, so its 33 False rows are plotted.
- Panel A: `mac` on a log axis from 10 to 1200.
- Panel B: `dt`, from 0 to max + 3%.
- x-axis: r_μ from 1 to 6.

**The CSV was curated by a script that is not in the repository.** From its contents:
- Rows per (β, τ): (0, 0) 666, (0, 0.5) 1215, (0, 0.998) 744, (0.5, 0) 661, (0.5, 0.5) 1209, (0.5, 0.998) 759.
- In 670 rows, `was_dup` is True: another run with the same (β, τ, η, ρ) existed in a different sweep. `chosen_from` names the sweep that was kept, and `alt_mac`/`alt_dt` give the other run's values. This is how the warm-started "wrong basin" re-runs replace originals.
- The selection rule and `d_score` (0.006–0.42) are undocumented.
- 54 `mc_beta*_s` runs with τ = 0.5 and r_μ of 5.1–5.5% are absent, although Fig 4 counts them as converged. This is presumably the thinning that "balanced" refers to.

`data/si_datafiles/fig5_data.xlsx` equals this CSV.

### 16.6 The discount rate r_μ

r_μ is the x-axis of Fig 5 and SI Fig 3. It is precomputed in the audit CSVs; the code that computed it is missing. The SI Fig 3 docstring says "ρ + η·g_μ from each run's distribution output". The following formula is reconstructed from the stored numbers:

r_μ(2030) [%/yr] = 100·ρ + 100·η·Σ_j w_j·ln(c_j(2031)/c_j(2030)),
with w_j = Fwi_j·c_j(2030)^(−η) / Σ_k Fwi_k·c_k(2030)^(−η) and c_j = (1 − s)·y_net,j.

This is marginal-utility-weighted log growth of per-node consumption. For G = 0 it reduces to 100ρ + 100η·ln(c̄(2031)/c̄(2030)).

| Growth definition | g (%/yr) | Difference from the stored value |
|---|---|---|
| Stored value: ρ-sweep row nearest the base case (ρ = 0.0249964), (r_μ − 100ρ)/η | 2.118704 | |
| Log growth (above), for gid-1 | 2.118655 | 5e-5 pp, within the run-to-run difference |
| Simple growth (as SI Fig 2's g_μ) | 2.141833 | 0.023 pp |
| −(1/η)·ln(M(t+1)/M(t)), with M = Σ Fwi·c^(−η) | 2.117809 | 9e-4 pp |

Fig 5 cells near the base case agree with the log definition to 0.002–0.004 pp in all six (β, τ) groups. **Treat the formula as inferred, not documented.**

### 16.7 SI Fig 2: consumption growth across the distribution

Data from gid-1 (τ = 0, β = 0):
- the distributions xlsx sheets `Quadrature_Info` (Fi, Fwi) and `y_net_yi`;
- the results column `s`;
- η from the config.

**Growth.** c_j(t) = y_net,j(t)·(1 − s(t)), and g_j(t) = 100·(c_j(t+1)/c_j(t) − 1), a forward difference labelled at year t.

**2030 statistics.** ḡ = Σ Fwi_j·g_j = 1.794094, and g_μ = Σ w_j·g_j = 2.141833, with w_j as in 16.6. The lowest node grows at 2.779334 and the highest at 1.064509.

**Contour panel.**
- g on 2030…2100 (71 years) × 100·Fi (32 nodes) is interpolated with `RegularGridInterpolator(method='cubic')` onto 400 × 400 points. That interpolator uses an iterative solver with tolerance about 1e-5, which limits the precision of the plotted field.
- It is then smoothed with `gaussian_filter(sigma=(12, 8))`.
- Fill levels 1.0:0.25:3.0; lines at 1.50:0.25:2.50.

`figs2_data.xlsx` holds the unsmoothed 71 × 32 matrix.

### 16.8 SI Fig 3: comparison with DICE-2023

**Data.**
- `dice_si_gini0_runs_2030.csv`: (ρ, η, r_mu_2030_pct, opc_2030) for the 31 `dice_comparison` runs.
- `dice_si_rho_sweep_2030.csv`: (r_mu_pct, opc_2030) for 840 τ = 0, β = 0 runs: the 500 ρ values of `single_param_beta0_s_rho` plus the 340 of `_rho_lowrmu`. Duplicates were resolved by best objective, which differs from Fig 4's rule.
- DICE-2023 values are hard-coded: r = 1, 2, 3, 4, 5% → 595, 224, 114, 66, 43 $/tCO2 (2030 values from the DICE-2023 workbook, which is on `dev-history` under `papers/barrage_nordhaus_2023/`).

**Curves.**
- **G = 0:** `PchipInterpolator` through the rows with r_μ ≤ 5.1 (30 of 31), drawn over linspace(1, 5, 200).
- **Empirical Gini:** `PchipInterpolator` through the rows with 0.5 ≤ r_μ ≤ 5.5 (567 of 840), evaluated at r = 1…5 and joined by straight lines.

| r_μ (%/yr) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| COIN, G = 0 ($/tCO2) | 396.4041 | 213.9908 | 112.1164 | 62.3345 | 38.4314 |
| COIN, empirical Gini ($/tCO2) | 650.9428 | 380.9458 | 204.7403 | 107.1736 | 59.6478 |

These match `figs3_data.xlsx` to < 1e-6.

### 16.9 SI data files

The scripts that wrote `data/si_datafiles/*.xlsx` are not in the repository. Checked against repository data:
- `fig2_data` equals the 9cases CSVs for 2025–2100;
- `fig5_data` equals the cloud CSV;
- `figs2_data` and `figs3_data` equal the recomputations in 16.7 and 16.8;
- `fig3_data` and `fig4_data` need the archive; their corner and central values agree with the 9cases runs to about 1e-5.

**Inconsistencies between figures, kept as published:**
- Fig 4 filters on convergence, Fig 3 does not, and Fig 5 plots 33 non-converged rows.
- Fig 4 resolves duplicates by most evaluations, and SI Fig 3 by best objective.
- SI Fig 2's g_μ uses simple growth, whereas r_μ uses log growth.

## 17. Verifying a reimplementation

### 17.1 Regression suite

`tests/` holds a golden baseline generated from the pre-cleanup release code (`baseline_meta.json`: commit `fb5965c3`). `.venv/bin/python -m pytest` runs it in about a minute. A reimplementation can be checked against the same files. Arrays are compared scale-aware: `max|new − base| ≤ rtol·max|base|`.

| Case | What | rtol |
|---|---|---|
| R1–R9 | the 9 reference runs, replayed with their stored f, s; vs. `tests/baseline/R*.npz` (all outputs, including per-quantile arrays, except the removed keys below) | 1e-12 (τ = 0), 1e-9 (τ > 0) |
| R1–R9 published | the same replays vs. the published `*_results.csv` (all columns except `r_consumption`) | 1e-9 |
| D1 | DICE-comparison config `dice_reference_r2_2030_gid-1_20260720_015723.json` (G ≡ 0) with gid-1's controls | 1e-12 |
| B1–B6 | Pareto Lorenz; income-dependent aggregate damage; η = 1; η = 0.8 with τ = 0.9; 10 GtCO2 emission pulse and $10B consumption pulse in 2030; logistic L, piecewise-linear θ₁, constant emission_ratio = 1.0 (configs in `tests/cases/`, controls of gid-5) | 1e-9 (B3: 1e-12) |
| O1 | full horizon, 4 f + 4 s points, 1 iteration, 300 evaluations | 1e-12, same evaluation count |
| O2 | 2020–2070, 4 → 6 points, 2 iterations, run to XTOL convergence | 10× the spread under a ±1e-10 initial-guess perturbation |
| O3 | prescribed s = 0.256, f at explicit years 2020, 2030, 2050, 2100, 2200, 2420, 300 evaluations | 1e-12, same evaluation count |
| K | kernels: tax solve, post-tax schedule, Lorenz derivatives, control times, PCHIP. Their inputs are defined in `tests/regression_cases.py:run_kernels`. The `control_times_*` arrays hold the times, then r₀, then a placeholder 0. | |

The baseline files also hold seven variables removed in 2026, which a reimplementation should skip (`REMOVED_OUTPUT_KEYS` in `tests/regression_cases.py`):
- `Gini` (a duplicate of `gini`);
- `Lambda` (an alias of `lambda_abate`);
- `dEcum_dt` (e·L, *without* the emission pulse);
- the four welfare-adjusted-MAC columns: `mu_dollar_numeraire`, `mu_dollar_cost`, `welfare_adjusted_mac`, `mac_over_wmac`.

**The published CSVs vs. the current code.** Published time-function columns differ from the current evaluation by 1 ulp at some points (A, L, θ₁, gini, emission_ratio), and by up to 9.4e-13 relative for σ. That is 3.4e-13 scale-aware, the worst of all published columns. The cause has not been identified. Every other published column agrees to 1.6e-14 or better.

O1 and O3 replay NLopt's exact path. They pass only for an implementation whose objective values agree to the last bit or nearly so. A reimplementation that differs at round-off should be expected to fail them, while still matching O2 and the forward cases.

**Clean-room test (2026-10-26).** An agent with access only to this document, the JSON configurations, the published CSVs and the baseline files (not `src/`, `scripts/` or `tests/*.py`) reimplemented Sections 4–12. Results:
- R1–R9, D1 and B1–B6: bit-identical to the baseline in every variable (worst difference 0.0);
- O1, O2 and O3: bit-identical, with the same evaluation counts;
- the control-time grids, the Section 17.2 checkpoint values and the objectives J: all matched.

Its questions about ambiguous passages have been resolved in the text: the operation order in `r_consumption`, which ω is output as `omega_yi`, the duplicate-time rule, the initial guess for explicit years, and the removed baseline keys.

### 17.2 Checkpoint values

Replay of the stored controls, printed with `%.15e`, at selected years. The objective J is from Section 10.

**gid-1 (τ = 0, β = 0):** J = −6487938701.598822 (published: −6.487938701599e-04 × 1e13)

| Year | f | s | K | Ecum | ΔT | y_gross | y_net | μ | Ω | log ΔU | U |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | 1.543376572092825 | 0.4786271018324223 | 2.95e14 | 2.327128215854423e12 | 1.16966 | 1.750372714528077e4 | 1.741645953981179e4 | 3.996318342682483e-2 | 4.743218355585199e-3 | −8.319879522643591 | −2.360533615801890e8 |
| 2030 | 1.629652538702295 | 0.2296397516562726 | 3.402759806869976e14 | 2.910845600510554e12 | 1.470360532030760 | 2.027360929257489e4 | 2.010579385432332e4 | 1.280010855705555e-1 | 7.495515646431158e-3 | −7.145717787234554 | −1.915743595569843e8 |
| 2100 | 2.395393167065027 | 0.2300201664610637 | 1.343287212994556e15 | 6.301202429344204e12 | 3.107513349030306 | 6.805266729437064e4 | 6.519107748253394e4 | 6.313729676087462e-1 | 3.347956815633018e-2 | −4.720976989850267 | −1.180399699134512e8 |
| 2420 | 1.932879549079158 | 2.775557561562892e-17 | 5.034453598921592e16 | 3.320522809349987e12 | 1.215602839595733 | 3.922956480656647e6 | 3.901326955944916e6 | 4.059728434018099e-1 | 5.123152144016334e-3 | −7.842941103688044 | −1.281976749031308e7 |

**gid-5 (τ = 0.5, β = 0.5):** J = −6508855233.019977 (published: −6.508855233020e-04 × 1e13)

| Year | f | ΔT | y_net | μ | Ω | log ΔU | U |
|---|---|---|---|---|---|---|---|
| 2030 | 2.009054232020102 | 1.455380714058155 | 2.004131335713467e4 | 2.209732540863020e-1 | 7.343567190229361e-3 | −6.192432774503468 | −1.922759044738247e8 |
| 2100 | 2.586382103696630 | 2.573472452959354 | 6.545325083882004e4 | 8.311015053237301e-1 | 2.296111053610958e-2 | −4.366620496532393 | −1.181557879701685e8 |

**gid-9 (τ = 0.998, β = 1):** J = −6510344441.554529 (published: −6.510344441555e-04 × 1e13)

| Year | f | ΔT | y_net | μ | Ω | log ΔU | U |
|---|---|---|---|---|---|---|---|
| 2030 | 2.427708597518568 | 1.425295047154427 | 1.977648257535698e4 | 4.036441750960487e-1 | 7.043092522992671e-3 | −584.9238997323782 | −1.929314677373643e8 |
| 2100 | 2.784386550723835 | 1.684852455816776 | 6.492391341398402e4 | 1.105113408535717 | 9.841869275221600e-3 | −462.4754453865223 | −1.168769728067995e8 |

In every run, ΔT(2020) = 1.16966 and y_gross(2020) = 1.750372714528077e4, since both are independent of the controls.

### 17.3 Re-optimization

Section 1 gives the acceptance criteria for a full re-optimization: objective within about 1e-13 relative, and f(t) within a few 1e-6. A full manuscript optimization (5 iterations, up to 100,000 evaluations in the last) took 14–89 minutes per run with the current code on one core (gid-1, -5, -9). Over the 2020–2070 horizon of test O2, the objective is flat near the optimum, so controls agree only to about 1e-3 while the objective agrees to about 1e-10.

## 18. Numerical caveats

These are properties of the published method, documented rather than changed. A faithful reimplementation must keep them.

1. **Lagged damage distribution.** Each step uses the previous step's income distribution of damage, rescaled by this step's Ω_base (Section 7.2). The first step uses uniform damage.
2. **Mean of pre-tax income.** The pre-tax income nodes do not average exactly to y_damaged (Section 7.7). The flat-tax closed form assumes they do (Section 7.8).
3. **Quadrature error.** The empirical Lorenz income mean has a relative quadrature error of 3–7e-7 at n_quad = 32 (Section 6.1). For the Pareto curve, which is not used for results, the error is 2–10%.
4. **Threshold comparisons.** `eta == 1` is an exact comparison in the utility, whereas |η_eff − 1| < ε is used in the tax. Many guards compare against ε = 1e-12 in absolute terms, applied to quantities of very different scale (dollars per person, fractions).
5. **Infeasible trajectories** score −1e17 (a caught exception) or −1e30 (a non-finite objective) at the NLopt interface (Section 12.3). Exceptions include a Gini above G_b, which would be a configuration error, not an infeasible policy.
6. **Ecum ≥ 0** is the only state constraint (Section 8). μ is effectively uncapped (μ_max = 1e20).
7. **`U` vs. `aggregate_utility`.** The CSV labels of these columns are misleading (Section 14).
