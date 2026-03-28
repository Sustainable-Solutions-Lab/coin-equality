# Config 037: Validation Rerun of Config 937

**Purpose:** Validate that config_937 results are reproducible and working properly

**Created:** March 28, 2026

**Source:** Duplicated from config_937 (100k evaluation variants only)

---

## Files Created (6 configs)

### Tax Policy: r-uniform (Uniform fractional tax)
- `config_037_r-uniform_f-f-f-f_100k_6000-3.7_base.json`
- `config_037_r-uniform_t-f-f-f_100k_6000-3.7_base.json`

### Tax Policy: u-max (Progressive tax - tax richest)
- `config_037_u-max_f-f-f-f_100k_6000-3.7_base.json`
- `config_037_u-max_t-f-f-f_100k_6000-3.7_base.json`

### Tax Policy: u-uniform (Equal utility loss taxation)
- `config_037_u-uniform_f-f-f-f_100k_6000-3.7_base.json`
- `config_037_u-uniform_t-f-f-f_100k_6000-3.7_base.json`

---

## Scenarios Covered

**f-f-f-f** (no income-dependent features):
- All 3 tax policies × 100k evals = 3 files

**t-f-f-f** (income-dependent damage only):
- All 3 tax policies × 100k evals = 3 files

**Total:** 6 configs (3 tax policies × 2 scenarios)

---

## Naming Convention Explanation

**Filename format:** `config_037_{tax_policy}_{scenario}_100k_6000-3.7_base.json`

- **037**: Config number (validation rerun of 937)
- **tax_policy**: `r-uniform` | `u-max` | `u-uniform`
- **scenario**: 4-flag system (e.g., `f-f-f-f`, `t-f-f-f`)
- **100k**: 100,000 optimization evaluations
- **6000**: theta1 initial value (≈6035.6 $/tCO2 backstop cost)
- **3.7**: Historical reference (unclear mapping to current parameters)

### Scenario Flags (4 flags)
```
Position:  1st 2nd 3rd 4th
Example:   f - f - f - f
           │   │   │   └─ income_dependent_redistribution_policy
           │   │   └───── income_redistribution
           │   └───────── income_dependent_aggregate_damage
           └───────────── income_dependent_damage_distribution
```

---

## Validation Plan

1. **Run all 6 configs** using `scripts/run_optimization.py`
2. **Compare results** with original config_937 outputs
3. **Check reproducibility**: MAC, SCC, utility values should match within numerical precision
4. **Expected differences**: Only timestamps and random seed variations (if any)

---

## What to Compare

For each matching config pair (037 vs 937):
- Optimized objective function value
- Final MAC values (marginal abatement cost)
- Final μ trajectory (abatement fraction)
- Final SCC values (if computed)
- Convergence behavior

---

## Running the Validation

### Single config:
```bash
python scripts/run_optimization.py json/config_037_r-uniform_f-f-f-f_100k_6000-3.7_base.json
```

### All 6 configs in parallel:
```bash
python scripts/run_parallel.py json/config_037_r-uniform*.json json/config_037_u-max*.json json/config_037_u-uniform*.json
```

Or simply:
```bash
python scripts/run_parallel.py json/config_037*.json
```

---

## Notes

- Only 100k evaluation variants included (3k variants excluded)
- All configs use clean DICE inversion (no rate constraints)
- `mu_max = 1e20` (essentially unlimited)
- Focus: tax policy comparison (r-uniform vs u-max vs u-uniform)
- Anomaly file (5-flag format) removed — only clean config_937 format retained

---

## Current Parameter Values (Baseline)

All 6 config_037 files use identical parameter values for validation:

| Parameter | Symbol | Current Value | Units | Description |
|-----------|--------|---------------|-------|-------------|
| **Pure rate of time preference** | ρ (rho) | **0.026** | yr⁻¹ | Discount rate (2.6% per year) |
| **Coefficient of relative risk aversion** | η (eta) | **1.45** | — | Exponent on utility function |
| **GDP-dependency of climate damage** | — | **0.41** | — | Exponent for income-dependent damage distribution |

### Additional Key Parameters
| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Abatement cost exponent | θ₂ | 2.6 | Controls steepness of MAC curve |
| Initial backstop cost | θ₁(0) | 6035.6 | Initial marginal abatement cost ($/tCO₂) |
| Capital share | α | 0.3 | Output elasticity of capital |
| Depreciation rate | δ | 0.1 | Capital depreciation (10% per year) |

---

## Planned Sensitivity Analysis (Future Work)

After validation, sensitivity tests will explore these parameter ranges:

| Parameter | Baseline (037) | Proposed Range | Notes |
|-----------|---------------|----------------|-------|
| **ρ (rho)** | 0.026 | 0.00 – 0.05 | Pure rate of time preference (discount rate) |
| **η (eta)** | 1.45 | 0.5 – 2.0 | Coefficient of relative risk aversion |
| **y_damage_exp** | 0.41 | 0.0 – 1.0 | Exponent on GDP-dependency of climate damage |

### Sensitivity Test Interpretation

**ρ (discount rate):**
- **Lower values** (ρ → 0): More weight on future welfare → more aggressive climate action
- **Higher values** (ρ → 0.05): Less weight on future → less climate action today
- **ρ = 0.026**: DICE baseline (2.6% pure time preference)

**η (risk aversion):**
- **Lower values** (η → 0.5): Less concerned about inequality → less redistribution
- **Higher values** (η → 2.0): More concerned about inequality → more redistribution
- **η = 1.0**: Logarithmic utility (special case)
- **η = 1.45**: DICE baseline

**y_damage_exp (damage distribution):**
- **0.0**: Income-independent damage (everyone equally affected)
- **0.41**: Baseline (poor more vulnerable to climate damage)
- **1.0**: Strong income-dependence (damage scales inversely with income)

**Note:** Final ranges TBD after validation results are reviewed.
