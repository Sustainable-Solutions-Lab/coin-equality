# Sensitivity Analysis Plan

**Created:** March 28, 2026
**Status:** Planning phase (after config_037 validation)
**Purpose:** Document planned sensitivity tests for key model parameters

---

## Phase 1: Validation (Current)

**Config 037** — Validate reproducibility of config_937 results
- 6 configs (3 tax policies × 2 scenarios)
- 100k evaluations each
- All use baseline parameter values

**Status:** Ready to run

---

## Phase 2: Sensitivity Analysis (Next)

### Three Key Parameters to Test

#### 1. Pure Rate of Time Preference (Discount Rate)

**Parameter:** `rho` (ρ)
**Baseline:** 0.026 (2.6% per year)
**Proposed range:** 0.00 – 0.05
**Units:** yr⁻¹

**Interpretation:**
- **Lower values** (ρ → 0): More weight on future welfare → more aggressive climate action
- **Higher values** (ρ → 0.05): Less weight on future → less climate action today
- **ρ = 0.026**: DICE baseline (Barrage & Nordhaus 2023)

**Economic meaning:**
Pure time preference represents how much society values consumption today vs. consumption in the future, independent of growth. Lower values imply stronger intergenerational equity concerns.

**Suggested test points:** 0.00, 0.01, 0.02, 0.026 (baseline), 0.03, 0.04, 0.05

---

#### 2. Coefficient of Relative Risk Aversion

**Parameter:** `eta` (η)
**Baseline:** 1.45
**Proposed range:** 0.5 – 2.0
**Units:** dimensionless

**Interpretation:**
- **Lower values** (η → 0.5): Less concerned about inequality → less redistribution
- **Higher values** (η → 2.0): More concerned about inequality → more redistribution
- **η = 1.0**: Logarithmic utility (special case, marginal utility = 1/c)
- **η = 1.45**: DICE baseline

**Utility function:**
- For η ≠ 1: `U(c) = c^(1-η) / (1-η)`
- For η = 1: `U(c) = ln(c)`

**Economic meaning:**
η measures both risk aversion and inequality aversion. Higher η means marginal utility declines faster with consumption, making redistribution more valuable.

**Suggested test points:** 0.5, 0.75, 1.0, 1.25, 1.45 (baseline), 1.75, 2.0

---

#### 3. Exponent on GDP-Dependency of Climate Damage

**Parameter:** `y_damage_distribution_exponent`
**Baseline:** 0.41
**Proposed range:** 0.0 – 1.0
**Units:** dimensionless

**Interpretation:**
- **0.0**: Income-independent damage (everyone equally affected by climate change)
- **0.41**: Baseline (poor moderately more vulnerable)
- **1.0**: Strong income-dependence (damage scales inversely with income)

**Damage formula:**
`ω(y_net) = Ω_base × (y_net / y_ref)^(-y_damage_exp)`

Where:
- `ω(y_net)`: Damage fraction for income level y_net
- `Ω_base`: Base damage from temperature
- `y_ref`: Reference income level (17,210.79 $/person/yr)

**Economic meaning:**
Controls how much climate damage is concentrated on low-income populations. Higher values mean poor people suffer disproportionately more from climate impacts.

**Suggested test points:** 0.0, 0.2, 0.41 (baseline), 0.6, 0.8, 1.0

---

## Implementation Strategy: Structured Approach

### Recommended Phased Strategy

We recommend a **3-phase approach** that builds understanding incrementally:

---

### Phase 1: Core Validation (6 configs)
**Purpose:** Confirm config_037 reproduces config_937 results

| Config | Tax Policy | Scenario | Evals | Parameters |
|--------|-----------|----------|-------|------------|
| 037_1 | r-uniform | f-f-f-f | 100k | ρ=0.026, η=1.45, y_dmg=0.41 |
| 037_2 | r-uniform | t-f-f-f | 100k | ρ=0.026, η=1.45, y_dmg=0.41 |
| 037_3 | u-max | f-f-f-f | 100k | ρ=0.026, η=1.45, y_dmg=0.41 |
| 037_4 | u-max | t-f-f-f | 100k | ρ=0.026, η=1.45, y_dmg=0.41 |
| 037_5 | u-uniform | f-f-f-f | 100k | ρ=0.026, η=1.45, y_dmg=0.41 |
| 037_6 | u-uniform | t-f-f-f | 100k | ρ=0.026, η=1.45, y_dmg=0.41 |

**Estimated runtime:** 6 configs × ~1 hour = 6 hours

---

### Phase 2: One-at-a-Time Sensitivity (60 configs)
**Purpose:** Understand individual parameter effects on tax policy comparison

For each parameter, test multiple values while holding others at baseline. Run **only 1 scenario** (f-f-f-f, no income effects) and **all 3 tax policies** to see how tax policy ranking changes.

#### Phase 2a: Discount Rate Sensitivity (21 configs)
**Vary:** ρ = 0.00, 0.01, 0.02, 0.026, 0.03, 0.04, 0.05 (7 values)
**Fixed:** η = 1.45, y_damage_exp = 0.41
**Scenarios:** f-f-f-f only
**Tax policies:** All 3 (r-uniform, u-max, u-uniform)
**Total:** 7 ρ values × 3 tax policies = **21 configs**

**Key questions:**
- How does discount rate affect optimal MAC trajectory?
- Which tax policy is best at high vs low discount rates?
- Does u-uniform maintain welfare advantage across discount rates?

#### Phase 2b: Risk Aversion Sensitivity (21 configs)
**Vary:** η = 0.5, 0.75, 1.0, 1.25, 1.45, 1.75, 2.0 (7 values)
**Fixed:** ρ = 0.026, y_damage_exp = 0.41
**Scenarios:** f-f-f-f only
**Tax policies:** All 3
**Total:** 7 η values × 3 tax policies = **21 configs**

**Key questions:**
- How does risk aversion affect tax policy comparison?
- At what η does u-uniform show clear advantage over r-uniform?
- Does u-max become more attractive at high η?

#### Phase 2c: Damage Distribution Sensitivity (18 configs)
**Vary:** y_damage_exp = 0.0, 0.2, 0.41, 0.6, 0.8, 1.0 (6 values)
**Fixed:** ρ = 0.026, η = 1.45
**Scenarios:** f-f-f-f only (no income-dependent damage, so this tests model behavior)
**Tax policies:** All 3
**Total:** 6 y_damage_exp values × 3 tax policies = **18 configs**

**Note:** This parameter only matters when income_dependent_damage_distribution = true (scenario t-f-f-f), but we test it in f-f-f-f first to establish baseline behavior.

**Phase 2 total:** 21 + 21 + 18 = **60 configs**
**Estimated runtime:** 60 configs × ~1 hour = 60 hours (~2.5 days)

---

### Phase 3: Two-Parameter Interactions (Optional, 72 configs)
**Purpose:** Test critical parameter interactions where we expect non-linear effects

#### Phase 3a: Risk Aversion × Damage Distribution (18 configs)
**Motivation:** Does the benefit of progressive taxation (u-uniform, u-max) depend on both inequality aversion (η) AND vulnerability distribution (y_damage_exp)?

**Test combinations:**
- η = {0.5, 1.45, 2.0} (low, baseline, high)
- y_damage_exp = {0.0, 0.41, 1.0} (none, baseline, strong)
- Scenario: **t-f-f-f** (income-dependent damage enabled)
- Tax policies: All 3

**Total:** 3 η × 3 y_damage_exp × 3 tax policies = **27 configs**

**Key questions:**
- Is progressive taxation only valuable when both η and y_damage_exp are high?
- Does u-uniform dominate across all combinations?

#### Phase 3b: Discount Rate × Damage Distribution (18 configs)
**Motivation:** Does future climate damage vulnerability affect the optimal discount rate choice?

**Test combinations:**
- ρ = {0.00, 0.026, 0.05} (zero, baseline, high)
- y_damage_exp = {0.0, 0.41, 1.0}
- Scenario: **t-f-f-f**
- Tax policies: All 3

**Total:** 3 ρ × 3 y_damage_exp × 3 tax policies = **27 configs**

**Key questions:**
- Does vulnerability distribution shift the optimal discount rate?
- Do patient societies (low ρ) benefit more from progressive taxation?

#### Phase 3c: Discount Rate × Risk Aversion (18 configs)
**Motivation:** Classic interaction in optimal growth theory

**Test combinations:**
- ρ = {0.00, 0.026, 0.05}
- η = {0.5, 1.45, 2.0}
- Scenario: f-f-f-f
- Tax policies: All 3

**Total:** 3 ρ × 3 η × 3 tax policies = **27 configs**

**Phase 3 total:** 27 + 27 + 27 = **81 configs**
**Estimated runtime:** 81 configs × ~1 hour = 81 hours (~3.4 days)

---

## Summary of Phased Approach

| Phase | Purpose | Configs | Runtime | Priority |
|-------|---------|---------|---------|----------|
| **Phase 1** | Validation | 6 | 6 hrs | **Required** |
| **Phase 2a** | Discount rate (ρ) | 21 | 21 hrs | **High** |
| **Phase 2b** | Risk aversion (η) | 21 | 21 hrs | **High** |
| **Phase 2c** | Damage distribution | 18 | 18 hrs | **Medium** |
| **Phase 3a** | η × y_damage_exp | 27 | 27 hrs | **Low** |
| **Phase 3b** | ρ × y_damage_exp | 27 | 27 hrs | **Low** |
| **Phase 3c** | ρ × η | 27 | 27 hrs | **Low** |
| **TOTAL** | | **147** | **147 hrs** | |

**Recommended minimum:** Phase 1 + Phase 2a + Phase 2b = **48 configs, 48 hours**

---

## Alternative: Reduced Test Set (30 configs)

If computational budget is limited, test only the most critical parameter points:

### Minimal Sensitivity Set (30 configs)

**Discount Rate (12 configs):**
- ρ = {0.00, 0.026, 0.05} × 3 tax policies × f-f-f-f scenario + t-f-f-f scenario
- But run only 1 tax policy (u-uniform) for t-f-f-f: 3 × 3 + 3 × 1 = **12 configs**

**Risk Aversion (12 configs):**
- η = {0.5, 1.45, 2.0} × 3 tax policies × f-f-f-f + t-f-f-f
- Same logic: 3 × 3 + 3 × 1 = **12 configs**

**Damage Distribution (6 configs):**
- y_damage_exp = {0.0, 0.41, 1.0} × 3 tax policies × t-f-f-f only
- Only matters when damage distribution enabled: 3 × 3 × 1 = **9 configs**

**Total:** 12 + 12 + 6 = **30 configs**, ~30 hours

---

## Rejected Approaches

### Full Factorial (NOT RECOMMENDED)
- 7 ρ × 7 η × 6 y_damage_exp × 3 tax policies × 2 scenarios = **1,764 configs**
- Would take 1,764 hours (~73 days) at 100k evals each
- Most combinations would provide little insight
- Parameter interactions are likely monotonic, don't need full coverage

**Decision:** Use phased approach (Phase 1 + Phase 2) or minimal set (30 configs)

---

## Current Parameter Values (Config 037 Baseline)

| Parameter | Symbol | Value | Units | Source |
|-----------|--------|-------|-------|--------|
| **Sensitivity test parameters** |
| Pure rate of time preference | ρ | 0.026 | yr⁻¹ | DICE 2023 |
| Risk aversion coefficient | η | 1.45 | — | DICE 2023 |
| Damage distribution exponent | — | 0.41 | — | Calibrated |
| **Other key parameters** |
| Abatement cost exponent | θ₂ | 2.6 | — | DICE 2023 |
| Initial backstop cost | θ₁(0) | 6035.6 | $/tCO₂ | DICE 2023 |
| Capital share | α | 0.3 | — | Standard |
| Depreciation rate | δ | 0.1 | yr⁻¹ | Standard |
| Temperature sensitivity | k_climate | 5.69×10⁻¹³ | K/tCO₂ | DICE 2023 |
| Quadratic damage coeff | ψ₂ | 0.003467 | °C⁻² | DICE 2023 |

---

## Questions to Answer with Sensitivity Analysis

1. **How sensitive are optimal carbon prices (MAC) to discount rate?**
   - Expected: Lower ρ → higher MAC (more climate action)

2. **How does risk aversion affect the tax policy comparison?**
   - Expected: Higher η → u-uniform performs better (welfare-efficient taxation)

3. **Does damage distribution exponent change optimal abatement levels?**
   - Expected: Higher exponent → more abatement if redistribution enabled

4. **Are there parameter interactions?**
   - E.g., does the benefit of redistribution depend on both η and y_damage_exp?

5. **Which parameters have the largest impact on results?**
   - Identify most critical uncertainties for policy recommendations

---

## Metrics to Track

For each sensitivity run, record:
- **Objective function value** (discounted total utility)
- **Mean MAC** over time
- **Peak MAC** (when does it occur?)
- **Total abatement** (cumulative ΔT avoided)
- **Utility gain from redistribution** (compare redistribution on/off)
- **Tax policy ranking** (which performs best for this parameter set?)
- **Convergence quality** (optimizer success)

---

## Timeline

**Now → Next session:**
- [ ] Run config_037 validation (6 configs)
- [ ] Compare results with config_937
- [ ] Verify reproducibility

**After validation:**
- [ ] Finalize sensitivity test ranges
- [ ] Choose implementation strategy (A, B, or C)
- [ ] Generate config files for sensitivity runs
- [ ] Run sensitivity analysis
- [ ] Create comparison plots and tables

---

## Notes for Tomorrow

1. **Validation first** — Make sure config_037 reproduces config_937 before expanding to sensitivity
2. **Start small** — One-at-a-time (Option B) is safer for initial exploration
3. **Check convergence** — With 100k evals, optimizer should converge well
4. **Document everything** — Each sensitivity run should have clear parameter labels
5. **Consider computational cost** — 120 configs × 100k evals × ~1 hour each = significant runtime

---

## References

- Barrage, L., & Nordhaus, W. (2023). *Policies, projections, and the social cost of carbon: Results from the DICE-2023 model*
- DICE model documentation: Standard parameter values and calibration
- Dennig et al. (2015): Income-dependent climate damage and optimal carbon pricing

---

*This is a planning document. Update as decisions are made and results come in.*
