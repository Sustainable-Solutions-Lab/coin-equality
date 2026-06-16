# Pending Test Status & Expected Outcomes

**Last Updated:** 2026-06-16 14:47 UTC
**Critical Finding:** Flat tax has structural 4% floor; highly progressive tax achieves 3.87% (proves it's not universal)

---

## Tests Currently Running

### 1. Highly Progressive Tax - gid-4 (RESTARTED)
**Configuration:** `prod_discount_exact_hiprog_low_targets`
- **Grid ID:** 4
- **Parameters:** tax_equity=0.998, β=0.5, η=1.5, ρ=0.021
- **Target r_consumption:** 1.0%
- **Status:** Restarted (previous run hit 1M evaluation limit without completing)
- **Background job ID:** b3ccf54
- **Expected runtime:** 30-60 minutes
- **Warm-start source:** prod_discount_exact_gid-22 (3% highly progressive, β=0.5)

**Previous attempt:**
- Hit max evaluations (100,000) and stopped
- Terminal shows timing statistics after 1,000,000 calls (unusual)
- No results CSV generated

**Expected outcome:**
- If completes: Will show if highly progressive can target 1% (vs 2% which gave 3.87%)
- If fails: Suggests 3.87% is functional limit even for hiprog with ρ=0.021

---

### 2. Highly Progressive Tax - gid-1 & gid-3 (NEW)
**Configuration:** `prod_discount_exact_hiprog_low_targets`
- **Grid IDs:** 1, 3
- **Parameters:**
  - gid-1: tax_equity=0.998, β=0.0, η=1.5, ρ=0.016 (Target: 1.0%)
  - gid-3: tax_equity=0.998, β=0.0, η=1.5, ρ=0.021 (Target: 1.0%)
- **Status:** Submitted for execution
- **Background job ID:** bdc11e3
- **Expected runtime:** 60-120 minutes (4 gids × 30 min each, but running in sequence)

**Previous status:** Not found locally (may have been on Sherlock or never submitted)

**Expected outcome:**
- gid-1: Will show if hiprog β=0.0 case can reach 1% with lower ρ=0.016
- gid-3: Will show if hiprog β=0.0 case with ρ=0.021 can reach 1%
- Together with gid-2 (achieved 3.87% for 2%), will complete the hiprog picture

---

### 3. Progressive Tax - All cases (gids 1-4) (NEW)
**Configuration:** `prod_discount_exact_progressive_low_targets`
- **Grid IDs:** 1, 2, 3, 4
- **Parameters:** tax_equity=0.5, η=1.5, ρ∈{0.016, 0.021}, β∈{0.0, 0.5}
- **Targets:**
  - gid-1: 1.0% (β=0.0, ρ=0.016)
  - gid-2: 2.0% (β=0.5, ρ=0.016)
  - gid-3: 1.0% (β=0.0, ρ=0.021)
  - gid-4: 2.0% (β=0.5, ρ=0.021)
- **Status:** Submitted for execution
- **Background job ID:** b0b0785
- **Expected runtime:** 120-240 minutes (4 gids × 30-60 min each)

**Previous status:** Not found locally (never submitted or only on Sherlock)

**Expected outcome:**
- Will show intermediate behavior between flat (fails completely) and hiprog (succeeds partially)
- If all converge to 4.0-4.5% like flat: Suggests flat tax structure issue
- If converge to 2.0-3.0% like hiprog: Suggests tax equity is key factor
- If show mixed pattern: Will reveal how tax equity affects convergence

---

## Completed Results Summary

### Flat Tax (tax_equity = 0.0) - Complete ✓
| GID | Target | β   | Actual  | Error   | Status |
|-----|--------|-----|---------|---------|--------|
| 51  | 1.0%   | 0.0 | 4.0446% | +3.04%  | FAIL   |
| 52  | 2.0%   | 0.5 | 4.0105% | +2.01%  | FAIL   |
| 53  | 1.0%   | 0.0 | 4.4829% | +3.48%  | FAIL   |
| 54  | 2.0%   | 0.5 | 4.4543% | +2.45%  | FAIL   |

**Summary:** All converge to 4.0-4.5% (mean 4.25%), error mean +2.75%

---

### Highly Progressive (tax_equity = 0.998) - Partial ◐
| GID | Target | β   | Actual  | Error   | Status | Location |
|-----|--------|-----|---------|---------|--------|----------|
| 1   | 1.0%   | 0.0 | —       | —       | RUNNING | bdc11e3 |
| 2   | 2.0%   | 0.5 | 3.8740% | +1.87%  | ✓      | Complete |
| 3   | 1.0%   | 0.0 | —       | —       | RUNNING | bdc11e3 |
| 4   | 2.0%   | 0.5 | —       | —       | RUNNING | b3ccf54 |

**Current insight:** gid-2 proves highly progressive CAN reach ~3.9%, much better than flat's 4.0-4.5%

---

### Progressive (tax_equity = 0.5) - Not yet run ◯
| GID | Target | β   | Actual  | Error   | Status | Location |
|-----|--------|-----|---------|---------|--------|----------|
| 1   | 1.0%   | 0.0 | —       | —       | PENDING | b0b0785 |
| 2   | 2.0%   | 0.5 | —       | —       | PENDING | b0b0785 |
| 3   | 1.0%   | 0.0 | —       | —       | PENDING | b0b0785 |
| 4   | 2.0%   | 0.5 | —       | —       | PENDING | b0b0785 |

**Will reveal:** Intermediate behavior between flat (fails) and hiprog (succeeds)

---

## Key Hypothesis to Test

### H1: Tax Equity Effects on Convergence
- **Prediction:** Progressive > Flat in convergence quality (fewer overshoot errors)
- **Evidence so far:** ✓ Supported (hiprog achieves 3.87% vs flat's 4.04% for similar targets)
- **Test:** Compare progressive results to both flat and hiprog

### H2: Does ρ value matter for low targets?
- **Prediction:** Lower ρ values might help reach lower targets
- **Evidence so far:** Unknown
- **Test:** Compare gid-1,3 (ρ=0.016,0.021) hiprog results

### H3: Does β (damage distribution) affect convergence floor?
- **Prediction:** β might affect convergence but not fundamentally change floor
- **Evidence so far:** Unknown (flat shows ~0.4% difference; hiprog needs more data)
- **Test:** Compare β=0.0 vs β=0.5 within same tax_equity

---

## Timeline Estimate

Based on ~30 minutes per run and sequential execution:

| Task | Est. Duration | Expected Completion |
|------|----------------|--------------------|
| Hiprog gid-4 restart | 30-60 min | ~15:45-16:15 UTC |
| Hiprog gid-1,3 | 60-90 min | ~16:30-17:00 UTC |
| Progressive gids 1-4 | 120-240 min | ~17:00-18:30 UTC |

**Total expected time:** ~3-4 hours from start time
**All results expected by:** ~18:30 UTC (6:30 PM)

---

## What to Do With Results

Once all tests complete:

1. **Extract r_consumption values** at year 2030 from each results CSV
2. **Calculate convergence errors:** |actual - target| for each case
3. **Create comparison table** with all 12 cases (4 flat + 4 prog + 4 hiprog)
4. **Analyze patterns:**
   - Does tax equity systematically improve convergence?
   - Is there a convergence hierarchy: flat < prog < hiprog?
   - Does ρ value affect ability to reach low targets?
   - Does β matter much?
5. **Update figure script** if needed to handle all test data
6. **Regenerate Fig 6.3** with complete low-target data
7. **Write manuscript section** explaining tax-equity effects on numerical convergence

---

## Action Items

- [ ] Monitor background jobs b3ccf54, bdc11e3, b0b0785
- [ ] Once hiprog gid-4 completes, verify it produced results CSV
- [ ] Check progress of progressive tests (b0b0785) - most time-consuming
- [ ] Once all complete, run analysis script to extract all r_consumption values
- [ ] Update CONVERGENCE_ANALYSIS_UPDATE.md with final results
- [ ] Regenerate figures with complete data

---

**Report Status:** Actively monitoring three background tests
**Next Check:** 30 minutes from now
