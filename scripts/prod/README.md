# Production Scripts

**These scripts ARE tracked in git** - they are essential tools for the project.

## Purpose

Scripts that are used in actual workflows:
- Selecting/filtering data points
- Processing results from optimizations
- Creating publication figures
- Data transformations and analysis
- Building datasets for analysis
- Preparing outputs for manuscript

## Examples

- `select_points_*.py` - Select specific optimization results
- `build_results_*.py` - Compile results into datasets
- `create_*.py` - Create publication figures
- `process_*.py` - Process raw outputs
- `prepare_*.py` - Prepare data for analysis
- `extract_*.py` - Extract values from results
- `*.sh` - Submission or execution scripts for production runs

## Guidelines

✓ **Use this folder for:**
- Scripts that are part of standard workflows
- Scripts that other people/future-you will use
- Scripts that produce publication-ready outputs
- Scripts that are documented and maintained

❌ **Don't use this folder for:**
- One-off exploratory scripts → use `Lamprini/scripts/`
- Debug/diagnostic scripts → use `Lamprini/scripts/`
- Temporary analysis → use `Lamprini/analysis/`

## Standards

Production scripts should:
1. Have clear docstrings explaining purpose
2. Handle errors gracefully
3. Be reproducible (no hard-coded paths)
4. Follow project coding conventions
5. Be documented in commit messages

---

**All files in this folder are automatically tracked in git.**
