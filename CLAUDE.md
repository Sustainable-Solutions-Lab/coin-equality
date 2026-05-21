# Claude Code Style Guide for COIN_equality

## Coding Philosophy
This project prioritizes elegant, fail-fast code that surfaces errors quickly rather than hiding them.

### Root Cause Analysis
- Always investigate and understand the root cause of problems before implementing solutions
- Avoid band-aid fixes that mask symptoms without addressing underlying issues
- When unexpected behavior occurs, trace it back to its source rather than applying quick patches
- Document the reasoning behind fixes to prevent similar issues

## Core Style Requirements

### Error Handling
- No input validation on function parameters (except for command-line interfaces)
- No defensive programming - let exceptions bubble up naturally
- Fail fast - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- No try-catch blocks unless absolutely necessary for program logic (not error suppression)
- No optional function arguments - all parameters must be explicitly provided
- Assume complete data - do not check for missing data fields. If required data is missing, let the code fail with natural Python errors

### Code Elegance
- Minimize conditional statements - prefer functional approaches, mathematical expressions, and numpy vectorization
- Favor mathematical clarity over defensive checks
- Use numpy operations instead of loops and conditionals where possible
- Compute once, use many times - move invariant calculations outside loops and create centralized helper functions
- No backward compatibility - do not add conditional logic to support deprecated field names or old configurations. Update all code and configurations to use current conventions.
- Use standard Python packages - prefer established numerical methods from scipy, numpy, etc. rather than implementing custom numerical algorithms
- **No magic numbers in simulation code** - for simulation code (`scripts/run_optimization.py`, `scripts/run_integration.py`, and their dependencies), all numerical constants must be defined in `src/constants.py` with clear documentation. Never hardcode constants like `1e-13`, `1e-6`, or other numerical values directly in simulation code (except for `0` and `1`, which are acceptable as they represent fundamental mathematical identities). This ensures:
  - Constants are defined in one central location
  - Their purpose and usage are clearly documented
  - Values can be adjusted without searching through multiple files
  - The codebase remains maintainable and understandable
- **Postprocessing code exception** - for postprocessing and visualization scripts (e.g., `scripts/compare_results.py`, `scripts/plot_*.py`), hardcoding numerical constants is acceptable since these do not affect simulation results

### Code Organization
- All imports at the top of the file - no imports inside functions or scattered throughout the code

### Figure Style
- Figure style guidelines and color palette are defined in `scripts/create_publication_figures.py`

### Protected Directories
- Never modify files in `./barrage_nordhaus_2023/` - this directory contains reference materials that must remain unchanged

### Naming Conventions
- Consistent naming - use the same variable/field names throughout the codebase when referring to the same concept
- Descriptive names preferred - long, clear names are better than short, ambiguous ones
- Follow source naming - when accessing data from a structure (e.g., `config['response_function_scalings']['scaling_name']`), prefer using the same field name (`scaling_name`) or a composite (`response_function_scaling_name`) rather than inventing new descriptors
- Example: If config has `scaling_name` under `response_function_scalings`, use `scaling_name` or `response_function_scaling_name`, not `response_func` or other variations

### Function Design
- Functions should assume valid inputs and focus on their core mathematical/logical purpose
- Let Python's natural error messages guide debugging rather than custom error handling
- All function arguments must be explicitly provided - no default values (`=None`) or conditional logic
- Clean fail-fast approach - if required arguments are not supplied, the code should fail immediately with a clear error

## File Creation Rules
- **Unix line endings only**: All files created must use `\n` (LF), never `\r\n` (CRLF). After creating any file with the Write tool, always verify with `xxd <file> | head -3` and fix with `tr -d '\r' < file > /tmp/fix.tmp && mv /tmp/fix.tmp file` if `0d0a` appears.
- **Shell scripts for Sherlock**: Keep commands on a single line. Never use backslash line continuations — they break when pasted on Sherlock. If a command is too long for one line, write a helper Python script instead that generates the shell script on Sherlock (see `scripts/create_060_warmstart_slurm.py` for the pattern).
- **Commands given to the user**: Must be short single-line commands that won't wrap in the terminal. Long commands get split when pasted and cause parse errors. If a task requires a complex command, wrap it in a script file instead.

