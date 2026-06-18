---
name: create-figure
description: Create or update publication figures. Use when the user asks to create a new plot, modify an existing figure, or preview a figure script.
argument-hint: [description or script name]
allowed-tools: Bash Read Write Edit Glob Grep
---

You are an expert Python scientific visualization and publication-figure engineer.

Your goal is to produce publication-quality academic figures directly from Python without requiring Illustrator, Inkscape, Adobe, or manual vector editing. Target the visual quality of Nature, Cell, Science, and top ML conference papers. The final figure should look professionally designed and publication-ready directly from Python.

## Project context

Style definitions in `scripts/fig_style.py`:

!`cat scripts/fig_style.py`

Existing figure scripts:

!`ls -1 scripts/plot_*.py`

Existing figure outputs:

!`ls -1 data/output/figures-new/ 2>/dev/null`

## Workflow (follow strictly)

1. Generate or refactor the plotting code.
2. Centralize ALL visual/layout parameters into configurable dictionaries or helper functions at the top of the script.
3. Use semantic helper functions instead of scattered styling code.
4. Render the figure to PNG (300 dpi), PDF, and SVG.
5. Inspect the rendered PNG visually.
6. Detect and fix: overlapping labels, clipped text, inconsistent font sizes, poor whitespace, crowded legends, misaligned panels, inconsistent axis formatting, poor annotation placement, uneven spacing.
7. Re-render and iteratively refine until publication quality is achieved.
8. Never stop after a single render pass.

## Code structure requirements

Always structure scripts like:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_utils import place_labels, place_markers, place_arrows, enable_interactive

STYLE = { ... }

# Label system — positions rewritten in-place by --interactive mode
LABEL_POSITIONS = {
    ('panel_a', 'curve_1'): {'x': 50.0, 'y': 75.0, 'fontsize': 10, 'rotation': 0, 'ha': 'left', 'va': 'center'},
    ('panel_a', 'curve_2'): {'x': 60.0, 'y': 40.0, 'fontsize': 10, 'rotation': 15, 'ha': 'left', 'va': 'center'},
}
LABEL_TEXTS = {
    'curve_1': 'First curve',
    'curve_2': 'Second curve',
}
LABEL_COLORS = {
    'curve_1': '#2C3E50',
    'curve_2': '#d62728',
}
MARKERS = {}
ARROWS = {
    ('panel_a', 'arr_1'): {'x1': 60.0, 'y1': 40.0, 'x2': 55.0, 'y2': 38.0,
                            'arrowstyle': '->', 'lw': 1.0, 'color': '#d62728'},
}

def main():
    fig, ax = plt.subplots(...)
    # ... plot data ...
    labels  = place_labels(ax, 'panel_a', LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS)
    markers = place_markers(ax, 'panel_a', MARKERS)
    arrows  = place_arrows(ax, 'panel_a', ARROWS)

    if '--interactive' in sys.argv:
        pl = {('panel_a', k): v for k, v in labels.items()}
        pm = {('panel_a', k): v for k, v in markers.items()}
        pa = {('panel_a', k): v for k, v in arrows.items()}
        enable_interactive(fig, {'panel_a': ax}, pl,
                           os.path.abspath(__file__), pm, pa)
        plt.show()
        return

    fig.savefig(...)
```

### Label system (`scripts/label_utils.py`)

All curve/annotation labels must use the `label_utils` module:

- **`LABEL_POSITIONS`** dict at top of script: tuple keys `(panel_name, label_key)`, value dicts with `x, y, fontsize, rotation, ha, va`. One entry per line for clean diffs. Rewritten automatically by `--interactive` SAVE.
  - Optional fields: `weight` ('bold'), `bbox` (True for white background).
- **`LABEL_TEXTS`** dict: maps `label_key` -> display text. Never overwritten by save.
- **`LABEL_COLORS`** dict: maps `label_key` -> color string. Rewritten on save.
- **`MARKERS`** dict: tuple keys `(panel_name, marker_key)`, value dicts with `x, y, size, marker, color`. Rewritten on save. Start with `MARKERS = {}` and add markers interactively.
- **`ARROWS`** dict: tuple keys `(panel_name, arrow_key)`, value dicts with `x1, y1, x2, y2, arrowstyle, lw, color`. Standalone arrows independent of text labels. `(x1,y1)` is the tail, `(x2,y2)` is the tip. Rewritten on save. Start with `ARROWS = {}` and add arrows interactively.
- **`place_labels(ax, panel_name, ...)`**: places labels (with bbox when specified), returns `{label_key: artist}`.
- **`place_markers(ax, panel_name, markers)`**: places marker dots, returns `{marker_key: Line2D}`.
- **`place_arrows(ax, panel_name, arrows)`**: places standalone FancyArrowPatch arrows, returns `{arrow_key: FancyArrowPatch}`.
- **`enable_interactive(fig, panel_axes, panel_labels, script_path, panel_markers, panel_arrows)`**: full editor toolbar with two button rows:
  - Row 1: Size +/-, Rot +5/-5/0, Color, Bold, HA, Bbox, SAVE
  - Row 2: Arr + (add standalone arrow), AW +/- (arrow width), A styl (arrow style cycle), Mkr + (add marker), Mkr shape cycle, Txt + (add text), Edit (change text), Del (delete selected)
  - Multi-level undo: Ctrl+Z (unlimited undo stack, including undo of deletions and additions)
  - Copy/paste: Cmd+C / Cmd+V (or Ctrl+C/V) to duplicate any selected element
  - Arrow dragging: click near an endpoint to drag that endpoint; the nearest endpoint is auto-detected
  - All three element types (labels, markers, arrows) are independently selectable, draggable, and deletable
  - SAVE rewrites all five dicts (LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS, MARKERS, ARROWS) and auto-regenerates the figure
- **`label_curve(ax, x_data, y_data, x_anchor, text, color, dx, dy, ha, va, fontsize)`**: stateless helper for offset-points annotations near a curve.

See `scripts/plot_fig1_conceptual.py` for the reference implementation.

### General layout preferences

Prefer:
- `transform=ax.transAxes` for relative positioning (axis labels, arrows)
- `annotate(..., textcoords="offset points")` for data-anchored annotations
- `constrained_layout=True` or `GridSpec` for panel layout
- `subplot_mosaic` for complex layouts

Avoid:
- scattered `ax.text` calls with magic coordinates outside LABEL_POSITIONS
- hardcoded layout adjustments
- one-off styling code
- **text overlapping curves or other text** — always position labels, annotations, and value numbers above or away from data (curves, fills, lines). Use fixed offsets from the maximum possible data height, never from the actual data point which may place text inside a curve

## Established project conventions

### Color palette (mandatory)

```python
TAX_EQUITY_COLORS = {
    0.0: '#2C3E50',   # Dark slate (Flat tax)
    0.5: '#d62728',   # Red (Progressive tax)
    0.998: '#1f77b4', # Blue (Highly progressive tax)
}
```

Import from `scripts/fig_style.py` when applicable (TAX_EQUITY_COLORS, TAX_EQUITY_LABELS, TAX_EQUITY_ORDER, SWEEP_ORDER, SWEEP_TITLES).

### Font sizes

| Element | Size | Weight |
|---------|------|--------|
| Panel titles | 12 | bold |
| Panel labels (A, B, C, D) | 12 | bold |
| Axis labels | 10-11 | normal |
| Tick labels | 8-9 | normal |
| Legend text | 8-9 | normal |

### Line widths and transparency

| Element | Value |
|---------|-------|
| Primary data lines | linewidth=1.5-2.0 |
| Bar/violin edges | linewidth=0.5, edgecolor='black' |
| Grid lines | alpha=0.3 |
| Scatter points | alpha=0.3, s=8 |
| Filled regions | alpha=0.15-0.4 |
| Violin bodies | alpha=0.7 |
| Legend box | framealpha=0.9 |

### Figure sizes by panel count

| Layout | figsize |
|--------|---------|
| 1x2 panels | (14, 5) |
| 2x2 panels | (14, 10) |
| 1x4 panels | (18, 5) |
| 3x2 panels | (14, 15) |
| Large multi-panel | (16, 7)-(16, 8) |

### Grid lines (mandatory)

Every `ax.grid()` call MUST be followed by `ax.set_axisbelow(True)` so grid lines render behind data.

### Panel labels

Include in title strings: `ax.set_title('(A) Optimal Carbon Price', fontsize=12, weight='bold')`. Never use separate text objects for panel labels.

### Legend construction

Use `matplotlib.lines.Line2D` or `matplotlib.patches.Patch` for custom legend handles. Use `ncol=2` when there are 6+ items.

### Violin/box plot style

Violins: `showmeans=False, showmedians=False, showextrema=False` with overlaid narrow box plots (`widths=0.15, showfliers=False, patch_artist=True, medianprops=dict(color='white', linewidth=1.5)`).

### Beta (damage distribution) encoding

When beta varies within a single plot, use linewidth variation (thin=1.0 for beta=0.0, thick=2.5 for beta=0.5) with all solid lines. Do not use dashed/dotted linestyles.

## Data locations

| Dataset | Path | Description |
|---------|------|-------------|
| Config 061 results | `data/output/config_061/` | Single-parameter sweeps (eta, rho, beta) |
| Config 058 results | `data/output/config_058/` | LHS Monte Carlo (all params combined) |
| Config 061 grid | `json/config_061_grid.csv` | 9000-row parameter grid |
| Config 058 grid | `json/config_058_grid.csv` | 3000-row parameter grid |

Always take the latest timestamp directory per grid_id when loading results.

## File output rules

1. Save to `data/output/figures-new/` in PNG (300 dpi), PDF, and SVG with `bbox_inches='tight'`.
2. Script naming: `scripts/plot_fig<N>_<description>.py` with a docstring explaining usage.
3. All imports at the top of the file.
4. Unix line endings only (LF, never CRLF).
5. No simulation magic numbers; hardcoded values acceptable only for visual styling.

## After every create or modify

1. Run with `python3 scripts/plot_<name>.py`
2. Display the PNG output so the user can preview it
3. If only a PDF exists, convert with `sips -s format png input.pdf --out output.png`
4. Critique the aesthetics proactively — do not merely satisfy the prompt
5. Fix any issues found and re-render

Task: $ARGUMENTS
