"""
Reusable interactive label editor for publication figures.

Public API:

- place_labels(ax, panel_name, label_positions, label_texts, label_colors)
- place_markers(ax, panel_name, markers)
- place_arrows(ax, panel_name, arrows)
- enable_interactive(fig, panel_axes, panel_labels, script_path, panel_markers,
                     panel_arrows)
- label_curve(ax, x_data, y_data, x_anchor, text, color, dx, dy, ha, va,
              fontsize)

Usage pattern:

    from label_utils import place_labels, place_markers, place_arrows,
                            enable_interactive

    LABEL_POSITIONS = {
        ('panel_a', 'curve_1'): {'x': 50, 'y': 75, 'fontsize': 10,
                                  'rotation': 0, 'ha': 'left', 'va': 'center'},
    }
    LABEL_TEXTS  = {'curve_1': 'My label'}
    LABEL_COLORS = {'curve_1': '#2C3E50'}
    MARKERS = {}
    ARROWS  = {}

    fig, ax = plt.subplots()
    labels  = place_labels(ax, 'panel_a', LABEL_POSITIONS, LABEL_TEXTS,
                           LABEL_COLORS)
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
"""

import os
import subprocess
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from matplotlib.widgets import Button

_FALLBACK_COLORS = [
    '#2C3E50', '#d62728', '#1f77b4', '#6ab06a',
    '#C06040', '#9467bd', '#e377c2', '#333333',
]

_HA_CYCLE = ['left', 'center', 'right']
_ARROW_STYLES = ['->', '-|>', 'fancy', 'simple', '-']
_MARKER_SHAPES = ['o', 's', 'D', '^', 'v', '*', 'P', 'X']


def _ask_text_dialog(prompt, initial=''):
    """Show a text input dialog. Uses native macOS dialog when available,
    falls back to tkinter in a subprocess on other platforms."""
    if sys.platform == 'darwin':
        escaped = initial.replace('\\', '\\\\').replace('"', '\\"')
        script = (f'text returned of (display dialog '
                  f'"{prompt}" default answer "{escaped}")')
        r = subprocess.run(['osascript', '-e', script],
                           capture_output=True, text=True, timeout=120)
        return r.stdout.strip() if r.returncode == 0 else None
    env = os.environ.copy()
    env['_LABEL_INITIAL_TEXT'] = initial
    dialog_code = (
        'import os, tkinter as tk\n'
        'from tkinter import simpledialog\n'
        'initial = os.environ.get("_LABEL_INITIAL_TEXT", "")\n'
        'root = tk.Tk()\n'
        'root.withdraw()\n'
        'root.attributes("-topmost", True)\n'
        'root.after(100, lambda: root.focus_force())\n'
        't = simpledialog.askstring("Label", "' + prompt + '",\n'
        '                           initialvalue=initial, parent=root)\n'
        'root.destroy()\n'
        'print(t if t else "")\n'
    )
    r = subprocess.run([sys.executable, '-c', dialog_code],
                       capture_output=True, text=True, timeout=60, env=env)
    text = r.stdout.strip()
    return text if text and text != 'None' else None


def _arrow_picker(artist, event):
    """Custom picker for standalone arrows — distance to line segment."""
    if event.xdata is None or event.ydata is None:
        return False, {}
    posA = artist._custom_posA
    posB = artist._custom_posB
    ax = artist.axes
    tr = ax.transData
    pA = np.array(tr.transform(posA))
    pB = np.array(tr.transform(posB))
    pM = np.array([event.x, event.y])
    AB = pB - pA
    AB_sq = np.dot(AB, AB)
    if AB_sq < 1e-6:
        dist = np.linalg.norm(pM - pA)
    else:
        t = np.clip(np.dot(pM - pA, AB) / AB_sq, 0, 1)
        proj = pA + t * AB
        dist = np.linalg.norm(pM - proj)
    return dist <= 20, {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def place_labels(ax, panel_name, label_positions, label_texts, label_colors):
    """Place text labels on *ax* from a LABEL_POSITIONS dict.

    Supports optional fields in each entry:
    - weight: 'bold' or 'normal' (default 'normal')
    - bbox: True to draw a white background pad
    - arrow, x_anchor, y_anchor, arrowstyle, arrowlw: draw leader arrow
    """
    artists = {}
    for (pname, label_key), props in label_positions.items():
        if pname != panel_name:
            continue

        color = label_colors[label_key]
        bbox_dict = (dict(facecolor='white', edgecolor='none',
                          alpha=0.75, pad=1.5)
                     if props.get('bbox', False) else None)

        if props.get('arrow', False):
            ann = ax.annotate(
                label_texts[label_key],
                xy=(props['x_anchor'], props['y_anchor']),
                xytext=(props['x'], props['y']),
                fontsize=props['fontsize'],
                rotation=props['rotation'],
                color=color,
                ha=props['ha'],
                va=props['va'],
                weight=props.get('weight', 'normal'),
                clip_on=False,
                bbox=bbox_dict,
                arrowprops=dict(
                    arrowstyle=props.get('arrowstyle', '->'),
                    lw=props.get('arrowlw', 1.0),
                    color=color,
                ),
            )
            artists[label_key] = ann
        else:
            kwargs = dict(
                fontsize=props['fontsize'],
                rotation=props['rotation'],
                color=color,
                ha=props['ha'],
                va=props['va'],
                weight=props.get('weight', 'normal'),
                clip_on=False,
            )
            if bbox_dict is not None:
                kwargs['bbox'] = bbox_dict
            txt = ax.text(props['x'], props['y'],
                          label_texts[label_key], **kwargs)
            artists[label_key] = txt
    return artists


def place_markers(ax, panel_name, markers):
    """Place marker dots on *ax* from a MARKERS dict.

    Each entry: ('panel_name', 'key'): {'x', 'y', 'size', 'marker', 'color'}
    Returns {marker_key: Line2D artist}.
    """
    artists = {}
    for (pname, marker_key), props in markers.items():
        if pname != panel_name:
            continue
        line, = ax.plot(
            props['x'], props['y'],
            marker=props.get('marker', 'o'),
            ms=props.get('size', 8),
            color=props['color'],
            linestyle='none',
            zorder=5,
            clip_on=False,
        )
        artists[marker_key] = line
    return artists


def place_arrows(ax, panel_name, arrows):
    """Place standalone arrows on *ax* from an ARROWS dict.

    Each entry: ('panel_name', 'key'): {'x1', 'y1', 'x2', 'y2',
                                         'arrowstyle', 'lw', 'color'}
    (x1,y1) is the tail, (x2,y2) is the tip/head.
    Returns {arrow_key: FancyArrowPatch artist}.
    """
    artists = {}
    for (pname, arrow_key), props in arrows.items():
        if pname != panel_name:
            continue
        posA = (props['x1'], props['y1'])
        posB = (props['x2'], props['y2'])
        style = props.get('arrowstyle', '->')
        lw = props.get('lw', 1.0)
        color = props['color']
        arrow = FancyArrowPatch(
            posA, posB,
            arrowstyle=style,
            mutation_scale=15,
            lw=lw,
            color=color,
            clip_on=False,
            zorder=4,
        )
        arrow._custom_posA = posA
        arrow._custom_posB = posB
        arrow._custom_style = style
        arrow._custom_lw = lw
        arrow._custom_color = color
        ax.add_patch(arrow)
        artists[arrow_key] = arrow
    return artists


def enable_interactive(fig, panel_axes, panel_labels, script_path,
                       panel_markers=None, panel_arrows=None,
                       dict_prefix=''):
    """Make labels, markers, and arrows draggable with editing toolbar.

    Parameters
    ----------
    fig : Figure
    panel_axes : dict  {panel_name: Axes}
    panel_labels : dict  {(panel_name, label_key): Text or Annotation}
    script_path : str  absolute path — SAVE rewrites dicts here
    panel_markers : dict or None  {(panel_name, marker_key): Line2D}
    panel_arrows : dict or None  {(panel_name, arrow_key): FancyArrowPatch}
    dict_prefix : str  prefix for dict names, e.g. 'SCATTER_'
    """
    if panel_markers is None:
        panel_markers = {}
    if panel_arrows is None:
        panel_arrows = {}

    all_axes = list(panel_axes.values())
    drag_labels = list(panel_labels.values())
    drag_markers = list(panel_markers.values())
    drag_arrows = list(panel_arrows.values())

    for t in drag_labels:
        t.set_picker(10)
    for m in drag_markers:
        m.set_picker(15)
    for a in drag_arrows:
        a.set_picker(_arrow_picker)
    for ax in all_axes:
        ax.set_navigate(False)

    # Unbind default matplotlib keys that conflict with our shortcuts
    for keymap in ('keymap.back', 'keymap.forward', 'keymap.copy'):
        try:
            plt.rcParams[keymap] = []
        except (KeyError, ValueError):
            pass

    # Color palette from figure + fallbacks
    color_palette = _collect_axes_colors(all_axes)
    for c in _FALLBACK_COLORS:
        if c.lower() not in {cc.lower() for cc in color_palette}:
            color_palette.append(c)

    # Reverse lookup: artist -> (panel_name, key)
    label_key_for = {v: k for k, v in panel_labels.items()}
    marker_key_for = {v: k for k, v in panel_markers.items()}
    arrow_key_for = {v: k for k, v in panel_arrows.items()}

    state = {
        'dragged': None,
        'selected': None,
        'sel_type': None,       # 'label', 'marker', or 'arrow'
        'off_x': 0, 'off_y': 0,
        'undo_stack': [],       # multi-level undo
        'clipboard': None,      # (sel_type, snapshot, panel_name)
        'mkr_counter': len(panel_markers),
        'txt_counter': 0,
        'arr_counter': len(panel_arrows),
        'background': None,
        'arrow_drag_end': None, # 'A' or 'B' for arrow endpoint
        'active_panel': list(panel_axes.keys())[0],  # last-clicked panel
    }

    # ---- two-row button bar ------------------------------------------------
    fig.subplots_adjust(bottom=0.26)
    bw, bh = 0.058, 0.035
    gap = 0.004
    y_row1 = 0.075
    y_row2 = 0.02

    def _btn_row(y):
        x = [0.02]
        def next_rect(w=bw, extra_gap=0):
            rect = [x[0], y, w, bh]
            x[0] += w + gap + extra_gap
            return rect
        return next_rect

    r1 = _btn_row(y_row1)
    ax_size_up = fig.add_axes(r1())
    ax_size_dn = fig.add_axes(r1(extra_gap=gap * 2))
    ax_rot_cw  = fig.add_axes(r1())
    ax_rot_ccw = fig.add_axes(r1())
    ax_rot_0   = fig.add_axes(r1(extra_gap=gap * 2))
    ax_color   = fig.add_axes(r1())
    ax_bold    = fig.add_axes(r1())
    ax_ha      = fig.add_axes(r1())
    ax_bbox    = fig.add_axes(r1(extra_gap=gap * 2))
    ax_save    = fig.add_axes(r1(w=0.065))

    r2 = _btn_row(y_row2)
    ax_arrow   = fig.add_axes(r2())
    ax_arrw_up = fig.add_axes(r2())
    ax_arrw_dn = fig.add_axes(r2())
    ax_arrstyl = fig.add_axes(r2(extra_gap=gap * 2))
    ax_mkr_add = fig.add_axes(r2())
    ax_mkr_shp = fig.add_axes(r2(extra_gap=gap * 2))
    ax_txt_add = fig.add_axes(r2())
    ax_edit    = fig.add_axes(r2(extra_gap=gap * 2))
    ax_del     = fig.add_axes(r2())

    btn_size_up = Button(ax_size_up, 'Size +')
    btn_size_dn = Button(ax_size_dn, 'Size \u2212')
    btn_rot_cw  = Button(ax_rot_cw,  'Rot +5')
    btn_rot_ccw = Button(ax_rot_ccw, 'Rot \u22125')
    btn_rot_0   = Button(ax_rot_0,   'Rot 0')
    btn_color   = Button(ax_color,   'Color')
    btn_bold    = Button(ax_bold,    'Bold')
    btn_ha      = Button(ax_ha,      'HA')
    btn_bbox    = Button(ax_bbox,    'Bbox')
    btn_save    = Button(ax_save,    'SAVE', color='#90EE90',
                         hovercolor='#50C878')

    btn_arrow   = Button(ax_arrow,   'Arr +')
    btn_arrw_up = Button(ax_arrw_up, 'AW +')
    btn_arrw_dn = Button(ax_arrw_dn, 'AW \u2212')
    btn_arrstyl = Button(ax_arrstyl, 'A styl')
    btn_mkr_add = Button(ax_mkr_add, 'Mkr +')
    btn_mkr_shp = Button(ax_mkr_shp, 'Mkr \u25cf')
    btn_txt_add = Button(ax_txt_add, 'Txt +')
    btn_edit    = Button(ax_edit, 'Edit')
    btn_del     = Button(ax_del, 'Del', color='#FFB0B0',
                         hovercolor='#FF7070')

    # Status text (rest of row 2)
    status_x = r2(w=0)
    sx = status_x[0]
    ax_status = fig.add_axes([sx, y_row2, 1.0 - sx - 0.01, bh])
    ax_status.set_axis_off()
    status_text = ax_status.text(0.0, 0.5, 'Click a label, marker, or arrow',
                                 fontsize=8, va='center',
                                 transform=ax_status.transAxes)

    # ---- helpers -----------------------------------------------------------

    def _is_bold(artist):
        w = artist.get_weight()
        return w == 'bold' or (isinstance(w, int) and w >= 700)

    def _has_arrow(artist):
        return isinstance(artist, Annotation) and artist.arrowprops is not None

    def _snapshot(artist):
        """Capture current artist state for undo."""
        if isinstance(artist, FancyArrowPatch) and hasattr(artist,
                                                            '_custom_posA'):
            return {'type': 'arrow',
                    'posA': artist._custom_posA,
                    'posB': artist._custom_posB,
                    'style': artist._custom_style,
                    'lw': artist._custom_lw,
                    'color': artist._custom_color}
        if isinstance(artist, Line2D):
            xd, yd = artist.get_data()
            return {'type': 'marker', 'x': float(xd[0]), 'y': float(yd[0]),
                    'ms': artist.get_markersize(),
                    'color': artist.get_color(),
                    'marker': artist.get_marker()}
        snap = {'type': 'label',
                'pos': artist.get_position(),
                'text': artist.get_text(),
                'fontsize': artist.get_fontsize(),
                'rotation': artist.get_rotation(),
                'ha': artist.get_ha(),
                'va': artist.get_va(),
                'weight': artist.get_weight(),
                'color': artist.get_color(),
                'bbox': artist.get_bbox_patch() is not None}
        if _has_arrow(artist):
            snap['arrow'] = True
            snap['xy'] = artist.xy
            snap['arrowprops'] = dict(artist.arrowprops)
            snap['arrow_hidden'] = getattr(artist, '_arrow_hidden', False)
        return snap

    def _restore(artist, snap):
        """Restore artist state from a snapshot."""
        if snap['type'] == 'arrow':
            artist._custom_posA = snap['posA']
            artist._custom_posB = snap['posB']
            artist.set_positions(snap['posA'], snap['posB'])
            artist._custom_style = snap['style']
            artist.set_arrowstyle(snap['style'])
            artist._custom_lw = snap['lw']
            artist.set_linewidth(snap['lw'])
            artist._custom_color = snap['color']
            artist.set_color(snap['color'])
        elif snap['type'] == 'marker':
            artist.set_data([snap['x']], [snap['y']])
            artist.set_markersize(snap['ms'])
            artist.set_color(snap['color'])
            artist.set_marker(snap['marker'])
        else:
            artist.set_position(snap['pos'])
            artist.set_text(snap['text'])
            artist.set_fontsize(snap['fontsize'])
            artist.set_rotation(snap['rotation'])
            artist.set_ha(snap['ha'])
            artist.set_va(snap['va'])
            artist.set_weight(snap['weight'])
            artist.set_color(snap['color'])
            if snap['bbox']:
                artist.set_bbox(dict(facecolor='white', edgecolor='none',
                                     alpha=0.75, pad=1.5))
            else:
                artist.set_bbox(None)
            if snap.get('arrow'):
                artist.xy = snap['xy']
                artist.arrowprops = snap['arrowprops']
                hidden = snap.get('arrow_hidden', False)
                artist._arrow_hidden = hidden
                if artist.arrow_patch is not None:
                    artist.arrow_patch.set_visible(not hidden)
                    artist.arrow_patch.set_color(
                        snap['arrowprops'].get('color', artist.get_color()))
                    artist.arrow_patch.set_linewidth(
                        snap['arrowprops'].get('lw', 1.0))
                    artist.arrow_patch.set_arrowstyle(
                        snap['arrowprops'].get('arrowstyle', '->'))
        fig.canvas.draw_idle()

    def _save_undo():
        if state['selected'] is not None:
            state['undo_stack'].append(
                ('modify', state['selected'], _snapshot(state['selected'])))

    def update_status():
        sel = state['selected']
        if sel is None:
            return
        if state['sel_type'] == 'marker':
            status_text.set_text(
                f'Marker  sz={sel.get_markersize():.0f}  '
                f'shape={sel.get_marker()}')
        elif state['sel_type'] == 'arrow':
            status_text.set_text(
                f'Arrow  lw={sel._custom_lw:.1f}  '
                f'style={sel._custom_style}')
        else:
            txt = sel.get_text().replace('\n', ' ')
            parts = [f'"{txt}"  sz={sel.get_fontsize():.0f}',
                     f'rot={sel.get_rotation():.0f}',
                     f'ha={sel.get_ha()}']
            if _is_bold(sel):
                parts.append('bold')
            if sel.get_bbox_patch() is not None:
                parts.append('bbox')
            if _has_arrow(sel):
                parts.append('arrow')
            status_text.set_text('  '.join(parts))
        fig.canvas.draw_idle()

    def _find_panel_for(artist):
        """Return panel name for an artist."""
        key = (label_key_for.get(artist) or marker_key_for.get(artist)
               or arrow_key_for.get(artist))
        if key:
            return key[0]
        return list(panel_axes.keys())[0]

    # ---- pick / drag (with blitting for speed) ------------------------------

    def _start_blit(artist):
        """Cache the background for fast drag updates."""
        artist.set_animated(True)
        fig.canvas.draw()
        state['background'] = fig.canvas.copy_from_bbox(fig.bbox)
        artist.axes.draw_artist(artist)
        fig.canvas.blit(fig.bbox)

    def on_click(event):
        """Track which panel was last clicked in."""
        if event.inaxes in all_axes:
            for pname, pax in panel_axes.items():
                if pax is event.inaxes:
                    state['active_panel'] = pname
                    break

    def on_pick(event):
        a = event.artist
        # Update active panel from pick
        if a.axes in all_axes:
            for pname, pax in panel_axes.items():
                if pax is a.axes:
                    state['active_panel'] = pname
                    break
        if a in drag_labels:
            state['selected'] = a
            state['sel_type'] = 'label'
            state['dragged'] = a
            x0, y0 = a.get_position()
            mx = event.mouseevent.xdata
            my = event.mouseevent.ydata
            state['off_x'] = (x0 - mx) if mx is not None else 0
            state['off_y'] = (y0 - my) if my is not None else 0
            state['arrow_drag_end'] = None
            update_status()
            _start_blit(a)
        elif a in drag_markers:
            state['selected'] = a
            state['sel_type'] = 'marker'
            state['dragged'] = a
            xd, yd = a.get_data()
            mx = event.mouseevent.xdata
            my = event.mouseevent.ydata
            state['off_x'] = (float(xd[0]) - mx) if mx is not None else 0
            state['off_y'] = (float(yd[0]) - my) if my is not None else 0
            state['arrow_drag_end'] = None
            update_status()
            _start_blit(a)
        elif a in drag_arrows:
            state['selected'] = a
            state['sel_type'] = 'arrow'
            state['dragged'] = a
            posA = a._custom_posA
            posB = a._custom_posB
            cx = event.mouseevent.xdata if event.mouseevent.xdata is not None else posA[0]
            cy = event.mouseevent.ydata if event.mouseevent.ydata is not None else posA[1]
            dist_a = (cx - posA[0])**2 + (cy - posA[1])**2
            dist_b = (cx - posB[0])**2 + (cy - posB[1])**2
            state['arrow_drag_end'] = 'A' if dist_a <= dist_b else 'B'
            state['off_x'] = 0
            state['off_y'] = 0
            update_status()
            _start_blit(a)

    def _move_to_axes(artist, new_ax):
        """Move an artist from its current axes to *new_ax*."""
        old_ax = artist.axes
        if old_ax is new_ax:
            return
        # Find old and new panel names
        old_panel = new_panel = None
        for pname, pax in panel_axes.items():
            if pax is old_ax:
                old_panel = pname
            if pax is new_ax:
                new_panel = pname
        if not old_panel or not new_panel:
            return
        # Update tracking dicts
        if state['sel_type'] == 'label':
            old_key = label_key_for.pop(artist)
            del panel_labels[old_key]
            new_key = (new_panel, old_key[1])
            panel_labels[new_key] = artist
            label_key_for[artist] = new_key
        elif state['sel_type'] == 'marker':
            old_key = marker_key_for.pop(artist)
            del panel_markers[old_key]
            new_key = (new_panel, old_key[1])
            panel_markers[new_key] = artist
            marker_key_for[artist] = new_key
        elif state['sel_type'] == 'arrow':
            old_key = arrow_key_for.pop(artist)
            del panel_arrows[old_key]
            new_key = (new_panel, old_key[1])
            panel_arrows[new_key] = artist
            arrow_key_for[artist] = new_key
        # Move between axes
        artist.set_animated(False)
        artist.remove()
        if state['sel_type'] == 'marker':
            new_ax.add_line(artist)
        elif state['sel_type'] == 'arrow':
            new_ax.add_patch(artist)
        else:
            new_ax.add_artist(artist)
            artist.set_transform(new_ax.transData)
        state['active_panel'] = new_panel
        # Restart blitting in new axes
        artist.set_animated(True)
        fig.canvas.draw()
        state['background'] = fig.canvas.copy_from_bbox(fig.bbox)
        # Reset drag offset since coordinate system changed
        state['off_x'] = 0
        state['off_y'] = 0

    def on_motion(event):
        if state['dragged'] is None:
            return
        # Determine data coordinates — use inaxes if available,
        # otherwise project display coords into the dragged artist's axes
        if event.inaxes is not None and event.inaxes in all_axes:
            ax_for_coords = event.inaxes
            xdata, ydata = event.xdata, event.ydata
            # Cross-panel drag
            if ax_for_coords is not state['dragged'].axes:
                _move_to_axes(state['dragged'], ax_for_coords)
        else:
            # Outside any axes — project into the artist's own axes
            ax_for_coords = state['dragged'].axes
            try:
                disp_point = (event.x, event.y)
                data_point = ax_for_coords.transData.inverted().transform(disp_point)
                xdata, ydata = data_point[0], data_point[1]
            except Exception:
                return
        if state['sel_type'] == 'marker':
            nx = xdata + state['off_x']
            ny = ydata + state['off_y']
            state['dragged'].set_data([nx], [ny])
        elif state['sel_type'] == 'arrow':
            a = state['dragged']
            if state['arrow_drag_end'] == 'A':
                a._custom_posA = (xdata, ydata)
            else:
                a._custom_posB = (xdata, ydata)
            a.set_positions(a._custom_posA, a._custom_posB)
        else:
            nx = xdata + state['off_x']
            ny = ydata + state['off_y']
            state['dragged'].set_position((nx, ny))
        if state['background'] is not None:
            fig.canvas.restore_region(state['background'])
            state['dragged'].axes.draw_artist(state['dragged'])
            fig.canvas.blit(fig.bbox)
        else:
            fig.canvas.draw_idle()

    def on_release(event):
        if state['dragged'] is not None:
            state['dragged'].set_animated(False)
            state['background'] = None
            fig.canvas.draw_idle()
        state['dragged'] = None

    # ---- label buttons -----------------------------------------------------

    def size_up(event):
        sel = state['selected']
        if sel is None:
            return
        _save_undo()
        if state['sel_type'] == 'marker':
            sel.set_markersize(sel.get_markersize() + 1)
        elif state['sel_type'] == 'arrow':
            sel._custom_lw = min(5.0, sel._custom_lw + 0.5)
            sel.set_linewidth(sel._custom_lw)
        else:
            sel.set_fontsize(sel.get_fontsize() + 1)
        update_status()

    def size_dn(event):
        sel = state['selected']
        if sel is None:
            return
        _save_undo()
        if state['sel_type'] == 'marker':
            sel.set_markersize(max(2, sel.get_markersize() - 1))
        elif state['sel_type'] == 'arrow':
            sel._custom_lw = max(0.5, sel._custom_lw - 0.5)
            sel.set_linewidth(sel._custom_lw)
        else:
            sel.set_fontsize(max(4, sel.get_fontsize() - 1))
        update_status()

    def rot_cw(event):
        if state['selected'] and state['sel_type'] == 'label':
            _save_undo()
            state['selected'].set_rotation(
                (state['selected'].get_rotation() + 5) % 360)
            update_status()

    def rot_ccw(event):
        if state['selected'] and state['sel_type'] == 'label':
            _save_undo()
            state['selected'].set_rotation(
                (state['selected'].get_rotation() - 5) % 360)
            update_status()

    def rot_reset(event):
        if state['selected'] and state['sel_type'] == 'label':
            _save_undo()
            state['selected'].set_rotation(0)
            update_status()

    def cycle_color(event):
        sel = state['selected']
        if sel is None:
            return
        _save_undo()
        if state['sel_type'] == 'arrow':
            cur = _normalize_color(sel._custom_color)
        else:
            cur = _normalize_color(sel.get_color())
        idx = 0
        for i, c in enumerate(color_palette):
            if _normalize_color(c) == cur:
                idx = (i + 1) % len(color_palette)
                break
        new_color = color_palette[idx]
        if state['sel_type'] == 'arrow':
            sel.set_color(new_color)
            sel._custom_color = new_color
        else:
            sel.set_color(new_color)
            if state['sel_type'] == 'label' and _has_arrow(sel):
                sel.arrowprops['color'] = new_color
                if sel.arrow_patch is not None:
                    sel.arrow_patch.set_color(new_color)
        update_status()
        fig.canvas.draw_idle()

    def toggle_bold(event):
        if state['selected'] and state['sel_type'] == 'label':
            _save_undo()
            state['selected'].set_weight(
                'normal' if _is_bold(state['selected']) else 'bold')
            update_status()
            fig.canvas.draw_idle()

    def cycle_ha(event):
        if state['selected'] and state['sel_type'] == 'label':
            _save_undo()
            cur = state['selected'].get_ha()
            idx = (_HA_CYCLE.index(cur) + 1) % len(_HA_CYCLE)
            state['selected'].set_ha(_HA_CYCLE[idx])
            update_status()
            fig.canvas.draw_idle()

    def toggle_bbox(event):
        if state['selected'] and state['sel_type'] == 'label':
            _save_undo()
            sel = state['selected']
            if sel.get_bbox_patch() is not None:
                sel.set_bbox(None)
            else:
                sel.set_bbox(dict(facecolor='white', edgecolor='none',
                                  alpha=0.75, pad=1.5))
            update_status()
            fig.canvas.draw_idle()

    # ---- arrow buttons -----------------------------------------------------

    def add_arrow(event):
        """Add a new standalone arrow to the figure."""
        pname = (_find_panel_for(state['selected'])
                 if state['selected'] else list(panel_axes.keys())[0])
        ax = panel_axes[pname]
        xl, xr = ax.get_xlim()
        yl, yr = ax.get_ylim()
        cx = (xl + xr) / 2
        cy = (yl + yr) / 2
        dx = (xr - xl) * 0.1

        state['arr_counter'] += 1
        akey = f'arr_{state["arr_counter"]}'
        posA = (cx - dx, cy)
        posB = (cx + dx, cy)
        c = color_palette[0] if color_palette else '#000000'

        arrow = FancyArrowPatch(
            posA, posB,
            arrowstyle='->',
            mutation_scale=15,
            lw=1.0,
            color=c,
            clip_on=False, zorder=4)
        arrow._custom_posA = posA
        arrow._custom_posB = posB
        arrow._custom_style = '->'
        arrow._custom_lw = 1.0
        arrow._custom_color = c
        ax.add_patch(arrow)
        arrow.set_picker(_arrow_picker)

        full_key = (pname, akey)
        panel_arrows[full_key] = arrow
        arrow_key_for[arrow] = full_key
        drag_arrows.append(arrow)
        state['undo_stack'].append(('add', arrow, full_key, 'arrow'))
        state['selected'] = arrow
        state['sel_type'] = 'arrow'
        update_status()
        fig.canvas.draw_idle()

    def arrow_width_up(event):
        sel = state['selected']
        if sel is None:
            return
        if state['sel_type'] == 'arrow':
            _save_undo()
            sel._custom_lw = min(5.0, sel._custom_lw + 0.5)
            sel.set_linewidth(sel._custom_lw)
            update_status()
            fig.canvas.draw_idle()
        elif (state['sel_type'] == 'label' and _has_arrow(sel)
              and not getattr(sel, '_arrow_hidden', False)):
            _save_undo()
            new_lw = min(5.0, sel.arrowprops.get('lw', 1.0) + 0.5)
            sel.arrowprops['lw'] = new_lw
            if sel.arrow_patch is not None:
                sel.arrow_patch.set_linewidth(new_lw)
            update_status()
            fig.canvas.draw_idle()

    def arrow_width_dn(event):
        sel = state['selected']
        if sel is None:
            return
        if state['sel_type'] == 'arrow':
            _save_undo()
            sel._custom_lw = max(0.5, sel._custom_lw - 0.5)
            sel.set_linewidth(sel._custom_lw)
            update_status()
            fig.canvas.draw_idle()
        elif (state['sel_type'] == 'label' and _has_arrow(sel)
              and not getattr(sel, '_arrow_hidden', False)):
            _save_undo()
            new_lw = max(0.5, sel.arrowprops.get('lw', 1.0) - 0.5)
            sel.arrowprops['lw'] = new_lw
            if sel.arrow_patch is not None:
                sel.arrow_patch.set_linewidth(new_lw)
            update_status()
            fig.canvas.draw_idle()

    def arrow_style_cycle(event):
        sel = state['selected']
        if sel is None:
            return
        if state['sel_type'] == 'arrow':
            _save_undo()
            cur = sel._custom_style
            idx = 0
            if cur in _ARROW_STYLES:
                idx = (_ARROW_STYLES.index(cur) + 1) % len(_ARROW_STYLES)
            sel._custom_style = _ARROW_STYLES[idx]
            sel.set_arrowstyle(_ARROW_STYLES[idx])
            update_status()
            fig.canvas.draw_idle()
        elif (state['sel_type'] == 'label' and _has_arrow(sel)
              and not getattr(sel, '_arrow_hidden', False)):
            _save_undo()
            cur = sel.arrowprops.get('arrowstyle', '->')
            idx = 0
            if cur in _ARROW_STYLES:
                idx = (_ARROW_STYLES.index(cur) + 1) % len(_ARROW_STYLES)
            new_style = _ARROW_STYLES[idx]
            sel.arrowprops['arrowstyle'] = new_style
            if sel.arrow_patch is not None:
                sel.arrow_patch.set_arrowstyle(new_style)
            update_status()
            fig.canvas.draw_idle()

    # ---- marker buttons ----------------------------------------------------

    def add_marker(event):
        pname = (_find_panel_for(state['selected'])
                 if state['selected'] else list(panel_axes.keys())[0])
        ax = panel_axes[pname]
        xl, xr = ax.get_xlim()
        yl, yr = ax.get_ylim()
        cx, cy = (xl + xr) / 2, (yl + yr) / 2

        state['mkr_counter'] += 1
        mkey = f'mkr_{state["mkr_counter"]}'
        line, = ax.plot(cx, cy, marker='o', ms=8,
                        color=color_palette[0], linestyle='none',
                        zorder=5, clip_on=False)
        line.set_picker(10)
        full_key = (pname, mkey)
        panel_markers[full_key] = line
        marker_key_for[line] = full_key
        drag_markers.append(line)
        state['undo_stack'].append(('add', line, full_key, 'marker'))
        state['selected'] = line
        state['sel_type'] = 'marker'
        update_status()
        fig.canvas.draw_idle()

    def cycle_marker_shape(event):
        sel = state['selected']
        if sel and state['sel_type'] == 'marker':
            _save_undo()
            cur = sel.get_marker()
            idx = 0
            if cur in _MARKER_SHAPES:
                idx = (_MARKER_SHAPES.index(cur) + 1) % len(_MARKER_SHAPES)
            sel.set_marker(_MARKER_SHAPES[idx])
            update_status()
            fig.canvas.draw_idle()

    # ---- add text ----------------------------------------------------------

    def add_text(event):
        text_str = _ask_text_dialog('Enter label text:')
        if not text_str:
            return

        pname = (_find_panel_for(state['selected'])
                 if state['selected'] else list(panel_axes.keys())[0])
        ax = panel_axes[pname]
        xl, xr = ax.get_xlim()
        yl, yr = ax.get_ylim()
        cx, cy = (xl + xr) / 2, (yl + yr) / 2

        state['txt_counter'] += 1
        lkey = f'txt_{state["txt_counter"]}'
        txt = ax.text(cx, cy, text_str,
                      fontsize=10, rotation=0,
                      color='#000000', ha='left', va='center',
                      weight='normal', clip_on=False)
        txt.set_picker(10)
        full_key = (pname, lkey)
        panel_labels[full_key] = txt
        label_key_for[txt] = full_key
        drag_labels.append(txt)
        state['undo_stack'].append(('add', txt, full_key, 'label'))
        state['selected'] = txt
        state['sel_type'] = 'label'
        update_status()
        fig.canvas.draw_idle()

    # ---- edit text ---------------------------------------------------------

    def edit_text(event):
        sel = state['selected']
        if sel is None or state['sel_type'] != 'label':
            return
        text_str = _ask_text_dialog('Edit label text:', sel.get_text())
        if not text_str:
            return
        _save_undo()
        sel.set_text(text_str)
        update_status()
        fig.canvas.draw_idle()

    # ---- delete selected ---------------------------------------------------

    def delete_selected(event):
        """Remove the currently selected element. Supports undo."""
        sel = state['selected']
        if sel is None:
            return
        if state['sel_type'] == 'marker':
            key = marker_key_for.pop(sel)
            del panel_markers[key]
            drag_markers.remove(sel)
            state['undo_stack'].append(
                ('delete', sel, key, 'marker', sel.axes))
        elif state['sel_type'] == 'arrow':
            key = arrow_key_for.pop(sel)
            del panel_arrows[key]
            drag_arrows.remove(sel)
            state['undo_stack'].append(
                ('delete', sel, key, 'arrow', sel.axes))
        else:
            key = label_key_for.pop(sel)
            del panel_labels[key]
            drag_labels.remove(sel)
            state['undo_stack'].append(
                ('delete', sel, key, 'label', sel.axes))
        sel.remove()
        state['selected'] = None
        state['sel_type'] = None
        status_text.set_text('Deleted. (Z to undo)')
        fig.canvas.draw_idle()

    # ---- keyboard shortcuts ------------------------------------------------

    def on_key(event):
        key = event.key
        # Show key in status bar for feedback
        if key not in (None, 'shift', 'control', 'alt', 'cmd', 'super',
                       'ctrl+shift', 'shift+control'):
            status_text.set_text(f'Key: {key!r}')
            fig.canvas.draw_idle()
        # Undo (multi-level) — z key (plain or with modifier)
        if key in ('z', 'ctrl+z', 'super+z') and state['undo_stack']:
            action = state['undo_stack'].pop()
            if action[0] == 'modify':
                _, artist, snap = action
                _restore(artist, snap)
                state['selected'] = artist
                state['sel_type'] = snap['type']
                update_status()
            elif action[0] == 'delete':
                _, artist, key, sel_type, axes_ref = action
                if sel_type == 'marker':
                    axes_ref.add_line(artist)
                    panel_markers[key] = artist
                    marker_key_for[artist] = key
                    drag_markers.append(artist)
                    artist.set_picker(10)
                elif sel_type == 'arrow':
                    axes_ref.add_patch(artist)
                    panel_arrows[key] = artist
                    arrow_key_for[artist] = key
                    drag_arrows.append(artist)
                    artist.set_picker(_arrow_picker)
                else:
                    axes_ref.add_artist(artist)
                    panel_labels[key] = artist
                    label_key_for[artist] = key
                    drag_labels.append(artist)
                    artist.set_picker(10)
                state['selected'] = artist
                state['sel_type'] = sel_type
                update_status()
                status_text.set_text('Undo: restored deleted item')
            elif action[0] == 'add':
                _, artist, key, sel_type = action
                if sel_type == 'marker':
                    marker_key_for.pop(artist, None)
                    panel_markers.pop(key, None)
                    if artist in drag_markers:
                        drag_markers.remove(artist)
                elif sel_type == 'arrow':
                    arrow_key_for.pop(artist, None)
                    panel_arrows.pop(key, None)
                    if artist in drag_arrows:
                        drag_arrows.remove(artist)
                else:
                    label_key_for.pop(artist, None)
                    panel_labels.pop(key, None)
                    if artist in drag_labels:
                        drag_labels.remove(artist)
                artist.remove()
                state['selected'] = None
                state['sel_type'] = None
                status_text.set_text('Undo: removed added item')
            fig.canvas.draw_idle()

        # Copy — c key (plain or with modifier)
        elif key in ('c', 'super+c', 'ctrl+c'):
            sel = state['selected']
            if sel is not None:
                state['clipboard'] = (
                    state['sel_type'], _snapshot(sel), _find_panel_for(sel))
                status_text.set_text('Copied!')
                fig.canvas.draw_idle()

        # Paste — v key (plain or with modifier)
        elif key in ('v', 'super+v', 'ctrl+v'):
            clip = state['clipboard']
            if clip is None:
                return
            sel_type, snap, src_panel = clip
            # Paste into the last-clicked panel
            target_panel = state['active_panel']
            cross_panel = (target_panel != src_panel)
            ax = panel_axes.get(target_panel,
                                list(panel_axes.values())[0])
            xl, xr = ax.get_xlim()
            yl, yr = ax.get_ylim()
            cx, cy = (xl + xr) / 2, (yl + yr) / 2
            dx = (xr - xl) * 0.05
            dy = (yr - yl) * 0.05

            if sel_type == 'marker':
                state['mkr_counter'] += 1
                mkey = f'mkr_{state["mkr_counter"]}'
                if cross_panel:
                    px, py = cx, cy
                else:
                    px, py = snap['x'] + dx, snap['y'] + dy
                line, = ax.plot(
                    px, py,
                    marker=snap['marker'], ms=snap['ms'],
                    color=snap['color'], linestyle='none',
                    zorder=5, clip_on=False)
                line.set_picker(10)
                full_key = (target_panel, mkey)
                panel_markers[full_key] = line
                marker_key_for[line] = full_key
                drag_markers.append(line)
                state['undo_stack'].append(
                    ('add', line, full_key, 'marker'))
                state['selected'] = line
                state['sel_type'] = 'marker'

            elif sel_type == 'arrow':
                state['arr_counter'] += 1
                akey = f'arr_{state["arr_counter"]}'
                if cross_panel:
                    new_posA = (cx - dx * 2, cy)
                    new_posB = (cx + dx * 2, cy)
                else:
                    new_posA = (snap['posA'][0] + dx, snap['posA'][1] + dy)
                    new_posB = (snap['posB'][0] + dx, snap['posB'][1] + dy)
                arrow = FancyArrowPatch(
                    new_posA, new_posB,
                    arrowstyle=snap['style'],
                    mutation_scale=15,
                    lw=snap['lw'],
                    color=snap['color'],
                    clip_on=False, zorder=4)
                arrow._custom_posA = new_posA
                arrow._custom_posB = new_posB
                arrow._custom_style = snap['style']
                arrow._custom_lw = snap['lw']
                arrow._custom_color = snap['color']
                ax.add_patch(arrow)
                arrow.set_picker(_arrow_picker)
                full_key = (target_panel, akey)
                panel_arrows[full_key] = arrow
                arrow_key_for[arrow] = full_key
                drag_arrows.append(arrow)
                state['undo_stack'].append(
                    ('add', arrow, full_key, 'arrow'))
                state['selected'] = arrow
                state['sel_type'] = 'arrow'

            elif sel_type == 'label':
                state['txt_counter'] += 1
                lkey = f'txt_{state["txt_counter"]}'
                if cross_panel:
                    px, py = cx, cy
                else:
                    x, y = snap['pos']
                    px, py = x + dx, y + dy
                txt = ax.text(
                    px, py, snap['text'],
                    fontsize=snap['fontsize'],
                    rotation=snap['rotation'],
                    color=snap['color'],
                    ha=snap['ha'], va=snap['va'],
                    weight=snap['weight'],
                    clip_on=False)
                if snap['bbox']:
                    txt.set_bbox(dict(facecolor='white', edgecolor='none',
                                      alpha=0.75, pad=1.5))
                txt.set_picker(10)
                full_key = (target_panel, lkey)
                panel_labels[full_key] = txt
                label_key_for[txt] = full_key
                drag_labels.append(txt)
                state['undo_stack'].append(
                    ('add', txt, full_key, 'label'))
                state['selected'] = txt
                state['sel_type'] = 'label'

            update_status()
            fig.canvas.draw_idle()

        # Delete — delete or backspace key
        elif key in ('delete', 'backspace'):
            delete_selected(event)

    # ---- save --------------------------------------------------------------

    def save_positions(event):
        _rewrite_all(script_path, panel_labels, panel_markers, panel_arrows,
                     dict_prefix)
        _regenerate_figure(script_path)
        status_text.set_text('Saved & regenerated!')
        fig.canvas.draw_idle()

    # ---- wire everything ---------------------------------------------------

    btn_size_up.on_clicked(size_up)
    btn_size_dn.on_clicked(size_dn)
    btn_rot_cw.on_clicked(rot_cw)
    btn_rot_ccw.on_clicked(rot_ccw)
    btn_rot_0.on_clicked(rot_reset)
    btn_color.on_clicked(cycle_color)
    btn_bold.on_clicked(toggle_bold)
    btn_ha.on_clicked(cycle_ha)
    btn_bbox.on_clicked(toggle_bbox)
    btn_save.on_clicked(save_positions)

    btn_arrow.on_clicked(add_arrow)
    btn_arrw_up.on_clicked(arrow_width_up)
    btn_arrw_dn.on_clicked(arrow_width_dn)
    btn_arrstyl.on_clicked(arrow_style_cycle)
    btn_mkr_add.on_clicked(add_marker)
    btn_mkr_shp.on_clicked(cycle_marker_shape)
    btn_txt_add.on_clicked(add_text)
    btn_edit.on_clicked(edit_text)
    btn_del.on_clicked(delete_selected)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # prevent garbage collection
    fig._label_utils_widgets = [
        btn_size_up, btn_size_dn, btn_rot_cw, btn_rot_ccw, btn_rot_0,
        btn_color, btn_bold, btn_ha, btn_bbox, btn_save,
        btn_arrow, btn_arrw_up, btn_arrw_dn, btn_arrstyl,
        btn_mkr_add, btn_mkr_shp, btn_txt_add, btn_edit, btn_del,
    ]

    print('Interactive mode: drag labels/markers/arrows, use toolbar.')
    print('Keys: Z=undo  C=copy  V=paste  Delete=remove.  SAVE to persist.')


def label_curve(ax, x_data, y_data, x_anchor, text, color, dx, dy, ha, va,
                fontsize):
    """Place a label near a curve at x=x_anchor with offset in points."""
    i = np.argmin(np.abs(x_data - x_anchor))
    ann = ax.annotate(
        text,
        xy=(x_data[i], y_data[i]),
        xytext=(dx, dy),
        textcoords='offset points',
        color=color,
        fontsize=fontsize,
        ha=ha,
        va=va,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5),
        clip_on=False,
    )
    return ann


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _regenerate_figure(script_path):
    """Re-run the script in non-interactive mode to regenerate output files."""
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True)
    if result.returncode == 0:
        print('Figure regenerated.')
    else:
        print(f'Regeneration failed:\n{result.stderr}')


def _normalize_color(c):
    """Convert any matplotlib color to lowercase hex for comparison."""
    return mcolors.to_hex(c).lower()


def _collect_axes_colors(axes_list):
    """Collect unique colors from lines and collections on axes."""
    seen = set()
    colors = []
    for ax in axes_list:
        for line in ax.get_lines():
            c = _normalize_color(line.get_color())
            if c not in seen:
                seen.add(c)
                colors.append(c)
        for coll in ax.collections:
            fc = coll.get_facecolors()
            if len(fc) > 0:
                c = _normalize_color(fc[0])
                if c not in seen:
                    seen.add(c)
                    colors.append(c)
    return colors


def _find_dict_block(lines, dict_name):
    """Return (start, end) line indices of a top-level dict assignment."""
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(dict_name) and '=' in stripped and '{' in stripped:
            start = i
            break
    if start is None:
        return None, None

    depth = 0
    for j in range(start, len(lines)):
        depth += lines[j].count('{') - lines[j].count('}')
        if depth == 0:
            return start, j + 1
    return start, len(lines)


def _generate_label_positions_text(panel_labels, dict_name='LABEL_POSITIONS'):
    """Generate LABEL_POSITIONS dict text from current artists."""
    entries = []
    for (panel_name, label_key), artist in panel_labels.items():
        x, y = artist.get_position()
        fs = int(artist.get_fontsize())
        rot = int(artist.get_rotation())
        ha = artist.get_ha()
        va = artist.get_va()

        extras = ''
        if _is_bold_static(artist):
            extras += ", 'weight': 'bold'"
        if artist.get_bbox_patch() is not None:
            extras += ", 'bbox': True"
        if isinstance(artist, Annotation) and artist.arrowprops is not None:
            hidden = getattr(artist, '_arrow_hidden', False)
            if not hidden:
                xa, ya = artist.xy
                astyle = artist.arrowprops.get('arrowstyle', '->')
                alw = artist.arrowprops.get('lw', 1.0)
                extras += (f", 'arrow': True, 'x_anchor': {xa:.4f}, "
                           f"'y_anchor': {ya:.4f}, "
                           f"'arrowstyle': '{astyle}', "
                           f"'arrowlw': {alw:.1f}")

        entry = (
            f"    ('{panel_name}', '{label_key}'): "
            f"{{'x': {x:.4f}, 'y': {y:.4f}, "
            f"'fontsize': {fs}, 'rotation': {rot}, "
            f"'ha': '{ha}', 'va': '{va}'{extras}}},"
        )
        entries.append(entry)
    return f'{dict_name} = {{\n' + '\n'.join(entries) + '\n}\n'


def _generate_label_texts_text(panel_labels, dict_name='LABEL_TEXTS'):
    """Generate LABEL_TEXTS dict text from current artists."""
    seen = {}
    for (panel_name, label_key), artist in panel_labels.items():
        seen[label_key] = artist.get_text()
    entries = []
    for label_key, text in seen.items():
        escaped = text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
        entries.append(f"    '{label_key}': '{escaped}',")
    return f'{dict_name} = {{\n' + '\n'.join(entries) + '\n}\n'


def _generate_label_colors_text(panel_labels, dict_name='LABEL_COLORS'):
    """Generate LABEL_COLORS dict text from current artists."""
    seen = {}
    for (panel_name, label_key), artist in panel_labels.items():
        seen[label_key] = artist.get_color()
    entries = [f"    '{k}': '{v}'," for k, v in seen.items()]
    return f'{dict_name} = {{\n' + '\n'.join(entries) + '\n}\n'


def _generate_markers_text(panel_markers, dict_name='MARKERS'):
    """Generate MARKERS dict text from current Line2D artists."""
    entries = []
    for (panel_name, marker_key), artist in panel_markers.items():
        xd, yd = artist.get_data()
        x, y = float(xd[0]), float(yd[0])
        sz = int(artist.get_markersize())
        mk = artist.get_marker()
        color = _normalize_color(artist.get_color())
        entry = (
            f"    ('{panel_name}', '{marker_key}'): "
            f"{{'x': {x:.4f}, 'y': {y:.4f}, "
            f"'size': {sz}, 'marker': '{mk}', 'color': '{color}'}},"
        )
        entries.append(entry)
    return f'{dict_name} = {{\n' + '\n'.join(entries) + '\n}\n'


def _generate_arrows_text(panel_arrows, dict_name='ARROWS'):
    """Generate ARROWS dict text from current FancyArrowPatch artists."""
    entries = []
    for (panel_name, arrow_key), artist in panel_arrows.items():
        x1, y1 = artist._custom_posA
        x2, y2 = artist._custom_posB
        style = artist._custom_style
        lw = artist._custom_lw
        color = _normalize_color(artist._custom_color)
        entry = (
            f"    ('{panel_name}', '{arrow_key}'): "
            f"{{'x1': {x1:.4f}, 'y1': {y1:.4f}, "
            f"'x2': {x2:.4f}, 'y2': {y2:.4f}, "
            f"'arrowstyle': '{style}', 'lw': {lw:.1f}, "
            f"'color': '{color}'}},"
        )
        entries.append(entry)
    return f'{dict_name} = {{\n' + '\n'.join(entries) + '\n}\n'


def _rewrite_dict(lines, dict_name, new_block):
    """Replace a dict block in *lines* and return the new list."""
    start, end = _find_dict_block(lines, dict_name)
    if start is None:
        return lines
    return lines[:start] + [new_block] + lines[end:]


def _rewrite_all(script_path, panel_labels, panel_markers,
                 panel_arrows=None, dict_prefix=''):
    """Rewrite LABEL_POSITIONS, LABEL_TEXTS, LABEL_COLORS, MARKERS,
    and ARROWS (optionally prefixed, e.g. 'SCATTER_')."""
    if panel_arrows is None:
        panel_arrows = {}
    with open(script_path, 'r') as f:
        lines = f.readlines()

    lp = f'{dict_prefix}LABEL_POSITIONS'
    lt = f'{dict_prefix}LABEL_TEXTS'
    lc = f'{dict_prefix}LABEL_COLORS'
    mk = f'{dict_prefix}MARKERS'
    ar = f'{dict_prefix}ARROWS'

    lines = _rewrite_dict(
        lines, lp,
        _generate_label_positions_text(panel_labels, lp))
    lines = _rewrite_dict(
        lines, lt,
        _generate_label_texts_text(panel_labels, lt))
    lines = _rewrite_dict(
        lines, lc,
        _generate_label_colors_text(panel_labels, lc))
    lines = _rewrite_dict(
        lines, mk,
        _generate_markers_text(panel_markers, mk))

    if panel_arrows or _find_dict_block(lines, ar)[0] is not None:
        arrows_text = _generate_arrows_text(panel_arrows, ar)
        start, _ = _find_dict_block(lines, ar)
        if start is not None:
            lines = _rewrite_dict(lines, ar, arrows_text)
        else:
            _, markers_end = _find_dict_block(lines, mk)
            if markers_end is not None:
                lines = lines[:markers_end] + [arrows_text] + lines[markers_end:]

    with open(script_path, 'w') as f:
        f.writelines(lines)


def _is_bold_static(artist):
    """Check if an artist has bold weight (standalone, no closure needed)."""
    w = artist.get_weight()
    return w == 'bold' or (isinstance(w, int) and w >= 700)
