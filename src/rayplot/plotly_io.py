"""Read 3-D scatter points out of a notebook's saved plotly figures (no re-computation)."""
import base64
import json
import re
import warnings
from pathlib import Path

import numpy as np

from .colours import _parse

# Plotly colour scales that are the same as matplotlib's colormap of the lowercase name. (Others
# share names but not colours: plotly's 'RdBu' runs blue to red, matplotlib's red to blue.)
SAME_AS_MATPLOTLIB = {'viridis', 'plasma', 'inferno', 'magma', 'cividis'}


def _arr(v):
    if isinstance(v, dict) and 'bdata' in v:
        return np.frombuffer(base64.b64decode(v['bdata']), dtype=v['dtype'])
    return np.asarray(v)


def _html_figures(html):
    """(data, layout) of every Plotly.newPlot(id, data, layout, ...) call in the HTML."""
    dec, out = json.JSONDecoder(), []
    for m in re.finditer(r'Plotly\.newPlot\(', html):
        start = html.find('[', m.end())
        if start < 0:
            continue
        data, end = dec.raw_decode(html, start)
        layout = {}
        sep = re.compile(r'\s*,\s*').match(html, end)
        if sep and html.startswith('{', sep.end()):
            layout = dec.raw_decode(html, sep.end())[0]
        out.append((data, layout))
    return out


def _scale(values, axis):
    """Numbers through a plotly colour axis (colorscale, cmin/cmax, reversescale), as (n, 3)
    sRGB; None if the scale is not one we can reproduce."""
    scale = axis.get('colorscale')
    if scale is None:
        return None
    v = np.asarray(values, dtype=np.float64)
    lo = axis.get('cmin', np.nanmin(v))
    hi = axis.get('cmax', np.nanmax(v))
    t = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(v)
    if axis.get('reversescale'):
        t = 1 - t
    if isinstance(scale, str):
        if scale.lower() not in SAME_AS_MATPLOTLIB:
            return None
        import matplotlib
        return np.asarray(matplotlib.colormaps[scale.lower()](t))[:, :3]
    stops = np.array([float(s[0]) for s in scale])
    rgb = np.array([_parse(s[1]) for s in scale])
    return np.stack([np.interp(t, stops, rgb[:, i]) for i in range(3)], 1)   # as plotly does


def _colour(trace, layout):
    """(colour as the figure shows it, the raw numbers behind it or None)."""
    marker = trace.get('marker') or {}
    colour = marker.get('color')
    if isinstance(colour, dict):
        colour = _arr(colour)
    if colour is None or isinstance(colour, str):
        return colour, None
    arr = np.asarray(colour)
    if arr.dtype.kind in 'USO':
        return list(arr), None
    # A trace that names a colour axis (plotly express does) takes its scale from the layout;
    # a figure that sets no scale at all gets its template's default.
    axis = dict(layout.get(marker['coloraxis']) or {}) if marker.get('coloraxis') else marker
    if axis.get('colorscale') is None:
        default = (((layout.get('template') or {}).get('layout') or {}).get('colorscale')
                   or {}).get('sequential')
        axis = {**axis, 'colorscale': default}
    rgb = _scale(arr, axis)
    if rgb is None:
        if axis.get('colorscale') is not None:
            warnings.warn(f'rayplot: colour scale {axis["colorscale"]!r} not recognised; '
                          'returning the numbers (they will go through `cmap`)', stacklevel=4)
        return arr, arr
    return rgb, arr


def notebook_traces(path):
    """Every scatter3d trace in the notebook's saved outputs, as dicts:
    name, xyz, colour (as the figure shows it: colour strings, an (n, 3) RGB array, or numbers
    when the figure has no colour scale we can reproduce) and values (the numbers behind a
    colour scale, else None)."""
    out = []
    for cell in json.loads(Path(path).read_text())['cells']:
        for o in cell.get('outputs', []):
            data = o.get('data', {})
            figs = []
            if 'application/vnd.plotly.v1+json' in data:
                fig = data['application/vnd.plotly.v1+json']
                figs.append((fig.get('data', []), fig.get('layout') or {}))
            figs += _html_figures(''.join(data.get('text/html', '')))
            for traces, layout in figs:
                for t in traces:
                    if t.get('type') != 'scatter3d':
                        continue
                    xyz = np.stack([_arr(t[k]).astype(np.float64) for k in 'xyz'], 1)
                    colour, values = _colour(t, layout)
                    out.append(dict(name=t.get('name'), xyz=xyz, colour=colour, values=values))
    return out


def points_from_notebook(path, trace=0):
    """(xyz, colours) of one scatter3d trace, chosen by name (e.g. 'layer 28') or index.
    `colours` is as the figure shows them (see `notebook_traces`); `cmap` applies only when it
    is numbers."""
    traces = notebook_traces(path)
    if not traces:
        raise ValueError(f'no saved scatter3d figures in {path}')
    if isinstance(trace, str):
        hits = [t for t in traces if t['name'] == trace]
        if not hits:
            names = sorted({str(t['name']) for t in traces})
            raise ValueError(f'no trace named {trace!r}; have {names}')
        t = hits[0]
    else:
        t = traces[trace]
    return t['xyz'], t['colour'] if t['colour'] is not None else 'tab:blue'
