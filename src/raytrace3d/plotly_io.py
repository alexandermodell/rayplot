"""Read 3-D scatter points out of a notebook's saved plotly figures (no re-computation)."""
import base64
import json
from pathlib import Path

import numpy as np


def _arr(v):
    if isinstance(v, dict) and 'bdata' in v:
        return np.frombuffer(base64.b64decode(v['bdata']), dtype=v['dtype'])
    return np.asarray(v)


def notebook_traces(path):
    """Every scatter3d trace in the notebook's saved outputs, as dicts (name, xyz, colour)."""
    out = []
    for cell in json.loads(Path(path).read_text())['cells']:
        for o in cell.get('outputs', []):
            data = o.get('data', {})
            figs = []
            if 'application/vnd.plotly.v1+json' in data:
                figs.append(data['application/vnd.plotly.v1+json'].get('data', []))
            html = ''.join(data.get('text/html', ''))
            i = html.find('Plotly.newPlot(')
            if i >= 0:
                figs.append(json.JSONDecoder().raw_decode(html[html.find('[', i):])[0])
            for traces in figs:
                for t in traces:
                    if t.get('type') != 'scatter3d':
                        continue
                    xyz = np.stack([_arr(t[k]).astype(np.float64) for k in 'xyz'], 1)
                    colour = (t.get('marker') or {}).get('color')
                    if isinstance(colour, dict):
                        colour = _arr(colour)
                    out.append(dict(name=t.get('name'), xyz=xyz, colour=colour))
    return out


def points_from_notebook(path, trace=0):
    """(xyz, colours) of one scatter3d trace, chosen by name (e.g. 'layer 28') or index.
    `colours` is whatever the figure stored: colour strings or numbers (use `cmap`)."""
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
    colour = t['colour']
    if colour is None or isinstance(colour, str):
        colour = colour or 'tab:blue'
    return t['xyz'], colour
