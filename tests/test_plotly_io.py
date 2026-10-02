import json

import numpy as np

from rayplot import notebook_traces, points_from_notebook

XYZ = dict(x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 2])


def _nb(tmp_path, *outputs):
    cells = [{'cell_type': 'code', 'source': '', 'outputs': [{'data': o} for o in outputs]}]
    path = tmp_path / 'nb.ipynb'
    path.write_text(json.dumps({'cells': cells}))
    return path


def _html(*figs):
    calls = ''.join(f'Plotly.newPlot("id{i}", {json.dumps(d)}, {json.dumps(l)}, {{}});'
                    for i, (d, l) in enumerate(figs))
    return f'<div></div><script>{calls}</script>'


def test_json_output_with_strings(tmp_path):
    t = dict(type='scatter3d', name='a', marker=dict(color=['red', 'green', 'blue']), **XYZ)
    path = _nb(tmp_path, {'application/vnd.plotly.v1+json': {'data': [t]}})
    xyz, col = points_from_notebook(path, 'a')
    assert xyz.shape == (3, 3) and col == ['red', 'green', 'blue']


def test_every_newplot_in_an_html_output(tmp_path):
    a = dict(type='scatter3d', name='a', **XYZ)
    b = dict(type='scatter3d', name='b', **XYZ)
    path = _nb(tmp_path, {'text/html': _html(([a], {}), ([b], {}))})
    assert [t['name'] for t in notebook_traces(path)] == ['a', 'b']
    assert points_from_notebook(path, 'b')[1] == 'tab:blue'


def test_explicit_colorscale_is_applied(tmp_path):
    scale = [[0, 'rgb(0,0,0)'], [1, 'rgb(255,0,0)']]
    t = dict(type='scatter3d', name='a', marker=dict(color=[0, 5, 10], colorscale=scale), **XYZ)
    path = _nb(tmp_path, {'application/vnd.plotly.v1+json': {'data': [t]}})
    tr = notebook_traces(path)[0]
    assert np.allclose(tr['colour'], [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]])
    assert np.array_equal(tr['values'], [0, 5, 10])


def test_coloraxis_in_layout_with_cmin_and_reverse(tmp_path):
    scale = [[0, '#000000'], [1, '#0000ff']]
    t = dict(type='scatter3d', name='a', marker=dict(color=[0, 5, 10], coloraxis='coloraxis'),
             **XYZ)
    layout = {'coloraxis': {'colorscale': scale, 'cmin': 0, 'cmax': 20, 'reversescale': True}}
    path = _nb(tmp_path, {'text/html': _html(([t], layout))})
    assert np.allclose(notebook_traces(path)[0]['colour'][:, 2], [1, 0.75, 0.5])


def test_unknown_named_scale_returns_numbers(tmp_path, recwarn):
    t = dict(type='scatter3d', name='a', marker=dict(color=[0, 5, 10], colorscale='RdBu'), **XYZ)
    path = _nb(tmp_path, {'application/vnd.plotly.v1+json': {'data': [t]}})
    assert np.array_equal(points_from_notebook(path)[1], [0, 5, 10])
    assert 'not recognised' in str(recwarn.pop(UserWarning).message)
