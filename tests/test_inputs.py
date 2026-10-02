import numpy as np
import pytest

import rayplot as rp
from rayplot import _prepare


def test_wider_points_use_first_three_columns():
    pts = np.arange(20.0).reshape(4, 5)
    xyz, rgb = _prepare(pts, 'red', 'viridis', None, None)
    assert np.array_equal(xyz, pts[:, :3]) and rgb.shape == (4, 3)


@pytest.mark.parametrize('shape', [(5,), (5, 2), (2, 3, 3)])
def test_bad_point_shapes(shape):
    with pytest.raises(ValueError):
        _prepare(np.zeros(shape), 'red', 'viridis', None, None)


def test_non_finite_points_dropped_with_a_warning():
    pts = np.array([[0, 0, 0], [np.nan, 0, 0], [1, 1, 1.0]])
    with pytest.warns(UserWarning, match='dropping 1'):
        xyz, rgb = _prepare(pts, ['red', 'green', 'blue'], 'viridis', None, None)
    assert len(xyz) == 2 and np.allclose(rgb[1], [0, 0, 1])


def test_all_non_finite_is_an_error():
    with pytest.raises(ValueError):
        _prepare(np.full((3, 3), np.nan), 'red', 'viridis', None, None)


def test_unknown_keyword_is_an_error():
    with pytest.raises(TypeError):
        rp.render(np.zeros((3, 3)), 'red', elevaton=10)


def test_pure_hue_is_gone():
    assert not hasattr(rp, 'pure_hue')
    with pytest.raises(TypeError):
        rp.render(np.zeros((3, 3)), 'red', pure_hues=True)


def test_page_escapes_title():
    from rayplot.server import page
    html = page('</script><b>x & y</b>')
    assert '<title>&lt;/script&gt;&lt;b&gt;x &amp; y&lt;/b&gt;</title>' in html
    assert 'title: "\\u003c/script>\\u003cb>x & y\\u003c/b>"' in html
