import numpy as np
import pytest

from rayplot.colours import to_rgb


def test_int_zero_one_triples_are_unit_scale():
    assert to_rgb([[1, 0, 0], [0, 1, 0]], 2).tolist() == [[1, 0, 0], [0, 1, 0]]


def test_floats_slightly_over_one_are_clipped_not_rescaled():
    assert np.allclose(to_rgb(np.array([[1.02, 0.5, 0.0]]), 1), [[1, 0.5, 0]])


def test_0_255_arrays():
    assert np.allclose(to_rgb([[255, 128, 0]], 1), [[1, 128 / 255, 0]])
    assert np.allclose(to_rgb(np.array([[255.0, 0, 0]]), 1), [[1, 0, 0]])
    assert np.allclose(to_rgb(np.array([[1, 0, 0]], dtype=np.uint8), 1), [[1 / 255, 0, 0]])


def test_tuple_is_one_colour_even_for_three_points():
    assert np.allclose(to_rgb((1.0, 0.5, 0.0), 3), [[1, 0.5, 0]] * 3)
    assert np.allclose(to_rgb((255, 0, 0), 3), [[1, 0, 0]] * 3)


def test_three_numbers_for_three_points_go_through_cmap():
    rgb = to_rgb([0.0, 0.5, 1.0], 3, cmap='viridis')
    assert not np.allclose(rgb[0], rgb[2])


def test_list_of_three_for_other_counts_is_one_colour():
    assert np.allclose(to_rgb([1.0, 0.5, 0.0], 4), [[1, 0.5, 0]] * 4)


def test_strings():
    rgb = to_rgb(['#ff0000', 'rgb(0, 255, 0)', 'hsl(240, 100%, 50%)', 'white'], 4)
    assert np.allclose(rgb, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    assert np.allclose(to_rgb('#00f', 2), [[0, 0, 1]] * 2)


def test_alpha_ignored():
    assert np.allclose(to_rgb(np.array([[1, 0, 0, 0.3]]), 1), [[1, 0, 0]])


def test_wrong_lengths():
    with pytest.raises(ValueError):
        to_rgb(['red', 'blue'], 3)
    with pytest.raises(ValueError):
        to_rgb(np.zeros(5), 3)
