import numpy as np
from numpy.testing import assert_allclose

from nuki.histogram import edges_centers, edges_volumes, edges_widths, integral


def test_edges_centers():
    rng = np.random.default_rng(0)
    # Test for 1D homogeneous histogram
    lower = rng.uniform(-1, 1, size=25)
    upper = lower + 1
    expected = lower + 0.5
    result = edges_centers(lower, upper)
    assert_allclose(result, expected)

    # Test for 1D inhomogeneous histogram
    lower = rng.uniform(-1, 1, size=25)
    width = rng.uniform(-1, 1, size=25)
    upper = lower + width
    expected = lower + width / 2
    result = edges_centers(lower, upper)
    assert_allclose(result, expected)

    # Test for 2D homogeneous histogram
    lower = rng.uniform(-1, 1, size=(25, 25))
    upper = lower + 1
    expected = lower + 0.5
    result = edges_centers(lower, upper)
    assert_allclose(result, expected)

    # Test for 2D inhomogeneous histogram
    lower = rng.uniform(-1, 1, size=(25, 25))
    width = rng.uniform(-1, 1, size=(25, 25))
    upper = lower + width
    expected = lower + width / 2
    result = edges_centers(lower, upper)
    assert_allclose(result, expected)

    # Test for batched 1D histogram
    lower = np.stack([np.arange(8), np.linspace(0, 100, 8)])
    upper = lower + 1
    expected = lower + 0.5
    result = edges_centers(lower, upper)
    assert_allclose(result, expected)


def test_edges_volumes():
    rng = np.random.default_rng(0)
    # Test for 2D homogeneous histogram
    x_widths = np.ones(25)
    y_widths = np.ones_like(x_widths)
    expected = x_widths
    result = edges_volumes(x_widths, y_widths)
    assert_allclose(result, expected)

    # Test for 2D inhomogeneous histogram
    x_widths = np.linspace(0, 1, 25)
    y_widths = np.linspace(0, 1, 25)
    expected = x_widths**2
    result = edges_volumes(x_widths, y_widths)
    assert_allclose(result, expected)

    # Test for 2D inhomogeneous random histogram
    x_widths = rng.uniform(-1, 1, size=25)
    y_widths = np.ones_like(x_widths)
    expected = x_widths
    result = edges_volumes(x_widths, y_widths)
    assert_allclose(result, expected)

    # Test for batched 1D histogram
    x_widths = np.stack([np.arange(8), np.linspace(0, 100, 8)])
    y_widths = np.stack([np.ones(8), np.ones(8)])
    expected = x_widths
    result = edges_volumes(x_widths, y_widths)
    assert_allclose(result, expected)


def test_edges_widths():
    rng = np.random.default_rng(0)
    # Test for 1D homogeneous histogram
    lower = rng.uniform(-1, 1, size=25)
    upper = lower + 1
    expected = 1
    result = edges_widths(lower, upper)
    assert_allclose(result, expected)

    # Test for 1D inhomogeneous histogram
    lower = rng.uniform(-1, 1, size=25)
    width = rng.uniform(-1, 1, size=25)
    upper = lower + width
    expected = width
    result = edges_widths(lower, upper)
    assert_allclose(result, expected)

    # Test for 2D homogeneous histogram
    lower = rng.uniform(-1, 1, size=(25, 25))
    upper = lower + 1
    expected = 1
    result = edges_widths(lower, upper)
    assert_allclose(result, expected)

    # Test for 2D inhomogeneous histogram
    lower = rng.uniform(-1, 1, size=(25, 25))
    width = rng.uniform(-1, 1, size=(25, 25))
    upper = lower + width
    expected = width
    result = edges_widths(lower, upper)
    assert_allclose(result, expected)

    # Test for batched 1D histogram
    lower = np.stack([np.arange(8), np.linspace(0, 100, 8)])
    upper = lower + 1
    expected = 1
    result = edges_widths(lower, upper)
    assert_allclose(result, expected)


def test_integral():
    rng = np.random.default_rng(0)
    # Test for 1D homogeneous histogram
    bin_widths = np.ones(25)
    bin_content = rng.uniform(0, 1, 25)
    expected = np.sum(bin_content)
    result = integral(bin_content, bin_widths)
    assert_allclose(result, expected)

    # Test for 2D homogeneous histogram
    x_widths = np.ones(25)
    y_widths = np.ones(25)
    bin_content = rng.uniform(0, 1, 25)
    expected = np.sum(bin_content)
    result = integral(bin_content, x_widths, y_widths)
    assert_allclose(result, expected)

    # Test for 2D inhomogeneous random histogram
    x_widths = np.linspace(0, 1, 25)
    y_widths = np.ones(25)
    bin_content = rng.uniform(0, 1, 25)
    expected = np.sum(bin_content * x_widths)
    result = integral(bin_content, x_widths, y_widths)
    assert_allclose(result, expected)

    # Test for batched 1D histogram
    x_widths = np.stack([np.arange(8), np.linspace(0, 100, 8)])
    y_widths = np.stack([np.ones(8), np.ones(8)])
    bin_content = rng.uniform(0, 1, 8)
    expected = np.asarray(
        [
            np.sum(np.arange(8) * bin_content),
            np.sum(np.linspace(0, 100, 8) * bin_content),
        ]
    )
    result = integral(bin_content, x_widths, y_widths, axis=1)
    assert_allclose(result, expected)
