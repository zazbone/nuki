import numpy as np
from array_api_compat import is_numpy_array, numpy
from pytest import fixture
from scipy.stats import norm

from nuki.distribution import normal


@fixture
def array_backends():
    backends = [(numpy, is_numpy_array)]
    try:
        from array_api_compat import is_jax_array, jax

        backends.append((jax.numpy, is_jax_array))
    except ImportError:
        pass
    return backends


def test_normal_cdf():
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 24).reshape(-1, 1, 1)
    mean = rng.uniform(-1, 1, 24).reshape(1, -1, 1)
    scale = rng.uniform(0, 1, 24).reshape(1, 1, -1)
    np.testing.assert_allclose(
        norm.cdf(x, loc=mean, scale=scale),
        normal.cdf(x, mean=mean, scale=scale),
    )


def test_normal_integral(array_backends):
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 24).reshape(-1, 1, 1)
    mean = rng.uniform(-1, 1, 24).reshape(1, -1, 1)
    scale = rng.uniform(0, 1, 24).reshape(1, 1, -1)
    np.testing.assert_allclose(
        norm.cdf(x, loc=mean, scale=scale),
        normal.cdf(x, mean=mean, scale=scale),
    )


def test_normal_pdf(array_backends):
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 24).reshape(-1, 1, 1)
    mean = rng.uniform(-1, 1, 24).reshape(1, -1, 1)
    scale = rng.uniform(0, 1, 24).reshape(1, 1, -1)
    for xp, is_xp in array_backends:
        expected = norm.pdf(x, loc=mean, scale=scale)
        results = normal.pdf(
            xp.asarray(x), mean=xp.asarray(mean), scale=xp.asarray(scale)
        )
        np.testing.assert_allclose(expected, results)
        assert is_xp(results)
