import numpy as np
from array_api_compat import is_numpy_array, numpy
from pytest import fixture
from scipy.special import erf, ndtr

from nuki import special


@fixture
def array_backends():
    backends = [(numpy, is_numpy_array)]
    try:
        from array_api_compat import is_jax_array, jax

        backends.append((jax.numpy, is_jax_array))
    except ImportError:
        pass
    return backends


def test_erf(array_backends):
    for xp, is_xp in array_backends:
        x = xp.linspace(-1, 2, 24)
        np.testing.assert_allclose(erf(x), special.erf(x))
        assert is_xp(special.erf(x))


def test_ndtr(array_backends):
    for xp, is_xp in array_backends:
        x = xp.linspace(-2, 2, 24)
        np.testing.assert_allclose(ndtr(x), special.ndtr(x))
        assert is_xp(special.ndtr(x))
