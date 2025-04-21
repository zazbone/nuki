import importlib

import array_api_compat
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nuki import compat

try:
    import jax
except ModuleNotFoundError:
    jax = None
MODULES_NAMES = (
    "cupy",
    "dask.array",
    "jax",
    "numpy",
    "torch",
)


@pytest.fixture
def array_modules():
    modules = []
    for module_name in MODULES_NAMES:
        try:
            modules.append(importlib.import_module(f"array_api_compat.{module_name}"))
        except ModuleNotFoundError:
            pass
    return modules


def test_privates_functions():
    assert not compat.is_not_python_number(10)
    assert not compat.is_not_python_number(3.14)
    assert not compat.is_not_python_number(2 + 3j)
    assert compat.is_not_python_number(np.array([1, 2, 3]))
    assert compat.is_not_python_number("string")

    iterable = [1, 2.0, 3 + 4j, np.array([4, 5, 6]), "Hello world !"]
    results = list(compat.filter_numbers(iterable))
    expected = [np.array([4, 5, 6]), "Hello world !"]
    assert len(results) == len(expected)
    assert_allclose(results[0], expected[0])
    assert results[1] == expected[1]


def test_array_namespace(array_modules):
    for xp in array_modules:
        scalar1 = 12.8
        arr1 = xp.asarray([1, 2, 3])
        arr2 = xp.asarray([4, 5, 6])
        namespace = compat.array_namespace(scalar1, arr1, arr2)
        assert namespace is xp

    scalar1 = 12.8
    scalar2 = 10
    scalar3 = 1.0 + 7j
    namespace = compat.array_namespace(scalar1, scalar2, scalar3)
    assert namespace is np
