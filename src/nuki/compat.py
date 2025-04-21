"""
compat
=====

Module Overview
---------------
The compat.py module provides a collection of utility functions
that facilitate compatibility between various array libraries,
including numpy, jax, and others (in future).
The goal is to enable the writing of framework-independent code
by abstracting away differences between these libraries.
The module also includes special handling to ensure that Python scalar
values do not interfere with operations that expect array-like inputs.

Available Functions
-------------------
array_namespace(*xs, api_version=DEFAULT_ARRAY_API_VERSION, use_compat=None)
    Get the appropriate array namespace for the input arrays.
erf(z)
    Compute the error function (erf) of the input.
erfc(z)
    Compute the complementary error function (erfc) of the input.
"""

from numbers import Number

import array_api_compat as aa
import numpy

__all__ = ("array_namespace",)

DEFAULT_ARRAY_API_VERSION = "v2022.12"


def is_not_python_number(x):
    """Check if `x` is not a Python scalar number.

    This function determines whether the input `x`
    is not an instance of a Python scalar number,
    such as an `int`, `float`, or `complex`.

    Parameters
    ----------
    x: Any
        The input value to check.

    Returns
    -------
    bool:
        True if `x` is not a Python scalar number, False otherwise.
    """
    return not isinstance(x, Number)


def filter_numbers(it):
    """Filter out Python scalar numbers from an iterable.

    This function removes scalar numbers (e.g., `int`, `float`, `complex`)
    from an iterable, ensuring only array-like objects remain.

    Parameters
    ----------
    it: Iterable[Array | Number]
        An iterable containing array-like objects and/or Python scalar numbers.

    Returns
    -------
    Iterator[Array]:
        An iterator over the array-like objects.
    """
    return filter(is_not_python_number, it)


def array_namespace(*xs, api_version=DEFAULT_ARRAY_API_VERSION, use_compat=None):
    """Get the appropriate array namespace for the input arrays.

    This function acts as a wrapper for `array_api_compat.array_namespace`
    to avoid issues with Python scalar values.
    It filters out scalar values from the input before determining the
    correct array namespace.

    Parameters
    ----------
    *xs: Array
        A sequence of arrays or array-like objects.
    api_version: str, optional
        The version of the Array API standard to use (default is "v2022.12").
    use_compat: bool, optional
        Whether to use compatibility shims (default is None).

    Returns
    -------
    module:
        The appropriate array namespace module (e.g., `numpy`, `jax`, etc.).

    Notes
    -----
    If no array-like objects are found, the function returns the `numpy` namespace.
    """
    fxs = filter_numbers(xs)
    try:
        first = next(fxs)
    except StopIteration:
        return numpy
    return aa.array_namespace(
        first, *fxs, api_version=api_version, use_compat=use_compat
    )
