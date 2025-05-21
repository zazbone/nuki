"""
nuki.stats.norm
===============

Module Overview
---------------
This module provides functions related to the normal (Gaussian) distribution,
including the cumulative distribution function (CDF),
probability density function (PDF),.
The functions are designed to be compatible with array libraries
supporting array api standard, allowing framework-independent code.

Available Functions
-------------------
cdf(x, mean=0, scale=1)
    Normal distribution cumulative density function (CDF).
pdf(x, mean=0, scale=1)
    Normal distribution probability density function (PDF).
integral(lower, upper, mean=0, scale=1)
    Integral of the normal distribution PDF between two bounds.
    Simply defined as cdf(upper) - cdf(lower)
"""

from numpy.typing import ArrayLike

from nuki.compat import array_namespace
from nuki.special import ndtr

__all__ = ("cdf", "integral", "pdf")

INV_SQRT_2PI = 0.3989422804014327
INV_SQRT_2 = 0.7071067811865476


def cdf(x: ArrayLike, /, *, mean=0, scale=1):
    """Normal distribution cumulative density function (CDF).

    Computes the CDF of the normal distribution given input `x`,
    mean `mean`, and standard deviation `scale`.

    Parameters
    ----------
    x: ArrayLike
        Input value(s) for which to compute the CDF.
    mean: ArrayLike
        The mean of the normal distribution.
        Default 0
    scale: ArrayLike
        The standard deviation (scale) of the normal distribution.
        Default 1

    Returns
    -------
    ArrayLike:
        The CDF value(s) corresponding to the input `x`.
    """
    beta = (x - mean) / scale
    return ndtr(beta)


def integral(lower: ArrayLike, upper: ArrayLike, /, *, mean=0, scale=1):
    """Integral of the normal distribution PDF between two bounds.
    Simply defined as cdf(upper) - cdf(lower).

    Computes the integral of the normal distribution's PDF
    between the `lower` and `upper` bounds, for a given `mean`
    and standard deviation `scale`.

    Parameters
    ----------
    lower: ArrayLike
        The lower bound of the integral.
    upper: ArrayLike
        The upper bound of the integral.
    mean: ArrayLike
        The mean of the normal distribution.
    std: ArrayLike
        The standard deviation of the normal distribution.
    Returns
    -------
    ArrayLike:
        The integral of the PDF between `xa` and `xb`.
    """
    return cdf(upper, mean=mean, scale=scale) - cdf(lower, mean=mean, scale=scale)


def pdf(x: ArrayLike, /, *, mean=0, scale=1):
    """Normal distribution probability density function (PDF).

    Computes the PDF of the normal distribution for a given `x`,
    mean `mean`, and standard deviation `scale`.

    Parameters
    ----------
    x: ArrayLike
        Input value(s) for which to compute the PDF.
    mean: ArrayLike
        The mean of the normal distribution.
    scale: ArrayLike
        The standard deviation (scale) of the normal distribution.

    Returns
    -------
    ArrayLike:
        The PDF value(s) corresponding to the input `x`.
    """
    beta = (x - mean) / scale
    xp = array_namespace(beta)
    return INV_SQRT_2PI * xp.exp(-0.5 * beta**2) / scale
