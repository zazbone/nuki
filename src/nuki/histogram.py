"""
Histogram Utilities for NDArray-based Representations
=====================================================

This module provides utility routines to operate on NDArray arrays representing
histogram components. These include bin edges (lower and upper), widths,
centers, volumes, and the numerical integral of histogram contents.

All inputs and outputs are based on `NDArray` types that implement the
Python array API. The routines are compatible with multi-dimensional
histograms, assuming bin edges and widths are broadcastable as needed.

Functions
---------
edges_centers
    Compute the center values of histogram bins from lower and upper edges.
edges_widths
    Compute the width of each bin from its lower and upper edge.
edges_volumes
    Compute the bin volumes as a product of widths across dimensions.
integral
    Compute the total integral (e.g., weighted sum) of histogram contents.

Examples
--------
Compute bin centers for a 1D histogram:

>>> import numpy as np
>>> import nuki.histogram as hist
>>> lower = np.array([0, 1, 2])
>>> upper = np.array([1, 2, 3])
>>> hist.edges_centers(lower, upper)
array([0.5, 1.5, 2.5])

Compute the integral of histogram contents:

>>> contents = np.array([2, 3, 4])
>>> widths = hist.edges_widths(lower, upper)
>>> hist.integral(contents, widths)
9
"""

from functools import reduce
from operator import mul

from numpy.typing import NDArray

__all__ = ("edges_centers", "edges_volumes", "edges_widths", "integral")


def edges_centers(lower_edges: NDArray, upper_edges: NDArray):
    """
    Compute the center values of histogram bins.

    Parameters
    ----------
    lower_edges : NDArray
        The lower edges of the histogram bins.
    upper_edges : NDArray
        The upper edges of the histogram bins.

    Returns
    -------
    NDArray
        The centers of the histogram bins.

    Examples
    --------
    Given arrays of lower and upper bin edgesrepresenting a 1D histogram edges,
    the function returns the midpoint of each bin.
    >>> import numpy as np
    >>> import nuki.histogram as hist
    >>> lower_edges = np.arange(5)
    >>> upper_edges = np.arange(5) + 1
    >>> hist.edges_centers(lower_edges, upper_edges)
    array([0.5, 1.5, 2.5, 3.5, 4.5])
    The lower_edges and upper_edges are provided as stacked meshgrids representing
    bin edges across two dimensions.
    In this case, each bin's center is computed independently for each dimension.
    >>> lower_edges= np.stack(np.meshgrid(np.arange(3), np.arange(5, 8)))
    >>> upper_edges = np.stack(np.meshgrid(np.arange(3), np.arange(5, 8))) + 1
    >>> hist.edges_centers(lower_edges, upper_edges)
    array([[[0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5]],

       [[5.5, 5.5, 5.5],
        [6.5, 6.5, 6.5],
        [7.5, 7.5, 7.5]]])
    This example calculates the centers of two 1D histogram bins using edges_centers.
    The lower_edges and upper_edges arrays represent the bin edges for two histograms
    (each with 3 bins).
    >>> lower_edges= np.stack((np.arange(3), np.arange(5, 8)))
    >>> upper_edges = np.stack((np.arange(3), np.arange(5, 8))) + 1
    >>> hist.edges_centers(lower_edges, upper_edges)
    array([[0.5, 1.5, 2.5],
        [5.5, 6.5, 7.5]])
    """
    return (upper_edges + lower_edges) / 2


def edges_volumes(*widths: NDArray):
    """
    Compute the volumes of histogram bins given a set of widths.

    Parameters
    ----------
    *widths: NDArray
        Arrays containing the widths of multiples bin edges.

    Returns
    -------
    NDArray
        The volumes of the histogram bins.

    Examples
    --------
    >>> import numpy as np
    >>> import nuki.histogram as hist
    >>> x_widths = np.ones(5)
    >>> hist.edges_volumes(x_widths, x_widths)
    array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> x_widths = np.arange(5)
    >>> y_widths = np.ones(5)
    >>> hist.edges_volumes(x_widths, y_widths)
    array([0.0, 1.0, 2.0, 3.0, 4.0])
    """
    return reduce(mul, widths)


def edges_widths(lower_edges: NDArray, upper_edges: NDArray):
    """
    Compute the widths of histogram bins.

    Parameters
    ----------
    lower_edges : NDArray
        The lower edges of the histogram bins.
    upper_edges : NDArray
        The upper edges of the histogram bins.

    Returns
    -------
    NDArray
        The widths of the histogram bins.

    Examples
    --------
    The lower_edges and upper_edges arrays represent the bin edges,
    and the function computes the difference between the upper and
    lower edges for each bin.
    >>> lower_edges = np.arange(5)
    >>> upper_edges = np.arange(5) + 1
    >>> hist.edges_widths(lower_edges, upper_edges)
    array([1, 1, 1, 1, 1])
    Here, the lower_edges and upper_edges arrays are provided as stacked meshgrids,
    representing bin edges across two dimensions.
    The function computes the widths independently for each dimension.
    >>> lower_edges= np.stack(np.meshgrid(np.arange(3), np.arange(5, 8)))
    >>> upper_edges = np.stack(np.meshgrid(np.arange(3), np.arange(5, 8))) + 1
    >>> hist.edges_widths(lower_edges, upper_edges)
    array([[[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],

        [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]])
    In this case, the lower_edges and upper_edges arrays represent two separate
    1D histograms, each with 3 bins.
    The function calculates the widths of the bins for both histograms.
    >>> lower_edges= np.stack((np.arange(3), np.arange(5, 8)))
    >>> upper_edges = np.stack((np.arange(3), np.arange(5, 8))) + 1
    >>> hist.edges_widths(lower_edges, upper_edges)
    array([[1, 1, 1],
        [1, 1, 1]])
    """
    return upper_edges - lower_edges


def integral(contents: NDArray, /, *widths: NDArray, axis=None, keepdims=False):
    """
    Compute the integral, as the sum of weighted contents, of histogram bins.

    Parameters
    ----------
    contents : NDArray
        The contents (e.g., counts or values) of the histogram bins.
    *widths: NDArray
        Arrays containing the widths of multiples bin edges.
    axis : int or tuple of int, optional
        The axis or axes along which to perform the summation. Default is None (sum over all).
    keepdims : bool, optional
        If True, retains reduced dimensions with length 1. Default is False.

    Returns
    -------
    NDArray
        The integral of the histogram.

    Examples
    --------
    >>> import numpy as np
    >>> import nuki.histogram as hist
    >>> lower_edges = np.arange(5)
    >>> upper_edges = np.arange(5) + 1
    >>> widths = hist.edges_widths(lower_edges, upper_edges)
    >>> hist.integral(np.arange(5), widths)
    np.int64(10)
    """
    return (contents * edges_volumes(*widths)).sum(axis, keepdims=keepdims)
