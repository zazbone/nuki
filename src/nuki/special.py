import scipy
from array_api_compat import is_cupy_array, is_jax_array, is_torch_array
from numpy.typing import NDArray

try:
    import cupyx as cp
except ImportError:
    cp = None

try:
    import jax
except ImportError:
    jax = None

try:
    import torch
except ImportError:
    torch = None


__all__ = ("erf",)


def erf(x: NDArray):
    """
    Compute the Gauss error function, dispatching to the appropriate array backend.

    This function serves as a unified wrapper for computing the error function `erf(x)`
    across multiple array libraries that implement the python array API. It supports
    numpy, cupy, jax, and torch arrays and dispatches the call to the corresponding
    backend implementation.

    Parameters
    ----------
    x : array-like
        Input array. Can be a numpy compatible, cupy, jax, or torch array.

    Returns
    -------
    array-like
        The computed values of the error function, with the same type and shape
        as the input array.

    Notes
    -----
    The output must match the scipy.special.erf result.
    - cupy: uses `cupyx.scipy.special.erf`
    - jax: uses `jax.scipy.special.erf`
    - torch: uses `torch.erf`
    - Default: uses `scipy.special.erf` (assumed numpy compatible input)
    """
    if is_cupy_array(x):
        return cp.scipy.special.erf(x)
    elif is_jax_array(x):
        return jax.scipy.special.erf(x)
    elif is_torch_array(x):
        return torch.erf(x)
    else:
        return scipy.special.erf(x)
