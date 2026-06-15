"""Python API for :mod:`kern._core`."""

import operator

import numpy as np

from . import _core


KERNELS = ("gaussian", "epanechnikov", "triangular", "uniform", "cosine")
PARALLEL_MODES = ("auto", "grid", "evaluation")
MEMORY_MODES = ("low", "high", "auto")

_CORE_PARALLEL_MODES = {
    "auto": "auto",
    "grid": "grid",
    "evaluation": "evaluation",
    "bandwidths": "grid",
    "kernels": "evaluation",
}
_MEMORY_MODE_IDS = {"low": 0, "high": 1, "auto": 2}


def has_openmp():
    """Return whether the current build uses OpenMP."""
    return bool(_core.has_openmp())


def has_blas():
    """Return whether the current build uses BLAS."""
    return bool(_core.has_blas())


def _array_1d(values, name, *, nonempty=True):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to float64") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if nonempty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _array_2d(values, name, *, nonempty=True):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to float64") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[1] == 0 or (nonempty and array.shape[0] == 0):
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _bandwidth(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("bandwidth must be positive") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("bandwidth must be positive")
    return value


def _bandwidth_grid(values):
    grid = _array_1d(values, "h_grid")
    if np.any(grid <= 0.0):
        raise ValueError("all bandwidths must be positive")
    return grid


def _kernel(value):
    if value not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    return value


def _parallel(value):
    try:
        return _CORE_PARALLEL_MODES[value]
    except (KeyError, TypeError) as error:
        raise ValueError(f"parallel must be one of {PARALLEL_MODES}") from error


def _positive_integer(value, name, *, minimum=1):
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _bounded(array, name):
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain values inside [0, 1]")
    return array


def _sorted(array):
    if np.any(array[1:] < array[:-1]):
        raise ValueError("sorted_data must be sorted")
    return array


def _approximation_options(kernel, cutoff, max_neighbors, fast_gaussian,
                           memory, *, core_compatible=True):
    if cutoff is None:
        cutoff_value = 0.0
    else:
        try:
            cutoff_value = float(cutoff)
        except (TypeError, ValueError) as error:
            raise ValueError("cutoff must be positive or None") from error
        if (not np.isfinite(cutoff_value) or cutoff_value < 0.0
                or (cutoff_value == 0.0 and not core_compatible)):
            raise ValueError("cutoff must be positive or None")

    if max_neighbors is None:
        neighbor_count = 0
    else:
        neighbor_count = _positive_integer(
            max_neighbors, "max_neighbors", minimum=0 if core_compatible else 1
        )

    if not isinstance(fast_gaussian, (bool, np.bool_)):
        raise ValueError("fast_gaussian must be a boolean")
    if fast_gaussian and kernel != "gaussian":
        raise ValueError("fast_gaussian is only available for Gaussian KDE")

    if core_compatible and not isinstance(memory, str):
        memory_id = _positive_integer(memory, "memory", minimum=0)
        if memory_id > 2:
            raise ValueError("memory must be 0, 1, 2, or a named memory mode")
    else:
        try:
            memory_id = _MEMORY_MODE_IDS[memory]
        except (KeyError, TypeError) as error:
            raise ValueError(f"memory must be one of {MEMORY_MODES}") from error

    return cutoff_value, neighbor_count, bool(fast_gaussian), memory_id


def kernel_is_symmetric(kernel):
    return bool(_core.kernel_is_symmetric(_kernel(kernel)))


def kernel_default_cutoff(kernel):
    return float(_core.kernel_default_cutoff(_kernel(kernel)))


def kde(data, xs, bandwidth, kernel="gaussian"):
    return _core.kde(
        _array_1d(data, "data"),
        _array_1d(xs, "xs", nonempty=False),
        _bandwidth(bandwidth),
        _kernel(kernel),
    )


def kde_self(data, bandwidth, kernel="gaussian"):
    return _core.kde_self(
        _array_1d(data, "data"), _bandwidth(bandwidth), _kernel(kernel)
    )


def kde_beta_ext(data, xs, bandwidth):
    return _core.kde_beta_ext(
        _bounded(_array_1d(data, "data"), "data"),
        _bounded(_array_1d(xs, "xs", nonempty=False), "xs"),
        _bandwidth(bandwidth),
    )


def kde_beta_self(data, bandwidth):
    return _core.kde_beta_self(
        _bounded(_array_1d(data, "data"), "data"), _bandwidth(bandwidth)
    )


def kde_reflected_ext(data, xs, bandwidth, kernel="gaussian"):
    return _core.kde_reflected_ext(
        _bounded(_array_1d(data, "data"), "data"),
        _bounded(_array_1d(xs, "xs", nonempty=False), "xs"),
        _bandwidth(bandwidth),
        _kernel(kernel),
    )


def kde_reflected_self(data, bandwidth, kernel="gaussian"):
    return _core.kde_reflected_self(
        _bounded(_array_1d(data, "data"), "data"),
        _bandwidth(bandwidth),
        _kernel(kernel),
    )


def bandwidth_loo(data, h_grid, kernel="gaussian", parallel="auto"):
    return _core.bandwidth_loo(
        _array_1d(data, "data"),
        _bandwidth_grid(h_grid),
        _kernel(kernel),
        _parallel(parallel),
    )


def bandwidth_kfold(data, h_grid, k_folds, kernel="gaussian",
                    parallel="auto"):
    samples = _array_1d(data, "data")
    folds = _positive_integer(k_folds, "k_folds", minimum=2)
    if folds > samples.size:
        raise ValueError("k_folds must not exceed the number of samples")
    return _core.bandwidth_kfold(
        samples, _bandwidth_grid(h_grid), folds, _kernel(kernel),
        _parallel(parallel)
    )


def bandwidth_grid(data, h_grid, kernel="gaussian", k_folds=1,
                   parallel="auto"):
    samples = _array_1d(data, "data")
    folds = _positive_integer(k_folds, "k_folds")
    if folds != 1 and (folds < 2 or folds > samples.size):
        raise ValueError("k_folds must be 1 or between 2 and the sample count")
    return _core.bandwidth_grid(
        samples, _bandwidth_grid(h_grid), _kernel(kernel), folds,
        _parallel(parallel)
    )


def kde_approx(sorted_data, xs, bandwidth, kernel="gaussian", cutoff=None,
               max_neighbors=None, fast_gaussian=False, memory="auto"):
    kernel = _kernel(kernel)
    options = _approximation_options(
        kernel, cutoff, max_neighbors, fast_gaussian, memory
    )
    return _core.kde_approx(
        _sorted(_array_1d(sorted_data, "sorted_data")),
        _array_1d(xs, "xs", nonempty=False),
        _bandwidth(bandwidth),
        kernel,
        *options,
    )


def kde_approx_self(sorted_data, bandwidth, kernel="gaussian", cutoff=None,
                    max_neighbors=None, fast_gaussian=False, memory="auto"):
    kernel = _kernel(kernel)
    options = _approximation_options(
        kernel, cutoff, max_neighbors, fast_gaussian, memory
    )
    return _core.kde_approx_self(
        _sorted(_array_1d(sorted_data, "sorted_data")),
        _bandwidth(bandwidth),
        kernel,
        *options,
    )


def kde_multivariate(data, xs, bandwidth, kernel="gaussian", block_size=32):
    samples = _array_2d(data, "data")
    points = _array_2d(xs, "xs", nonempty=False)
    if points.shape[1] != samples.shape[1]:
        raise ValueError("data and xs must have the same number of features")
    return _core.kde_multivariate(
        samples, points, _bandwidth(bandwidth), _kernel(kernel),
        _positive_integer(block_size, "block_size")
    )


def kde_multivariate_self(data, bandwidth, kernel="gaussian", block_size=32):
    return _core.kde_multivariate_self(
        _array_2d(data, "data"),
        _bandwidth(bandwidth),
        _kernel(kernel),
        _positive_integer(block_size, "block_size"),
    )


__all__ = [
    "bandwidth_grid",
    "bandwidth_kfold",
    "bandwidth_loo",
    "has_blas",
    "has_openmp",
    "kde",
    "kde_approx",
    "kde_approx_self",
    "kde_beta_ext",
    "kde_beta_self",
    "kde_multivariate",
    "kde_multivariate_self",
    "kde_reflected_ext",
    "kde_reflected_self",
    "kde_self",
    "kernel_default_cutoff",
    "kernel_is_symmetric",
]
