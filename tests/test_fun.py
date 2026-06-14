import math

import numpy as np
import pytest

import kern
from kern import _core, _fun


KERNELS = _fun.KERNELS


def kernel_value(name, value):
    value = np.asarray(value)
    if name == "gaussian":
        return np.exp(-0.5 * value**2) / np.sqrt(2.0 * np.pi)

    inside = np.abs(value) <= 1.0
    if name == "epanechnikov":
        result = 0.75 * (1.0 - value**2)
    elif name == "triangular":
        result = 1.0 - np.abs(value)
        inside = np.abs(value) < 1.0
    elif name == "uniform":
        result = np.full_like(value, 0.5, dtype=np.float64)
    elif name == "cosine":
        result = (np.pi / 4.0) * np.cos((np.pi / 2.0) * value)
    else:
        raise AssertionError(name)
    return np.where(inside, result, 0.0)


def reference_kde(data, xs, bandwidth, kernel):
    data = np.asarray(data)
    xs = np.asarray(xs)
    values = kernel_value(kernel, (xs[:, None] - data[None, :]) / bandwidth)
    return values.mean(axis=1) / bandwidth


def reference_self(data, bandwidth, kernel):
    data = np.asarray(data)
    values = kernel_value(
        kernel, (data[:, None] - data[None, :]) / bandwidth
    )
    np.fill_diagonal(values, 0.0)
    return values.sum(axis=1) / ((data.size - 1) * bandwidth)


def reference_reflected(data, xs, bandwidth, kernel, self_density=False):
    data = np.asarray(data)
    xs = np.asarray(xs)
    values = (
        kernel_value(kernel, (xs[:, None] - data[None, :]) / bandwidth)
        + kernel_value(kernel, (xs[:, None] + data[None, :]) / bandwidth)
        + kernel_value(
            kernel, (xs[:, None] - (2.0 - data[None, :])) / bandwidth
        )
    )
    if self_density:
        np.fill_diagonal(values, 0.0)
        divisor = (data.size - 1) * bandwidth
    else:
        divisor = data.size * bandwidth
    return values.sum(axis=1) / divisor


def beta_value(x, sample, bandwidth):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    a = sample / bandwidth + 1.0
    b = (1.0 - sample) / bandwidth + 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp(
        (a - 1.0) * math.log(x)
        + (b - 1.0) * math.log1p(-x)
        - log_beta
    )


def reference_beta(data, xs, bandwidth, self_density=False):
    result = []
    for i, x in enumerate(xs):
        values = [
            beta_value(x, sample, bandwidth)
            for j, sample in enumerate(data)
            if not self_density or i != j
        ]
        result.append(np.mean(values))
    return np.asarray(result)


def reference_multivariate(data, xs, bandwidth, kernel, self_density=False):
    data = np.asarray(data)
    xs = np.asarray(xs)
    result = []
    for i, point in enumerate(xs):
        values = []
        for j, sample in enumerate(data):
            if self_density and i == j:
                continue
            values.append(
                np.prod(kernel_value(kernel, (point - sample) / bandwidth))
            )
        result.append(np.mean(values) / bandwidth**data.shape[1])
    return np.asarray(result)


def reference_kfold(data, bandwidth, kernel, folds):
    data = np.asarray(data)
    total = 0.0
    for fold in range(folds):
        start = fold * data.size // folds
        end = (fold + 1) * data.size // folds
        train = np.concatenate((data[:start], data[end:]))
        density = reference_kde(train, data[start:end], bandwidth, kernel)
        total += np.log(density).sum()
    return total / data.size


def test_all_core_functions_are_publicly_exported():
    core_functions = {name for name in dir(_core) if not name.startswith("_")}
    assert set(_fun.__all__) == core_functions
    assert set(_fun.__all__) <= set(kern.__all__)
    assert all(getattr(kern, name) is getattr(_fun, name) for name in _fun.__all__)


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_metadata(kernel):
    assert _fun.kernel_is_symmetric(kernel) is True
    expected = 4.0 if kernel == "gaussian" else 1.0
    assert _fun.kernel_default_cutoff(kernel) == expected


@pytest.mark.parametrize("kernel", KERNELS)
def test_kde_and_kde_self_match_reference(kernel):
    data = [0.0, 0.3, 0.8, 1.1]
    xs = [-0.2, 0.25, 0.9]
    bandwidth = 0.4

    np.testing.assert_allclose(
        _fun.kde(data, xs, bandwidth, kernel),
        reference_kde(data, xs, bandwidth, kernel),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _fun.kde_self(data, bandwidth, kernel),
        reference_self(data, bandwidth, kernel),
        rtol=1e-12,
        atol=1e-12,
    )


def test_beta_kde_functions_match_reference():
    data = [0.05, 0.3, 0.7, 0.95]
    xs = [0.0, 0.2, 0.5, 1.0]
    bandwidth = 0.2

    np.testing.assert_allclose(
        _fun.kde_beta_ext(data, xs, bandwidth),
        reference_beta(data, xs, bandwidth),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _fun.kde_beta_self(data, bandwidth),
        reference_beta(data, data, bandwidth, self_density=True),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_reflected_kde_functions_match_reference(kernel):
    data = [0.05, 0.3, 0.7, 0.95]
    xs = [0.0, 0.2, 0.8, 1.0]
    bandwidth = 0.2

    np.testing.assert_allclose(
        _fun.kde_reflected_ext(data, xs, bandwidth, kernel),
        reference_reflected(data, xs, bandwidth, kernel),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _fun.kde_reflected_self(data, bandwidth, kernel),
        reference_reflected(
            data, data, bandwidth, kernel, self_density=True
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_bandwidth_functions_match_reference_and_parallel_aliases():
    data = np.asarray([-1.1, -0.4, 0.0, 0.35, 1.2])
    grid = np.asarray([0.25, 0.6, 1.0])
    loo = np.asarray(
        [np.log(reference_self(data, h, "gaussian")).mean() for h in grid]
    )
    kfold = np.asarray(
        [reference_kfold(data, h, "gaussian", 2) for h in grid]
    )

    np.testing.assert_allclose(_fun.bandwidth_loo(data, grid), loo)
    np.testing.assert_allclose(_fun.bandwidth_kfold(data, grid, 2), kfold)
    np.testing.assert_allclose(_fun.bandwidth_grid(data, grid), loo)
    np.testing.assert_allclose(
        _fun.bandwidth_grid(data, grid, k_folds=2), kfold
    )
    np.testing.assert_allclose(
        _fun.bandwidth_grid(data, grid, parallel="grid"), loo
    )
    np.testing.assert_allclose(
        _fun.bandwidth_grid(data, grid, parallel="evaluation"), loo
    )
    np.testing.assert_allclose(
        _fun.bandwidth_grid(data, grid, parallel="bandwidths"), loo
    )
    np.testing.assert_allclose(
        _fun.bandwidth_grid(data, grid, parallel="kernels"), loo
    )


@pytest.mark.parametrize("memory", _fun.MEMORY_MODES)
def test_approximate_kde_functions(memory):
    data = np.asarray([-1.0, -0.3, 0.2, 0.8, 1.4])
    xs = np.asarray([-0.7, 0.0, 1.0])
    bandwidth = 0.35

    np.testing.assert_allclose(
        _fun.kde_approx(
            data, xs, bandwidth, cutoff=100.0, memory=memory
        ),
        _fun.kde(data, xs, bandwidth),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _fun.kde_approx_self(
            data, bandwidth, cutoff=100.0, memory=memory
        ),
        _fun.kde_self(data, bandwidth),
        rtol=1e-12,
        atol=1e-12,
    )

    limited = _fun.kde_approx(
        data, xs, bandwidth, max_neighbors=1, fast_gaussian=True, memory=memory
    )
    assert limited.shape == xs.shape
    assert np.all(np.isfinite(limited))


def test_approximate_kde_accepts_core_sentinel_values():
    data = [-1.0, 0.0, 1.0]
    points = [-0.5, 0.5]

    np.testing.assert_allclose(
        _fun.kde_approx(data, points, 0.4, "gaussian", 0.0, 0, False, 2),
        _fun.kde_approx(data, points, 0.4),
    )
    np.testing.assert_allclose(
        _fun.kde_approx_self(data, 0.4, "gaussian", 0.0, 0, False, 2),
        _fun.kde_approx_self(data, 0.4),
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_multivariate_kde_functions_match_reference(kernel):
    data = [[0.0, 0.0], [0.3, 0.7], [1.0, 0.2]]
    xs = [[0.1, 0.2], [0.8, 0.4]]
    bandwidth = 0.5

    np.testing.assert_allclose(
        _fun.kde_multivariate(data, xs, bandwidth, kernel, block_size=2),
        reference_multivariate(data, xs, bandwidth, kernel),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _fun.kde_multivariate_self(data, bandwidth, kernel, block_size=2),
        reference_multivariate(
            data, data, bandwidth, kernel, self_density=True
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_array_like_inputs_are_converted_and_empty_queries_are_supported():
    result = _fun.kde(np.asarray([0, 1], dtype=np.int32), [], 0.5)
    assert result.dtype == np.float64
    assert result.shape == (0,)

    matrix_result = _fun.kde_multivariate(
        [[0, 0], [1, 1]], np.empty((0, 2)), 0.5
    )
    assert matrix_result.shape == (0,)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: _fun.kde([], [0.0], 1.0), "must not be empty"),
        (lambda: _fun.kde([0.0], [0.0], 0.0), "bandwidth must be positive"),
        (lambda: _fun.kde([np.nan], [0.0], 1.0), "finite"),
        (lambda: _fun.kde([0.0], [0.0], 1.0, "bad"), "kernel must be"),
        (lambda: _fun.kde_beta_ext([-0.1], [0.0], 1.0), r"\[0, 1\]"),
        (lambda: _fun.kde_reflected_self([1.1], 1.0), r"\[0, 1\]"),
        (lambda: _fun.bandwidth_loo([0.0, 1.0], [0.0]), "positive"),
        (
            lambda: _fun.bandwidth_kfold([0.0, 1.0], [1.0], 3),
            "must not exceed",
        ),
        (
            lambda: _fun.bandwidth_grid([0.0, 1.0], [1.0], parallel="bad"),
            "parallel must be",
        ),
        (
            lambda: _fun.kde_approx([1.0, 0.0], [0.0], 1.0),
            "must be sorted",
        ),
        (
            lambda: _fun.kde_approx(
                [0.0, 1.0], [0.0], 1.0, cutoff=np.inf
            ),
            "cutoff must be",
        ),
        (
            lambda: _fun.kde_approx(
                [0.0, 1.0], [0.0], 1.0, max_neighbors=1.5
            ),
            "must be an integer",
        ),
        (
            lambda: _fun.kde_approx(
                [0.0, 1.0], [0.0], 1.0, kernel="uniform",
                fast_gaussian=True
            ),
            "only available",
        ),
        (
            lambda: _fun.kde_approx([0.0, 1.0], [0.0], 1.0, memory="bad"),
            "memory must be",
        ),
        (
            lambda: _fun.kde_multivariate([[0.0, 1.0]], [[0.0]], 1.0),
            "same number of features",
        ),
        (
            lambda: _fun.kde_multivariate_self([[0.0]], 1.0, block_size=0),
            "at least 1",
        ),
        (lambda: _fun.kernel_default_cutoff("bad"), "kernel must be"),
    ],
)
def test_validation_errors(call, message):
    with pytest.raises(ValueError, match=message):
        call()
