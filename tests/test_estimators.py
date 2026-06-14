import numpy as np
import pytest

from kern import (
    ApproximateKernelDensity,
    BandwidthSelector,
    BoundedKernelDensity,
    KernelDensity,
    MultivariateKernelDensity,
    default_bandwidth_grid,
)
from kern import _fun


def test_default_bandwidth_grid_properties_and_validation():
    grid = default_bandwidth_grid([0.0, 0.2, 0.8, 1.0], size=8)
    assert grid.shape == (8,)
    assert grid.dtype == np.float64
    assert grid.flags.c_contiguous
    assert np.all(np.diff(grid) > 0.0)
    assert np.all(grid > 0.0)

    explicit = default_bandwidth_grid(
        [0.0, 1.0], size=3, minimum=0.1, maximum=0.9
    )
    np.testing.assert_allclose(explicit, [0.1, 0.3, 0.9])

    with pytest.raises(ValueError, match="at least two samples"):
        default_bandwidth_grid([1.0])
    with pytest.raises(ValueError, match="size must be at least 2"):
        default_bandwidth_grid([0.0, 1.0], size=1)
    with pytest.raises(ValueError, match="maximum must be greater"):
        default_bandwidth_grid([0.0, 1.0], minimum=1.0, maximum=0.5)


def test_kernel_density_lifecycle_scoring_and_params():
    data = [0.0, 0.3, 0.8]
    points = [[0.1], [0.6]]
    model = KernelDensity(bandwidth=0.4, kernel="triangular")

    assert model.get_params() == {"bandwidth": 0.4, "kernel": "triangular"}
    assert model.set_params(kernel="gaussian") is model
    with pytest.raises(ValueError, match="unknown parameter"):
        model.set_params(unknown=1)
    with pytest.raises(ValueError, match="fit must be called"):
        model.evaluate(points)

    assert model.fit(np.asarray(data, dtype=np.float32), y=[1, 2, 3]) is model
    assert model.data_.dtype == np.float64
    assert model.n_features_in_ == 1
    np.testing.assert_allclose(
        model.evaluate(points), _fun.kde(data, [0.1, 0.6], 0.4)
    )
    np.testing.assert_allclose(model.self_density(), _fun.kde_self(data, 0.4))
    np.testing.assert_allclose(
        model.score_samples(points), np.log(model.evaluate(points))
    )
    assert model.score(points) == pytest.approx(model.score_samples(points).sum())


@pytest.mark.parametrize(
    "model",
    [
        KernelDensity(kernel="bad"),
        KernelDensity(bandwidth=0.0),
        KernelDensity(),
    ],
)
def test_kernel_density_fit_validation(model):
    data = [[0.0, 1.0]] if model.kernel == "gaussian" and model.bandwidth == 1.0 else [0.0, 1.0]
    with pytest.raises(ValueError):
        model.fit(data)


def test_approximate_kernel_density_sorts_and_restores_original_order():
    data = np.asarray([0.8, -0.5, 0.2, 1.1])
    points = [-0.2, 0.7]
    model = ApproximateKernelDensity(
        bandwidth=0.3, cutoff=100.0, memory="high"
    ).fit(data)

    np.testing.assert_array_equal(model.data_, np.sort(data))
    np.testing.assert_array_equal(model.sort_order_, np.argsort(data, kind="mergesort"))
    np.testing.assert_allclose(
        model.evaluate(points), _fun.kde(data, points, 0.3)
    )
    np.testing.assert_allclose(
        model.self_density(), _fun.kde_self(data, 0.3)
    )
    assert model.score(points) == pytest.approx(model.score_samples(points).sum())


@pytest.mark.parametrize(
    "params",
    [
        {"cutoff": 0.0},
        {"cutoff": np.inf},
        {"max_neighbors": 0},
        {"max_neighbors": 1.5},
        {"kernel": "uniform", "fast_gaussian": True},
        {"memory": "bad"},
    ],
)
def test_approximate_kernel_density_validation(params):
    with pytest.raises(ValueError):
        ApproximateKernelDensity(**params).fit([0.0, 1.0])


@pytest.mark.parametrize("method", ["reflected", "beta"])
def test_bounded_kernel_density_methods(method):
    data = [0.05, 0.3, 0.8]
    points = [0.0, 0.4, 1.0]
    model = BoundedKernelDensity(bandwidth=0.2, method=method).fit(data)

    expected = (
        _fun.kde_beta_ext(data, points, 0.2)
        if method == "beta"
        else _fun.kde_reflected_ext(data, points, 0.2)
    )
    expected_self = (
        _fun.kde_beta_self(data, 0.2)
        if method == "beta"
        else _fun.kde_reflected_self(data, 0.2)
    )
    np.testing.assert_allclose(model.evaluate(points), expected)
    np.testing.assert_allclose(model.self_density(), expected_self)
    assert model.score(points) == pytest.approx(model.score_samples(points).sum())


def test_bounded_kernel_density_validation():
    with pytest.raises(ValueError, match="method must be"):
        BoundedKernelDensity(method="bad").fit([0.5])
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        BoundedKernelDensity().fit([-0.1])
    model = BoundedKernelDensity().fit([0.5])
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        model.evaluate([1.1])


def test_multivariate_kernel_density_lifecycle_and_validation():
    data = [[0.0, 0.0], [0.3, 0.7], [1.0, 0.2]]
    points = [[0.1, 0.2], [0.8, 0.4]]
    model = MultivariateKernelDensity(
        bandwidth=0.5, kernel="cosine", block_size=2
    ).fit(data)

    assert model.n_features_in_ == 2
    np.testing.assert_allclose(
        model.evaluate(points),
        _fun.kde_multivariate(data, points, 0.5, "cosine", 2),
    )
    np.testing.assert_allclose(
        model.self_density(),
        _fun.kde_multivariate_self(data, 0.5, "cosine", 2),
    )
    assert model.score(points) == pytest.approx(model.score_samples(points).sum())

    with pytest.raises(ValueError, match="different number of features"):
        model.evaluate([[0.0]])
    with pytest.raises(ValueError, match="block_size must be positive"):
        MultivariateKernelDensity(block_size=0).fit(data)
    with pytest.raises(ValueError, match="shape"):
        MultivariateKernelDensity().fit([0.0, 1.0])


@pytest.mark.parametrize(("cv", "parallel"), [("loo", "grid"), (2, "evaluation")])
def test_bandwidth_selector_matches_low_level_scores(cv, parallel):
    data = [-1.0, -0.2, 0.1, 0.6, 1.2]
    grid = [0.2, 0.5, 0.9]
    selector = BandwidthSelector(
        grid=grid, kernel="gaussian", cv=cv, parallel=parallel
    ).fit(data)
    folds = 1 if cv == "loo" else cv
    expected = _fun.bandwidth_grid(
        data, grid, "gaussian", folds, parallel
    )

    np.testing.assert_allclose(selector.scores_, expected)
    assert selector.best_bandwidth_ == grid[int(np.argmax(expected))]
    assert selector.best_score_ == pytest.approx(np.max(expected))
    assert selector.kernel_ == "gaussian"
    assert isinstance(selector.best_estimator_, KernelDensity)
    np.testing.assert_allclose(
        selector.evaluate([0.0]), selector.best_estimator_.evaluate([0.0])
    )
    np.testing.assert_allclose(
        selector.self_density(), selector.best_estimator_.self_density()
    )
    assert selector.score([0.0]) == selector.best_estimator_.score([0.0])


def test_bandwidth_selector_default_grid_and_validation():
    selector = BandwidthSelector().fit([-1.0, -0.2, 0.4, 1.0])
    assert selector.grid_.shape == (64,)

    with pytest.raises(ValueError, match="at least two samples"):
        BandwidthSelector().fit([0.0])
    with pytest.raises(ValueError, match="positive"):
        BandwidthSelector(grid=[0.0, 1.0]).fit([0.0, 1.0])
    with pytest.raises(ValueError, match="cv must be"):
        BandwidthSelector(cv=3).fit([0.0, 1.0])
    with pytest.raises(ValueError, match="parallel must be"):
        BandwidthSelector(parallel="bad").fit([0.0, 1.0])


@pytest.mark.parametrize(
    "model",
    [
        ApproximateKernelDensity(),
        BoundedKernelDensity(),
        MultivariateKernelDensity(),
        BandwidthSelector(),
    ],
)
def test_estimators_reject_evaluation_before_fit(model):
    with pytest.raises(ValueError, match="fit must be called"):
        model.score_samples([[0.0]] if isinstance(model, MultivariateKernelDensity) else [0.0])
