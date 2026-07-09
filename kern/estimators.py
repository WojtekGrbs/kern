"""Scikit-learn-like estimators built on the C extension."""

import numpy as np

from . import _fun


KERNELS = _fun.KERNELS


def _one_dimensional(values, name="X"):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or have shape (n, 1)")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _two_dimensional(values, name="X"):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _check_kernel(kernel):
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")


def _check_bounded_method(method):
    if method not in (None, "reflected", "beta"):
        raise ValueError("method must be None, 'reflected', or 'beta'")
    return method


def _check_bandwidth(bandwidth):
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    return bandwidth


def _log_density(density):
    return np.log(np.maximum(density, 1e-300))


def _bandwidth_log_density(density):
    density = np.asarray(density, dtype=np.float64)
    result = np.full(density.shape, -700.0, dtype=np.float64)
    positive = density > 1e-300
    result[positive] = np.log(density[positive])
    return result


def default_bandwidth_grid(X, size=64, minimum=None, maximum=None):
    """Create a data-scaled logarithmic bandwidth grid.

    The default range is centered on Silverman's robust normal-reference
    bandwidth :ref:`[Silverman 1986] <reference-silverman>` and spans a factor
    of four in each direction.

    Parameters
    ----------
    X : array-like
        One-dimensional training samples.
    size : int, default=64
        Number of grid values.
    minimum : float or None, default=None
        Smallest bandwidth. Uses one quarter of the reference bandwidth when
        omitted.
    maximum : float or None, default=None
        Largest bandwidth. Uses four times the reference bandwidth when
        omitted.

    Returns
    -------
    numpy.ndarray
        C-contiguous positive ``float64`` bandwidths.

    Examples
    --------
    >>> grid = default_bandwidth_grid([0.0, 0.2, 0.8, 1.0], size=8)
    >>> grid.shape
    (8,)
    """
    data = _one_dimensional(X)
    if data.size < 2:
        raise ValueError("a default grid needs at least two samples")

    size = int(size)
    if size < 2:
        raise ValueError("size must be at least 2")

    std = float(np.std(data, ddof=1))
    q25, q75 = np.percentile(data, [25.0, 75.0])
    robust = float((q75 - q25) / 1.349)
    candidates = [value for value in (std, robust) if value > 0.0]
    scale = min(candidates) if candidates else float(np.ptp(data))
    if scale <= 0.0:
        scale = max(abs(float(data[0])), 1.0)

    reference = 0.9 * scale * data.size ** -0.2
    lower = reference / 4.0 if minimum is None else _check_bandwidth(minimum)
    upper = reference * 4.0 if maximum is None else _check_bandwidth(maximum)
    if upper <= lower:
        raise ValueError("maximum must be greater than minimum")

    return np.ascontiguousarray(np.geomspace(lower, upper, size),
                                dtype=np.float64)


class _Estimator:
    def _require_fitted(self):
        if not hasattr(self, "data_"):
            raise ValueError("fit must be called before evaluating the estimator")

    def get_params(self):
        """Return constructor parameters."""
        return {name: getattr(self, name) for name in self._parameter_names}

    def set_params(self, **params):
        """Set constructor parameters and return the estimator."""
        for name, value in params.items():
            if name not in self._parameter_names:
                raise ValueError(f"unknown parameter {name!r}")
            setattr(self, name, value)
        return self


class KernelDensity(_Estimator):
    """Exact one-dimensional kernel density estimator.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel bandwidth.
    kernel : str, default="gaussian"
        One of ``gaussian``, ``epanechnikov``, ``triangular``, ``uniform``,
        or ``cosine``.

    Examples
    --------
    >>> model = KernelDensity(bandwidth=0.3).fit([0.0, 0.2, 1.0])
    >>> model.evaluate([0.1, 0.5]).shape
    (2,)
    """

    _parameter_names = ("bandwidth", "kernel")

    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y=None):
        """Store the training samples."""
        del y
        _check_kernel(self.kernel)
        self.bandwidth = _check_bandwidth(self.bandwidth)
        self.data_ = _one_dimensional(X)
        self.n_features_in_ = 1
        return self

    def evaluate(self, X):
        """Return density values at X."""
        self._require_fitted()
        return _fun.kde(self.data_, _one_dimensional(X), self.bandwidth,
                        self.kernel)

    def self_density(self):
        """Return leave-one-out density at each training sample."""
        self._require_fitted()
        return _fun.kde_self(self.data_, self.bandwidth, self.kernel)

    def score_samples(self, X):
        """Return log-density values at X."""
        return _log_density(self.evaluate(X))

    def score(self, X, y=None):
        """Return the total log-likelihood of X."""
        del y
        return float(self.score_samples(X).sum())


class ApproximateKernelDensity(_Estimator):
    """Sorted one-dimensional KDE with distance and neighbor cutoffs.

    The samples are sorted once by :meth:`fit`. ``cutoff`` is measured in
    bandwidths. When it is ``None``, compact kernels use 1 and Gaussian uses
    4. ``max_neighbors`` limits work on each side of a point.

    Set ``fast_gaussian=True`` to use Schraudolph's exponential approximation
    :ref:`[Schraudolph 1999] <reference-schraudolph>`.
    ``memory="high"`` evaluates symmetric self-pairs once with per-thread
    buffers and caches external cutoff bounds. ``"low"`` avoids those buffers,
    while ``"auto"`` chooses based on problem size.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel bandwidth.
    kernel : str, default="gaussian"
        Kernel name.
    cutoff : float or None, default=None
        Distance cutoff measured in bandwidths. If ``None``, compact kernels
        use 1 and Gaussian uses 4.
    max_neighbors : int or None, default=None
        Maximum number of neighbors to evaluate on each side of a point.
    fast_gaussian : bool, default=False
        Whether to use Schraudolph's exponential approximation for Gaussian
        kernels.
    memory : {"auto", "high", "low"}, default="auto"
        Memory strategy. ``"high"`` evaluates symmetric self-pairs once with
        per-thread buffers and caches external cutoff bounds. ``"low"`` avoids
        those buffers. ``"auto"`` chooses based on problem size and cutoff
        density.

    Examples
    --------
    Limit the Gaussian kernel to nearby samples:

    >>> model = ApproximateKernelDensity(
    ...     bandwidth=0.2, cutoff=3.0, max_neighbors=2
    ... ).fit([0.0, 0.1, 0.4, 0.9, 1.0])
    >>> model.evaluate([0.2, 0.8]).shape
    (2,)
    >>> model.self_density().shape
    (5,)
    """
    _parameter_names = (
        "bandwidth", "kernel", "cutoff", "max_neighbors", "fast_gaussian",
        "memory"
    )

    def __init__(self, bandwidth=1.0, kernel="gaussian", cutoff=None,
                 max_neighbors=None, fast_gaussian=False, memory="auto"):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.fast_gaussian = fast_gaussian
        self.memory = memory

    def fit(self, X, y=None):
        """Sort and store the training samples."""
        del y
        _check_kernel(self.kernel)
        self.bandwidth = _check_bandwidth(self.bandwidth)
        _fun._approximation_options(
            self.kernel, self.cutoff, self.max_neighbors, self.fast_gaussian,
            self.memory, core_compatible=False
        )

        data = _one_dimensional(X)
        self.sort_order_ = np.argsort(data, kind="mergesort")
        self.data_ = np.ascontiguousarray(data[self.sort_order_])
        self.n_features_in_ = 1
        return self

    def evaluate(self, X):
        """Return approximate density values at X."""
        self._require_fitted()
        return _fun.kde_approx(
            self.data_, _one_dimensional(X), self.bandwidth, self.kernel,
            self.cutoff, self.max_neighbors, bool(self.fast_gaussian),
            self.memory
        )

    def self_density(self):
        """Return approximate leave-one-out density in original input order."""
        self._require_fitted()
        sorted_density = _fun.kde_approx_self(
            self.data_, self.bandwidth, self.kernel, self.cutoff,
            self.max_neighbors, bool(self.fast_gaussian), self.memory
        )
        density = np.empty_like(sorted_density)
        density[self.sort_order_] = sorted_density
        return density

    def score_samples(self, X):
        """Return approximate log-density values at X."""
        return _log_density(self.evaluate(X))

    def score(self, X, y=None):
        """Return the total approximate log-likelihood of X."""
        del y
        return float(self.score_samples(X).sum())


class BoundedKernelDensity(_Estimator):
    """One-dimensional KDE for data on the [0, 1] unit interval.

    ``method=None`` uses the regular unbounded kernel density estimator.
    ``method="reflected"`` works with every symmetric standard kernel.
    It uses the reflection construction for support constraints
    :ref:`[Schuster 1985] <reference-schuster>`. ``method="beta"`` uses the
    sample-centered Beta kernel.

    Parameters
    ----------
    bandwidth : float, default=0.1
        Kernel bandwidth.
    kernel : str, default="gaussian"
        Kernel name used by the reflected and regular methods.
    method : {None, "reflected", "beta"}, default="reflected"
        Boundary correction method. ``None`` uses the regular unbounded KDE.
        ``"reflected"`` works with every symmetric standard kernel.
        ``"beta"`` uses the boundary-aware Beta kernel.

    Examples
    --------
    Fit a reflected KDE on samples inside the unit interval:

    >>> model = BoundedKernelDensity(bandwidth=0.1).fit(
    ...     [0.05, 0.2, 0.6, 0.95]
    ... )
    >>> model.evaluate([0.0, 0.5, 1.0]).shape
    (3,)

    Use the boundary-aware Beta kernel instead:

    >>> beta_model = BoundedKernelDensity(method="beta", bandwidth=0.05).fit(
    ...     [0.1, 0.4, 0.8]
    ... )
    >>> beta_model.score_samples([0.2, 0.7]).shape
    (2,)
    """

    _parameter_names = ("bandwidth", "kernel", "method")

    def __init__(self, bandwidth=0.1, kernel="gaussian", method="reflected"):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.method = method

    def fit(self, X, y=None):
        """Store samples after checking the selected domain."""
        del y
        self.bandwidth = _check_bandwidth(self.bandwidth)
        self.method = _check_bounded_method(self.method)
        if self.method != "beta":
            _check_kernel(self.kernel)
        self.data_ = _one_dimensional(X)
        if self.method is not None and np.any(
                (self.data_ < 0.0) | (self.data_ > 1.0)):
            raise ValueError("bounded KDE data must be inside [0, 1]")
        self.n_features_in_ = 1
        return self

    def evaluate(self, X):
        """Return density values at X."""
        self._require_fitted()
        points = _one_dimensional(X)
        if self.method is None:
            return _fun.kde(self.data_, points, self.bandwidth, self.kernel)
        if np.any((points < 0.0) | (points > 1.0)):
            raise ValueError("bounded KDE evaluation points must be inside [0, 1]")
        if self.method == "beta":
            return _fun.kde_beta_ext(self.data_, points, self.bandwidth)
        return _fun.kde_reflected_ext(
            self.data_, points, self.bandwidth, self.kernel
        )

    def self_density(self):
        """Return leave-one-out density at each training sample."""
        self._require_fitted()
        if self.method is None:
            return _fun.kde_self(self.data_, self.bandwidth, self.kernel)
        if self.method == "beta":
            return _fun.kde_beta_self(self.data_, self.bandwidth)
        return _fun.kde_reflected_self(
            self.data_, self.bandwidth, self.kernel
        )

    def score_samples(self, X):
        """Return log-density values at X."""
        return _log_density(self.evaluate(X))

    def score(self, X, y=None):
        """Return the total log-likelihood of X."""
        del y
        return float(self.score_samples(X).sum())


class MultivariateKernelDensity(_Estimator):
    """Multivariate KDE.

    ``block_size`` controls how many query rows reuse each cached data block.
    Values from 16 to 64 are usually useful; the default is 32.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel bandwidth.
    kernel : str, default="gaussian"
        Kernel name.
    block_size : int, default=32
        Number of query rows that reuse each cached data block. Values from
        16 to 64 are usually useful.

    Examples
    --------
    Fit a two-dimensional estimator and evaluate query rows:

    >>> X = [[0.0, 0.0], [0.2, 0.1], [1.0, 0.9]]
    >>> model = MultivariateKernelDensity(bandwidth=0.3).fit(X)
    >>> model.evaluate([[0.1, 0.1], [0.8, 0.8]]).shape
    (2,)
    >>> model.n_features_in_
    2
    """

    _parameter_names = ("bandwidth", "kernel", "block_size")

    def __init__(self, bandwidth=1.0, kernel="gaussian", block_size=32):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.block_size = block_size

    def fit(self, X, y=None):
        """Store a row-major sample matrix."""
        del y
        _check_kernel(self.kernel)
        self.bandwidth = _check_bandwidth(self.bandwidth)
        self.block_size = int(self.block_size)
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        self.data_ = _two_dimensional(X)
        self.n_features_in_ = self.data_.shape[1]
        return self

    def evaluate(self, X):
        """Return density values at rows of X."""
        self._require_fitted()
        points = _two_dimensional(X)
        if points.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features")
        return _fun.kde_multivariate(
            self.data_, points, self.bandwidth, self.kernel, self.block_size
        )

    def self_density(self):
        """Return leave-one-out density at each training row."""
        self._require_fitted()
        return _fun.kde_multivariate_self(
            self.data_, self.bandwidth, self.kernel, self.block_size
        )

    def score_samples(self, X):
        """Return log-density values at rows of X."""
        return _log_density(self.evaluate(X))

    def score(self, X, y=None):
        """Return the total log-likelihood of X."""
        del y
        return float(self.score_samples(X).sum())


class BandwidthSelector(_Estimator):
    """Select a bandwidth for one kernel using log-likelihood.

    Parameters
    ----------
    grid : array-like or None, default=None
        Candidate positive bandwidths. When omitted, :func:`default_bandwidth_grid`
        creates a data-scaled logarithmic grid during :meth:`fit`.
    kernel : str, default="gaussian"
        Kernel used for every bandwidth candidate except Beta-kernel bounded
        selection.
    cv : "loo" or int, default="loo"
        Leave-one-out or the number of folds.
    parallel : {"auto", "grid", "evaluation"}, default="auto"
        Parallelize bandwidth candidates or each individual LOO/k-fold
        evaluation.
    bounded : bool, default=False
        Whether to select bandwidths for :class:`BoundedKernelDensity`.
    bounded_method : {None, "reflected", "beta"}, default="reflected"
        Boundary correction method used when ``bounded=True``.

    Examples
    --------
    Use the default grid:

    >>> selector = BandwidthSelector(kernel="cosine").fit(
    ...     [0.0, 0.1, 0.4, 0.8, 1.0]
    ... )
    >>> selector.best_bandwidth_ in selector.grid_
    True

    Or provide an explicit grid:

    >>> selector = BandwidthSelector(grid=[0.1, 0.2, 0.4]).fit(
    ...     [0.0, 0.1, 0.4, 0.8, 1.0]
    ... )
    """

    _parameter_names = (
        "grid", "kernel", "cv", "parallel", "bounded", "bounded_method"
    )

    def __init__(self, grid=None, kernel="gaussian", cv="loo",
                 parallel="auto", bounded=False, bounded_method="reflected"):
        self.grid = grid
        self.kernel = kernel
        self.cv = cv
        self.parallel = parallel
        self.bounded = bounded
        self.bounded_method = bounded_method

    @staticmethod
    def _bounded_density(data, points, bandwidth, kernel, method):
        if method is None:
            return _fun.kde(data, points, bandwidth, kernel)
        if method == "beta":
            return _fun.kde_beta_ext(data, points, bandwidth)
        return _fun.kde_reflected_ext(data, points, bandwidth, kernel)

    @staticmethod
    def _bounded_self_density(data, bandwidth, kernel, method):
        if method is None:
            return _fun.kde_self(data, bandwidth, kernel)
        if method == "beta":
            return _fun.kde_beta_self(data, bandwidth)
        return _fun.kde_reflected_self(data, bandwidth, kernel)

    @classmethod
    def _bounded_bandwidth_grid(cls, data, grid, kernel, k_folds, method):
        scores = np.empty(grid.shape, dtype=np.float64)
        for index, bandwidth in enumerate(grid):
            if k_folds == 1:
                scores[index] = float(
                    _bandwidth_log_density(
                        cls._bounded_self_density(
                            data, bandwidth, kernel, method
                        )
                    ).mean()
                )
                continue

            total_score = 0.0
            for fold in range(k_folds):
                start = fold * data.size // k_folds
                end = (fold + 1) * data.size // k_folds
                train = np.concatenate((data[:start], data[end:]))
                density = cls._bounded_density(
                    train, data[start:end], bandwidth, kernel, method
                )
                total_score += float(_bandwidth_log_density(density).sum())
            scores[index] = total_score / data.size
        return scores

    def fit(self, X, y=None):
        """Score the grid and fit the best exact one-dimensional estimator."""
        del y
        data = _one_dimensional(X)
        if data.size < 2:
            raise ValueError("bandwidth selection needs at least two samples")
        grid = (default_bandwidth_grid(data) if self.grid is None
                else _one_dimensional(self.grid, "grid"))
        if np.any(grid <= 0.0):
            raise ValueError("all grid bandwidths must be positive")

        if not isinstance(self.bounded, (bool, np.bool_)):
            raise ValueError("bounded must be a boolean")
        bounded = bool(self.bounded)

        bounded_method = _check_bounded_method(self.bounded_method)
        if (not bounded) or bounded_method != "beta":
            _check_kernel(self.kernel)
        if bounded and bounded_method is not None and np.any(
                (data < 0.0) | (data > 1.0)):
            raise ValueError("bounded KDE data must be inside [0, 1]")

        if self.cv == "loo":
            k_folds = 1
        else:
            k_folds = int(self.cv)
            if k_folds < 2 or k_folds > data.size:
                raise ValueError("cv must be 'loo' or an integer between 2 and n")
        if self.parallel not in ("auto", "grid", "evaluation"):
            raise ValueError("parallel must be 'auto', 'grid', or 'evaluation'")

        if bounded:
            self.scores_ = self._bounded_bandwidth_grid(
                data, grid, self.kernel, k_folds, bounded_method
            )
        else:
            self.scores_ = _fun.bandwidth_grid(
                data, grid, self.kernel, k_folds, self.parallel
            )
        best_bandwidth_index = int(np.argmax(self.scores_))

        self.data_ = data
        self.grid_ = grid
        self.kernel_ = self.kernel
        self.bounded_ = bounded
        self.bounded_method_ = bounded_method
        self.best_score_ = float(self.scores_[best_bandwidth_index])
        self.best_bandwidth_ = float(grid[best_bandwidth_index])
        if bounded:
            self.best_estimator_ = BoundedKernelDensity(
                bandwidth=self.best_bandwidth_,
                kernel=self.kernel_,
                method=self.bounded_method_,
            ).fit(data)
        else:
            self.best_estimator_ = KernelDensity(
                bandwidth=self.best_bandwidth_, kernel=self.kernel_
            ).fit(data)
        self.n_features_in_ = 1
        return self

    def evaluate(self, X):
        """Evaluate the selected estimator."""
        self._require_fitted()
        return self.best_estimator_.evaluate(X)

    def score_samples(self, X):
        """Return log-density values from the selected estimator."""
        self._require_fitted()
        return self.best_estimator_.score_samples(X)

    def self_density(self):
        """Return leave-one-out density from the selected estimator."""
        self._require_fitted()
        return self.best_estimator_.self_density()

    def score(self, X, y=None):
        """Return total log-likelihood from the selected estimator."""
        self._require_fitted()
        return self.best_estimator_.score(X, y)
