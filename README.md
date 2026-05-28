# kern

Fast one-dimensional Kernel Density Estimation implemented as a NumPy C
extension.

Build locally:

```bash
python3 setup.py build_ext --inplace
```

```python
import numpy as np
import kern

x = np.array([1.0, 2.0, 3.0, 4.0])

# Biased estimator: divide by n.
density = kern.kde(x, h=0.4)

# Unbiased leave-one-out estimator: skip self and divide by n - 1.
loo_density = kern.kde(x, h=0.4, unbiased=True)

# Search h by leave-one-out log-likelihood, then evaluate the density.
density, h = kern.kde(x, optimize=True, return_bandwidth=True)

# Search only, using an explicit logarithmic grid.
h = kern.optimal_bandwidth(x, h_min=0.05, h_max=2.0, h_steps=128)
```

The inner pairwise loops are written in C. If OpenMP is available at build time,
the extension parallelizes across sample points and adds SIMD reduction hints for
the inner summation loop. If OpenMP is not available, the same code builds as a
serial extension.

Supported kernels:

`gaussian`, `uniform`, `triangular`, `epanechnikov`, `quartic`/`biweight`,
`triweight`, `tricube`, `cosine`, `logistic`, `sigmoid`.
