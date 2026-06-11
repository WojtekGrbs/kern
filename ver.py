import numpy as np
import kern

data = np.random.default_rng(0).normal(size=100).astype(np.float64)
xs = np.linspace(-3, 3, 200).astype(np.float64)
h_grid = np.linspace(0.05, 1.0, 20).astype(np.float64)

h = h_grid[np.argmax(kern.bandwidth_loo(data, h_grid, "loglik"))]

for kernel in ["gaussian", "epanechnikov", "triangular", "uniform", "cosine"]:
    print(kernel, kern.kernel_is_symmetric(kernel))
    print(kern.kde(data, xs, h, kernel)[:5])
    print(kern.kde_self(data, h, kernel)[:5])

bounded_data = np.random.default_rng(1).beta(2, 5, size=100).astype(np.float64)
bounded_xs = np.linspace(0.001, 0.999, 200).astype(np.float64)

print(kern.kde_beta_ext(bounded_data, bounded_xs, 0.05)[:5])
print(kern.kde_beta_self(bounded_data, 0.05)[:5])
print(kern.kde_reflected_ext(bounded_data, bounded_xs, 0.05, "gaussian")[:5])
print(kern.kde_reflected_self(bounded_data, 0.05, "gaussian")[:5])