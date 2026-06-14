"""Select a bandwidth for one kernel using log-likelihood."""

import numpy as np

from kern import BandwidthSelector, default_bandwidth_grid


rng = np.random.default_rng(1)
data = rng.normal(size=1_000)

# Omit grid to use the default.
automatic = BandwidthSelector(
    kernel="gaussian",
    cv="loo",
).fit(data)

# Or construct and pass a custom grid.
grid = default_bandwidth_grid(data, size=64, minimum=0.05, maximum=0.8)
custom = BandwidthSelector(
    grid=grid,
    kernel="epanechnikov",
    cv=5,
    parallel="evaluation",
).fit(data)

print(automatic.kernel_, automatic.best_bandwidth_)
print(custom.kernel_, custom.best_bandwidth_)
