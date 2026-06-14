"""Fit and evaluate an exact one-dimensional KDE."""

import numpy as np

from kern import KernelDensity


rng = np.random.default_rng(0)
data = rng.normal(size=1_000)
points = np.linspace(-3.0, 3.0, 200)

model = KernelDensity(bandwidth=0.25, kernel="gaussian").fit(data)
density = model.evaluate(points)

print(density[:5])
