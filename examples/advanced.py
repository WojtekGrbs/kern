"""Use approximate and multivariate KDE."""

import numpy as np

from kern import ApproximateKernelDensity, MultivariateKernelDensity


rng = np.random.default_rng(2)
data = rng.normal(size=2_000)
points = np.linspace(-3.0, 3.0, 200)

approximate = ApproximateKernelDensity(
    bandwidth=0.25,
    cutoff=3.5,
    memory="auto",
).fit(data)
print(approximate.evaluate(points)[:5])
# Out: [0.00523714 0.00582754 0.00646412 0.00715279 0.00789581]

matrix = rng.normal(size=(1_000, 3))
multivariate = MultivariateKernelDensity(
    bandwidth=0.4,
    kernel="gaussian",
    block_size=32,
).fit(matrix)
print(multivariate.evaluate(matrix[:5]))
# [0.01052811 0.01548928 0.00208585 0.02895958 0.03646734]
