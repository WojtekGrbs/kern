API Reference
=============
All estimators assume `X` is an object that can be casted to a `np.ndarray` with `dtype=np.float64`.

Estimators
----------

.. autoclass:: kern.KernelDensity
   :members:

.. autoclass:: kern.ApproximateKernelDensity
   :members:

.. autoclass:: kern.BoundedKernelDensity
   :members:

.. autoclass:: kern.MultivariateKernelDensity
   :members:

.. autoclass:: kern.BandwidthSelector
   :members:

Bandwidth grids
---------------

.. autofunction:: kern.default_bandwidth_grid
