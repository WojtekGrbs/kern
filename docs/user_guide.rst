User Guide
==========

Exact one-dimensional KDE
-------------------------

.. literalinclude:: ../examples/basic.py
   :language: python

Approximate KDE
---------------

:class:`kern.ApproximateKernelDensity` sorts training samples once during
``fit``. A distance cutoff and optional neighbor limit reduce the number of
kernel evaluations.

``memory="high"`` uses per-thread partial sums for symmetric self-KDE and can
cache external cutoff bounds. ``memory="low"`` avoids those buffers.
``memory="auto"`` chooses based on the workload.

Bounded KDE
-----------

:class:`kern.BoundedKernelDensity` supports reflected standard kernels and the
Beta kernel for samples on ``[0, 1]``.

Multivariate KDE
----------------

:class:`kern.MultivariateKernelDensity` uses blocked product-kernel evaluation.
The ``block_size`` parameter controls how many query rows reuse each data
block.
