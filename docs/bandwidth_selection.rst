Bandwidth Selection
===================

Bandwidth selection maximizes leave-one-out or k-fold log-likelihood.

Default grid
------------

When ``grid=None``, :class:`kern.BandwidthSelector` calls
:func:`kern.default_bandwidth_grid`. The default is a 64-point logarithmic grid
centered on Silverman's rule-of-thumb bandwidth
:ref:`[Silverman 1986] <reference-silverman>`.

.. literalinclude:: ../examples/bandwidth_selection.py
   :language: python

Custom grid
-----------

Pass any positive one-dimensional array as ``grid``:

.. code-block:: python

   selector = BandwidthSelector(
       grid=[0.05, 0.1, 0.2, 0.4],
       kernel="cosine",
       cv=5,
   ).fit(data)

Bandwidth selection always uses exactly one kernel. Mappings are not accepted
for ``grid`` or ``kernel``.

Bounded KDE
-----------

Pass ``bounded=True`` to select a bandwidth for
:class:`kern.BoundedKernelDensity`. ``bounded_method`` accepts the same values
as :class:`kern.BoundedKernelDensity`: ``"reflected"``, ``"beta"``, or
``None`` for regular unbounded KDE behavior.

.. code-block:: python

   selector = BandwidthSelector(
       grid=[0.05, 0.1, 0.2],
       bounded=True,
       bounded_method="beta",
   ).fit(data)

Parallel loop
-------------

``parallel="grid"`` evaluates bandwidth candidates concurrently. Each
individual LOO or k-fold evaluation is serial.

``parallel="evaluation"`` walks the grid serially and parallelizes each
individual LOO or k-fold KDE evaluation.

``parallel="auto"`` uses evaluation parallelism for a single bandwidth and
for small grids over large datasets. It otherwise parallelizes the grid.

Bounded KDE bandwidth selection uses the bounded density functions directly
and does not use the C-level parallel scorer.
