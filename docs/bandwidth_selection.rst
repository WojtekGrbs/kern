Bandwidth Selection
===================

Bandwidth selection maximizes leave-one-out or k-fold log-likelihood.

Default grid
------------

When ``grid=None``, :class:`kern.BandwidthSelector` calls
:func:`kern.default_bandwidth_grid`. The default is a 32-point logarithmic grid
centered on a robust rule-of-thumb bandwidth.

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

Parallel loop
-------------

``parallel="grid"`` evaluates bandwidth candidates concurrently. Each
individual LOO or k-fold evaluation is serial.

``parallel="evaluation"`` walks the grid serially and parallelizes each
individual LOO or k-fold KDE evaluation.

``parallel="auto"`` uses evaluation parallelism for a single bandwidth and
for small grids over large datasets. It otherwise parallelizes the grid.
