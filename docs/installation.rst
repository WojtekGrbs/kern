Installation
============

Install a prebuilt wheel from PyPI:

.. code-block:: console

   python -m pip install kern_kde

When a wheel is not available, pip builds the C extension from source. The
build detects CBLAS and OpenMP by compiling and linking small test programs.
Both accelerators are optional: the extension uses scalar and serial fallbacks
when they are unavailable.

Build options
-------------

``KERN_USE_BLAS`` and ``KERN_USE_OPENMP`` accept ``auto`` (the default),
``required``/``1``, or ``disabled``/``0``. For a CBLAS installation in a
non-standard location, provide one or more of:

* ``KERN_BLAS_INCLUDE_DIRS``
* ``KERN_BLAS_LIBRARY_DIRS``
* ``KERN_BLAS_LIBRARIES``
* ``KERN_BLAS_COMPILE_ARGS``
* ``KERN_BLAS_LINK_ARGS``

For example:

.. code-block:: console

   KERN_USE_BLAS=required \
   KERN_BLAS_INCLUDE_DIRS=/opt/openblas/include \
   KERN_BLAS_LIBRARY_DIRS=/opt/openblas/lib \
   KERN_BLAS_LIBRARIES=openblas \
   python -m pip install --no-binary=kern-kde kern_kde

On macOS, Accelerate is detected as the preferred CBLAS implementation.
OpenMP uses Homebrew ``libomp`` when it is installed. To point to a different
macOS OpenMP installation, set ``KERN_LIBOMP_PREFIX``.
