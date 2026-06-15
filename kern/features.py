"""Public build-feature helpers."""

from . import _core


def has_openmp():
    """Return whether the current build uses OpenMP."""
    return bool(_core.has_openmp())


def has_blas():
    """Return whether the current build uses BLAS."""
    return bool(_core.has_blas())


__all__ = ["has_blas", "has_openmp"]
