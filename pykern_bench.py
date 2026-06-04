import numpy as np
from sklearn.neighbors import KDTree

SQRT_2PI_INV = 0.3989422804014327


def _col(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def kde(data, xs, h, leaf_size=64):
    data = _col(data)
    xs = _col(xs)

    tree = KDTree(data, leaf_size=leaf_size)
    return tree.kernel_density(
        xs,
        h=h,
        kernel="gaussian",
        return_log=False,
    ) / len(data)


def kde_self(data, h, leaf_size=64):
    data = _col(data)
    n = len(data)

    tree = KDTree(data, leaf_size=leaf_size)

    summed = tree.kernel_density(
        data,
        h=h,
        kernel="gaussian",
        return_log=False,
    )
    self_contrib = SQRT_2PI_INV / h
    return (summed - self_contrib) / (n-1)


def kde_reflected(data, xs, h, *, leaf_size=64):
    data_1d = np.asarray(data, dtype=np.float64)
    xs = _col(xs)
    n = len(data_1d)

    reflected = np.concatenate([
        data_1d,
        -data_1d,
        2.0 - data_1d,
    ])

    tree = KDTree(_col(reflected), leaf_size=leaf_size)
    return tree.kernel_density(
        xs,
        h=h,
        kernel="gaussian",
        return_log=False,
    ) / n


def kde_reflected_self(data, h, leaf_size=64):
    data_1d = np.asarray(data, dtype=np.float64)
    data_col = _col(data_1d)
    n = len(data_1d)

    reflected = np.concatenate([
        data_1d,
        -data_1d,
        2.0 - data_1d,
    ])

    tree = KDTree(_col(reflected), leaf_size=leaf_size)

    summed = tree.kernel_density(
        data_col,
        h=h,
        kernel="gaussian",
        return_log=False,
    )
    self_reflected_contrib = (
        SQRT_2PI_INV / h
        + SQRT_2PI_INV * np.exp(-0.5 * (2 * data_1d / h) ** 2) / h
        + SQRT_2PI_INV * np.exp(-0.5 * ((2* data_1d - 2) / h) ** 2) / h
    )

    return (summed - self_reflected_contrib) / (n-1)