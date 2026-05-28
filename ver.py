import kern
import numpy as np
from scipy.stats import gaussian_kde, norm
import time
import numpy as np
from scipy.stats import gaussian_kde
from math import sqrt, pi
np.random.seed(42)
CONST = 1 / sqrt(2 * pi)

def loo_kde_1d_fast(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    h = 0.5

    z = (x[:, None] - x[None, :]) / h
    K = np.exp(-0.5 * z*z) * CONST

    np.fill_diagonal(K, 0.0)

    return K.sum(axis=1) / ((n - 1) * h)

def loo_kde_1d_chunked(x, chunk=2000):
    x = np.asarray(x, dtype=float)
    n = x.size
    h = 0.5

    out = np.empty(n)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        z = (x[start:stop, None] - x[None, :]) / h
        K = np.exp(-0.5 * z*z) * CONST

        rows = np.arange(start, stop)
        K[np.arange(stop - start), rows] = 0.0

        out[start:stop] = K.sum(axis=1) / ((n - 1) * h)

    return out

x = np.random.random(20000)

print('-'*15)
t1 = time.perf_counter()
print(kern.kde(x)[:5])
t2 = time.perf_counter()
print(t2-t1)

print('-'*15)
t1 = time.perf_counter()
print(loo_kde_1d_fast(x)[:5])
t2 = time.perf_counter()
print(t2-t1)

print('-'*15)
t1 = time.perf_counter()
print(loo_kde_1d_chunked(x)[:5])
t2 = time.perf_counter()
print(t2-t1)
print('-'*15)

