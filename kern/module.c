#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bandwidth.h"
#include "kde.h"
#include "kde_approx.h"
#include "kde_multivariate.h"
#include "kernels.h"

static int require_float64_1d(PyArrayObject *arr, const char *name) {
    if (PyArray_NDIM(arr) != 1 || PyArray_TYPE(arr) != NPY_DOUBLE ||
        !PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a C-contiguous 1D float64 array", name);
        return 0;
    }
    if (PyArray_SIZE(arr) > INT_MAX) {
        PyErr_Format(PyExc_ValueError, "%s is too large", name);
        return 0;
    }
    return 1;
}

static int require_float64_2d(PyArrayObject *arr, const char *name) {
    if (PyArray_NDIM(arr) != 2 || PyArray_TYPE(arr) != NPY_DOUBLE ||
        !PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a C-contiguous 2D float64 array", name);
        return 0;
    }
    if (PyArray_DIM(arr, 0) > INT_MAX || PyArray_DIM(arr, 1) > INT_MAX) {
        PyErr_Format(PyExc_ValueError, "%s is too large", name);
        return 0;
    }
    return 1;
}

static int require_nonempty(PyArrayObject *arr, const char *name) {
    if (PyArray_DIM(arr, 0) == 0) {
        PyErr_Format(PyExc_ValueError, "%s must not be empty", name);
        return 0;
    }
    return 1;
}

static int require_bandwidth(double h) {
    if (!isfinite(h) || h <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "bandwidth must be positive");
        return 0;
    }
    return 1;
}

static int require_bandwidth_grid(PyArrayObject *arr) {
    if (!require_nonempty(arr, "h_grid")) return 0;

    double *values = (double *)PyArray_DATA(arr);
    int n = (int)PyArray_SIZE(arr);
    for (int i = 0; i < n; i++) {
        if (!isfinite(values[i]) || values[i] <= 0.0) {
            PyErr_SetString(PyExc_ValueError,
                            "all bandwidths must be positive");
            return 0;
        }
    }
    return 1;
}

static int require_sorted(PyArrayObject *arr, const char *name) {
    double *values = (double *)PyArray_DATA(arr);
    int n = (int)PyArray_SIZE(arr);
    for (int i = 1; i < n; i++) {
        if (values[i] < values[i - 1]) {
            PyErr_Format(PyExc_ValueError, "%s must be sorted", name);
            return 0;
        }
    }
    return 1;
}

static int metric_from_name(const char *metric) {
    if (strcmp(metric, "loglik") == 0) return BANDWIDTH_LOG_LIKELIHOOD;
    if (strcmp(metric, "ise") == 0) return BANDWIDTH_ISE;
    PyErr_SetString(PyExc_ValueError, "metric must be 'loglik' or 'ise'");
    return -1;
}

static int parallel_from_name(const char *parallel) {
    if (strcmp(parallel, "auto") == 0) return BANDWIDTH_PARALLEL_AUTO;
    if (strcmp(parallel, "bandwidths") == 0) {
        return BANDWIDTH_PARALLEL_BANDWIDTHS;
    }
    if (strcmp(parallel, "kernels") == 0) return BANDWIDTH_PARALLEL_KERNELS;
    PyErr_SetString(PyExc_ValueError,
                    "parallel must be 'auto', 'bandwidths', or 'kernels'");
    return -1;
}

static const kernel1d_info *require_kernel(const char *kernel_name) {
    const kernel1d_info *kernel = get_kernel(kernel_name);
    if (!kernel) {
        PyErr_SetString(PyExc_ValueError, "Unknown kernel name");
        return NULL;
    }
    return kernel;
}

static const kernel1d_info *require_symmetric_kernel(const char *kernel_name) {
    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    if (!kernel->is_symmetric) {
        PyErr_SetString(PyExc_ValueError, "Reflected KDE requires a symmetric kernel");
        return NULL;
    }
    return kernel;
}

static PyObject *py_kernel_is_symmetric(PyObject *self, PyObject *args) {
    (void)self;

    const char *kernel_name;
    if (!PyArg_ParseTuple(args, "s", &kernel_name)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    if (kernel->is_symmetric) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *py_kernel_default_cutoff(PyObject *self, PyObject *args) {
    (void)self;

    const char *kernel_name;
    if (!PyArg_ParseTuple(args, "s", &kernel_name)) return NULL;

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;
    return PyFloat_FromDouble(kernel->default_cutoff);
}

static PyObject *py_kde_infinite_self(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    double h;
    const char *kernel_name;

    if (!PyArg_ParseTuple(args, "O!ds", &PyArray_Type, &data_arr, &h, &kernel_name)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    double *data = (double *)PyArray_DATA(data_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    double *out = (double *)PyArray_DATA(out_arr);

    if (strcmp(kernel_name, "gaussian") == 0) {
        self_kde_gaussian(data, out, n, h);
    } else {
        self_kde_generic(data, out, n, h, kernel);
    }

    return (PyObject *)out_arr;
}

static PyObject *py_kde_infinite_ext(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *xs_arr;
    double h;
    const char *kernel_name;

    if (!PyArg_ParseTuple(args, "O!O!ds", &PyArray_Type, &data_arr,
                          &PyArray_Type, &xs_arr, &h, &kernel_name)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_float64_1d(xs_arr, "xs") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    int m = (int)PyArray_SIZE(xs_arr);
    double *data = (double *)PyArray_DATA(data_arr);
    double *xs   = (double *)PyArray_DATA(xs_arr);

    npy_intp dims[1] = {m};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    double *out = (double *)PyArray_DATA(out_arr);

    if (strcmp(kernel_name, "gaussian") == 0) {
        ext_kde_gaussian(data, n, xs, out, m, h);
    } else {
        ext_kde_generic(data, n, xs, out, m, h, kernel);
    }

    return (PyObject *)out_arr;
}

static PyObject *py_kde_beta_self(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    double h;

    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &data_arr, &h)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }

    int n = (int)PyArray_SIZE(data_arr);
    double *data = (double *)PyArray_DATA(data_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    self_kde_beta(data, (double *)PyArray_DATA(out_arr), n, h);
    return (PyObject *)out_arr;
}

static PyObject *py_kde_beta_ext(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *xs_arr;
    double h;

    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &data_arr,
                          &PyArray_Type, &xs_arr, &h)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_float64_1d(xs_arr, "xs") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }

    int n = (int)PyArray_SIZE(data_arr);
    int m = (int)PyArray_SIZE(xs_arr);

    npy_intp dims[1] = {m};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    ext_kde_beta((double *)PyArray_DATA(data_arr), n,
                 (double *)PyArray_DATA(xs_arr),
                 (double *)PyArray_DATA(out_arr), m, h);

    return (PyObject *)out_arr;
}

static PyObject *py_kde_reflected_self(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    double h;
    const char *kernel_name;

    if (!PyArg_ParseTuple(args, "O!ds", &PyArray_Type, &data_arr, &h, &kernel_name)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_symmetric_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    npy_intp dims[1] = {n};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    self_kde_reflected((double *)PyArray_DATA(data_arr),
                       (double *)PyArray_DATA(out_arr), n, h, kernel);

    return (PyObject *)out_arr;
}

static PyObject *py_kde_reflected_ext(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *xs_arr;
    double h;
    const char *kernel_name;

    if (!PyArg_ParseTuple(args, "O!O!ds", &PyArray_Type, &data_arr,
                          &PyArray_Type, &xs_arr, &h, &kernel_name)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_float64_1d(xs_arr, "xs") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_symmetric_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    int m = (int)PyArray_SIZE(xs_arr);
    npy_intp dims[1] = {m};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    ext_kde_reflected((double *)PyArray_DATA(data_arr), n,
                      (double *)PyArray_DATA(xs_arr),
                      (double *)PyArray_DATA(out_arr), m, h, kernel);

    return (PyObject *)out_arr;
}

static PyObject *py_bandwidth_loo(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *h_arr;
    const char *metric;
    const char *kernel_name = "gaussian";
    const char *parallel = "auto";

    if (!PyArg_ParseTuple(args, "O!O!s|ss", &PyArray_Type, &data_arr,
                          &PyArray_Type, &h_arr, &metric,
                          &kernel_name, &parallel)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_float64_1d(h_arr, "h_grid") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth_grid(h_arr)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    int nh = (int)PyArray_SIZE(h_arr);
    double *data  = (double *)PyArray_DATA(data_arr);
    double *hgrid = (double *)PyArray_DATA(h_arr);
    int metric_id = metric_from_name(metric);
    int parallel_id = parallel_from_name(parallel);
    if (metric_id < 0 || parallel_id < 0) return NULL;

    npy_intp dims[1] = {nh};
    PyArrayObject *scores_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!scores_arr) return NULL;

    double *scores = (double *)PyArray_DATA(scores_arr);
    Py_BEGIN_ALLOW_THREADS
    bandwidth_score_grid(data, n, hgrid, nh, &kernel, 1, 1, metric_id,
                         parallel_id, scores);
    Py_END_ALLOW_THREADS

    return (PyObject *)scores_arr;
}

static PyObject *py_bandwidth_kfold(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *h_arr;
    int k_folds;
    const char *metric;
    const char *kernel_name = "gaussian";
    const char *parallel = "auto";

    if (!PyArg_ParseTuple(args, "O!O!is|ss", &PyArray_Type, &data_arr,
                          &PyArray_Type, &h_arr, &k_folds, &metric,
                          &kernel_name, &parallel)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_float64_1d(h_arr, "h_grid") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth_grid(h_arr)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    if (k_folds < 2 || k_folds > n) {
        PyErr_SetString(PyExc_ValueError, "k_folds must be between 2 and n");
        return NULL;
    }

    int nh = (int)PyArray_SIZE(h_arr);
    double *data  = (double *)PyArray_DATA(data_arr);
    double *hgrid = (double *)PyArray_DATA(h_arr);
    int metric_id = metric_from_name(metric);
    int parallel_id = parallel_from_name(parallel);
    if (metric_id < 0 || parallel_id < 0) return NULL;

    npy_intp dims[1] = {nh};
    PyArrayObject *scores_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!scores_arr) return NULL;

    double *scores = (double *)PyArray_DATA(scores_arr);
    Py_BEGIN_ALLOW_THREADS
    bandwidth_score_grid(data, n, hgrid, nh, &kernel, 1, k_folds, metric_id,
                         parallel_id, scores);
    Py_END_ALLOW_THREADS

    return (PyObject *)scores_arr;
}

static PyObject *py_bandwidth_grid(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *h_arr;
    PyObject *kernel_names;
    int k_folds;
    const char *metric;
    const char *parallel;

    if (!PyArg_ParseTuple(args, "O!O!Oiss", &PyArray_Type, &data_arr,
                          &PyArray_Type, &h_arr, &kernel_names,
                          &k_folds, &metric, &parallel)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") ||
        !require_float64_1d(h_arr, "h_grid") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth_grid(h_arr)) {
        return NULL;
    }

    int n = (int)PyArray_SIZE(data_arr);
    if (k_folds != 1 && (k_folds < 2 || k_folds > n)) {
        PyErr_SetString(PyExc_ValueError,
                        "k_folds must be 1 for LOO or between 2 and n");
        return NULL;
    }

    int metric_id = metric_from_name(metric);
    int parallel_id = parallel_from_name(parallel);
    if (metric_id < 0 || parallel_id < 0) return NULL;

    PyObject *names = PySequence_Fast(kernel_names,
                                      "kernels must be a sequence of names");
    if (!names) return NULL;

    Py_ssize_t nk_size = PySequence_Fast_GET_SIZE(names);
    if (nk_size < 1 || nk_size > INT_MAX) {
        Py_DECREF(names);
        PyErr_SetString(PyExc_ValueError, "kernels must not be empty");
        return NULL;
    }

    int nk = (int)nk_size;
    const kernel1d_info **kernels =
        (const kernel1d_info **)malloc((size_t)nk * sizeof(*kernels));
    if (!kernels) {
        Py_DECREF(names);
        return PyErr_NoMemory();
    }

    for (int i = 0; i < nk; i++) {
        const char *name = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(names, i));
        if (!name) {
            free(kernels);
            Py_DECREF(names);
            return NULL;
        }
        kernels[i] = require_kernel(name);
        if (!kernels[i]) {
            free(kernels);
            Py_DECREF(names);
            return NULL;
        }
    }

    int nh = (int)PyArray_SIZE(h_arr);
    npy_intp dims[2] = {nk, nh};
    PyArrayObject *scores_arr =
        (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!scores_arr) {
        free(kernels);
        Py_DECREF(names);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    bandwidth_score_grid((double *)PyArray_DATA(data_arr), n,
                         (double *)PyArray_DATA(h_arr), nh, kernels, nk,
                         k_folds, metric_id, parallel_id,
                         (double *)PyArray_DATA(scores_arr));
    Py_END_ALLOW_THREADS

    free(kernels);
    Py_DECREF(names);
    return (PyObject *)scores_arr;
}

static PyObject *py_kde_approx_self(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    double h;
    const char *kernel_name;
    double cutoff;
    int max_neighbors;
    int fast_gaussian;
    int memory_mode = KDE_APPROX_MEMORY_AUTO;

    if (!PyArg_ParseTuple(args, "O!dsdip|i", &PyArray_Type, &data_arr, &h,
                          &kernel_name, &cutoff, &max_neighbors,
                          &fast_gaussian, &memory_mode)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "sorted_data") ||
        !require_nonempty(data_arr, "sorted_data") ||
        !require_sorted(data_arr, "sorted_data") || !require_bandwidth(h)) {
        return NULL;
    }
    if (isnan(cutoff) || cutoff < 0.0 || max_neighbors < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "cutoff and max_neighbors must be non-negative");
        return NULL;
    }
    if (memory_mode < KDE_APPROX_MEMORY_LOW ||
        memory_mode > KDE_APPROX_MEMORY_AUTO) {
        PyErr_SetString(PyExc_ValueError, "invalid approximation memory mode");
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    npy_intp dims[1] = {n};
    PyArrayObject *out_arr =
        (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    Py_BEGIN_ALLOW_THREADS
    approx_self_kde((double *)PyArray_DATA(data_arr),
                    (double *)PyArray_DATA(out_arr), n, h, kernel,
                    cutoff, max_neighbors, fast_gaussian, memory_mode);
    Py_END_ALLOW_THREADS
    return (PyObject *)out_arr;
}

static PyObject *py_kde_approx_ext(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *xs_arr;
    double h;
    const char *kernel_name;
    double cutoff;
    int max_neighbors;
    int fast_gaussian;
    int memory_mode = KDE_APPROX_MEMORY_AUTO;

    if (!PyArg_ParseTuple(args, "O!O!dsdip|i", &PyArray_Type, &data_arr,
                          &PyArray_Type, &xs_arr, &h, &kernel_name,
                          &cutoff, &max_neighbors, &fast_gaussian,
                          &memory_mode)) {
        return NULL;
    }
    if (memory_mode < KDE_APPROX_MEMORY_LOW ||
        memory_mode > KDE_APPROX_MEMORY_AUTO) {
        PyErr_SetString(PyExc_ValueError, "invalid approximation memory mode");
        return NULL;
    }
    if (!require_float64_1d(data_arr, "sorted_data") ||
        !require_float64_1d(xs_arr, "xs") ||
        !require_nonempty(data_arr, "sorted_data") ||
        !require_sorted(data_arr, "sorted_data") || !require_bandwidth(h)) {
        return NULL;
    }
    if (isnan(cutoff) || cutoff < 0.0 || max_neighbors < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "cutoff and max_neighbors must be non-negative");
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_SIZE(data_arr);
    int m = (int)PyArray_SIZE(xs_arr);
    npy_intp dims[1] = {m};
    PyArrayObject *out_arr =
        (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    Py_BEGIN_ALLOW_THREADS
    approx_ext_kde((double *)PyArray_DATA(data_arr), n,
                   (double *)PyArray_DATA(xs_arr),
                   (double *)PyArray_DATA(out_arr), m, h, kernel,
                   cutoff, max_neighbors, fast_gaussian, memory_mode);
    Py_END_ALLOW_THREADS
    return (PyObject *)out_arr;
}

static PyObject *py_kde_multivariate_self(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    double h;
    const char *kernel_name;
    int block_size = 32;

    if (!PyArg_ParseTuple(args, "O!ds|i", &PyArray_Type, &data_arr,
                          &h, &kernel_name, &block_size)) {
        return NULL;
    }
    if (!require_float64_2d(data_arr, "data") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_DIM(data_arr, 0);
    int dimensions = (int)PyArray_DIM(data_arr, 1);
    if (dimensions < 1) {
        PyErr_SetString(PyExc_ValueError, "data must have at least one feature");
        return NULL;
    }
    if (block_size < 1) {
        PyErr_SetString(PyExc_ValueError, "block_size must be positive");
        return NULL;
    }

    npy_intp dims[1] = {n};
    PyArrayObject *out_arr =
        (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    Py_BEGIN_ALLOW_THREADS
    multivariate_self_kde((double *)PyArray_DATA(data_arr),
                          (double *)PyArray_DATA(out_arr),
                          n, dimensions, h, kernel, block_size);
    Py_END_ALLOW_THREADS
    return (PyObject *)out_arr;
}

static PyObject *py_kde_multivariate_ext(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *xs_arr;
    double h;
    const char *kernel_name;
    int block_size = 32;

    if (!PyArg_ParseTuple(args, "O!O!ds|i", &PyArray_Type, &data_arr,
                          &PyArray_Type, &xs_arr, &h, &kernel_name,
                          &block_size)) {
        return NULL;
    }
    if (!require_float64_2d(data_arr, "data") ||
        !require_float64_2d(xs_arr, "xs") ||
        !require_nonempty(data_arr, "data") || !require_bandwidth(h)) {
        return NULL;
    }
    if (PyArray_DIM(data_arr, 1) != PyArray_DIM(xs_arr, 1)) {
        PyErr_SetString(PyExc_ValueError,
                        "data and xs must have the same number of features");
        return NULL;
    }

    const kernel1d_info *kernel = require_kernel(kernel_name);
    if (!kernel) return NULL;

    int n = (int)PyArray_DIM(data_arr, 0);
    int m = (int)PyArray_DIM(xs_arr, 0);
    int dimensions = (int)PyArray_DIM(data_arr, 1);
    if (dimensions < 1) {
        PyErr_SetString(PyExc_ValueError, "data must have at least one feature");
        return NULL;
    }
    if (block_size < 1) {
        PyErr_SetString(PyExc_ValueError, "block_size must be positive");
        return NULL;
    }

    npy_intp dims[1] = {m};
    PyArrayObject *out_arr =
        (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;

    Py_BEGIN_ALLOW_THREADS
    multivariate_kde((double *)PyArray_DATA(data_arr), n, dimensions,
                     (double *)PyArray_DATA(xs_arr),
                     (double *)PyArray_DATA(out_arr), m, h, kernel,
                     block_size);
    Py_END_ALLOW_THREADS
    return (PyObject *)out_arr;
}

static PyMethodDef KernMethods[] = {
    /* Kernel metadata */
    {"kernel_is_symmetric", py_kernel_is_symmetric, METH_VARARGS,
     "Return whether a named kernel is symmetric. Args: kernel_name."},
    {"kernel_default_cutoff", py_kernel_default_cutoff, METH_VARARGS,
     "Return the default approximation cutoff in bandwidth units."},

    /* Infinite-support / standard kernels */
    {"kde_self", py_kde_infinite_self, METH_VARARGS,
     "Self-KDE with kernel (LOO). Args: data, h, kernel_name."},
    {"kde", py_kde_infinite_ext, METH_VARARGS,
     "External KDE with kernel. Args: data, xs, h, kernel_name."},

    /* [0,1]-support Beta kernel */
    {"kde_beta_self", py_kde_beta_self, METH_VARARGS,
     "Self-KDE with Beta kernel (native [0,1] support). Args: data, h."},
    {"kde_beta_ext", py_kde_beta_ext, METH_VARARGS,
     "External KDE with Beta kernel. Args: data, xs, h."},

    /* Boundary-reflected KDE */
    {"kde_reflected_self", py_kde_reflected_self, METH_VARARGS,
     "Self-KDE with boundary reflection for [0,1]. Args: data, h, kernel_name."},
    {"kde_reflected_ext", py_kde_reflected_ext, METH_VARARGS,
     "External KDE with boundary reflection. Args: data, xs, h, kernel_name."},

    /* Bandwidth optimization */
    {"bandwidth_loo", py_bandwidth_loo, METH_VARARGS,
     "LOO-CV scores over h_grid. Args: data, h_grid, metric ('loglik'|'ise')."},
    {"bandwidth_kfold", py_bandwidth_kfold, METH_VARARGS,
     "K-Fold CV scores over h_grid. Args: data, h_grid, k_folds, metric."},
    {"bandwidth_grid", py_bandwidth_grid, METH_VARARGS,
     "Scores for each kernel and bandwidth with selectable parallel loop."},

    /* Approximate sorted 1D KDE */
    {"kde_approx_self", py_kde_approx_self, METH_VARARGS,
     "Approximate self-KDE for sorted data."},
    {"kde_approx", py_kde_approx_ext, METH_VARARGS,
     "Approximate external KDE for sorted data."},

    /* Multivariate product-kernel KDE */
    {"kde_multivariate_self", py_kde_multivariate_self, METH_VARARGS,
     "Multivariate self-KDE with a product kernel."},
    {"kde_multivariate", py_kde_multivariate_ext, METH_VARARGS,
     "Multivariate external KDE with a product kernel."},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kern_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    NULL,
    -1,
    KernMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&kern_module);
}
