#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <string.h>

#include "bandwidth.h"
#include "kde.h"
#include "kernels.h"

static int require_float64_1d(PyArrayObject *arr, const char *name) {
    if (PyArray_NDIM(arr) != 1 || PyArray_TYPE(arr) != NPY_DOUBLE) {
        PyErr_Format(PyExc_ValueError, "%s must be a 1D float64 array", name);
        return 0;
    }
    return 1;
}

static int metric_from_name(const char *metric) {
    return (strcmp(metric, "ise") == 0) ? 1 : 0;
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

static PyObject *py_kde_infinite_self(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    double h;
    const char *kernel_name;

    if (!PyArg_ParseTuple(args, "O!ds", &PyArray_Type, &data_arr, &h, &kernel_name)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data")) {
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
    if (!require_float64_1d(data_arr, "data") || !require_float64_1d(xs_arr, "xs")) {
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
    if (!require_float64_1d(data_arr, "data")) {
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
    if (!require_float64_1d(data_arr, "data") || !require_float64_1d(xs_arr, "xs")) {
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
    if (!require_float64_1d(data_arr, "data")) {
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
    if (!require_float64_1d(data_arr, "data") || !require_float64_1d(xs_arr, "xs")) {
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

    if (!PyArg_ParseTuple(args, "O!O!s", &PyArray_Type, &data_arr,
                          &PyArray_Type, &h_arr, &metric)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") || !require_float64_1d(h_arr, "h_grid")) {
        return NULL;
    }

    int n = (int)PyArray_SIZE(data_arr);
    int nh = (int)PyArray_SIZE(h_arr);
    double *data  = (double *)PyArray_DATA(data_arr);
    double *hgrid = (double *)PyArray_DATA(h_arr);
    int metric_id = metric_from_name(metric);

    npy_intp dims[1] = {nh};
    PyArrayObject *scores_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!scores_arr) return NULL;

    double *scores = (double *)PyArray_DATA(scores_arr);
    for (int hi = 0; hi < nh; hi++) {
        scores[hi] = loo_score(data, n, hgrid[hi], metric_id);
    }

    return (PyObject *)scores_arr;
}

static PyObject *py_bandwidth_kfold(PyObject *self, PyObject *args) {
    (void)self;

    PyArrayObject *data_arr;
    PyArrayObject *h_arr;
    int k_folds;
    const char *metric;

    if (!PyArg_ParseTuple(args, "O!O!is", &PyArray_Type, &data_arr,
                          &PyArray_Type, &h_arr, &k_folds, &metric)) {
        return NULL;
    }
    if (!require_float64_1d(data_arr, "data") || !require_float64_1d(h_arr, "h_grid")) {
        return NULL;
    }

    int n = (int)PyArray_SIZE(data_arr);
    int nh = (int)PyArray_SIZE(h_arr);
    double *data  = (double *)PyArray_DATA(data_arr);
    double *hgrid = (double *)PyArray_DATA(h_arr);
    int metric_id = metric_from_name(metric);

    npy_intp dims[1] = {nh};
    PyArrayObject *scores_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!scores_arr) return NULL;

    double *scores = (double *)PyArray_DATA(scores_arr);
    for (int hi = 0; hi < nh; hi++) {
        scores[hi] = kfold_score(data, n, k_folds, hgrid[hi], metric_id);
    }

    return (PyObject *)scores_arr;
}

static PyMethodDef KernMethods[] = {
    /* Kernel metadata */
    {"kernel_is_symmetric", py_kernel_is_symmetric, METH_VARARGS,
     "Return whether a named kernel is symmetric. Args: kernel_name."},

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
