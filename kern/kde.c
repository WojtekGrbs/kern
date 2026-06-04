#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#if defined(_OPENMP)
  #include <omp.h>
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#define MATHCONST_1_SQRT2PI  0.3989422804014327   /* 1 / sqrt(2*pi) */
#define MATHCONST_PI_4     0.7853981633974483   /* pi/4            */

/* Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-0.5*u^2)  */
#if defined(_OPENMP)
#pragma omp declare simd
#endif
static inline double kernel_gaussian(double u) {
    return MATHCONST_1_SQRT2PI * exp(-0.5 * u * u);
}

/* Epanechnikov kernel: K(u) = 0.75*(1-u^2) for |u|<=1, else 0 */
static inline double kernel_epanechnikov(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return 0.75 * (1.0 - u * u);
}

/* Triangular kernel: K(u) = (1-|u|) for |u|<=1, else 0 */
static inline double kernel_triangular(double u) {
    double au = fabs(u);
    if (au >= 1.0) return 0.0;
    return 1.0 - au;
}

/* Uniform kernel: K(u) = 0.5 for |u|<=1, else 0 */
static inline double kernel_uniform(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return 0.5;
}

/* Cosine kernel: K(u) = (pi/4)*cos(pi/2 * u) for |u|<=1, else 0 */
static inline double kernel_cosine(double u) {
    if (u < -1.0 || u > 1.0) return 0.0;
    return MATHCONST_PI_4 * cos(MATHCONST_PI_4 * 2.0 * u);
}


//
//


/* lgamma-based log-beta for numerical stability */
static inline double log_beta(double a, double b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/* Beta kernel
 * log-space to avoid overflow, lgamma calls are the cost. */
static inline double kernel_beta(double x, double xi, double h) {
    if (x <= 0.0 || x >= 1.0) return 0.0;
    double a = xi / h + 1.0;
    double b = (1.0 - xi) / h + 1.0;
    double log_k = (a - 1.0) * log(x) + (b - 1.0) * log(1.0 - x) - log_beta(a, b);
    return exp(log_k);
}


    /* self-KDE with Gaussian kernel
    * out[i] = (1/(n-1)*h) * sum_{j!=i} K_gauss((xi - xj)/h) */
static void self_kde_gaussian(const double * restrict data, double * restrict out,
                               int n, double h) {
    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        double xi = data[i];
        double sum = 0.0;

        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < i; j++) {
            double u = (xi - data[j]) * inv_h;
            sum += kernel_gaussian(u);
        }

        #pragma omp simd reduction(+:sum)
        for (int j = i + 1; j < n; j++) {
            double u = (xi - data[j]) * inv_h;
            sum += kernel_gaussian(u);
        }

        out[i] = sum * inv_nh;
    }
}


typedef double (*kernel1d_fn)(double);

static void _self_kde(const double * restrict data, double * restrict out,
                              int n, double h, kernel1d_fn kfn) {
    double inv_nh = 1.0 / ((n - 1) * h);
    double inv_h  = 1.0 / h;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        double xi = data[i];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double diff = xi - data[j];
                sum += kfn(diff * inv_h);
            }
        }
        out[i] = sum * inv_nh;
    }
}


static void ext_kde_gaussian(const double * restrict data, int n,
                              const double * restrict xs, double * restrict out,
                              int m, double h) {
    double inv_nh = 1.0 / (n * h);
    double inv_h  = 1.0 / h;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        double xi = xs[i];
        double sum = 0.0;

        /* OpenMP SIMD replaces the handwritten AVX2 intrinsic loop. */
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; j++) {
            double u = (xi - data[j]) * inv_h;
            sum += kernel_gaussian(u);
        }

        out[i] = sum * inv_nh;
    }
}

static void _ext_kde(const double * restrict data, int n,
                             const double * restrict xs, double * restrict out,
                             int m, double h, kernel1d_fn kfn) {
    double inv_nh = 1.0 / (n * h);
    double inv_h  = 1.0 / h;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        double xi  = xs[i];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; j++) {
            double diff = xi - data[j];
            sum += kfn(diff * inv_h);
        }
        out[i] = sum * inv_nh;
    }
}


static kernel1d_fn get_kernel_fn(const char *name) {
    if (strcmp(name, "gaussian")     == 0) return kernel_gaussian;
    if (strcmp(name, "epanechnikov") == 0) return kernel_epanechnikov;
    if (strcmp(name, "triangular")   == 0) return kernel_triangular;
    if (strcmp(name, "uniform")      == 0) return kernel_uniform;
    if (strcmp(name, "cosine")       == 0) return kernel_cosine;
    return NULL;
}

static PyObject *py_kde_infinite_self(PyObject *self, PyObject *args) {
    PyArrayObject *data_arr;
    double h;
    const char *kernel_name;

    if (!PyArg_ParseTuple(args, "O!ds", &PyArray_Type, &data_arr, &h, &kernel_name))
        return NULL;

    if (PyArray_NDIM(data_arr) != 1 || PyArray_TYPE(data_arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "data must be a 1D float64 array");
        return NULL;
    }

    int n = (int)PyArray_SIZE(data_arr);
    double *data = (double *)PyArray_DATA(data_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!out_arr) return NULL;
    double *out = (double *)PyArray_DATA(out_arr);

    if (strcmp(kernel_name, "gaussian") == 0) {
        self_kde_gaussian(data, out, n, h);
    } else {
        kernel1d_fn kfn = get_kernel_fn(kernel_name);
        if (!kfn) { 
            PyErr_SetString(PyExc_ValueError, "Unknown kernel name");
            Py_DECREF(out_arr); return NULL;
        }
        _self_kde(data, out, n, h, kfn);
    }
    return (PyObject *)out_arr;
}

static PyObject *py_kde_infinite_ext(PyObject *self, PyObject *args) {
    PyArrayObject *data_arr, *xs_arr;
    double h;
    const char *kernel_name;

    if (!PyArg_ParseTuple(args, "O!O!ds", &PyArray_Type, &data_arr,
                           &PyArray_Type, &xs_arr, &h, &kernel_name))
        return NULL;

    if (PyArray_TYPE(data_arr) != NPY_DOUBLE || PyArray_TYPE(xs_arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Arrays must be float64");
        return NULL;
    }

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
        kernel1d_fn kfn = get_kernel_fn(kernel_name);
        if (!kfn) {
            PyErr_SetString(PyExc_ValueError, "Unknown kernel name");
            Py_DECREF(out_arr); return NULL;
        }
        _ext_kde(data, n, xs, out, m, h, kfn);
    }
    return (PyObject *)out_arr;
}

static PyMethodDef KernMethods[] = {

    // infinite - support kernels
    {"kde_self",     py_kde_infinite_self,     METH_VARARGS,
     "Self-KDE with infinite-support kernel (LOO). Args: data, h, kernel_name."},
    {"kde",      py_kde_infinite_ext,      METH_VARARGS,
     "External KDE with infinite-support kernel. Args: data, xs, h, kernel_name."},
    
    // [0,1] - support kernels (and methods)

    // Bandwidth optimization

    // optimization? precision performance tradeoff? torch?

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kern_module = {
    PyModuleDef_HEAD_INIT, "_core", NULL, -1, KernMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&kern_module);
}