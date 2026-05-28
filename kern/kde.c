#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

#include "kernels.h"

static PyObject* py_kde(PyObject* self, PyObject* args) {
    PyObject* input_obj;

    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }

    PyArrayObject* input = (PyArrayObject*)PyArray_FROM_OTF(
        input_obj,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY
    );

    if (input == NULL) {
        return NULL;
    }

    int ndim = PyArray_NDIM(input);
    npy_intp* dims = PyArray_DIMS(input);

    PyArrayObject* output = (PyArrayObject*)PyArray_SimpleNew(
        ndim,
        dims,
        NPY_DOUBLE
    );

    if (output == NULL) {
        Py_DECREF(input);
        return NULL;
    }

    npy_intp size = PyArray_SIZE(input);

    double* in_data = (double*)PyArray_DATA(input);
    double* out_data = (double*)PyArray_DATA(output);

    standard_kde(in_data, size, 0.5, out_data);
    
    Py_DECREF(input);

    return (PyObject*)output;
}

static PyMethodDef MyMethods[] = {
    {
        "kde",
        py_kde,
        METH_VARARGS,
        "1d KDE"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "First KDE extension",
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&mymodule);
}