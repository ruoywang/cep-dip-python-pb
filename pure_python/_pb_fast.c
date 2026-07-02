#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static PyObject *local_field_factor_fixed(PyObject *self, PyObject *args) {
    PyObject *e_obj = NULL;
    double alpha_pol, alpha0_rot, invalpha_sic, pbeta;
    if (!PyArg_ParseTuple(args, "Odddd", &e_obj, &alpha_pol, &alpha0_rot, &invalpha_sic, &pbeta)) {
        return NULL;
    }

    PyArrayObject *e_arr = (PyArrayObject *)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (e_arr == NULL) {
        return NULL;
    }

    npy_intp ndim = PyArray_NDIM(e_arr);
    npy_intp *dims = PyArray_DIMS(e_arr);
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    if (out_arr == NULL) {
        Py_DECREF(e_arr);
        return NULL;
    }

    const npy_intp n = PyArray_SIZE(e_arr);
    const double *e = (const double *)PyArray_DATA(e_arr);
    double *out = (double *)PyArray_DATA(out_arr);

    const double lo = 1.0 / (1.0 - alpha_pol * invalpha_sic);
    double hi = 1.0 / (1.0 - (alpha_pol + alpha0_rot) * invalpha_sic);
    hi *= 1.0 + 1.0e-8;
    const double tol = 1.0e-10 * fmax(1.0, hi);

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        out[i] = hi;
    }

    for (int iter = 0; iter < 80; ++iter) {
        double diff = 0.0;
        #pragma omp parallel for reduction(max:diff) schedule(static)
        for (npy_intp i = 0; i < n; ++i) {
            const double x = out[i] * pbeta * e[i];
            double grot;
            if (x < 2.0e-4) {
                grot = 1.0;
            } else {
                const double tx = tanh(x);
                grot = 3.0 * (x - tx) / (x * x * tx);
            }
            double val = 1.0 / (1.0 - (grot * alpha0_rot + alpha_pol) * invalpha_sic);
            if (val < lo) {
                val = lo;
            } else if (val > hi) {
                val = hi;
            }
            const double d = fabs(val - out[i]);
            if (d > diff) {
                diff = d;
            }
            out[i] = val;
        }
        if (diff <= tol) {
            break;
        }
    }

    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        if (pbeta * e[i] == 0.0) {
            out[i] = hi;
        }
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(e_arr);
    return (PyObject *)out_arr;
}

static PyMethodDef Methods[] = {
    {"local_field_factor_fixed", local_field_factor_fixed, METH_VARARGS, "OpenMP local field fixed-point solver."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_pb_fast",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__pb_fast(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
