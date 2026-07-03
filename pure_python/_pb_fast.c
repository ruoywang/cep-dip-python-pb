#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* All kernels assume C-contiguous, aligned float64 / complex128 arrays of a
 * common length. They fuse the elementwise passes that dominate the PB solve
 * so each pass streams every operand exactly once, in parallel. */

#define TPI (2.0 * M_PI)

static int check_carray(PyArrayObject *a, int typenum, const char *name) {
    if (!PyArray_ISCARRAY_RO(a) || PyArray_TYPE(a) != typenum) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous of the expected dtype", name);
        return 0;
    }
    return 1;
}

static int check_wcarray(PyArrayObject *a, int typenum, const char *name) {
    if (!PyArray_ISCARRAY(a) || PyArray_TYPE(a) != typenum) {
        PyErr_Format(PyExc_ValueError, "%s must be writeable C-contiguous of the expected dtype", name);
        return 0;
    }
    return 1;
}

/* ---------------- original local-field fixed point ---------------- */

static PyObject *local_field_factor_fixed(PyObject *self, PyObject *args) {
    PyObject *e_obj = NULL;
    double alpha_pol, alpha0_rot, invalpha_sic, pbeta;
    if (!PyArg_ParseTuple(args, "Odddd", &e_obj, &alpha_pol, &alpha0_rot, &invalpha_sic, &pbeta)) {
        return NULL;
    }
    PyArrayObject *e_arr = (PyArrayObject *)PyArray_FROM_OTF(e_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (e_arr == NULL) return NULL;
    npy_intp ndim = PyArray_NDIM(e_arr);
    npy_intp *dims = PyArray_DIMS(e_arr);
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    if (out_arr == NULL) { Py_DECREF(e_arr); return NULL; }
    const npy_intp n = PyArray_SIZE(e_arr);
    const double *e = (const double *)PyArray_DATA(e_arr);
    double *out = (double *)PyArray_DATA(out_arr);
    const double lo = 1.0 / (1.0 - alpha_pol * invalpha_sic);
    double hi = 1.0 / (1.0 - (alpha_pol + alpha0_rot) * invalpha_sic);
    hi *= 1.0 + 1.0e-8;
    const double tol = 1.0e-10 * fmax(1.0, hi);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) out[i] = hi;
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
            if (val < lo) val = lo;
            else if (val > hi) val = hi;
            const double d = fabs(val - out[i]);
            if (d > diff) diff = d;
            out[i] = val;
        }
        if (diff <= tol) break;
    }
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        if (pbeta * e[i] == 0.0) out[i] = hi;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(e_arr);
    return (PyObject *)out_arr;
}

/* ---------------- conj(a) * b for complex128 ---------------- */

static PyObject *conj_mul(PyObject *self, PyObject *args) {
    PyArrayObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b)) return NULL;
    if (!check_carray(a, NPY_CDOUBLE, "a") || !check_carray(b, NPY_CDOUBLE, "b")) return NULL;
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(a), PyArray_DIMS(a), NPY_CDOUBLE);
    if (out == NULL) return NULL;
    const npy_intp n = PyArray_SIZE(a);
    const double *pa = (const double *)PyArray_DATA(a);
    const double *pb = (const double *)PyArray_DATA(b);
    double *po = (double *)PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double ar = pa[2*i], ai = pa[2*i+1];
        const double br = pb[2*i], bi = pb[2*i+1];
        po[2*i]   = ar * br + ai * bi;
        po[2*i+1] = ar * bi - ai * br;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* ------------- (i*TPI*g) * f for three real g and one complex f ------------- */

static PyObject *grad_premul(PyObject *self, PyObject *args) {
    PyArrayObject *gx, *gy, *gz, *f;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &gx, &PyArray_Type, &gy,
                          &PyArray_Type, &gz, &PyArray_Type, &f)) return NULL;
    if (!check_carray(gx, NPY_DOUBLE, "gx") || !check_carray(gy, NPY_DOUBLE, "gy") ||
        !check_carray(gz, NPY_DOUBLE, "gz") || !check_carray(f, NPY_CDOUBLE, "f")) return NULL;
    const npy_intp n = PyArray_SIZE(f);
    PyArrayObject *ax = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(f), PyArray_DIMS(f), NPY_CDOUBLE);
    PyArrayObject *ay = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(f), PyArray_DIMS(f), NPY_CDOUBLE);
    PyArrayObject *az = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(f), PyArray_DIMS(f), NPY_CDOUBLE);
    if (!ax || !ay || !az) { Py_XDECREF(ax); Py_XDECREF(ay); Py_XDECREF(az); return NULL; }
    const double *px = (const double *)PyArray_DATA(gx);
    const double *py = (const double *)PyArray_DATA(gy);
    const double *pz = (const double *)PyArray_DATA(gz);
    const double *pf = (const double *)PyArray_DATA(f);
    double *ox = (double *)PyArray_DATA(ax);
    double *oy = (double *)PyArray_DATA(ay);
    double *oz = (double *)PyArray_DATA(az);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double fr = pf[2*i], fi = pf[2*i+1];
        const double sx = TPI * px[i], sy = TPI * py[i], sz = TPI * pz[i];
        ox[2*i] = -sx * fi; ox[2*i+1] = sx * fr;
        oy[2*i] = -sy * fi; oy[2*i+1] = sy * fr;
        oz[2*i] = -sz * fi; oz[2*i+1] = sz * fr;
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NNN", ax, ay, az);
}

/* ------------- |E| from three real components ------------- */

static PyObject *magnitude3(PyObject *self, PyObject *args) {
    PyArrayObject *ex, *ey, *ez;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &ex, &PyArray_Type, &ey, &PyArray_Type, &ez)) return NULL;
    if (!check_carray(ex, NPY_DOUBLE, "ex") || !check_carray(ey, NPY_DOUBLE, "ey") ||
        !check_carray(ez, NPY_DOUBLE, "ez")) return NULL;
    const npy_intp n = PyArray_SIZE(ex);
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ex), PyArray_DIMS(ex), NPY_DOUBLE);
    if (out == NULL) return NULL;
    const double *px = (const double *)PyArray_DATA(ex);
    const double *py = (const double *)PyArray_DATA(ey);
    const double *pz = (const double *)PyArray_DATA(ez);
    double *po = (double *)PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        po[i] = sqrt(px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i]);
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* ------------- in-place scale of 4 real arrays by a factor array ------------- */

static PyObject *scale4_inplace(PyObject *self, PyObject *args) {
    PyArrayObject *ex, *ey, *ez, *em, *f;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &ex, &PyArray_Type, &ey,
                          &PyArray_Type, &ez, &PyArray_Type, &em, &PyArray_Type, &f)) return NULL;
    if (!check_wcarray(ex, NPY_DOUBLE, "ex") || !check_wcarray(ey, NPY_DOUBLE, "ey") ||
        !check_wcarray(ez, NPY_DOUBLE, "ez") || !check_wcarray(em, NPY_DOUBLE, "emag") ||
        !check_carray(f, NPY_DOUBLE, "f")) return NULL;
    const npy_intp n = PyArray_SIZE(f);
    double *px = (double *)PyArray_DATA(ex);
    double *py = (double *)PyArray_DATA(ey);
    double *pz = (double *)PyArray_DATA(ez);
    double *pm = (double *)PyArray_DATA(em);
    const double *pf = (const double *)PyArray_DATA(f);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double s = pf[i];
        px[i] *= s; py[i] *= s; pz[i] *= s; pm[i] *= s;
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* ------------- tensor work: w_i = chi_perp*g_i + chi_factor*e_i*(e . grad) ------------- */

static PyObject *tensor_work(PyObject *self, PyObject *args) {
    PyArrayObject *cp, *cf, *ex, *ey, *ez, *g0, *g1, *g2;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!",
                          &PyArray_Type, &cp, &PyArray_Type, &cf,
                          &PyArray_Type, &ex, &PyArray_Type, &ey, &PyArray_Type, &ez,
                          &PyArray_Type, &g0, &PyArray_Type, &g1, &PyArray_Type, &g2)) return NULL;
    PyArrayObject *ins[8] = {cp, cf, ex, ey, ez, g0, g1, g2};
    const char *names[8] = {"chi_perp","chi_factor","ex","ey","ez","g0","g1","g2"};
    for (int k = 0; k < 8; ++k) if (!check_carray(ins[k], NPY_DOUBLE, names[k])) return NULL;
    const npy_intp n = PyArray_SIZE(ex);
    PyArrayObject *w0 = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ex), PyArray_DIMS(ex), NPY_DOUBLE);
    PyArrayObject *w1 = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ex), PyArray_DIMS(ex), NPY_DOUBLE);
    PyArrayObject *w2 = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ex), PyArray_DIMS(ex), NPY_DOUBLE);
    if (!w0 || !w1 || !w2) { Py_XDECREF(w0); Py_XDECREF(w1); Py_XDECREF(w2); return NULL; }
    const double *pcp = (const double *)PyArray_DATA(cp);
    const double *pcf = (const double *)PyArray_DATA(cf);
    const double *pex = (const double *)PyArray_DATA(ex);
    const double *pey = (const double *)PyArray_DATA(ey);
    const double *pez = (const double *)PyArray_DATA(ez);
    const double *pg0 = (const double *)PyArray_DATA(g0);
    const double *pg1 = (const double *)PyArray_DATA(g1);
    const double *pg2 = (const double *)PyArray_DATA(g2);
    double *pw0 = (double *)PyArray_DATA(w0);
    double *pw1 = (double *)PyArray_DATA(w1);
    double *pw2 = (double *)PyArray_DATA(w2);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double dot = pex[i]*pg0[i] + pey[i]*pg1[i] + pez[i]*pg2[i];
        const double c = pcp[i];
        const double fd = pcf[i] * dot;
        pw0[i] = c * pg0[i] + fd * pex[i];
        pw1[i] = c * pg1[i] + fd * pey[i];
        pw2[i] = c * pg2[i] + fd * pez[i];
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NNN", w0, w1, w2);
}

/* ------------- out += i*TPI*(gx*f0 + gy*f1 + gz*f2), f complex ------------- */

static PyObject *div_accum(PyObject *self, PyObject *args) {
    PyArrayObject *out, *gx, *gy, *gz, *f0, *f1, *f2;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
                          &PyArray_Type, &out,
                          &PyArray_Type, &gx, &PyArray_Type, &gy, &PyArray_Type, &gz,
                          &PyArray_Type, &f0, &PyArray_Type, &f1, &PyArray_Type, &f2)) return NULL;
    if (!check_wcarray(out, NPY_CDOUBLE, "out") ||
        !check_carray(gx, NPY_DOUBLE, "gx") || !check_carray(gy, NPY_DOUBLE, "gy") ||
        !check_carray(gz, NPY_DOUBLE, "gz") ||
        !check_carray(f0, NPY_CDOUBLE, "f0") || !check_carray(f1, NPY_CDOUBLE, "f1") ||
        !check_carray(f2, NPY_CDOUBLE, "f2")) return NULL;
    const npy_intp n = PyArray_SIZE(out);
    double *po = (double *)PyArray_DATA(out);
    const double *px = (const double *)PyArray_DATA(gx);
    const double *py = (const double *)PyArray_DATA(gy);
    const double *pz = (const double *)PyArray_DATA(gz);
    const double *p0 = (const double *)PyArray_DATA(f0);
    const double *p1 = (const double *)PyArray_DATA(f1);
    const double *p2 = (const double *)PyArray_DATA(f2);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double sr = px[i]*p0[2*i]   + py[i]*p1[2*i]   + pz[i]*p2[2*i];
        const double si = px[i]*p0[2*i+1] + py[i]*p1[2*i+1] + pz[i]*p2[2*i+1];
        po[2*i]   += -TPI * si;
        po[2*i+1] +=  TPI * sr;
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* ------------- l_op tail: out = scale*(w_b*lap - TPI^2*gsq*dphi) ------------- */

static PyObject *wb_combine(PyObject *self, PyObject *args) {
    PyArrayObject *wb, *lap, *gsq, *dphi;
    double scale;
    if (!PyArg_ParseTuple(args, "O!O!O!O!d", &PyArray_Type, &wb, &PyArray_Type, &lap,
                          &PyArray_Type, &gsq, &PyArray_Type, &dphi, &scale)) return NULL;
    if (!check_carray(wb, NPY_CDOUBLE, "w_b") || !check_carray(lap, NPY_CDOUBLE, "lap") ||
        !check_carray(gsq, NPY_DOUBLE, "gsq") || !check_carray(dphi, NPY_CDOUBLE, "dphi")) return NULL;
    const npy_intp n = PyArray_SIZE(wb);
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(wb), PyArray_DIMS(wb), NPY_CDOUBLE);
    if (out == NULL) return NULL;
    const double *pw = (const double *)PyArray_DATA(wb);
    const double *pl = (const double *)PyArray_DATA(lap);
    const double *pg = (const double *)PyArray_DATA(gsq);
    const double *pd = (const double *)PyArray_DATA(dphi);
    double *po = (double *)PyArray_DATA(out);
    const double t2 = TPI * TPI;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double wr = pw[2*i], wi = pw[2*i+1];
        const double lr = pl[2*i], li = pl[2*i+1];
        const double k = t2 * pg[i];
        po[2*i]   = scale * (wr*lr - wi*li - k * pd[2*i]);
        po[2*i+1] = scale * (wr*li + wi*lr - k * pd[2*i+1]);
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* ------------- dst += alpha * src (complex, real alpha) ------------- */

static PyObject *caxpy(PyObject *self, PyObject *args) {
    PyArrayObject *dst, *src;
    double alpha;
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &dst, &PyArray_Type, &src, &alpha)) return NULL;
    if (!check_wcarray(dst, NPY_CDOUBLE, "dst") || !check_carray(src, NPY_CDOUBLE, "src")) return NULL;
    const npy_intp n2 = 2 * PyArray_SIZE(dst);
    double *pd = (double *)PyArray_DATA(dst);
    const double *ps = (const double *)PyArray_DATA(src);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n2; ++i) pd[i] += alpha * ps[i];
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* ------------- p = z + beta * p (complex, real beta, in-place on p) ------------- */

static PyObject *cxpby(PyObject *self, PyObject *args) {
    PyArrayObject *p, *z;
    double beta;
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &p, &PyArray_Type, &z, &beta)) return NULL;
    if (!check_wcarray(p, NPY_CDOUBLE, "p") || !check_carray(z, NPY_CDOUBLE, "z")) return NULL;
    const npy_intp n2 = 2 * PyArray_SIZE(p);
    double *pp = (double *)PyArray_DATA(p);
    const double *pz = (const double *)PyArray_DATA(z);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n2; ++i) pp[i] = pz[i] + beta * pp[i];
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

/* ------------- z = precond(real) * r(complex) ------------- */

static PyObject *rmulc(PyObject *self, PyObject *args) {
    PyArrayObject *pre, *r;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &pre, &PyArray_Type, &r)) return NULL;
    if (!check_carray(pre, NPY_DOUBLE, "precond") || !check_carray(r, NPY_CDOUBLE, "r")) return NULL;
    const npy_intp n = PyArray_SIZE(r);
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(r), PyArray_DIMS(r), NPY_CDOUBLE);
    if (out == NULL) return NULL;
    const double *pp = (const double *)PyArray_DATA(pre);
    const double *pr = (const double *)PyArray_DATA(r);
    double *po = (double *)PyArray_DATA(out);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        po[2*i] = pp[i] * pr[2*i];
        po[2*i+1] = pp[i] * pr[2*i+1];
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* ------------- real part of vdot(b, a) with OpenMP reduction ------------- */

static PyObject *dprod_rc_fast(PyObject *self, PyObject *args) {
    PyArrayObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b)) return NULL;
    if (!check_carray(a, NPY_CDOUBLE, "a") || !check_carray(b, NPY_CDOUBLE, "b")) return NULL;
    const npy_intp n2 = 2 * PyArray_SIZE(a);
    const double *pa = (const double *)PyArray_DATA(a);
    const double *pb = (const double *)PyArray_DATA(b);
    double sum = 0.0;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (npy_intp i = 0; i < n2; ++i) sum += pa[i] * pb[i];
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(sum);
}

/* ------------- chi response: chi_perp, chi_factor from scaled |E| ------------- */

static PyObject *chi_response(PyObject *self, PyObject *args) {
    PyArrayObject *emag, *sdiel;
    double pbeta, alpha_pol, alpha0_rot, invalpha_sic, n_mol;
    if (!PyArg_ParseTuple(args, "O!O!ddddd", &PyArray_Type, &emag, &PyArray_Type, &sdiel,
                          &pbeta, &alpha_pol, &alpha0_rot, &invalpha_sic, &n_mol)) return NULL;
    if (!check_carray(emag, NPY_DOUBLE, "emag") || !check_carray(sdiel, NPY_DOUBLE, "s_diel")) return NULL;
    const npy_intp n = PyArray_SIZE(emag);
    PyArrayObject *cp = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(emag), PyArray_DIMS(emag), NPY_DOUBLE);
    PyArrayObject *cf = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(emag), PyArray_DIMS(emag), NPY_DOUBLE);
    if (!cp || !cf) { Py_XDECREF(cp); Py_XDECREF(cf); return NULL; }
    const double *pe = (const double *)PyArray_DATA(emag);
    const double *ps = (const double *)PyArray_DATA(sdiel);
    double *pcp = (double *)PyArray_DATA(cp);
    double *pcf = (double *)PyArray_DATA(cf);
    const double e_floor = 2.0e-4 / fmax(pbeta, 1.0e-300);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double x = pbeta * pe[i];
        double par, perp;
        if (x < 2.0e-4) {
            par = 1.0; perp = 1.0;
        } else if (x > 100.0) {
            par = 3.0 / (x * x);
            perp = 3.0 * (1.0 - 1.0 / x) / x;
        } else {
            const double sh = sinh(x);
            const double th = tanh(x);
            par = 3.0 * (1.0 / (x * x) - 1.0 / (sh * sh));
            perp = 3.0 * (1.0 / th - 1.0 / x) / x;
        }
        par = alpha_pol + alpha0_rot * par;
        perp = alpha_pol + alpha0_rot * perp;
        const double sd = n_mol * ps[i];
        par = sd / (1.0 / par - invalpha_sic);
        perp = sd / (1.0 / perp - invalpha_sic);
        double inv_e2 = 0.0;
        if (pe[i] >= e_floor) inv_e2 = 1.0 / (pe[i] * pe[i]);
        pcp[i] = perp;
        pcf[i] = (par - perp) * inv_e2;
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NN", cp, cf);
}

/* ------------- P/E vector: p_i = e_i * n_mol*s_diel*(a0rot_eps*g(y) + apol_eps) ------------- */

static PyObject *polarization_vec(PyObject *self, PyObject *args) {
    PyArrayObject *ex, *ey, *ez, *emag, *sdiel;
    double pbeta, alpha_pol_over_eps, alpha0_rot_over_eps, n_mol;
    int lnldiel;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!ddddi", &PyArray_Type, &ex, &PyArray_Type, &ey,
                          &PyArray_Type, &ez, &PyArray_Type, &emag, &PyArray_Type, &sdiel,
                          &pbeta, &alpha_pol_over_eps, &alpha0_rot_over_eps, &n_mol, &lnldiel)) return NULL;
    if (!check_carray(ex, NPY_DOUBLE, "ex") || !check_carray(ey, NPY_DOUBLE, "ey") ||
        !check_carray(ez, NPY_DOUBLE, "ez") || !check_carray(emag, NPY_DOUBLE, "emag") ||
        !check_carray(sdiel, NPY_DOUBLE, "s_diel")) return NULL;
    const npy_intp n = PyArray_SIZE(ex);
    PyArrayObject *p0 = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ex), PyArray_DIMS(ex), NPY_DOUBLE);
    PyArrayObject *p1 = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ex), PyArray_DIMS(ex), NPY_DOUBLE);
    PyArrayObject *p2 = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ex), PyArray_DIMS(ex), NPY_DOUBLE);
    if (!p0 || !p1 || !p2) { Py_XDECREF(p0); Py_XDECREF(p1); Py_XDECREF(p2); return NULL; }
    const double *px = (const double *)PyArray_DATA(ex);
    const double *py = (const double *)PyArray_DATA(ey);
    const double *pz = (const double *)PyArray_DATA(ez);
    const double *pm = (const double *)PyArray_DATA(emag);
    const double *ps = (const double *)PyArray_DATA(sdiel);
    double *o0 = (double *)PyArray_DATA(p0);
    double *o1 = (double *)PyArray_DATA(p1);
    double *o2 = (double *)PyArray_DATA(p2);
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        double g = 1.0;
        if (lnldiel) {
            const double y = pbeta * pm[i];
            if (y < 2.0e-4) {
                g = 1.0;
            } else if (y > 100.0) {
                g = 3.0 * (1.0 - 1.0 / y) / y;
            } else {
                g = 3.0 * (1.0 / tanh(y) - 1.0 / y) / y;
            }
        }
        const double pe = n_mol * ps[i] * (alpha0_rot_over_eps * g + alpha_pol_over_eps);
        o0[i] = pe * px[i];
        o1[i] = pe * py[i];
        o2[i] = pe * pz[i];
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NNN", p0, p1, p2);
}

/* ------------- nonlinear ion density values ------------- */

static PyObject *ion_density_values(PyObject *self, PyObject *args) {
    PyArrayObject *phi, *sion;
    double zbeta, theta, n_max, invbeta, volume;
    int lnlion;
    if (!PyArg_ParseTuple(args, "O!O!dddddi", &PyArray_Type, &phi, &PyArray_Type, &sion,
                          &zbeta, &theta, &n_max, &invbeta, &volume, &lnlion)) return NULL;
    if (!check_carray(phi, NPY_DOUBLE, "phi") || !check_carray(sion, NPY_DOUBLE, "s_ion")) return NULL;
    const npy_intp n = PyArray_SIZE(phi);
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(phi), PyArray_DIMS(phi), NPY_DOUBLE);
    if (out == NULL) return NULL;
    const double *pp = (const double *)PyArray_DATA(phi);
    const double *ps = (const double *)PyArray_DATA(sion);
    double *po = (double *)PyArray_DATA(out);
    const double pref = -n_max * invbeta * zbeta * volume;
    const double small_cut = sqrt(theta) * 2.0e-4;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        const double x = zbeta * pp[i];
        double nw;
        if (lnlion && theta > 0.0) {
            const double ax = fabs(x);
            if (ax > 100.0) {
                nw = (x > 0.0) ? 1.0 : -1.0;
            } else if (ax < small_cut) {
                nw = theta * x;
            } else {
                nw = theta * sinh(x) / (1.0 + theta * (cosh(x) - 1.0));
            }
        } else if (lnlion) {
            double xc = x;
            if (xc > 100.0) xc = 100.0;
            else if (xc < -100.0) xc = -100.0;
            nw = sinh(xc);
        } else {
            nw = x;
        }
        po[i] = pref * ps[i] * nw;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* ------------- ekappa2 for the ion response ------------- */

static PyObject *ekappa2_values(PyObject *self, PyObject *args) {
    PyArrayObject *phi, *sion;
    double zbeta, theta, n_max, alpha0_ion;
    int lnlion;
    if (!PyArg_ParseTuple(args, "O!O!ddddi", &PyArray_Type, &phi, &PyArray_Type, &sion,
                          &zbeta, &theta, &n_max, &alpha0_ion, &lnlion)) return NULL;
    if (!check_carray(phi, NPY_DOUBLE, "phi") || !check_carray(sion, NPY_DOUBLE, "s_ion")) return NULL;
    const npy_intp n = PyArray_SIZE(phi);
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(phi), PyArray_DIMS(phi), NPY_DOUBLE);
    if (out == NULL) return NULL;
    const double *pp = (const double *)PyArray_DATA(phi);
    const double *ps = (const double *)PyArray_DATA(sion);
    double *po = (double *)PyArray_DATA(out);
    const double pref = n_max * alpha0_ion;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static)
    for (npy_intp i = 0; i < n; ++i) {
        double val;
        if (lnlion) {
            const double x = zbeta * pp[i];
            const double ax = fabs(x);
            if (ax > 100.0) {
                val = 0.0;
            } else {
                double x2;
                if (ax < 2.0e-4) {
                    x2 = 0.5 * x * x;
                } else {
                    x2 = cosh(x) - 1.0;
                }
                const double d = 1.0 + theta * x2;
                val = (1.0 + (1.0 - theta) * x2) / (d * d);
            }
        } else {
            val = 1.0;
        }
        po[i] = pref * ps[i] * val;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

static PyMethodDef Methods[] = {
    {"local_field_factor_fixed", local_field_factor_fixed, METH_VARARGS, "OpenMP local field fixed-point solver."},
    {"conj_mul", conj_mul, METH_VARARGS, "conj(a)*b for complex128 arrays."},
    {"grad_premul", grad_premul, METH_VARARGS, "(i*2pi*g_k)*f for k=x,y,z in one pass."},
    {"magnitude3", magnitude3, METH_VARARGS, "sqrt(ex^2+ey^2+ez^2)."},
    {"scale4_inplace", scale4_inplace, METH_VARARGS, "in-place scale of ex,ey,ez,emag by f."},
    {"tensor_work", tensor_work, METH_VARARGS, "fused dielectric tensor-vector contraction."},
    {"div_accum", div_accum, METH_VARARGS, "out += i*2pi*(g.f) fused divergence accumulation."},
    {"wb_combine", wb_combine, METH_VARARGS, "scale*(w_b*lap - (2pi)^2*gsq*dphi)."},
    {"caxpy", caxpy, METH_VARARGS, "dst += alpha*src (complex, real alpha)."},
    {"cxpby", cxpby, METH_VARARGS, "p = z + beta*p (complex, real beta)."},
    {"rmulc", rmulc, METH_VARARGS, "real precond * complex r."},
    {"dprod_rc", dprod_rc_fast, METH_VARARGS, "real part of vdot(b,a) with OpenMP reduction."},
    {"chi_response", chi_response, METH_VARARGS, "fused chi_perp/chi_factor build."},
    {"polarization_vec", polarization_vec, METH_VARARGS, "fused P/E vector build."},
    {"ion_density_values", ion_density_values, METH_VARARGS, "fused nonlinear ion density."},
    {"ekappa2_values", ekappa2_values, METH_VARARGS, "fused ion response ekappa2."},
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
