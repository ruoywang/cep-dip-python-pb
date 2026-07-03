from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter

import numpy as np
from scipy.special import erfc

try:
    from . import _pb_fast
except Exception:  # pragma: no cover - optional acceleration extension.
    _pb_fast = None

from .grid import (
    BOLKEV,
    fused_kernel as _fused,
    EDEPS,
    MOLAR,
    TPI,
    Grid,
    convolve_real,
    exp_kernel_g,
    l0_op,
    l0_inv_op,
    normalized_gaussian_kernel_g,
)


@dataclass
class PBState:
    rho_ion_values: np.ndarray
    rho_bound_values: np.ndarray
    phi_values: np.ndarray
    s_ion: np.ndarray
    s_diel: np.ndarray


_PB_TIMING: dict[str, float] = {}
_PB_COUNTS: dict[str, int] = {}


def reset_pb_timing() -> None:
    _PB_TIMING.clear()
    _PB_COUNTS.clear()


def get_pb_timing() -> dict[str, tuple[float, int]]:
    return {key: (_PB_TIMING[key], _PB_COUNTS.get(key, 0)) for key in sorted(_PB_TIMING)}


def _profile_enabled() -> bool:
    return os.environ.get("PB_PROFILE_INNER", "0") not in ("", "0", "false", "False")


def _add_timing(label: str, seconds: float) -> None:
    if not _profile_enabled():
        return
    _PB_TIMING[label] = _PB_TIMING.get(label, 0.0) + seconds
    _PB_COUNTS[label] = _PB_COUNTS.get(label, 0) + 1


def derived_params(sol: dict) -> dict:
    out = dict(sol)
    inv_beta = BOLKEV * float(sol["SOLTEMP"])
    p_beta = float(sol["P_MOL"]) / inv_beta
    z_beta = float(sol["ZION"]) / inv_beta
    alpha0_rot = EDEPS * inv_beta * p_beta**2 / 3.0
    alpha_pol = alpha0_rot / (float(sol["EB_K"]) - float(sol["EPSILON_INF"])) * (float(sol["EPSILON_INF"]) - 1.0)
    invalpha_sic = ((float(sol["EB_K"]) - float(sol["EPSILON_INF"])) / alpha0_rot - float(sol["N_MOL"])) / (
        float(sol["EB_K"]) - 1.0
    )
    alpha0_ion = EDEPS * inv_beta * z_beta**2
    if float(sol["LAMBDA_D_K"]) > 0.0:
        c_ion_b = float(sol["EB_K"]) / (float(sol["LAMBDA_D_K"]) ** 2 * alpha0_ion)
    else:
        c_ion_b = 2.0 * float(sol["C_MOLAR"]) * MOLAR
    d_ion = float(sol["D_ION"])
    if d_ion < 0.0:
        d_ion = 2.0 ** (5.0 / 6.0) * float(sol["R_ION"])
    l_ion = c_ion_b > 0.0
    l_nlion = bool(sol["LNLION"]) and l_ion
    if l_nlion and d_ion > 0.0:
        n_max = 1.0 / d_ion**3
        theta_b = c_ion_b / n_max
        alpha0_ion = theta_b * alpha0_ion
    else:
        n_max = c_ion_b
        theta_b = 0.0
    out.update(
        invBETA=inv_beta,
        PBETA=p_beta,
        ZBETA=z_beta,
        alpha0_rot=alpha0_rot,
        alpha_pol=alpha_pol,
        invalpha_sic=invalpha_sic,
        alpha0_ion=alpha0_ion,
        c_ion_b=c_ion_b,
        d_ion=d_ion,
        n_max=n_max,
        theta_b=theta_b,
        LION=l_ion,
        LNLION=l_nlion,
    )
    return out


def shape_func(x: np.ndarray, sigma_k: float) -> np.ndarray:
    return 0.5 * erfc(x / (np.sqrt(2.0) * sigma_k))


def smooth_box(grid: Grid, z0: float, z1: float, sigma: float) -> np.ndarray:
    z = grid.cartesian_z_mesh()
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    mask = 0.5 * (erfc((z - z1) / (inv_sqrt2 * sigma)) - erfc((z - z0) / (inv_sqrt2 * sigma)))
    return np.clip(mask, 0.0, 1.0)


def create_cavity(
    n_e_density: np.ndarray,
    grid: Grid,
    params: dict,
    timings: list[tuple[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = perf_counter()
    sigma_k = float(params["SIGMA_K"])
    a_k = float(params["A_K"])
    n_min = float(params["N_MIN"])
    x_vdw = np.log(np.maximum(n_e_density / float(params["NC_K"]), n_min))
    s_vdw = shape_func(x_vdw, sigma_k)
    t = _mark(timings, "cavity_vdw_shape", t)

    def maybe_exp(radius: float):
        if radius <= 0.0:
            return None
        return exp_kernel_g(grid, radius, a_k / sigma_k)

    w_ion = maybe_exp(float(params["R_ION"]))
    w_solv = maybe_exp(float(params["R_SOLV"]))
    w_cav = maybe_exp(float(params["R_CAV"]))
    w_diel = maybe_exp(float(params["R_DIEL"]))
    t = _mark(timings, "cavity_kernels", t)

    if w_ion is not None:
        x_ion = np.log(np.maximum(convolve_real(1.0 - s_vdw, w_ion, grid), n_min))
        s_ion = shape_func(x_ion, sigma_k)
    else:
        s_ion = s_vdw.copy()
    t = _mark(timings, "cavity_ion", t)

    if w_solv is not None:
        x_solv = np.log(np.maximum(convolve_real(1.0 - s_vdw, w_solv, grid), n_min))
        s_solv = shape_func(x_solv, sigma_k)
    else:
        s_solv = s_vdw.copy()
    t = _mark(timings, "cavity_solv", t)

    if w_cav is not None:
        x_cav = np.log(np.maximum(convolve_real(s_solv, w_cav, grid), n_min))
        s_cav = 1.0 - shape_func(x_cav, sigma_k)
    else:
        s_cav = s_solv.copy()
    t = _mark(timings, "cavity_cav", t)

    if w_diel is not None:
        x_diel = np.log(np.maximum(convolve_real(s_solv, w_diel, grid), n_min))
        s_diel = 1.0 - shape_func(x_diel, sigma_k)
    else:
        s_diel = s_solv.copy()
    t = _mark(timings, "cavity_diel", t)

    if bool(params["LVAC"]) and float(params["SOL_Z1"]) > float(params["SOL_Z0"]):
        m_sol = smooth_box(grid, float(params["SOL_Z0"]), float(params["SOL_Z1"]), float(params["SOL_SIGMA"]))
        m_ion = smooth_box(
            grid,
            float(params["SOL_Z0"]) + float(params["D_STERN"]),
            float(params["SOL_Z1"]) - float(params["D_STERN"]),
            float(params["SOL_SIGMA"]),
        )
        s_ion = s_ion * m_ion
        s_cav = s_cav * m_ion
        s_diel = s_diel * m_sol
    _mark(timings, "cavity_vacuum_mask", t)
    return s_ion, s_diel, s_cav


def local_field_factor(e_mag: np.ndarray, params: dict) -> np.ndarray:
    alpha_pol = float(params["alpha_pol"])
    alpha0_rot = float(params["alpha0_rot"])
    invalpha_sic = float(params["invalpha_sic"])
    if not bool(params["LNLDIEL"]):
        return np.full_like(e_mag, 1.0 / (1.0 - (alpha_pol + alpha0_rot) * invalpha_sic))
    if _pb_fast is not None and os.environ.get("PB_DISABLE_FAST_LOCAL_FIELD", "0") in ("", "0", "false", "False"):
        return _pb_fast.local_field_factor_fixed(
            e_mag,
            alpha_pol,
            alpha0_rot,
            invalpha_sic,
            float(params["PBETA"]),
        )
    lo_scalar = 1.0 / (1.0 - alpha_pol * invalpha_sic)
    hi_scalar = 1.0 / (1.0 - (alpha_pol + alpha0_rot) * invalpha_sic)
    hi_scalar *= 1.0 + 1.0e-8
    x0 = float(params["PBETA"]) * e_mag

    def g_rot_array(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        small = x < 2.0e-4
        out[small] = 1.0
        xs = x[~small]
        out[~small] = 3.0 * (xs - np.tanh(xs)) / (xs * xs * np.tanh(xs))
        return out

    out = np.full_like(e_mag, hi_scalar)
    zero = x0 == 0.0
    for _ in range(80):
        gx = g_rot_array(out * x0)
        new = 1.0 / (1.0 - (gx * alpha0_rot + alpha_pol) * invalpha_sic)
        new = np.clip(new, lo_scalar, hi_scalar)
        diff = np.max(np.abs(new - out))
        out = new
        if diff <= 1.0e-10 * max(1.0, hi_scalar):
            break
    out[zero] = hi_scalar
    return out


def _mark(timings: list[tuple[str, float]] | None, label: str, t0: float) -> float:
    t1 = perf_counter()
    if timings is not None:
        timings.append((label, t1 - t0))
    return t1


def update_from_total_phi(
    phi: np.ndarray,
    n_e_density: np.ndarray,
    grid: Grid,
    solvation: dict,
    timings: list[tuple[str, float]] | None = None,
) -> PBState:
    t = perf_counter()
    params = derived_params(solvation)
    s_ion, s_diel, _ = create_cavity(n_e_density, grid, params, timings)
    t = _mark(timings, "create_cavity", t)
    state = update_from_total_phi_with_cavity(phi, s_ion, s_diel, grid, params, timings, t)
    return state


def ion_density_values_from_phi(phi: np.ndarray, s_ion: np.ndarray, grid: Grid, params: dict) -> np.ndarray:
    kern = _fused("ion_density_values")
    if kern is not None:
        return kern(
            np.ascontiguousarray(phi),
            np.ascontiguousarray(s_ion),
            float(params["ZBETA"]),
            float(params["theta_b"]),
            float(params["n_max"]),
            float(params["invBETA"]),
            grid.volume,
            1 if bool(params["LNLION"]) else 0,
        )
    x = float(params["ZBETA"]) * phi
    theta = float(params["theta_b"])
    if bool(params["LNLION"]) and theta > 0.0:
        n_work = np.empty_like(phi)
        large = np.abs(x) > 100.0
        small = np.abs(x) < np.sqrt(theta) * 2.0e-4
        mid = ~(large | small)
        n_work[large] = np.sign(x[large])
        n_work[small] = theta * x[small]
        denom = 1.0 + theta * (np.cosh(x[mid]) - 1.0)
        n_work[mid] = theta * np.sinh(x[mid]) / denom
    elif bool(params["LNLION"]):
        n_work = np.sinh(np.clip(x, -100.0, 100.0))
    else:
        n_work = x
    rho_ion = -float(params["n_max"]) * float(params["invBETA"]) * float(params["ZBETA"]) * s_ion * n_work
    return rho_ion * grid.volume


def bound_density_values_from_phi(phi: np.ndarray, s_diel: np.ndarray, grid: Grid, params: dict) -> np.ndarray:
    phi_g = grid.fft(phi)
    w_b = normalized_gaussian_kernel_g(grid, float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"]))
    ex, ey, ez, emag = grid.grad_from_recip(-np.conj(w_b) * phi_g)
    f_loc = local_field_factor(emag, params)
    ex *= f_loc
    ey *= f_loc
    ez *= f_loc
    emag *= f_loc

    y = float(params["PBETA"]) * emag
    if bool(params["LNLDIEL"]):
        g = np.empty_like(y)
        small = y < 2.0e-4
        large = y > 100.0
        mid = ~(small | large)
        g[small] = 1.0
        g[large] = 3.0 * (1.0 - 1.0 / y[large]) / y[large]
        g[mid] = 3.0 * (1.0 / np.tanh(y[mid]) - 1.0 / y[mid]) / y[mid]
    else:
        g = np.ones_like(y)
    polar_over_eps = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
    p_over_e = float(params["N_MOL"]) * s_diel * polar_over_eps
    div_p_g = grid.div_real_vector(p_over_e * ex, p_over_e * ey, p_over_e * ez)
    rho_bound_g = -w_b * div_p_g
    return grid.ifft_real(rho_bound_g) * grid.volume


def update_from_total_phi_with_cavity(
    phi: np.ndarray,
    s_ion: np.ndarray,
    s_diel: np.ndarray,
    grid: Grid,
    params: dict,
    timings: list[tuple[str, float]] | None = None,
    t_start: float | None = None,
) -> PBState:
    t = perf_counter() if t_start is None else t_start
    phi_g = grid.fft(phi)
    t = _mark(timings, "fft_phi", t)
    rho_ion_values = ion_density_values_from_phi(phi, s_ion, grid, params)
    t = _mark(timings, "ion_density", t)
    w_b = normalized_gaussian_kernel_g(grid, float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"]))
    ex, ey, ez, emag = grid.grad_from_recip(-np.conj(w_b) * phi_g)
    t = _mark(timings, "local_field_gradient", t)
    f_loc = local_field_factor(emag, params)
    t = _mark(timings, "local_field_factor", t)
    ex *= f_loc
    ey *= f_loc
    ez *= f_loc
    emag *= f_loc
    y = float(params["PBETA"]) * emag
    if bool(params["LNLDIEL"]):
        g = np.empty_like(y)
        small = y < 2.0e-4
        large = y > 100.0
        mid = ~(small | large)
        g[small] = 1.0
        g[large] = 3.0 * (1.0 - 1.0 / y[large]) / y[large]
        g[mid] = 3.0 * (1.0 / np.tanh(y[mid]) - 1.0 / y[mid]) / y[mid]
    else:
        g = np.ones_like(y)
    polar_over_eps = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
    p_over_e = float(params["N_MOL"]) * s_diel * polar_over_eps
    div_p_g = grid.div_real_vector(p_over_e * ex, p_over_e * ey, p_over_e * ez)
    rho_bound_g = -w_b * div_p_g
    rho_bound_values = grid.ifft_real(rho_bound_g) * grid.volume
    _mark(timings, "bound_density", t)
    return PBState(
        rho_ion_values=rho_ion_values,
        rho_bound_values=rho_bound_values,
        phi_values=phi.copy(),
        s_ion=s_ion,
        s_diel=s_diel,
    )


def poisson_potential_from_density_values(charge_values: np.ndarray, grid: Grid) -> np.ndarray:
    source_g = grid.fft(charge_values)
    phi_g = l0_inv_op(source_g, grid)
    return grid.ifft_real(phi_g)


def dprod_rc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.real(np.vdot(b, a)))


def set_g0(a: np.ndarray, value: float) -> None:
    a[(0,) * a.ndim] = value


def get_g0(a: np.ndarray) -> float:
    return float(np.real(a[(0,) * a.ndim]))


def lapl_tensor(phi_g: np.ndarray, response: tuple, grid: Grid) -> np.ndarray:
    t_all = perf_counter()
    gx, gy, gz, _ = grid.reciprocal_mesh()
    t = perf_counter()
    premul = _fused("grad_premul")
    if premul is not None:
        ax, ay, az = premul(gx, gy, gz, np.ascontiguousarray(phi_g))
        grads = [grid.ifft_real(ax), grid.ifft_real(ay), grid.ifft_real(az)]
    else:
        grads = [
            grid.ifft_real(1j * TPI * gx * phi_g),
            grid.ifft_real(1j * TPI * gy * phi_g),
            grid.ifft_real(1j * TPI * gz * phi_g),
        ]
    _add_timing("lapl_tensor_grad_ifft", perf_counter() - t)
    out = np.zeros(grid.shape, dtype=complex)
    g_list = [gx, gy, gz]
    kind = response[0]
    if kind == "scalar":
        scalar = response[1]
        for i in range(3):
            t = perf_counter()
            out += 1j * TPI * g_list[i] * grid.fft(scalar * grads[i])
            _add_timing("lapl_tensor_work_fft", perf_counter() - t)
    elif kind == "tensor_field":
        chi_perp, chi_factor, ex, ey, ez = response[1:]
        tw = _fused("tensor_work")
        acc = _fused("div_accum")
        if tw is not None and acc is not None:
            t = perf_counter()
            w0, w1, w2 = tw(chi_perp, chi_factor, ex, ey, ez, grads[0], grads[1], grads[2])
            _add_timing("lapl_tensor_work_build", perf_counter() - t)
            t = perf_counter()
            f0 = np.ascontiguousarray(grid.fft(w0))
            f1 = np.ascontiguousarray(grid.fft(w1))
            f2 = np.ascontiguousarray(grid.fft(w2))
            _add_timing("lapl_tensor_work_fft", perf_counter() - t)
            t = perf_counter()
            acc(out, gx, gy, gz, f0, f1, f2)
            _add_timing("lapl_tensor_dot", perf_counter() - t)
        else:
            e = [ex, ey, ez]
            t = perf_counter()
            dot = ex * grads[0] + ey * grads[1] + ez * grads[2]
            _add_timing("lapl_tensor_dot", perf_counter() - t)
            for i in range(3):
                t = perf_counter()
                work = chi_perp * grads[i] + chi_factor * e[i] * dot
                _add_timing("lapl_tensor_work_build", perf_counter() - t)
                t = perf_counter()
                out += 1j * TPI * g_list[i] * grid.fft(work)
                _add_timing("lapl_tensor_work_fft", perf_counter() - t)
    else:
        raise ValueError(f"unknown dielectric response kind: {kind}")
    _add_timing("lapl_tensor_total", perf_counter() - t_all)
    return out


def l_op(dphi_g: np.ndarray, response: tuple, ekappa2: np.ndarray | None, w_b: np.ndarray, grid: Grid) -> np.ndarray:
    t_all = perf_counter()
    _, _, _, gsq = grid.reciprocal_mesh()
    cm = _fused("conj_mul")
    comb = _fused("wb_combine")
    axpy = _fused("caxpy")
    if cm is not None and comb is not None and axpy is not None:
        dphi_c = np.ascontiguousarray(dphi_g)
        cwork = cm(w_b, dphi_c)
        lap = lapl_tensor(cwork, response, grid)
        out = comb(w_b, np.ascontiguousarray(lap), gsq, dphi_c, -grid.volume / EDEPS)
        if ekappa2 is not None:
            t = perf_counter()
            real = grid.ifft_real(dphi_c)
            axpy(out, np.ascontiguousarray(grid.fft(ekappa2 * real)), grid.volume / EDEPS)
            _add_timing("l_op_ekappa_fft", perf_counter() - t)
        _add_timing("l_op_total", perf_counter() - t_all)
        return out
    cwork = np.conj(w_b) * dphi_g
    lp = w_b * lapl_tensor(cwork, response, grid)
    lp += -(TPI**2) * gsq * dphi_g
    if ekappa2 is not None:
        t = perf_counter()
        real = grid.ifft_real(dphi_g)
        lp += grid.fft(-ekappa2 * real)
        _add_timing("l_op_ekappa_fft", perf_counter() - t)
    _add_timing("l_op_total", perf_counter() - t_all)
    return -grid.volume / EDEPS * lp


def nlpb_field_quantities(
    phi: np.ndarray,
    s_ion: np.ndarray,
    s_diel: np.ndarray,
    grid: Grid,
    params: dict,
    w_b: np.ndarray,
) -> dict:
    """Field-dependent quantities needed for trial residuals and for the response."""
    t_all = perf_counter()
    phi_c = np.ascontiguousarray(phi)
    t = perf_counter()
    phi_g = grid.fft(phi_c)
    _add_timing("nlpb_fft_phi", perf_counter() - t)
    t = perf_counter()
    cm = _fused("conj_mul")
    if cm is not None:
        cwork = cm(w_b, np.ascontiguousarray(phi_g))
        np.negative(cwork, out=cwork)
        ex, ey, ez, emag = grid.grad_from_recip(cwork)
    else:
        ex, ey, ez, emag = grid.grad_from_recip(-np.conj(w_b) * phi_g)
    _add_timing("nlpb_grad", perf_counter() - t)
    t = perf_counter()
    f_loc = local_field_factor(emag, params)
    _add_timing("nlpb_local_field_factor", perf_counter() - t)
    t = perf_counter()
    s4 = _fused("scale4_inplace")
    if s4 is not None:
        s4(ex, ey, ez, emag, f_loc)
    else:
        ex *= f_loc
        ey *= f_loc
        ez *= f_loc
        emag *= f_loc
    _add_timing("nlpb_scale_field", perf_counter() - t)

    t = perf_counter()
    n_ion_values = ion_density_values_from_phi(phi_c, s_ion, grid, params)
    _add_timing("nlpb_ion", perf_counter() - t)

    t = perf_counter()
    pv = _fused("polarization_vec")
    if pv is not None:
        px, py, pz = pv(
            ex, ey, ez, emag,
            np.ascontiguousarray(s_diel),
            float(params["PBETA"]),
            float(params["alpha_pol"]) / EDEPS,
            float(params["alpha0_rot"]) / EDEPS,
            float(params["N_MOL"]),
            1 if bool(params["LNLDIEL"]) else 0,
        )
    else:
        y = float(params["PBETA"]) * emag
        if bool(params["LNLDIEL"]):
            g = np.empty_like(y)
            small = y < 2.0e-4
            large = y > 100.0
            mid = ~(small | large)
            g[small] = 1.0
            g[large] = 3.0 * (1.0 - 1.0 / y[large]) / y[large]
            g[mid] = 3.0 * (1.0 / np.tanh(y[mid]) - 1.0 / y[mid]) / y[mid]
        else:
            g = np.ones_like(y)
        polar_over_eps = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
        p_over_e = float(params["N_MOL"]) * s_diel * polar_over_eps
        px = p_over_e * ex
        py = p_over_e * ey
        pz = p_over_e * ez
    _add_timing("nlpb_dielectric_scalar", perf_counter() - t)
    t = perf_counter()
    div_p_g = grid.div_real_vector(px, py, pz)
    n_b_values = grid.ifft_real(-w_b * div_p_g) * grid.volume
    _add_timing("nlpb_bound_fft", perf_counter() - t)
    _add_timing("nlpb_total_no_response", perf_counter() - t_all)
    return {
        "phi": phi_c,
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "emag": emag,
        "n_b": n_b_values,
        "n_ion": n_ion_values,
    }


def nlpb_response_from_fields(
    fields: dict,
    s_ion: np.ndarray,
    s_diel: np.ndarray,
    grid: Grid,
    params: dict,
) -> tuple[tuple, np.ndarray | None]:
    t_all = perf_counter()
    phi = fields["phi"]
    emag = fields["emag"]
    t = perf_counter()
    ekappa2 = None
    if bool(params["LION"]):
        ek = _fused("ekappa2_values")
        if ek is not None:
            ekappa2 = ek(
                phi,
                np.ascontiguousarray(s_ion),
                float(params["ZBETA"]),
                float(params["theta_b"]),
                float(params["n_max"]),
                float(params["alpha0_ion"]),
                1 if bool(params["LNLION"]) else 0,
            )
        else:
            x_ion = float(params["ZBETA"]) * phi
            theta = float(params["theta_b"])
            if bool(params["LNLION"]):
                ekappa2 = np.zeros_like(phi)
                not_large = np.abs(x_ion) <= 100.0
                x2 = np.empty_like(phi)
                small = np.abs(x_ion) < 2.0e-4
                x2[small] = 0.5 * x_ion[small] ** 2
                x2[~small] = np.cosh(np.clip(x_ion[~small], -100.0, 100.0)) - 1.0
                ekappa2[not_large] = (
                    1.0 + (1.0 - theta) * x2[not_large]
                ) / (1.0 + theta * x2[not_large]) ** 2
            else:
                ekappa2 = np.ones_like(phi)
            ekappa2 = float(params["n_max"]) * float(params["alpha0_ion"]) * s_ion * ekappa2
    _add_timing("nlpb_ekappa2", perf_counter() - t)

    t = perf_counter()
    if bool(params["LNLDIEL"]):
        cr = _fused("chi_response")
        if cr is not None:
            chi_perp, chi_factor = cr(
                emag,
                np.ascontiguousarray(s_diel),
                float(params["PBETA"]),
                float(params["alpha_pol"]),
                float(params["alpha0_rot"]),
                float(params["invalpha_sic"]),
                float(params["N_MOL"]),
            )
        else:
            x = float(params["PBETA"]) * emag
            chi_par = np.empty_like(phi)
            chi_perp = np.empty_like(phi)
            small = x < 2.0e-4
            large = x > 100.0
            mid = ~(small | large)
            chi_par[small] = 1.0
            chi_perp[small] = 1.0
            chi_par[large] = 3.0 / (x[large] ** 2)
            chi_perp[large] = 3.0 * (1.0 - 1.0 / x[large]) / x[large]
            chi_par[mid] = 3.0 * (1.0 / x[mid] ** 2 - 1.0 / np.sinh(x[mid]) ** 2)
            chi_perp[mid] = 3.0 * (1.0 / np.tanh(x[mid]) - 1.0 / x[mid]) / x[mid]
            chi_par = float(params["alpha_pol"]) + float(params["alpha0_rot"]) * chi_par
            chi_perp = float(params["alpha_pol"]) + float(params["alpha0_rot"]) * chi_perp
            chi_par = float(params["N_MOL"]) * s_diel / (1.0 / chi_par - float(params["invalpha_sic"]))
            chi_perp = float(params["N_MOL"]) * s_diel / (1.0 / chi_perp - float(params["invalpha_sic"]))
            inv_e2 = np.zeros_like(phi)
            nz = emag >= 2.0e-4 / max(float(params["PBETA"]), 1.0e-300)
            inv_e2[nz] = 1.0 / (emag[nz] ** 2)
            chi_factor = (chi_par - chi_perp) * inv_e2
        response = ("tensor_field", chi_perp, chi_factor, fields["ex"], fields["ey"], fields["ez"])
    else:
        alpha = float(params["alpha0_rot"]) + float(params["alpha_pol"])
        scalar = float(params["N_MOL"]) * s_diel / (1.0 / alpha - float(params["invalpha_sic"]))
        response = ("scalar", scalar)
    _add_timing("nlpb_response", perf_counter() - t)
    _add_timing("nlpb_total_response", perf_counter() - t_all)
    return response, ekappa2


def nlpb_quantities(
    phi: np.ndarray,
    s_ion: np.ndarray,
    s_diel: np.ndarray,
    grid: Grid,
    params: dict,
    w_b: np.ndarray | None = None,
    need_response: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple | None, np.ndarray | None, np.ndarray]:
    if w_b is None:
        t = perf_counter()
        w_b = normalized_gaussian_kernel_g(grid, float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"]))
        _add_timing("nlpb_build_wb", perf_counter() - t)
    fields = nlpb_field_quantities(phi, s_ion, s_diel, grid, params, w_b)
    if not need_response:
        return fields["n_b"], fields["n_ion"], None, None, w_b
    response, ekappa2 = nlpb_response_from_fields(fields, s_ion, s_diel, grid, params)
    return fields["n_b"], fields["n_ion"], response, ekappa2, w_b


def minimize_l(
    resid_g: np.ndarray,
    response: tuple,
    ekappa2: np.ndarray | None,
    w_b: np.ndarray,
    grid: Grid,
    tol: float,
    max_iter: int = 200,
) -> tuple[np.ndarray, float, int]:
    t_all = perf_counter()
    _, _, _, gsq = grid.reciprocal_mesh()
    precond = np.zeros(grid.shape, dtype=float)
    mask = gsq > 0.0
    precond[mask] = EDEPS / (TPI**2 * gsq[mask]) / grid.volume
    rm = _fused("rmulc")
    xpb = _fused("cxpby")
    axpy = _fused("caxpy")
    dpf = _fused("dprod_rc")

    def _zmul(rr):
        return rm(precond, rr) if rm is not None else precond * rr

    def _dot(a, b):
        return dpf(a, b) if dpf is not None else dprod_rc(a, b)

    dphi = np.zeros(grid.shape, dtype=complex)
    r = np.ascontiguousarray(resid_g).copy()
    z = _zmul(r)
    lp0 = None
    lambda0 = 0.0
    if ekappa2 is not None:
        lp0 = np.ascontiguousarray(grid.fft(ekappa2) * grid.volume / EDEPS)
        lambda0 = get_g0(lp0)
        r0 = get_g0(r)
        if abs(lambda0) > 0.0:
            alpha0 = r0 / lambda0
            set_g0(dphi, alpha0)
            if axpy is not None:
                axpy(r, lp0, -alpha0)
            else:
                r = r - alpha0 * lp0
            z = _zmul(r)
    p = None
    rmr_old = 0.0
    rms = float(np.sqrt(max(_dot(z, z), 0.0)))
    for iteration in range(1, max_iter + 1):
        rmr = _dot(r, z)
        if p is None:
            p = z.copy()
        elif xpb is not None:
            beta = rmr / rmr_old if rmr_old != 0.0 else 0.0
            xpb(p, z, beta)
        else:
            beta = rmr / rmr_old if rmr_old != 0.0 else 0.0
            p = z + beta * p
        if lp0 is not None:
            if abs(lambda0) > 0.0:
                lam = _dot(z, lp0)
                p0 = get_g0(p) - lam / lambda0
                set_g0(p, p0)
        lp = l_op(p, response, ekappa2, w_b, grid)
        plp = _dot(p, lp)
        if plp == 0.0:
            break
        alpha = rmr / plp
        if axpy is not None:
            axpy(dphi, p, alpha)
            axpy(r, lp, -alpha)
        else:
            dphi = dphi + alpha * p
            r = r - alpha * lp
        z = _zmul(r)
        rms = float(np.sqrt(max(_dot(z, z), 0.0)))
        if rms <= tol and iteration >= 4:
            _add_timing("minimize_l_total", perf_counter() - t_all)
            return dphi, rms, iteration
        rmr_old = rmr
    _add_timing("minimize_l_total", perf_counter() - t_all)
    return dphi, rms, max_iter


def residual_g(
    phi_solv_g: np.ndarray,
    n_b_values: np.ndarray,
    n_ion_values: np.ndarray,
    q_sol: float,
    grid: Grid,
) -> tuple[np.ndarray, float]:
    resid = l0_op(phi_solv_g, grid)
    set_g0(resid, -q_sol)
    resid = grid.fft(n_b_values + n_ion_values) - resid
    cwork = l0_inv_op(resid, grid)
    rms0 = get_g0(resid)
    rms = float(np.sqrt(rms0 * rms0 + dprod_rc(cwork, cwork)))
    return resid, rms
