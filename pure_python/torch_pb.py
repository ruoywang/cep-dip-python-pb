"""GPU (torch) port of the nonlinear Poisson-Boltzmann Newton-PCG solver.

Line-for-line translation of the numpy solver (`grid.py`, `pb.py`,
`solve_from_chgcar_newton.solve_nlpb_for_phi_sol`) onto torch tensors so
the whole solve runs on the GPU with no host round-trip. Physics, unit
conventions and the FFT amplitude normalization are identical to the numpy
path; only the array backend changes.

Scope: the per-solve hot path (cavity construction + Newton outer loop +
preconditioned-CG inner loop). Scalar parameter derivation
(`pb.derived_params`) is reused from the numpy module verbatim. The solute
potential setup (POTCAR Hartree/local-pseudopotential) stays on the numpy
side / is replaced by the MACE Gaussian surrogate; its result enters here
as the ready-made `phi_sol` tensor.

Design choices:
- both full and half spectrum (rfft) supported via TorchGrid(rspec=...);
  rspec=None reads PB_RFFT like the numpy Grid. Numerics match numpy
  bit-for-bit per mode (component test covers both).
- float64 by default to clear the cal_18 validation gate; float32 diverges
  on this ill-conditioned problem (documented, use f64).
- the fused C-kernel paths of the numpy solver are irrelevant here (pure
  torch); numerics follow the numpy fallback branches exactly.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch

from .grid import AUTOA, BOLKEV, EDEPS, MOLAR, RYTOEV, TPI

__all__ = [
    "TorchGrid",
    "create_cavity_torch",
    "solve_nlpb_for_phi_sol_torch",
    "ion_density_values_from_phi_torch",
    "make_numpy_io_solver",
]

_GRID_CACHE = {}


def make_numpy_io_solver(device="cpu", dtype=torch.float64):
    """Return a drop-in replacement for
    solve_from_chgcar_newton.solve_nlpb_for_phi_sol with the same numpy
    signature/return, running the solve on `device` via torch_pb. TorchGrid
    meshes are cached per (shape, id(cell-bytes)) so repeated fixsteps and
    the coarse grid don't rebuild reciprocal meshes."""

    def _grid_for(cell, shape):
        key = (tuple(shape), np.asarray(cell, dtype=float).tobytes(), str(device), str(dtype))
        g = _GRID_CACHE.get(key)
        if g is None:
            g = TorchGrid(cell, shape, device=device, dtype=dtype)
            _GRID_CACHE[key] = g
        return g

    def solver(phi_total, phi_sol, s_ion, s_diel, grid, params, q_sol,
               tol, max_outer, cg_max_iter, progress_path=None, fixstep=0):
        tg = _grid_for(grid.cell, grid.shape)
        pt, nb, ni, psg, hist = solve_nlpb_for_phi_sol_torch(
            torch.as_tensor(np.ascontiguousarray(phi_total), device=tg.device, dtype=dtype),
            torch.as_tensor(np.ascontiguousarray(phi_sol), device=tg.device, dtype=dtype),
            torch.as_tensor(np.ascontiguousarray(s_ion), device=tg.device, dtype=dtype),
            torch.as_tensor(np.ascontiguousarray(s_diel), device=tg.device, dtype=dtype),
            tg, params, float(q_sol), float(tol), int(max_outer), int(cg_max_iter),
        )
        to_np = lambda x: x.detach().cpu().numpy()
        return to_np(pt), to_np(nb), to_np(ni), to_np(psg), hist

    return solver


class TorchGrid:
    """torch mirror of pure_python.grid.Grid (full- and half-spectrum)."""

    def __init__(self, cell, shape, device=None, dtype=torch.float64, rspec=None):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.cell = torch.as_tensor(
            np.asarray(cell, dtype=float), device=self.device, dtype=self.dtype
        )
        self.shape = tuple(int(s) for s in shape)
        self.ngrid = int(np.prod(self.shape))
        self.volume = float(abs(np.linalg.det(np.asarray(cell, dtype=float))))
        if rspec is None:
            rspec = os.environ.get("PB_RFFT", "0") not in ("", "0", "false", "False")
        self.rspec = bool(rspec)
        nx, ny, nz = self.shape
        self.spec_shape = (nx, ny, nz // 2 + 1) if self.rspec else self.shape
        # b_i rows satisfy a_i . b_j = delta_ij (no 2pi), matching numpy Grid.
        self.recip_no_2pi = torch.as_tensor(
            np.linalg.inv(np.asarray(cell, dtype=float)).T,
            device=self.device,
            dtype=self.dtype,
        )
        self._build_meshes()

    # -- meshes -----------------------------------------------------------
    def _fftfreq_n(self, n: int) -> torch.Tensor:
        return torch.fft.fftfreq(n, device=self.device, dtype=self.dtype) * n

    def _build_meshes(self) -> None:
        nx, ny, nz = self.shape
        b = self.recip_no_2pi
        hx = self._fftfreq_n(nx)
        hy = self._fftfreq_n(ny)
        if self.rspec:
            hz = torch.arange(nz // 2 + 1, device=self.device, dtype=self.dtype)
        else:
            hz = self._fftfreq_n(nz)
        h, k, l = torch.meshgrid(hx, hy, hz, indexing="ij")
        gx = h * b[0, 0] + k * b[1, 0] + l * b[2, 0]
        gy = h * b[0, 1] + k * b[1, 1] + l * b[2, 1]
        gz = h * b[0, 2] + k * b[1, 2] + l * b[2, 2]
        self.gx, self.gy, self.gz = gx, gy, gz
        self.gsq = gx * gx + gy * gy + gz * gz

        if self.rspec:
            # Derivative premultipliers: zero each lattice axis's Nyquist
            # frequency (an i*h term there is anti-Hermitian, dropped by the
            # full-spectrum ifft(...).real; c2r would fold it back wrong).
            dhx = self._fftfreq_n(nx).clone()
            dhy = self._fftfreq_n(ny).clone()
            dhz = torch.arange(nz // 2 + 1, device=self.device, dtype=self.dtype).clone()
            if nx % 2 == 0:
                dhx[nx // 2] = 0.0
            if ny % 2 == 0:
                dhy[ny // 2] = 0.0
            if nz % 2 == 0:
                dhz[nz // 2] = 0.0
            dh, dk, dl = torch.meshgrid(dhx, dhy, dhz, indexing="ij")
            self.dgx = dh * b[0, 0] + dk * b[1, 0] + dl * b[2, 0]
            self.dgy = dh * b[0, 1] + dk * b[1, 1] + dl * b[2, 1]
            self.dgz = dh * b[0, 2] + dk * b[1, 2] + dl * b[2, 2]
            # inner-product multiplicity for the half spectrum
            w = torch.full(self.spec_shape, 2.0, device=self.device, dtype=self.dtype)
            w[:, :, 0] = 1.0
            if nz % 2 == 0:
                w[:, :, (nz // 2 + 1) - 1] = 1.0
            self.spectral_weight = w
        else:
            self.dgx, self.dgy, self.dgz = gx, gy, gz
            self.spectral_weight = None

        # cartesian z of each grid point (for the vacuum smooth box)
        fx = (torch.arange(nx, device=self.device, dtype=self.dtype) / nx)[:, None, None]
        fy = (torch.arange(ny, device=self.device, dtype=self.dtype) / ny)[None, :, None]
        fz = (torch.arange(nz, device=self.device, dtype=self.dtype) / nz)[None, None, :]
        self.z_mesh = (
            fx * self.cell[0, 2] + fy * self.cell[1, 2] + fz * self.cell[2, 2]
        )

        # periodic distance from origin (for real-space kernels)
        px = (torch.remainder(torch.arange(nx, device=self.device, dtype=self.dtype) / nx + 0.5, 1.0) - 0.5)[:, None, None]
        py = (torch.remainder(torch.arange(ny, device=self.device, dtype=self.dtype) / ny + 0.5, 1.0) - 0.5)[None, :, None]
        pz = (torch.remainder(torch.arange(nz, device=self.device, dtype=self.dtype) / nz + 0.5, 1.0) - 0.5)[None, None, :]
        x = px * self.cell[0, 0] + py * self.cell[1, 0] + pz * self.cell[2, 0]
        y = px * self.cell[0, 1] + py * self.cell[1, 1] + pz * self.cell[2, 1]
        z = px * self.cell[0, 2] + py * self.cell[1, 2] + pz * self.cell[2, 2]
        self.r_from_origin = torch.sqrt(x * x + y * y + z * z)

    # -- transforms (amplitude convention: /ngrid on forward) -------------
    def fft(self, real_values: torch.Tensor) -> torch.Tensor:
        if self.rspec:
            return torch.fft.rfftn(real_values.to(self.dtype)) / self.ngrid
        return torch.fft.fftn(real_values.to(self.cdtype)) / self.ngrid

    def ifft_real(self, recip_values: torch.Tensor) -> torch.Tensor:
        if self.rspec:
            return torch.fft.irfftn(recip_values * self.ngrid, s=self.shape).to(self.dtype)
        return torch.fft.ifftn(recip_values * self.ngrid).real.to(self.dtype)

    def to_tensor(self, a) -> torch.Tensor:
        if torch.is_tensor(a):
            return a.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(a, dtype=float), device=self.device, dtype=self.dtype)

    # -- vector calculus in reciprocal space ------------------------------
    def grad_from_recip(self, phi_g: torch.Tensor):
        ex = self.ifft_real(1j * TPI * self.dgx * phi_g)
        ey = self.ifft_real(1j * TPI * self.dgy * phi_g)
        ez = self.ifft_real(1j * TPI * self.dgz * phi_g)
        mag = torch.sqrt(ex * ex + ey * ey + ez * ez)
        return ex, ey, ez, mag

    def div_real_vector(self, vx, vy, vz) -> torch.Tensor:
        return 1j * TPI * (
            self.dgx * self.fft(vx) + self.dgy * self.fft(vy) + self.dgz * self.fft(vz)
        )

    # -- Poisson operators (l0) -------------------------------------------
    def l0_op(self, phi_solv_g: torch.Tensor) -> torch.Tensor:
        return phi_solv_g * (TPI ** 2 * self.gsq) * self.volume / EDEPS

    def l0_inv_op(self, source_g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(source_g)
        mask = self.gsq > 0.0
        out[mask] = source_g[mask] / (TPI ** 2 * self.gsq[mask]) * EDEPS / self.volume
        return out

    # -- G=0 helpers ------------------------------------------------------
    def get_g0(self, a: torch.Tensor) -> float:
        return float(a[0, 0, 0].real)

    def set_g0(self, a: torch.Tensor, value) -> None:
        a[0, 0, 0] = value

    def dprod_rc(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Full-spectrum inner product; half-spectrum doubles non-self-
        # conjugate modes via spectral_weight (matches numpy grid_dprod_rc).
        if self.spectral_weight is None:
            return (b.conj() * a).real.sum()
        return (self.spectral_weight * (a.real * b.real + a.imag * b.imag)).sum()


# -- real-space convolution kernels (torch) -------------------------------
def _normalized_gaussian_kernel_g(grid: TorchGrid, sigma: float) -> torch.Tensor:
    r = grid.r_from_origin
    real = torch.exp(-0.5 * (r / sigma) ** 2) / (sigma * np.sqrt(TPI)) ** 3
    real = real * (grid.ngrid / real.sum())
    return grid.fft(real)


def _exp_kernel_g(grid: TorchGrid, r_c: float, sigma: float) -> torch.Tensor:
    r = grid.r_from_origin
    cutoff = 100.0
    real = 1.0 / (1.0 / cutoff + torch.exp((r - r_c) / sigma))
    real = real / (4.0 * TPI * sigma ** 3)
    real = real * (4.0 / (2.0 + r_c / sigma))
    real = real * (grid.ngrid / grid.volume)
    return grid.fft(real)


def _convolve_real(field: torch.Tensor, kernel_g: torch.Tensor, grid: TorchGrid) -> torch.Tensor:
    return grid.ifft_real(grid.fft(field) * kernel_g)


def _shape_func(x: torch.Tensor, sigma_k: float) -> torch.Tensor:
    return 0.5 * torch.erfc(x / (np.sqrt(2.0) * sigma_k))


def _smooth_box(grid: TorchGrid, z0: float, z1: float, sigma: float) -> torch.Tensor:
    z = grid.z_mesh
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    mask = 0.5 * (
        torch.erfc((z - z1) / (inv_sqrt2 * sigma))
        - torch.erfc((z - z0) / (inv_sqrt2 * sigma))
    )
    return torch.clamp(mask, 0.0, 1.0)


def create_cavity_torch(
    n_e_density: torch.Tensor, grid: TorchGrid, params: dict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_k = float(params["SIGMA_K"])
    a_k = float(params["A_K"])
    n_min = float(params["N_MIN"])
    ne = grid.to_tensor(n_e_density)
    x_vdw = torch.log(torch.clamp(ne / float(params["NC_K"]), min=n_min))
    s_vdw = _shape_func(x_vdw, sigma_k)

    def maybe_exp(radius: float):
        if radius <= 0.0:
            return None
        return _exp_kernel_g(grid, radius, a_k / sigma_k)

    w_ion = maybe_exp(float(params["R_ION"]))
    w_solv = maybe_exp(float(params["R_SOLV"]))
    w_cav = maybe_exp(float(params["R_CAV"]))
    w_diel = maybe_exp(float(params["R_DIEL"]))

    if w_ion is not None:
        x_ion = torch.log(torch.clamp(_convolve_real(1.0 - s_vdw, w_ion, grid), min=n_min))
        s_ion = _shape_func(x_ion, sigma_k)
    else:
        s_ion = s_vdw.clone()
    if w_solv is not None:
        x_solv = torch.log(torch.clamp(_convolve_real(1.0 - s_vdw, w_solv, grid), min=n_min))
        s_solv = _shape_func(x_solv, sigma_k)
    else:
        s_solv = s_vdw.clone()
    if w_cav is not None:
        x_cav = torch.log(torch.clamp(_convolve_real(s_solv, w_cav, grid), min=n_min))
        s_cav = 1.0 - _shape_func(x_cav, sigma_k)
    else:
        s_cav = s_solv.clone()
    if w_diel is not None:
        x_diel = torch.log(torch.clamp(_convolve_real(s_solv, w_diel, grid), min=n_min))
        s_diel = 1.0 - _shape_func(x_diel, sigma_k)
    else:
        s_diel = s_solv.clone()

    if bool(params["LVAC"]) and float(params["SOL_Z1"]) > float(params["SOL_Z0"]):
        m_sol = _smooth_box(grid, float(params["SOL_Z0"]), float(params["SOL_Z1"]), float(params["SOL_SIGMA"]))
        m_ion = _smooth_box(
            grid,
            float(params["SOL_Z0"]) + float(params["D_STERN"]),
            float(params["SOL_Z1"]) - float(params["D_STERN"]),
            float(params["SOL_SIGMA"]),
        )
        s_ion = s_ion * m_ion
        s_cav = s_cav * m_ion
        s_diel = s_diel * m_sol
    return s_ion, s_diel, s_cav


def _local_field_factor(e_mag: torch.Tensor, params: dict) -> torch.Tensor:
    alpha_pol = float(params["alpha_pol"])
    alpha0_rot = float(params["alpha0_rot"])
    invalpha_sic = float(params["invalpha_sic"])
    if not bool(params["LNLDIEL"]):
        return torch.full_like(e_mag, 1.0 / (1.0 - (alpha_pol + alpha0_rot) * invalpha_sic))
    lo_scalar = 1.0 / (1.0 - alpha_pol * invalpha_sic)
    hi_scalar = 1.0 / (1.0 - (alpha_pol + alpha0_rot) * invalpha_sic)
    hi_scalar *= 1.0 + 1.0e-8
    x0 = float(params["PBETA"]) * e_mag

    def g_rot(x: torch.Tensor) -> torch.Tensor:
        out = torch.ones_like(x)
        small = x < 2.0e-4
        xs = x[~small]
        out[~small] = 3.0 * (xs - torch.tanh(xs)) / (xs * xs * torch.tanh(xs))
        return out

    out = torch.full_like(e_mag, hi_scalar)
    zero = x0 == 0.0
    for _ in range(80):
        gx = g_rot(out * x0)
        new = 1.0 / (1.0 - (gx * alpha0_rot + alpha_pol) * invalpha_sic)
        new = torch.clamp(new, lo_scalar, hi_scalar)
        diff = torch.max(torch.abs(new - out))
        out = new
        if float(diff) <= 1.0e-10 * max(1.0, hi_scalar):
            break
    out[zero] = hi_scalar
    return out


def ion_density_values_from_phi_torch(
    phi: torch.Tensor, s_ion: torch.Tensor, grid: TorchGrid, params: dict
) -> torch.Tensor:
    x = float(params["ZBETA"]) * phi
    theta = float(params["theta_b"])
    if bool(params["LNLION"]) and theta > 0.0:
        n_work = torch.empty_like(phi)
        ax = torch.abs(x)
        large = ax > 100.0
        small = ax < np.sqrt(theta) * 2.0e-4
        mid = ~(large | small)
        n_work[large] = torch.sign(x[large])
        n_work[small] = theta * x[small]
        xm = x[mid]
        denom = 1.0 + theta * (torch.cosh(xm) - 1.0)
        n_work[mid] = theta * torch.sinh(xm) / denom
    elif bool(params["LNLION"]):
        n_work = torch.sinh(torch.clamp(x, -100.0, 100.0))
    else:
        n_work = x
    rho_ion = -float(params["n_max"]) * float(params["invBETA"]) * float(params["ZBETA"]) * s_ion * n_work
    return rho_ion * grid.volume


def _dielectric_g(y: torch.Tensor, params: dict) -> torch.Tensor:
    if not bool(params["LNLDIEL"]):
        return torch.ones_like(y)
    g = torch.empty_like(y)
    small = y < 2.0e-4
    large = y > 100.0
    mid = ~(small | large)
    g[small] = 1.0
    yl = y[large]
    g[large] = 3.0 * (1.0 - 1.0 / yl) / yl
    ym = y[mid]
    g[mid] = 3.0 * (1.0 / torch.tanh(ym) - 1.0 / ym) / ym
    return g


def _field_quantities(phi, s_ion, s_diel, grid: TorchGrid, params: dict, w_b):
    phi_g = grid.fft(phi)
    ex, ey, ez, emag = grid.grad_from_recip(-torch.conj(w_b) * phi_g)
    f_loc = _local_field_factor(emag, params)
    ex = ex * f_loc
    ey = ey * f_loc
    ez = ez * f_loc
    emag = emag * f_loc
    n_ion = ion_density_values_from_phi_torch(phi, s_ion, grid, params)
    y = float(params["PBETA"]) * emag
    g = _dielectric_g(y, params)
    polar_over_eps = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
    p_over_e = float(params["N_MOL"]) * s_diel * polar_over_eps
    div_p_g = grid.div_real_vector(p_over_e * ex, p_over_e * ey, p_over_e * ez)
    n_b = grid.ifft_real(-w_b * div_p_g) * grid.volume
    return {"phi": phi, "ex": ex, "ey": ey, "ez": ez, "emag": emag, "n_b": n_b, "n_ion": n_ion}


def _response_from_fields(fields, s_ion, s_diel, grid: TorchGrid, params: dict):
    phi = fields["phi"]
    emag = fields["emag"]
    ekappa2 = None
    if bool(params["LION"]):
        x_ion = float(params["ZBETA"]) * phi
        theta = float(params["theta_b"])
        if bool(params["LNLION"]):
            ekappa2 = torch.zeros_like(phi)
            ax = torch.abs(x_ion)
            not_large = ax <= 100.0
            x2 = torch.empty_like(phi)
            small = ax < 2.0e-4
            x2[small] = 0.5 * x_ion[small] ** 2
            ns = ~small
            x2[ns] = torch.cosh(torch.clamp(x_ion[ns], -100.0, 100.0)) - 1.0
            val = (1.0 + (1.0 - theta) * x2) / (1.0 + theta * x2) ** 2
            ekappa2[not_large] = val[not_large]
        else:
            ekappa2 = torch.ones_like(phi)
        ekappa2 = float(params["n_max"]) * float(params["alpha0_ion"]) * s_ion * ekappa2

    if bool(params["LNLDIEL"]):
        x = float(params["PBETA"]) * emag
        chi_par = torch.empty_like(phi)
        chi_perp = torch.empty_like(phi)
        small = x < 2.0e-4
        large = x > 100.0
        mid = ~(small | large)
        chi_par[small] = 1.0
        chi_perp[small] = 1.0
        xl = x[large]
        chi_par[large] = 3.0 / (xl ** 2)
        chi_perp[large] = 3.0 * (1.0 - 1.0 / xl) / xl
        xm = x[mid]
        chi_par[mid] = 3.0 * (1.0 / xm ** 2 - 1.0 / torch.sinh(xm) ** 2)
        chi_perp[mid] = 3.0 * (1.0 / torch.tanh(xm) - 1.0 / xm) / xm
        ap = float(params["alpha_pol"])
        a0 = float(params["alpha0_rot"])
        isic = float(params["invalpha_sic"])
        nmol = float(params["N_MOL"])
        chi_par = ap + a0 * chi_par
        chi_perp = ap + a0 * chi_perp
        chi_par = nmol * s_diel / (1.0 / chi_par - isic)
        chi_perp = nmol * s_diel / (1.0 / chi_perp - isic)
        inv_e2 = torch.zeros_like(phi)
        thr = 2.0e-4 / max(float(params["PBETA"]), 1.0e-300)
        nz = emag >= thr
        inv_e2[nz] = 1.0 / (emag[nz] ** 2)
        chi_factor = (chi_par - chi_perp) * inv_e2
        response = ("tensor_field", chi_perp, chi_factor, fields["ex"], fields["ey"], fields["ez"])
    else:
        alpha = float(params["alpha0_rot"]) + float(params["alpha_pol"])
        scalar = float(params["N_MOL"]) * s_diel / (1.0 / alpha - float(params["invalpha_sic"]))
        response = ("scalar", scalar)
    return response, ekappa2


def _lapl_tensor(phi_g, response, grid: TorchGrid) -> torch.Tensor:
    g_list = [grid.dgx, grid.dgy, grid.dgz]
    grads = [
        grid.ifft_real(1j * TPI * grid.dgx * phi_g),
        grid.ifft_real(1j * TPI * grid.dgy * phi_g),
        grid.ifft_real(1j * TPI * grid.dgz * phi_g),
    ]
    out = torch.zeros(grid.spec_shape, dtype=grid.cdtype, device=grid.device)
    kind = response[0]
    if kind == "scalar":
        scalar = response[1]
        for i in range(3):
            out = out + 1j * TPI * g_list[i] * grid.fft(scalar * grads[i])
    elif kind == "tensor_field":
        chi_perp, chi_factor, ex, ey, ez = response[1:]
        e = [ex, ey, ez]
        dot = ex * grads[0] + ey * grads[1] + ez * grads[2]
        for i in range(3):
            work = chi_perp * grads[i] + chi_factor * e[i] * dot
            out = out + 1j * TPI * g_list[i] * grid.fft(work)
    else:
        raise ValueError(f"unknown dielectric response kind: {kind}")
    return out


def _l_op(dphi_g, response, ekappa2, w_b, grid: TorchGrid) -> torch.Tensor:
    cwork = torch.conj(w_b) * dphi_g
    lp = w_b * _lapl_tensor(cwork, response, grid)
    lp = lp - (TPI ** 2) * grid.gsq * dphi_g
    if ekappa2 is not None:
        real = grid.ifft_real(dphi_g)
        lp = lp + grid.fft(-ekappa2 * real)
    return -grid.volume / EDEPS * lp


def _residual_g(phi_solv_g, n_b, n_ion, q_sol, grid: TorchGrid):
    resid = grid.l0_op(phi_solv_g)
    grid.set_g0(resid, -q_sol)
    resid = grid.fft(n_b + n_ion) - resid
    cwork = grid.l0_inv_op(resid)
    rms0 = grid.get_g0(resid)
    rms = float(torch.sqrt(rms0 * rms0 + grid.dprod_rc(cwork, cwork)))
    return resid, rms


def _minimize_l(resid_g, response, ekappa2, w_b, grid: TorchGrid, tol, max_iter=200):
    precond = torch.zeros(grid.spec_shape, dtype=grid.dtype, device=grid.device)
    mask = grid.gsq > 0.0
    precond[mask] = EDEPS / (TPI ** 2 * grid.gsq[mask]) / grid.volume

    def zmul(rr):
        return precond * rr

    def dot(a, b):
        return grid.dprod_rc(a, b)

    dphi = torch.zeros(grid.spec_shape, dtype=grid.cdtype, device=grid.device)
    r = resid_g.clone()
    z = zmul(r)
    lp0 = None
    lambda0 = 0.0
    if ekappa2 is not None:
        lp0 = grid.fft(ekappa2) * grid.volume / EDEPS
        lambda0 = grid.get_g0(lp0)
        r0 = grid.get_g0(r)
        if abs(lambda0) > 0.0:
            alpha0 = r0 / lambda0
            grid.set_g0(dphi, alpha0)
            r = r - alpha0 * lp0
            z = zmul(r)
    p = None
    rmr_old = 0.0
    rms = float(torch.sqrt(torch.clamp(dot(z, z), min=0.0)))
    for iteration in range(1, max_iter + 1):
        rmr = dot(r, z)
        if p is None:
            p = z.clone()
        else:
            beta = rmr / rmr_old if rmr_old != 0.0 else torch.zeros((), device=grid.device, dtype=grid.dtype)
            p = z + beta * p
        if lp0 is not None and abs(lambda0) > 0.0:
            lam = dot(z, lp0)
            p0 = grid.get_g0(p) - lam / lambda0
            grid.set_g0(p, p0)
        lp = _l_op(p, response, ekappa2, w_b, grid)
        plp = dot(p, lp)
        if float(plp) == 0.0:
            break
        alpha = rmr / plp
        dphi = dphi + alpha * p
        r = r - alpha * lp
        z = zmul(r)
        rms = float(torch.sqrt(torch.clamp(dot(z, z), min=0.0)))
        if rms <= tol and iteration >= 4:
            return dphi, rms, iteration
        rmr_old = rmr
    return dphi, rms, max_iter


def solve_nlpb_for_phi_sol_torch(
    phi_total: torch.Tensor,
    phi_sol: torch.Tensor,
    s_ion: torch.Tensor,
    s_diel: torch.Tensor,
    grid: TorchGrid,
    params: dict,
    q_sol: float,
    tol: float,
    max_outer: int,
    cg_max_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """torch port of solve_from_chgcar_newton.solve_nlpb_for_phi_sol.

    All array inputs are torch tensors on grid.device. Returns
    (phi_total, n_b, n_ion, phi_solv_g, history) with history entries
    (outer, rms, cg_iter, post_step_rms).
    """
    phi_total = grid.to_tensor(phi_total)
    phi_sol = grid.to_tensor(phi_sol)
    s_ion = grid.to_tensor(s_ion)
    s_diel = grid.to_tensor(s_diel)
    r_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    w_b = _normalized_gaussian_kernel_g(grid, r_b)
    phi_solv_g = grid.fft(phi_total - phi_sol)
    history = []
    fields = None
    for outer in range(max_outer + 1):
        if fields is None:
            fields = _field_quantities(phi_total, s_ion, s_diel, grid, params, w_b)
        n_b = fields["n_b"]
        n_ion = fields["n_ion"]
        resid, rms = _residual_g(phi_solv_g, n_b, n_ion, q_sol, grid)
        if rms < tol and outer >= 1:
            history.append((outer, rms, 0, 0.0))
            break
        response, ekappa2 = _response_from_fields(fields, s_ion, s_diel, grid, params)
        dphi_g, cg_rms, cg_iter = _minimize_l(
            resid, response, ekappa2, w_b, grid, max(rms / 10.0, tol), cg_max_iter
        )
        dphi_real = grid.ifft_real(dphi_g)
        alpha = 1.0
        accepted_rms = float("inf")
        for _ in range(7):
            trial_phi = phi_total + alpha * dphi_real
            trial_phi_solv_g = phi_solv_g + alpha * dphi_g
            trial_fields = _field_quantities(trial_phi, s_ion, s_diel, grid, params, w_b)
            _, trial_rms = _residual_g(
                trial_phi_solv_g, trial_fields["n_b"], trial_fields["n_ion"], q_sol, grid
            )
            if trial_rms <= rms or alpha <= 1.0 / 64.0:
                phi_total = trial_phi
                phi_solv_g = trial_phi_solv_g
                fields = trial_fields
                accepted_rms = trial_rms
                break
            alpha *= 0.5
        history.append((outer, rms, cg_iter, accepted_rms))
    if fields is None:
        fields = _field_quantities(phi_total, s_ion, s_diel, grid, params, w_b)
    return phi_total, fields["n_b"], fields["n_ion"], phi_solv_g, history
