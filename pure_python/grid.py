from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import os

import numpy as np

try:
    from . import _pb_fast
except Exception:  # pragma: no cover - optional acceleration extension.
    _pb_fast = None

try:
    from scipy import fft as scipy_fft
except Exception:  # pragma: no cover - NumPy fallback for minimal environments.
    scipy_fft = None

try:
    import pyfftw
    from pyfftw.interfaces import numpy_fft as pyfftw_fft

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(300)
except Exception:  # pragma: no cover - optional acceleration backend.
    pyfftw = None
    pyfftw_fft = None

try:
    import mkl_fft
except Exception:  # pragma: no cover - optional acceleration backend.
    mkl_fft = None


TPI = 2.0 * np.pi
AUTOA = 0.529177249
RYTOEV = 13.605826
BOLKEV = 8.6173857e-5
EDEPS = 4.0 * np.pi * 2.0 * RYTOEV * AUTOA
MOLAR = 6.022e-4


def _fft_workers() -> int:
    for name in ("PB_FFT_WORKERS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1


def _fft_backend() -> str:
    return os.environ.get("PB_FFT_BACKEND", "scipy").strip().lower()


def fused_kernel(name: str):
    if _pb_fast is None:
        return None
    if os.environ.get("PB_DISABLE_FUSED", "0") not in ("", "0", "false", "False"):
        return None
    return getattr(_pb_fast, name, None)


@dataclass(frozen=True)
class Grid:
    cell: np.ndarray
    shape: tuple[int, int, int]

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def ngrid(self) -> int:
        return int(np.prod(self.shape))

    @cached_property
    def reciprocal_no_2pi(self) -> np.ndarray:
        # Rows b_i satisfy a_i dot b_j = delta_ij. Fortran multiplies by TPI separately.
        return np.linalg.inv(self.cell).T

    def fft(self, real_values: np.ndarray) -> np.ndarray:
        # VASP's FFT3D_RL2RC stores the normalized reciprocal coefficients.
        if _fft_backend() == "mkl" and mkl_fft is not None:
            return mkl_fft.fftn(real_values) / self.ngrid
        if _fft_backend() == "pyfftw" and pyfftw_fft is not None:
            return (
                pyfftw_fft.fftn(
                    real_values,
                    threads=_fft_workers(),
                    planner_effort="FFTW_ESTIMATE",
                )
                / self.ngrid
            )
        if scipy_fft is not None:
            return scipy_fft.fftn(real_values, workers=_fft_workers()) / self.ngrid
        return np.fft.fftn(real_values) / self.ngrid

    def ifft_real(self, recip_values: np.ndarray) -> np.ndarray:
        # Inverse of the normalized convention above. Returns a contiguous
        # array (the .real of a complex ifft is a strided view, which the
        # fused C kernels cannot accept).
        if _fft_backend() == "mkl" and mkl_fft is not None:
            return np.ascontiguousarray(mkl_fft.ifftn(recip_values * self.ngrid).real)
        if _fft_backend() == "pyfftw" and pyfftw_fft is not None:
            return np.ascontiguousarray(
                pyfftw_fft.ifftn(
                    recip_values * self.ngrid,
                    threads=_fft_workers(),
                    planner_effort="FFTW_ESTIMATE",
                ).real
            )
        if scipy_fft is not None:
            return np.ascontiguousarray(scipy_fft.ifftn(recip_values * self.ngrid, workers=_fft_workers()).real)
        return np.ascontiguousarray(np.fft.ifftn(recip_values * self.ngrid).real)

    def fractional_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.shape
        return np.meshgrid(
            np.arange(nx, dtype=float) / nx,
            np.arange(ny, dtype=float) / ny,
            np.arange(nz, dtype=float) / nz,
            indexing="ij",
        )

    def cartesian_z_mesh(self) -> np.ndarray:
        return self._cartesian_z_mesh

    @cached_property
    def _cartesian_z_mesh(self) -> np.ndarray:
        nx, ny, nz = self.shape
        fx = (np.arange(nx, dtype=float) / nx)[:, None, None]
        fy = (np.arange(ny, dtype=float) / ny)[None, :, None]
        fz = (np.arange(nz, dtype=float) / nz)[None, None, :]
        return fx * self.cell[0, 2] + fy * self.cell[1, 2] + fz * self.cell[2, 2]

    def reciprocal_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._reciprocal_mesh

    @cached_property
    def _reciprocal_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.shape
        hx = np.fft.fftfreq(nx) * nx
        hy = np.fft.fftfreq(ny) * ny
        hz = np.fft.fftfreq(nz) * nz
        h, k, l = np.meshgrid(hx, hy, hz, indexing="ij")
        b = self.reciprocal_no_2pi
        gx = h * b[0, 0] + k * b[1, 0] + l * b[2, 0]
        gy = h * b[0, 1] + k * b[1, 1] + l * b[2, 1]
        gz = h * b[0, 2] + k * b[1, 2] + l * b[2, 2]
        gsq = gx * gx + gy * gy + gz * gz
        return gx, gy, gz, gsq

    def periodic_distance_from_origin(self) -> np.ndarray:
        return self._periodic_distance_from_origin

    @cached_property
    def _periodic_distance_from_origin(self) -> np.ndarray:
        nx, ny, nz = self.shape
        fx = (np.mod(np.arange(nx, dtype=float) / nx + 0.5, 1.0) - 0.5)[:, None, None]
        fy = (np.mod(np.arange(ny, dtype=float) / ny + 0.5, 1.0) - 0.5)[None, :, None]
        fz = (np.mod(np.arange(nz, dtype=float) / nz + 0.5, 1.0) - 0.5)[None, None, :]
        x = fx * self.cell[0, 0] + fy * self.cell[1, 0] + fz * self.cell[2, 0]
        y = fx * self.cell[0, 1] + fy * self.cell[1, 1] + fz * self.cell[2, 1]
        z = fx * self.cell[0, 2] + fy * self.cell[1, 2] + fz * self.cell[2, 2]
        return np.sqrt(x * x + y * y + z * z)

    def grad_from_recip(self, phi_g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gx, gy, gz, _ = self.reciprocal_mesh()
        premul = fused_kernel("grad_premul")
        mag3 = fused_kernel("magnitude3")
        if premul is not None and mag3 is not None:
            ax, ay, az = premul(gx, gy, gz, np.ascontiguousarray(phi_g))
            ex = self.ifft_real(ax)
            ey = self.ifft_real(ay)
            ez = self.ifft_real(az)
            mag = mag3(ex, ey, ez)
            return ex, ey, ez, mag
        ex = self.ifft_real(1j * TPI * gx * phi_g)
        ey = self.ifft_real(1j * TPI * gy * phi_g)
        ez = self.ifft_real(1j * TPI * gz * phi_g)
        mag = np.sqrt(ex * ex + ey * ey + ez * ez)
        return ex, ey, ez, mag

    def div_real_vector(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        gx, gy, gz, _ = self.reciprocal_mesh()
        acc = fused_kernel("div_accum")
        if acc is not None:
            out = np.zeros(self.shape, dtype=complex)
            f0 = np.ascontiguousarray(self.fft(vx))
            f1 = np.ascontiguousarray(self.fft(vy))
            f2 = np.ascontiguousarray(self.fft(vz))
            acc(out, gx, gy, gz, f0, f1, f2)
            return out
        return 1j * TPI * (
            gx * self.fft(vx)
            + gy * self.fft(vy)
            + gz * self.fft(vz)
        )


def normalized_gaussian_kernel_g(grid: Grid, sigma: float) -> np.ndarray:
    r = grid.periodic_distance_from_origin()
    real = np.exp(-0.5 * (r / sigma) ** 2) / (sigma * np.sqrt(TPI)) ** 3
    real *= grid.ngrid / real.sum()
    return grid.fft(real)


def exp_kernel_g(grid: Grid, r_c: float, sigma: float) -> np.ndarray:
    r = grid.periodic_distance_from_origin()
    cutoff = 100.0
    real = 1.0 / (1.0 / cutoff + np.exp((r - r_c) / sigma))
    real /= 4.0 * TPI * sigma**3
    real *= 4.0 / (2.0 + r_c / sigma)
    real *= grid.ngrid / grid.volume
    return grid.fft(real)


def convolve_real(field: np.ndarray, kernel_g: np.ndarray, grid: Grid) -> np.ndarray:
    return grid.ifft_real(grid.fft(field) * kernel_g)


def xcorr_real(field: np.ndarray, kernel_g: np.ndarray, grid: Grid) -> np.ndarray:
    return grid.ifft_real(grid.fft(field) * np.conj(kernel_g))


def l0_op(phi_solv_g: np.ndarray, grid: Grid) -> np.ndarray:
    _, _, _, gsq = grid.reciprocal_mesh()
    return phi_solv_g * (TPI**2 * gsq) * grid.volume / EDEPS


def l0_inv_op(source_g: np.ndarray, grid: Grid) -> np.ndarray:
    _, _, _, gsq = grid.reciprocal_mesh()
    out = np.zeros_like(source_g, dtype=complex)
    mask = gsq > 0.0
    out[mask] = source_g[mask] / (TPI**2 * gsq[mask]) * EDEPS / grid.volume
    return out
