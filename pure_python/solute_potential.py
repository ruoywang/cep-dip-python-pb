from __future__ import annotations

from time import perf_counter

import numpy as np

from .grid import EDEPS, RYTOEV, AUTOA, TPI, Grid
from .potcar import PotcarEntry

FELECT = 2.0 * AUTOA * RYTOEV


def _mark(timings: list[tuple[str, float]] | None, label: str, t0: float) -> float:
    t1 = perf_counter()
    if timings is not None:
        timings.append((label, t1 - t0))
    return t1


def integer_mesh(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    hx = np.fft.fftfreq(nx) * nx
    hy = np.fft.fftfreq(ny) * ny
    hz = np.fft.fftfreq(nz) * nz
    return np.meshgrid(hx, hy, hz, indexing="ij")


def structure_factor_for_positions(shape: tuple[int, int, int], positions_direct: np.ndarray) -> np.ndarray:
    nx, ny, nz = shape
    if len(positions_direct) == 0:
        return np.zeros(shape, dtype=complex)
    hx = np.fft.fftfreq(nx) * nx
    hy = np.fft.fftfreq(ny) * ny
    hz = np.fft.fftfreq(nz) * nz
    positions_direct = np.asarray(positions_direct, dtype=float)
    ex = np.exp(-1j * TPI * positions_direct[:, 0, None] * hx[None, :])
    ey = np.exp(-1j * TPI * positions_direct[:, 1, None] * hy[None, :])
    ez = np.exp(-1j * TPI * positions_direct[:, 2, None] * hz[None, :])
    return np.einsum("ah,ak,al->hkl", ex, ey, ez, optimize=True)


def _interp_prho(values: np.ndarray, g_abs: np.ndarray, psgmax: float) -> np.ndarray:
    out = np.zeros_like(g_abs, dtype=float)
    dq = psgmax / len(values)
    argsc = len(values) / psgmax
    mask0 = g_abs == 0.0
    out[mask0] = values[0]
    mask = (g_abs != 0.0) & (g_abs < psgmax - 3.0 * dq)
    arg = g_abs[mask] * argsc + 1.0
    naddr = np.maximum(arg.astype(int), 2)
    rem = arg - naddr
    i = naddr - 1
    v1 = values[i - 1]
    v2 = values[i]
    v3 = values[i + 1]
    v4 = values[i + 2]
    t0 = v2
    t1 = ((6.0 * v3) - (2.0 * v1) - (3.0 * v2) - v4) / 6.0
    t2 = (v1 + v3 - (2.0 * v2)) / 2.0
    t3 = (v4 - v1 + (3.0 * (v2 - v3))) / 6.0
    out[mask] = t0 + rem * (t1 + rem * (t2 + rem * t3))
    return out


def _interp_psp(entry: PotcarEntry, g_abs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g_abs, dtype=float)
    dq = entry.psgmax / len(entry.psp_values)
    argsc = len(entry.psp_values) / entry.psgmax
    mask = (g_abs != 0.0) & (g_abs < entry.psgmax - dq)
    arg = g_abs[mask] * argsc
    i = arg.astype(int)
    rem = g_abs[mask] - entry.psp_spline[i, 0]
    p = entry.psp_spline
    out[mask] = p[i, 1] + rem * (p[i, 2] + rem * (p[i, 3] + rem * p[i, 4]))
    return out


def type_slices(counts: list[int]) -> list[slice]:
    out: list[slice] = []
    start = 0
    for count in counts:
        out.append(slice(start, start + count))
        start += count
    return out


def dencor_values(
    grid: Grid,
    entries: list[PotcarEntry],
    counts: list[int],
    positions_direct: np.ndarray,
    structure_factors: list[np.ndarray] | None = None,
) -> np.ndarray:
    _, _, _, gsq = grid.reciprocal_mesh_full()
    g_abs = np.sqrt(gsq) * TPI
    dens_g = np.zeros(grid.shape, dtype=complex)
    slices = type_slices(counts)
    if structure_factors is None:
        structure_factors = [structure_factor_for_positions(grid.shape, positions_direct[slc]) for slc in slices]
    for entry, sf in zip(entries, structure_factors):
        if entry.pspcor is None:
            continue
        dens_g += _interp_prho(entry.pspcor, g_abs, entry.psgmax) * sf
    return grid.ifft_real_full(dens_g)


def hartree_potential_g(charge_g: np.ndarray, grid: Grid) -> np.ndarray:
    _, _, _, gsq = grid.reciprocal_mesh_full()
    out = np.zeros_like(charge_g, dtype=complex)
    mask = gsq > 0.0
    scale = EDEPS / grid.volume / (TPI**2)
    out[mask] = charge_g[mask] / gsq[mask] * scale
    return out


def local_pseudopotential_g(
    grid: Grid,
    entries: list[PotcarEntry],
    counts: list[int],
    positions_direct: np.ndarray,
    structure_factors: list[np.ndarray] | None = None,
) -> np.ndarray:
    _, _, _, gsq = grid.reciprocal_mesh_full()
    g_abs = np.sqrt(gsq) * TPI
    cvps = np.zeros(grid.shape, dtype=complex)
    slices = type_slices(counts)
    if structure_factors is None:
        structure_factors = [structure_factor_for_positions(grid.shape, positions_direct[slc]) for slc in slices]
    for entry, sf in zip(entries, structure_factors):
        vpst = _interp_psp(entry, g_abs)
        zz = -4.0 * np.pi * entry.zval * FELECT
        term = np.zeros(grid.shape, dtype=float)
        mask = g_abs != 0.0
        term[mask] = (vpst[mask] + zz / (g_abs[mask] ** 2)) / grid.volume
        cvps += term * sf
    return cvps


def solute_potential_g(
    grid: Grid,
    valence_values: np.ndarray,
    entries: list[PotcarEntry],
    counts: list[int],
    positions_direct: np.ndarray,
    timings: list[tuple[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t = perf_counter()
    valence_g = grid.fft_full(valence_values)
    t = _mark(timings, "solute_fft_valence", t)
    slices = type_slices(counts)
    structure_factors = [structure_factor_for_positions(grid.shape, positions_direct[slc]) for slc in slices]
    t = _mark(timings, "solute_structure_factors", t)
    dencor = dencor_values(grid, entries, counts, positions_direct, structure_factors)
    t = _mark(timings, "solute_dencor", t)
    hartree_g = hartree_potential_g(valence_g, grid)
    t = _mark(timings, "solute_hartree", t)
    local_g = local_pseudopotential_g(grid, entries, counts, positions_direct, structure_factors)
    _mark(timings, "solute_local_potential", t)
    return (
        hartree_g + local_g,
        dencor,
    )
