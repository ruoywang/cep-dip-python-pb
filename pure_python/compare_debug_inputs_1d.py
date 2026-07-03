from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.vasp_volumetric import read_vasp_volumetric, write_profile

from .config import load_config
from .grid import EDEPS, TPI, Grid
from .potcar import read_potcar
from .solute_potential import FELECT, _interp_prho, _interp_psp, type_slices


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def reciprocal_z_line(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    nz = grid.shape[2]
    l = np.fft.fftfreq(nz) * nz
    b = grid.reciprocal_no_2pi
    gx = l * b[2, 0]
    gy = l * b[2, 1]
    gz = l * b[2, 2]
    gsq = gx * gx + gy * gy + gz * gz
    return l, gsq


def dencor_1d_raw(grid: Grid, entries, counts: list[int], positions_direct: np.ndarray) -> np.ndarray:
    nz = grid.shape[2]
    l, gsq = reciprocal_z_line(grid)
    g_abs = np.sqrt(gsq) * TPI
    dens_g = np.zeros(nz, dtype=complex)
    for entry, slc in zip(entries, type_slices(counts)):
        if entry.pspcor is None:
            continue
        z = positions_direct[slc, 2]
        sf = np.exp(-1j * TPI * np.outer(l, z)).sum(axis=1)
        dens_g += _interp_prho(entry.pspcor, g_abs, entry.psgmax) * sf
    return np.fft.ifft(dens_g * nz).real


def solute_phi_1d(
    grid: Grid,
    valence_z_raw: np.ndarray,
    entries,
    counts: list[int],
    positions_direct: np.ndarray,
) -> np.ndarray:
    nz = grid.shape[2]
    l, gsq = reciprocal_z_line(grid)
    g_abs = np.sqrt(gsq) * TPI

    val_g = np.fft.fft(valence_z_raw) / nz
    hartree = np.zeros(nz, dtype=complex)
    mask_g = gsq > 0.0
    hartree[mask_g] = val_g[mask_g] / gsq[mask_g] * EDEPS / grid.volume / (TPI**2)

    local = np.zeros(nz, dtype=complex)
    for entry, slc in zip(entries, type_slices(counts)):
        z = positions_direct[slc, 2]
        sf = np.exp(-1j * TPI * np.outer(l, z)).sum(axis=1)
        vpst = _interp_psp(entry, g_abs)
        zz = -4.0 * np.pi * entry.zval * FELECT
        term = np.zeros(nz, dtype=float)
        mask = g_abs != 0.0
        term[mask] = (vpst[mask] + zz / (g_abs[mask] ** 2)) / grid.volume
        local += term * sf
    return np.fft.ifft((hartree + local) * nz).real


def line(name: str, py: np.ndarray, ref: np.ndarray) -> list[str]:
    return [
        f"{name}_rmse\t{rmse(py, ref):.12e}",
        f"{name}_demean_rmse\t{rmse(py - py.mean(), ref - ref.mean()):.12e}",
        f"{name}_mean_py\t{py.mean():.12e}",
        f"{name}_mean_ref\t{ref.mean():.12e}",
        f"{name}_maxabs\t{np.max(np.abs(py - ref)):.12e}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--debug-dir", default="reproduce3d/debug_cal18_nelm1")
    parser.add_argument("--out-dir", default="pure_python/results/debug_input_compare_1d")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    grid = Grid(chg.cell, chg.grid)
    entries = read_potcar(args.potcar)
    positions = np.asarray(cfg["positions_direct"], dtype=float)
    counts = list(cfg["counts"])

    valence = chg.values.reshape(chg.grid, order="F")
    val_z = valence.mean(axis=(0, 1))
    dencor_z = dencor_1d_raw(grid, entries, counts, positions)
    phisol_z = solute_phi_1d(grid, val_z, entries, counts, positions)

    debug_dir = Path(args.debug_dir)
    dencor_ref = read_vasp_volumetric(debug_dir / "DBG_DENCOR").values.reshape(chg.grid, order="F").mean(axis=(0, 1))
    nval_ref = read_vasp_volumetric(debug_dir / "DBG_NVAL").values.reshape(chg.grid, order="F").mean(axis=(0, 1))
    ne_ref = read_vasp_volumetric(debug_dir / "DBG_NE").values.reshape(chg.grid, order="F").mean(axis=(0, 1))
    phisol_ref = read_vasp_volumetric(debug_dir / "DBG_PHISOL_IN").values.reshape(chg.grid, order="F").mean(axis=(0, 1))
    phi_final_ref = read_vasp_volumetric(debug_dir / "PHI").values.reshape(chg.grid, order="F").mean(axis=(0, 1))
    cvhar_ref = None
    cvdip_ref = None
    if (debug_dir / "DBG_CVHAR_IN").exists():
        cvhar_ref = read_vasp_volumetric(debug_dir / "DBG_CVHAR_IN").values.reshape(chg.grid, order="F").mean(axis=(0, 1))
    if (debug_dir / "DBG_CVDIP_IN").exists():
        cvdip_ref = read_vasp_volumetric(debug_dir / "DBG_CVDIP_IN").values.reshape(chg.grid, order="F").mean(axis=(0, 1))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines += line("nval_vs_chgcar_1d", val_z, nval_ref)
    lines += line("dencor_1d", dencor_z, dencor_ref)
    lines += line("ne_1d", val_z + dencor_z, ne_ref)
    lines += line("phisol_1d", phisol_z, phisol_ref)
    if cvhar_ref is not None:
        lines += line("cvhar_1d", phisol_z, cvhar_ref)
    if cvhar_ref is not None and cvdip_ref is not None:
        lines += line("cvhar_plus_cvdip_vs_phisol_1d", cvhar_ref + cvdip_ref, phisol_ref)
        lines += line("cvdip_1d", phisol_ref - cvhar_ref, cvdip_ref)
    lines += line("phisol_vs_final_phi_1d", phisol_ref, phi_final_ref)
    z = np.arange(chg.grid[2], dtype=float) * chg.length_z / chg.grid[2]
    write_profile(
        out / "profiles.tsv",
        {
            "z_A": z,
            "nval_CHGCAR": val_z,
            "nval_DBG": nval_ref,
            "dencor_py": dencor_z,
            "dencor_DBG": dencor_ref,
            "ne_py": val_z + dencor_z,
            "ne_DBG": ne_ref,
            "phisol_py_eV": phisol_z,
            "cvhar_DBG_eV": cvhar_ref if cvhar_ref is not None else phisol_ref * np.nan,
            "cvdip_DBG_eV": cvdip_ref if cvdip_ref is not None else phisol_ref * np.nan,
            "phisol_DBG_eV": phisol_ref,
            "phi_final_DBG_eV": phi_final_ref,
        },
    )
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(out / "summary.txt")


if __name__ == "__main__":
    main()
