from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.vasp_volumetric import read_vasp_volumetric, write_profile

from .config import load_config
from .grid import Grid
from .potcar import read_potcar
from .solute_potential import dencor_values, hartree_potential_g, local_pseudopotential_g


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def stats_line(name: str, py: np.ndarray, ref: np.ndarray, grid: Grid) -> list[str]:
    py_f = py.reshape(-1, order="F")
    ref_f = ref.reshape(-1, order="F")
    py_center = py_f - py_f.mean()
    ref_center = ref_f - ref_f.mean()
    py_z = py.mean(axis=(0, 1))
    ref_z = ref.mean(axis=(0, 1))
    return [
        f"{name}_raw_rmse\t{rmse(py_f, ref_f):.12e}",
        f"{name}_demean_rmse\t{rmse(py_center, ref_center):.12e}",
        f"{name}_raw_mean_py\t{py_f.mean():.12e}",
        f"{name}_raw_mean_ref\t{ref_f.mean():.12e}",
        f"{name}_z_rmse\t{rmse(py_z, ref_z):.12e}",
        f"{name}_integral_py\t{py_f.sum()/grid.ngrid:.12e}",
        f"{name}_integral_ref\t{ref_f.sum()/grid.ngrid:.12e}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--debug-dir", default="reproduce3d/debug_cal18_nelm1")
    parser.add_argument("--out-dir", default="pure_python/results/debug_input_compare")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    grid = Grid(chg.cell, chg.grid)
    entries = read_potcar(args.potcar)
    positions = np.asarray(cfg["positions_direct"], dtype=float)
    counts = list(cfg["counts"])

    valence = chg.values.reshape(chg.grid, order="F")
    dencor_py = dencor_values(grid, entries, counts, positions)
    phi_py_g = hartree_potential_g(grid.fft(valence), grid) + local_pseudopotential_g(
        grid, entries, counts, positions
    )
    phi_py = grid.ifft_real(phi_py_g)

    debug_dir = Path(args.debug_dir)
    dencor_ref = read_vasp_volumetric(debug_dir / "DBG_DENCOR").values.reshape(chg.grid, order="F")
    nval_ref = read_vasp_volumetric(debug_dir / "DBG_NVAL").values.reshape(chg.grid, order="F")
    ne_ref = read_vasp_volumetric(debug_dir / "DBG_NE").values.reshape(chg.grid, order="F")
    phisol_ref = read_vasp_volumetric(debug_dir / "DBG_PHISOL_IN").values.reshape(chg.grid, order="F")
    phi_final_ref = read_vasp_volumetric(debug_dir / "PHI").values.reshape(chg.grid, order="F")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines += stats_line("nval_vs_chgcar", valence, nval_ref, grid)
    lines += stats_line("dencor", dencor_py, dencor_ref, grid)
    lines += stats_line("ne_vs_val_plus_dencor", valence + dencor_py, ne_ref, grid)
    lines += stats_line("phisol", phi_py, phisol_ref, grid)
    lines += stats_line("phisol_vs_final_phi", phisol_ref, phi_final_ref, grid)

    z = np.arange(chg.grid[2], dtype=float) * grid.length_z / chg.grid[2]
    write_profile(
        out / "profiles.tsv",
        {
            "z_A": z,
            "nval_CHGCAR": valence.mean(axis=(0, 1)),
            "nval_DBG": nval_ref.mean(axis=(0, 1)),
            "dencor_py": dencor_py.mean(axis=(0, 1)),
            "dencor_DBG": dencor_ref.mean(axis=(0, 1)),
            "ne_py": (valence + dencor_py).mean(axis=(0, 1)),
            "ne_DBG": ne_ref.mean(axis=(0, 1)),
            "phisol_py_eV": phi_py.mean(axis=(0, 1)),
            "phisol_DBG_eV": phisol_ref.mean(axis=(0, 1)),
            "phi_final_DBG_eV": phi_final_ref.mean(axis=(0, 1)),
        },
    )
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(out / "summary.txt")


if __name__ == "__main__":
    main()
