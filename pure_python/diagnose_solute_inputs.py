from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.vasp_volumetric import read_vasp_volumetric, write_profile

from .config import load_config
from .grid import Grid
from .potcar import read_potcar
from .solute_potential import dencor_values, local_pseudopotential_g, hartree_potential_g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--out-dir", default="pure_python/results/solute_inputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    grid = Grid(chg.cell, chg.grid)
    valence_values = chg.values.reshape(chg.grid, order="F")
    entries = read_potcar(args.potcar)
    positions = np.asarray(cfg["positions_direct"], dtype=float)

    dencor = dencor_values(grid, entries, cfg["counts"], positions)
    valence_g = grid.fft(valence_values)
    v_hartree = grid.ifft_real(hartree_potential_g(valence_g, grid))
    v_local = grid.ifft_real(local_pseudopotential_g(grid, entries, cfg["counts"], positions))
    phi_sol = v_hartree + v_local

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z, val_z = chg.plane_average_density()
    write_profile(
        out / "profiles.tsv",
        {
            "z_A": z,
            "valence_CHGCAR_e_A3": val_z,
            "dencor_e_A3": dencor.mean(axis=(0, 1)) / grid.volume,
            "phi_hartree_eV": v_hartree.mean(axis=(0, 1)),
            "phi_local_eV": v_local.mean(axis=(0, 1)),
            "phi_sol_no_dipole_eV": phi_sol.mean(axis=(0, 1)),
        },
    )
    (out / "summary.txt").write_text(
        "\n".join(
            [
                f"valence_integral_e\t{valence_values.sum()/grid.ngrid:.12e}",
                f"dencor_integral_e\t{dencor.sum()/grid.ngrid:.12e}",
                f"phi_hartree_mean_eV\t{v_hartree.mean():.12e}",
                f"phi_local_mean_eV\t{v_local.mean():.12e}",
                f"phi_sol_no_dipole_mean_eV\t{phi_sol.mean():.12e}",
            ]
        )
        + "\n"
    )
    print(out / "summary.txt")


if __name__ == "__main__":
    main()
