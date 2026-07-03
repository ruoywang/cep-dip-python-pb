from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.vasp_volumetric import read_vasp_volumetric

from .config import load_config
from .grid import Grid
from .io import write_vasp_like
from .pb import poisson_potential_from_density_values, update_from_total_phi


def point_ion_values(config: dict, grid: Grid) -> np.ndarray:
    # First diagnostic replacement for POTION/DENCOR. This is not expected to be final.
    vals = np.zeros(grid.shape, dtype=float)
    zvals = config["zval"]
    elements = config["elements"]
    counts = config["counts"]
    pos = np.asarray(config["positions_direct"], dtype=float)
    idx = 0
    for elem, count in zip(elements, counts):
        zval = float(zvals[elem])
        for _ in range(count):
            ijk = np.floor(pos[idx] * np.asarray(grid.shape)).astype(int) % np.asarray(grid.shape)
            vals[tuple(ijk)] += zval * grid.ngrid
            idx += 1
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--out-dir", default="pure_python/results/from_chgcar")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    grid = Grid(chg.cell, chg.grid)
    valence_values = chg.values.reshape(chg.grid, order="F")
    n_e_density = valence_values / grid.volume
    charge_values = point_ion_values(cfg, grid) - valence_values
    phi_sol = poisson_potential_from_density_values(charge_values, grid)
    state = update_from_total_phi(phi_sol, n_e_density, grid, cfg["solvation"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_vasp_like(out / "PHI", chg, state.phi_values, "pure_python PHI")
    write_vasp_like(out / "RHOB", chg, state.rho_bound_values, "pure_python RHOB")
    write_vasp_like(out / "RHOION", chg, state.rho_ion_values, "pure_python RHOION")
    (out / "summary.txt").write_text(
        "\n".join(
            [
                "mode: CHGCAR + config only",
                "solute_potential_mode: point_ion_values - valence poisson; diagnostic placeholder",
                f"RHOB_integral_e\t{state.rho_bound_values.sum()/grid.ngrid:.12e}",
                f"RHOION_integral_e\t{state.rho_ion_values.sum()/grid.ngrid:.12e}",
            ]
        )
        + "\n"
    )
    print(out / "summary.txt")


if __name__ == "__main__":
    main()

