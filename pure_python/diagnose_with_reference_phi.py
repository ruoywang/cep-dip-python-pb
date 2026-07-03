from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.vasp_volumetric import read_vasp_volumetric, write_profile

from .config import load_config
from .grid import Grid
from .pb import update_from_total_phi


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--rhoion-ref", default="data/case_cal18/RHOION")
    parser.add_argument("--out-dir", default="pure_python/results/ref_phi_diag")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    phi_ref = read_vasp_volumetric(args.phi_ref)
    rhob_ref = read_vasp_volumetric(args.rhob_ref)
    rhoion_ref = read_vasp_volumetric(args.rhoion_ref)
    grid = Grid(chg.cell, chg.grid)
    chg_values = chg.values.reshape(chg.grid, order="F")
    n_e_density = chg_values / grid.volume
    phi = phi_ref.values.reshape(phi_ref.grid, order="F")
    timings: list[tuple[str, float]] = []
    state = update_from_total_phi(phi, n_e_density, grid, cfg["solvation"], timings=timings)
    rhob = rhob_ref.values.reshape(rhob_ref.grid, order="F")
    rhoion = rhoion_ref.values.reshape(rhoion_ref.grid, order="F")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = [
        "diagnostic: update_nlpb using reference PHI as total phi",
        "final solver is not allowed to use PHI; this isolates PB update/cavity errors",
        f"RHOB_raw_rmse\t{rmse(state.rho_bound_values, rhob):.12e}",
        f"RHOION_raw_rmse\t{rmse(state.rho_ion_values, rhoion):.12e}",
        f"RHOB_integral_ref\t{rhob.sum()/grid.ngrid:.12e}",
        f"RHOB_integral_py\t{state.rho_bound_values.sum()/grid.ngrid:.12e}",
        f"RHOION_integral_ref\t{rhoion.sum()/grid.ngrid:.12e}",
        f"RHOION_integral_py\t{state.rho_ion_values.sum()/grid.ngrid:.12e}",
    ]
    summary.extend(f"time_{label}_s\t{seconds:.6f}" for label, seconds in timings)
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    z, rhob_z = rhob_ref.plane_average_density()
    _, rhoion_z = rhoion_ref.plane_average_density()
    rhob_py_z = (state.rho_bound_values / grid.volume).mean(axis=(0, 1))
    rhoion_py_z = (state.rho_ion_values / grid.volume).mean(axis=(0, 1))
    write_profile(
        out / "profiles.tsv",
        {
            "z_A": z,
            "RHOB_DFT_e_A3": rhob_z,
            "RHOB_py_refphi_e_A3": rhob_py_z,
            "RHOION_DFT_e_A3": rhoion_z,
            "RHOION_py_refphi_e_A3": rhoion_py_z,
        },
    )
    print(out / "summary.txt")


if __name__ == "__main__":
    main()
