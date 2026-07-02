from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from src.vasp_volumetric import read_vasp_volumetric, write_profile

from .config import load_config
from .grid import Grid, l0_inv_op
from .pb import (
    create_cavity,
    derived_params,
    ion_density_values_from_phi,
    update_from_total_phi_with_cavity,
)
from .potcar import read_potcar
from .solute_potential import solute_potential_g


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def find_ion_offset(phi_base: np.ndarray, s_ion: np.ndarray, grid: Grid, params: dict, q_sol: float) -> float:
    target = -q_sol

    def charge(offset: float) -> float:
        vals = ion_density_values_from_phi(phi_base + offset, s_ion, grid, params)
        return vals.sum() / grid.ngrid - target

    lo, hi = -1.0, 1.0
    flo, fhi = charge(lo), charge(hi)
    for _ in range(30):
        if flo * fhi <= 0.0:
            return float(brentq(charge, lo, hi, xtol=1.0e-10, rtol=1.0e-10, maxiter=100))
        lo *= 2.0
        hi *= 2.0
        flo, fhi = charge(lo), charge(hi)
    raise RuntimeError(f"failed to bracket ion offset: f({lo})={flo}, f({hi})={fhi}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--rhoion-ref", default="data/case_cal18/RHOION")
    parser.add_argument("--out-dir", default="pure_python/results/from_chgcar_simple")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--source-sign", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    phi_ref = read_vasp_volumetric(args.phi_ref)
    rhob_ref = read_vasp_volumetric(args.rhob_ref)
    rhoion_ref = read_vasp_volumetric(args.rhoion_ref)
    grid = Grid(chg.cell, chg.grid)
    valence_values = chg.values.reshape(chg.grid, order="F")
    entries = read_potcar(args.potcar)
    positions = np.asarray(cfg["positions_direct"], dtype=float)
    phi_sol_g, dencor = solute_potential_g(grid, valence_values, entries, cfg["counts"], positions)
    phi_sol = grid.ifft_real(phi_sol_g)
    n_e_density = (valence_values + dencor) / grid.volume
    params = derived_params(cfg["solvation"])
    s_ion, s_diel, _ = create_cavity(n_e_density, grid, params)

    phi_ref_values = phi_ref.values.reshape(phi_ref.grid, order="F")
    rhob_ref_values = rhob_ref.values.reshape(rhob_ref.grid, order="F")
    rhoion_ref_values = rhoion_ref.values.reshape(rhoion_ref.grid, order="F")
    phi_solv = np.zeros(grid.shape, dtype=float)
    lines = ["iter\toffset_eV\tphi_rmse_eV\trhob_rmse_raw\trhoion_rmse_raw\trhoion_integral_e"]
    state = None
    phi = phi_sol.copy()
    for it in range(args.iterations):
        offset = find_ion_offset(phi_sol + phi_solv, s_ion, grid, params, cfg["q_sol"])
        phi = phi_sol + phi_solv + offset
        timings: list[tuple[str, float]] = []
        state = update_from_total_phi_with_cavity(phi, s_ion, s_diel, grid, params, timings=timings)
        source_g = args.source_sign * grid.fft(state.rho_bound_values + state.rho_ion_values)
        phi_solv_new = grid.ifft_real(l0_inv_op(source_g, grid))
        phi_solv = (1.0 - args.mix) * phi_solv + args.mix * phi_solv_new
        lines.append(
            f"{it}\t{offset:.12e}\t{rmse(phi, phi_ref_values):.12e}\t"
            f"{rmse(state.rho_bound_values, rhob_ref_values):.12e}\t"
            f"{rmse(state.rho_ion_values, rhoion_ref_values):.12e}\t"
            f"{state.rho_ion_values.sum()/grid.ngrid:.12e}"
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "iteration_summary.tsv").write_text("\n".join(lines) + "\n")
    z, phi_ref_z = phi_ref.plane_average_raw()
    _, rhob_ref_z = rhob_ref.plane_average_density()
    _, rhoion_ref_z = rhoion_ref.plane_average_density()
    assert state is not None
    write_profile(
        out / "profiles.tsv",
        {
            "z_A": z,
            "PHI_DFT_eV": phi_ref_z,
            "PHI_py_eV": phi.mean(axis=(0, 1)),
            "RHOB_DFT_e_A3": rhob_ref_z,
            "RHOB_py_e_A3": (state.rho_bound_values / grid.volume).mean(axis=(0, 1)),
            "RHOION_DFT_e_A3": rhoion_ref_z,
            "RHOION_py_e_A3": (state.rho_ion_values / grid.volume).mean(axis=(0, 1)),
        },
    )
    print(out / "iteration_summary.tsv")


if __name__ == "__main__":
    main()
