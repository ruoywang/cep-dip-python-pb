"""1D PB test: cavity built in 3D, PB solve on plane-averaged inputs.

The 1D solve reuses the production solver verbatim on a (1, 1, nz) grid, so
any deviation from the 3D reference isolates the physics of the reduction
(planar averaging does not commute with the nonlinear response) rather than
new code. The cavity (s_ion, s_diel) and the solute potential are computed on
the full 3D grid exactly as in production, then plane-averaged.

Usage (from the repo root):
  PYTHONPATH=. python tools/pb_check/solve_1d_from_3d_cavity.py --out-dir <run>/out
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.vasp_volumetric import read_vasp_volumetric, write_profile
from pure_python.config import load_config
from pure_python.dipole_correction import (
    EwaldDipoleMixer,
    cdipol_indmin_from_center,
    cdipol_potential_1d,
    solvent_moments,
    valence_ion_dipole_cart,
)
from pure_python.grid import Grid
from pure_python.pb import create_cavity, derived_params
from pure_python.potcar import read_potcar
from pure_python.solute_potential import solute_potential_g
from pure_python.solve_from_chgcar_newton import rmse, solve_nlpb_for_phi_sol


def plane_avg(a: np.ndarray) -> np.ndarray:
    """xy plane average, kept as a (1, 1, nz) array (preserves mean and z-moment)."""
    return np.ascontiguousarray(a.mean(axis=(0, 1)).reshape(1, 1, -1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--rhoion-ref", default="data/case_cal18/RHOION")
    parser.add_argument("--out-dir", default="pb_1d_test/out")
    parser.add_argument("--fixsol-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1.0e-3)
    parser.add_argument("--max-outer", type=int, default=12)
    parser.add_argument("--cg-max-iter", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage_lines = ["stage\tseconds"]
    t0 = perf_counter()

    def mark(label: str) -> None:
        nonlocal t0
        now = perf_counter()
        stage_lines.append(f"{label}\t{now - t0:.6f}")
        (out / "stage_times.tsv").write_text("\n".join(stage_lines) + "\n")
        t0 = now

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    phi_ref = read_vasp_volumetric(args.phi_ref)
    rhob_ref = read_vasp_volumetric(args.rhob_ref)
    rhoion_ref = read_vasp_volumetric(args.rhoion_ref)
    mark("read_inputs")

    grid3 = Grid(chg.cell, chg.grid)
    valence_values = chg.values.reshape(chg.grid, order="F")
    entries = read_potcar(args.potcar)
    positions = np.asarray(cfg["positions_direct"], dtype=float)
    counts = list(cfg["counts"])
    zvals = [entry.zval for entry in entries]
    cvhar_g, dencor = solute_potential_g(grid3, valence_values, entries, counts, positions, None)
    cvhar3 = grid3.ifft_real_full(cvhar_g)
    n_e_density = (valence_values + dencor) / grid3.volume
    mark("solute_potential_3d")

    params = derived_params(cfg["solvation"])
    s_ion3, s_diel3, _ = create_cavity(n_e_density, grid3, params, None)
    mark("cavity_3d")

    # ---- reduce to 1D: plane-averaged inputs on a (1, 1, nz) grid ----
    nz = grid3.shape[2]
    grid1 = Grid(chg.cell, (1, 1, nz))
    s_ion1 = plane_avg(s_ion3)
    s_diel1 = plane_avg(s_diel3)
    cvhar1 = plane_avg(cvhar3)
    mark("reduce_to_1d")

    # reference profiles
    phi_ref_z = phi_ref.values.reshape(phi_ref.grid, order="F").mean(axis=(0, 1))
    rhob_ref_z = (rhob_ref.values.reshape(rhob_ref.grid, order="F") / grid3.volume).mean(axis=(0, 1))
    rhoion_ref_z = (rhoion_ref.values.reshape(rhoion_ref.grid, order="F") / grid3.volume).mean(axis=(0, 1))

    val_ion_dipole = valence_ion_dipole_cart(valence_values, positions, zvals, counts, chg.cell)
    val_ion_dipole[0:2] = 0.0
    center_abs = 0.5 * chg.cell[0] + 0.5 * chg.cell[1] + 0.5 * chg.cell[2]
    mixer = EwaldDipoleMixer.fresh()
    qsol_cache = 0.0
    dsol_cache = np.zeros(3, dtype=float)
    indmin_z = cdipol_indmin_from_center(nz, 0.5)
    length_z = float(np.linalg.norm(chg.cell[2]))
    q_sol = float(cfg["q_sol"])

    phi_total = np.zeros(grid1.shape, dtype=float)
    n_b = np.zeros(grid1.shape, dtype=float)
    n_ion = np.zeros(grid1.shape, dtype=float)
    lines = [
        "fixstep\touter_last\trms_last\tEFz_direct\tqsol\tdsol_z\tphi_z_rmse_eV\trhob_z_rmse_e_A3\trhoion_z_rmse_e_A3\tseconds"
    ]
    for step in range(args.fixsol_steps):
        t_fix = perf_counter()
        dip_for_field = val_ion_dipole.copy()
        dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef_direct = mixer.ewald_dipol(dip_for_field, chg.cell, 3)
        cvdip_z = cdipol_potential_1d(nz, length_z, ef_direct[2], indmin_z)
        phi_sol = cvhar1 + cvdip_z[None, None, :]
        phi_total, n_b, n_ion, _, history = solve_nlpb_for_phi_sol(
            phi_total, phi_sol, s_ion1, s_diel1, grid1, params, q_sol,
            args.tol, args.max_outer, args.cg_max_iter, None, step,
        )
        qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, chg.cell)
        dt = perf_counter() - t_fix
        phi_z = phi_total[0, 0, :]
        rhob_z = n_b[0, 0, :] / grid1.volume
        rhoion_z = n_ion[0, 0, :] / grid1.volume
        last_outer, last_rms, _, _ = history[-1]
        lines.append(
            f"{step}\t{last_outer}\t{last_rms:.12e}\t{ef_direct[2]:.12e}\t{qsol_cache:.12e}\t{dsol_cache[2]:.12e}\t"
            f"{rmse(phi_z, phi_ref_z):.12e}\t{rmse(rhob_z, rhob_ref_z):.12e}\t{rmse(rhoion_z, rhoion_ref_z):.12e}\t{dt:.6f}"
        )
        (out / "fixstep_summary.tsv").write_text("\n".join(lines) + "\n")
    mark("fixsteps_1d")

    z = np.arange(nz) * length_z / nz
    write_profile(
        out / "profiles.tsv",
        {
            "z_A": z,
            "PHI_DFT_eV": phi_ref_z,
            "PHI_1d_eV": phi_total[0, 0, :],
            "RHOB_DFT_e_A3": rhob_ref_z,
            "RHOB_1d_e_A3": n_b[0, 0, :] / grid1.volume,
            "RHOION_DFT_e_A3": rhoion_ref_z,
            "RHOION_1d_e_A3": n_ion[0, 0, :] / grid1.volume,
        },
    )
    print(out / "fixstep_summary.tsv")


if __name__ == "__main__":
    main()
