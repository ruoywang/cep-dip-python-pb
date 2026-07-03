"""Production-source test for the affine 1D closure: extract from a COARSE 3D solve.

Three-way comparison on cal_18:
  (a) coarse 3D solve alone (half-resolution, full CDIPOL fixstep loop),
  (b) affine 1D with A/P_off extracted from the prolonged coarse potential,
  (c) reference = the validated full-3D route (numbers quoted from step3_rfft).

At prediction time there is no reference PHI; the coarse solve is the cheap,
available source. If (b) does not improve on (a), the 1D refinement adds no
value over just using the coarse solve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parent))

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
from pure_python.solve_from_chgcar_newton import (
    prolong_double,
    restrict_half,
    rmse,
    solve_nlpb_for_phi_sol,
)
from solve_1d_frozen_response import plane_avg, response_coefficient_3d, solve_frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--rhoion-ref", default="data/case_cal18/RHOION")
    parser.add_argument("--out-dir", default="pb_1d_test/coarse_extract")
    parser.add_argument("--fixsol-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1.0e-3)
    parser.add_argument("--max-outer", type=int, default=12)
    parser.add_argument("--cg-max-iter", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage_lines = ["stage\tseconds"]
    t0 = perf_counter()

    def mark(label):
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
    params = derived_params(cfg["solvation"])
    s_ion3, s_diel3, _ = create_cavity(n_e_density, grid3, params, None)
    mark("prep_3d")

    # reference profiles (full z axis)
    nz = grid3.shape[2]
    phi_ref_z = phi_ref.values.reshape(phi_ref.grid, order="F").mean(axis=(0, 1))
    rhob_ref_z = (rhob_ref.values.reshape(rhob_ref.grid, order="F") / grid3.volume).mean(axis=(0, 1))
    rhoion_ref_z = (rhoion_ref.values.reshape(rhoion_ref.grid, order="F") / grid3.volume).mean(axis=(0, 1))

    val_ion_dipole = valence_ion_dipole_cart(valence_values, positions, zvals, counts, chg.cell)
    val_ion_dipole[0:2] = 0.0
    center_abs = 0.5 * chg.cell[0] + 0.5 * chg.cell[1] + 0.5 * chg.cell[2]
    length_z = float(np.linalg.norm(chg.cell[2]))
    q_sol = float(cfg["q_sol"])

    # ---- (a) coarse 3D solve with its own CDIPOL fixstep loop ----
    grid_c = Grid(chg.cell, tuple(n // 2 for n in grid3.shape))
    nzc = grid_c.shape[2]
    s_ion_c = restrict_half(s_ion3)
    s_diel_c = restrict_half(s_diel3)
    cvhar_c = restrict_half(cvhar3)
    indmin_c = cdipol_indmin_from_center(nzc, 0.5)
    mixer = EwaldDipoleMixer.fresh()
    qsol_cache = 0.0
    dsol_cache = np.zeros(3, dtype=float)
    phi_c = np.zeros(grid_c.shape, dtype=float)
    coarse_lines = ["fixstep\trms_last\tEFz\tphi_z_rmse_eV\trhob_z_rmse_e_A3\trhoion_z_rmse_e_A3\tseconds"]
    ef_c = 0.0
    for step in range(args.fixsol_steps):
        t_fix = perf_counter()
        dip_for_field = val_ion_dipole.copy()
        dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef_direct = mixer.ewald_dipol(dip_for_field, chg.cell, 3)
        ef_c = ef_direct[2]
        cvdip_c = cdipol_potential_1d(nzc, length_z, ef_c, indmin_c)
        phi_sol_c = cvhar_c + cvdip_c[None, None, :]
        phi_c, nb_c, ni_c, _, hist = solve_nlpb_for_phi_sol(
            phi_c, phi_sol_c, s_ion_c, s_diel_c, grid_c, params, q_sol,
            args.tol, args.max_outer, args.cg_max_iter, None, step,
        )
        qsol_cache, dsol_cache = solvent_moments(nb_c + ni_c, chg.cell)
        dt = perf_counter() - t_fix
        coarse_lines.append(
            f"{step}\t{hist[-1][1]:.6e}\t{ef_c:.6e}\t"
            f"{rmse(phi_c.mean(axis=(0,1)), phi_ref_z[::2]):.6e}\t"
            f"{rmse((nb_c/grid_c.volume).mean(axis=(0,1)), rhob_ref_z[::2]):.6e}\t"
            f"{rmse((ni_c/grid_c.volume).mean(axis=(0,1)), rhoion_ref_z[::2]):.6e}\t{dt:.3f}"
        )
        (out / "coarse3d_summary.tsv").write_text("\n".join(coarse_lines) + "\n")
    mark("coarse3d_solve")

    # ---- (b) extract affine closure from the prolonged coarse potential ----
    phi3_from_coarse = prolong_double(phi_c, grid_c, grid3)
    a3, ez3 = response_coefficient_3d(phi3_from_coarse, s_diel3, grid3, params)
    a1 = plane_avg(a3)
    p_off = plane_avg(a3 * ez3) - a1 * plane_avg(ez3)
    mark("extract_from_coarse")

    grid1 = Grid(chg.cell, (1, 1, nz))
    s_ion1 = plane_avg(s_ion3)
    cvhar1 = plane_avg(cvhar3)
    indmin_z = cdipol_indmin_from_center(nz, 0.5)
    mixer = EwaldDipoleMixer.fresh()
    qsol_cache = 0.0
    dsol_cache = np.zeros(3, dtype=float)
    phi_total = np.zeros(grid1.shape, dtype=float)
    n_b = np.zeros(grid1.shape, dtype=float)
    n_ion = np.zeros(grid1.shape, dtype=float)
    lines = ["fixstep\trms_last\tEFz\tphi_z_rmse_eV\trhob_z_rmse_e_A3\trhoion_z_rmse_e_A3\tseconds"]
    for step in range(args.fixsol_steps):
        t_fix = perf_counter()
        dip_for_field = val_ion_dipole.copy()
        dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef_direct = mixer.ewald_dipol(dip_for_field, chg.cell, 3)
        cvdip_z = cdipol_potential_1d(nz, length_z, ef_direct[2], indmin_z)
        phi_sol = cvhar1 + cvdip_z[None, None, :]
        phi_total, n_b, n_ion, history = solve_frozen(
            phi_total, phi_sol, s_ion1, a1, grid1, params, q_sol,
            args.tol, args.max_outer, args.cg_max_iter, p_off,
        )
        qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, chg.cell)
        dt = perf_counter() - t_fix
        lines.append(
            f"{step}\t{history[-1][1]:.6e}\t{ef_direct[2]:.6e}\t"
            f"{rmse(phi_total[0,0,:], phi_ref_z):.6e}\t"
            f"{rmse(n_b[0,0,:]/grid1.volume, rhob_ref_z):.6e}\t"
            f"{rmse(n_ion[0,0,:]/grid1.volume, rhoion_ref_z):.6e}\t{dt:.3f}"
        )
        (out / "affine1d_summary.tsv").write_text("\n".join(lines) + "\n")
    mark("fixsteps_1d_affine")

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
            "A_frozen": a1[0, 0, :],
            "P_off": p_off[0, 0, :],
        },
    )
    print(out / "affine1d_summary.tsv")


if __name__ == "__main__":
    main()
