"""Mixed-source test for the affine 1D closure: which ingredient carries the
coarse-extraction error — the slope A(z) or the anchor offset P_off(z)?

2x2 matrix on cal_18: A from {fine reference, coarse solve} x anchor fields
(<P_z>, <E_z>) from {fine reference, coarse solve}; P_off = <P> - A*<E> with
the chosen pair. The diagonal reproduces the two known results
(frozen_affine ~1.6e-3 eV, coarse_extract ~9.4e-3 eV) as a built-in check.
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

from tools.vasp_volumetric import read_vasp_volumetric
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
from pure_python.grid import EDEPS
from solve_1d_frozen_response import plane_avg, response_coefficient_3d, solve_frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--rhoion-ref", default="data/case_cal18/RHOION")
    parser.add_argument("--out-dir", default="pb_1d_test/mixed_sources")
    parser.add_argument("--fixsol-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1.0e-3)
    parser.add_argument("--max-outer", type=int, default=12)
    parser.add_argument("--cg-max-iter", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    phi_ref = read_vasp_volumetric(args.phi_ref)
    rhob_ref = read_vasp_volumetric(args.rhob_ref)
    rhoion_ref = read_vasp_volumetric(args.rhoion_ref)

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

    nz = grid3.shape[2]
    phi_ref3 = phi_ref.values.reshape(phi_ref.grid, order="F")
    phi_ref_z = phi_ref3.mean(axis=(0, 1))
    rhob_ref_z = (rhob_ref.values.reshape(rhob_ref.grid, order="F") / grid3.volume).mean(axis=(0, 1))
    rhoion_ref_z = (rhoion_ref.values.reshape(rhoion_ref.grid, order="F") / grid3.volume).mean(axis=(0, 1))

    val_ion_dipole = valence_ion_dipole_cart(valence_values, positions, zvals, counts, chg.cell)
    val_ion_dipole[0:2] = 0.0
    center_abs = 0.5 * chg.cell[0] + 0.5 * chg.cell[1] + 0.5 * chg.cell[2]
    length_z = float(np.linalg.norm(chg.cell[2]))
    q_sol = float(cfg["q_sol"])

    # ---- fine-source response fields from the reference PHI ----
    a3_f, ez3_f = response_coefficient_3d(phi_ref3, s_diel3, grid3, params)

    # ---- coarse 3D solve (same protocol as solve_1d_coarse_extract) ----
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
    t0 = perf_counter()
    for step in range(args.fixsol_steps):
        dip_for_field = val_ion_dipole.copy()
        dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef_direct = mixer.ewald_dipol(dip_for_field, chg.cell, 3)
        cvdip_c = cdipol_potential_1d(nzc, length_z, ef_direct[2], indmin_c)
        phi_sol_c = cvhar_c + cvdip_c[None, None, :]
        phi_c, nb_c, ni_c, _, _hist = solve_nlpb_for_phi_sol(
            phi_c, phi_sol_c, s_ion_c, s_diel_c, grid_c, params, q_sol,
            args.tol, args.max_outer, args.cg_max_iter, None, step,
        )
        qsol_cache, dsol_cache = solvent_moments(nb_c + ni_c, chg.cell)
    t_coarse = perf_counter() - t0
    phi3_from_coarse = prolong_double(phi_c, grid_c, grid3)
    a3_c, ez3_c = response_coefficient_3d(phi3_from_coarse, s_diel3, grid3, params)

    # ---- closure ingredients from each source ----
    # "formula" A: zero-field linear response on the (averaged) cavity — no 3D
    # solve at all; at zero field the formula is linear in s_diel so building
    # in 3D and plane-averaging equals the pure-1D formula exactly.
    a3_z, _ez_z = response_coefficient_3d(np.zeros_like(phi_ref3), s_diel3, grid3, params)
    A = {"fine": plane_avg(a3_f), "coarse": plane_avg(a3_c),
         "formula": plane_avg(a3_z)}
    # ---- tier-1 physical PRIOR anchors from the VACUUM solute field ----
    # No PB solve at all: the lateral field structure is dominated by the
    # analytically-known solute potential (cvhar). "vac" evaluates the same
    # correlation with the raw vacuum field (saturation factors self-limit
    # where the bare field is huge); "vacscr" screens the field by the local
    # dielectric eps(r) = 1 + EDEPS*chi0 (D-continuity heuristic) and uses the
    # zero-field chi there.
    a3_v, ez3_v = response_coefficient_3d(cvhar3, s_diel3, grid3, params)
    eps3 = 1.0 + EDEPS * a3_z
    ez3_scr = ez3_v / eps3
    ANCHOR = {
        "fine": (plane_avg(a3_f * ez3_f), plane_avg(ez3_f)),
        "coarse": (plane_avg(a3_c * ez3_c), plane_avg(ez3_c)),
        "vac": (plane_avg(a3_v * ez3_v), plane_avg(ez3_v)),
        "vacscr": (plane_avg(a3_z * ez3_scr), plane_avg(ez3_scr)),
    }
    # prior-vs-exact P_off profile deviation (A = formula in all cases)
    p_exact = ANCHOR["fine"][0] - A["formula"] * ANCHOR["fine"][1]
    for k in ("coarse", "vac", "vacscr"):
        p_prior = ANCHOR[k][0] - A["formula"] * ANCHOR[k][1]
        print(f"P_off[{k}] vs exact: rmse={rmse(p_prior, p_exact):.3e}"
              f"  (peak |exact|={np.abs(p_exact).max():.3e})", flush=True)

    # ---- 2x2 matrix of 1D solves ----
    grid1 = Grid(chg.cell, (1, 1, nz))
    s_ion1 = plane_avg(s_ion3)
    cvhar1 = plane_avg(cvhar3)
    indmin_z = cdipol_indmin_from_center(nz, 0.5)
    lines = [
        "A_source\tanchor_source\tphi_z_rmse_eV\trhob_z_rmse_e_A3\trhoion_z_rmse_e_A3\tseconds"
    ]
    for a_src in ("fine", "coarse", "formula"):
        for anch_src in ("fine", "coarse", "vac", "vacscr"):
            a1 = A[a_src]
            pz_avg, ez_avg = ANCHOR[anch_src]
            p_off = pz_avg - a1 * ez_avg
            mixer = EwaldDipoleMixer.fresh()
            qsol_cache = 0.0
            dsol_cache = np.zeros(3, dtype=float)
            phi_total = np.zeros(grid1.shape, dtype=float)
            n_b = np.zeros(grid1.shape, dtype=float)
            n_ion = np.zeros(grid1.shape, dtype=float)
            t1 = perf_counter()
            for step in range(args.fixsol_steps):
                dip_for_field = val_ion_dipole.copy()
                dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
                _, ef_direct = mixer.ewald_dipol(dip_for_field, chg.cell, 3)
                cvdip_z = cdipol_potential_1d(nz, length_z, ef_direct[2], indmin_z)
                phi_sol = cvhar1 + cvdip_z[None, None, :]
                phi_total, n_b, n_ion, _history = solve_frozen(
                    phi_total, phi_sol, s_ion1, a1, grid1, params, q_sol,
                    args.tol, args.max_outer, args.cg_max_iter, p_off,
                )
                qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, chg.cell)
            dt = perf_counter() - t1
            row = (
                f"{a_src}\t{anch_src}\t"
                f"{rmse(phi_total[0, 0, :], phi_ref_z):.6e}\t"
                f"{rmse(n_b[0, 0, :] / grid1.volume, rhob_ref_z):.6e}\t"
                f"{rmse(n_ion[0, 0, :] / grid1.volume, rhoion_ref_z):.6e}\t{dt:.3f}"
            )
            lines.append(row)
            print(row, flush=True)
    lines.append(f"# coarse 3D solve: {t_coarse:.3f}s")
    (out / "mixed_sources_summary.tsv").write_text("\n".join(lines) + "\n")
    print(out / "mixed_sources_summary.tsv")


if __name__ == "__main__":
    main()
