"""Does evaluating the saturation factors at the SCREENED-VACUUM field (instead
of zero field) improve the formula A?  cal_18, three A sources:

  A_form : zero-field chi0 (current design)
  A_scr  : saturation at |E_scr| = |grad cvhar|_wb / eps(r)   (candidate)
  A_ext  : extracted from the DFT reference field             (target)

Reports: A profile rms vs A_ext; the absorption term (A_ext-A)*<Ez>_ref in
P_off units; paired-anchor 1-D solve phi_z rmse; prior quality with each A.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_HERE.parent))
import os
DATA = Path(os.environ.get("CAL18_DATA", "data/case_cal18"))

from tools.vasp_volumetric import read_vasp_volumetric
from pure_python.config import load_config
from pure_python.dipole_correction import (
    EwaldDipoleMixer,
    cdipol_indmin_from_center,
    cdipol_potential_1d,
    solvent_moments,
    valence_ion_dipole_cart,
)
from pure_python.grid import EDEPS, Grid, normalized_gaussian_kernel_g
from pure_python.pb import create_cavity, derived_params, local_field_factor
from pure_python.potcar import read_potcar
from pure_python.solute_potential import solute_potential_g
from pure_python.solve_from_chgcar_newton import rmse
from solve_1d_frozen_response import (
    langevin_g,
    plane_avg,
    response_coefficient_3d,
    solve_frozen,
)

FIXSOL_STEPS = 5
TOL = 1.0e-3
MAX_OUTER = 12
CG_MAX_ITER = 40


def a_from_emag(emag, s_diel3, params):
    f_loc = local_field_factor(emag, params)
    y = float(params["PBETA"]) * emag * f_loc
    g = langevin_g(y) if bool(params["LNLDIEL"]) else np.ones_like(y)
    poe = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
    return f_loc * float(params["N_MOL"]) * s_diel3 * poe


def solve_1d(a1, p_off, cvhar1, s_ion1, grid1, params, q_sol, val_ion_dipole,
             center_abs, length_z, nz):
    indmin_z = cdipol_indmin_from_center(nz, 0.5)
    mixer = EwaldDipoleMixer.fresh()
    qsol_cache = 0.0
    dsol_cache = np.zeros(3, dtype=float)
    phi_total = np.zeros(grid1.shape, dtype=float)
    for _step in range(FIXSOL_STEPS):
        dip_for_field = val_ion_dipole.copy()
        dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef_direct = mixer.ewald_dipol(dip_for_field, grid1.cell, 3)
        cvdip_z = cdipol_potential_1d(nz, length_z, ef_direct[2], indmin_z)
        phi_sol = cvhar1 + cvdip_z[None, None, :]
        phi_total, n_b, n_ion, _hist = solve_frozen(
            phi_total, phi_sol, s_ion1, a1, grid1, params, q_sol,
            TOL, MAX_OUTER, CG_MAX_ITER, p_off,
        )
        qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, grid1.cell)
    return phi_total[0, 0, :], n_b[0, 0, :] / grid1.volume


def main() -> None:
    cfg = load_config(str(REPO / "pure_python/configs/cal18.json"))
    chg = read_vasp_volumetric(str(DATA / "CHGCAR"))
    phi_ref = read_vasp_volumetric(str(DATA / "PHI"))
    rhob_ref = read_vasp_volumetric(str(DATA / "RHOB"))

    grid3 = Grid(chg.cell, chg.grid)
    valence_values = chg.values.reshape(chg.grid, order="F")
    entries = read_potcar(str(DATA / "POTCAR"))
    positions = np.asarray(cfg["positions_direct"], dtype=float)
    counts = list(cfg["counts"])
    zvals = [entry.zval for entry in entries]
    cvhar_g, dencor = solute_potential_g(grid3, valence_values, entries, counts,
                                         positions, None)
    cvhar3 = grid3.ifft_real_full(cvhar_g)
    n_e_density = (valence_values + dencor) / grid3.volume
    del dencor
    params = derived_params(cfg["solvation"])
    s_ion3, s_diel3, _ = create_cavity(n_e_density, grid3, params, None)
    del n_e_density

    nz = grid3.shape[2]
    phi_ref3 = phi_ref.values.reshape(phi_ref.grid, order="F")
    phi_ref_z = phi_ref3.mean(axis=(0, 1))

    # reference-field response (target) + anchor pair
    a3_f, ez3_f = response_coefficient_3d(phi_ref3, s_diel3, grid3, params)
    del phi_ref3
    A_ext = plane_avg(a3_f)
    pz_ref, ez_ref = plane_avg(a3_f * ez3_f), plane_avg(ez3_f)
    del a3_f, ez3_f

    # zero-field response (current design)
    a3_z, _ = response_coefficient_3d(np.zeros_like(cvhar3), s_diel3, grid3, params)
    A_form = plane_avg(a3_z)
    eps3 = 1.0 + EDEPS * a3_z

    # screened-vacuum-field response (candidate): same w_b-filtered gradient
    # as response_coefficient_3d, then isotropic pointwise screening by eps(r)
    sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    w_b3 = normalized_gaussian_kernel_g(grid3, sigma_b)
    ex, ey, ez_v, emag_v = grid3.grad_from_recip(-np.conj(w_b3) * grid3.fft(cvhar3))
    del ex, ey, w_b3
    emag_scr = emag_v / eps3
    ez_scr3 = ez_v / eps3
    del emag_v, ez_v
    a3_scr = a_from_emag(emag_scr, s_diel3, params)
    del emag_scr
    A_scr = plane_avg(a3_scr)

    # ---- report 1: A profiles ----
    print("== A(z) vs extracted (target) ==")
    print(f"rms(A_form-A_ext) = {rmse(A_form, A_ext):.4f}   max {np.abs(A_form-A_ext).max():.4f}")
    print(f"rms(A_scr -A_ext) = {rmse(A_scr, A_ext):.4f}   max {np.abs(A_scr-A_ext).max():.4f}")

    # ---- report 2: absorption term (A_ext-A)*<Ez>_ref in P_off units ----
    for name, A in (("form", A_form), ("scr", A_scr)):
        absorb = (A_ext - A) * ez_ref
        print(f"absorption[{name}]: rms {np.sqrt(np.mean(absorb**2)):.3e}  "
              f"max {np.abs(absorb).max():.3e}")

    # ---- report 3: prior quality with each pairing ----
    for name, a3x, Ax in (("chi0 +A_form (current)", a3_z, A_form),
                          ("a_scr+A_scr  (candidate)", a3_scr, A_scr)):
        p_prior = plane_avg(a3x * ez_scr3) - Ax * plane_avg(ez_scr3)
        p_exact = pz_ref - Ax * ez_ref
        print(f"prior[{name}]: rms vs exact {rmse(p_prior, p_exact):.3e}  "
              f"(peak |exact|={np.abs(p_exact).max():.3e})")

    # ---- report 4: end-to-end 1-D solves ----
    val_ion_dipole = valence_ion_dipole_cart(valence_values, positions, zvals,
                                             counts, chg.cell)
    val_ion_dipole[0:2] = 0.0
    del valence_values
    center_abs = 0.5 * chg.cell[0] + 0.5 * chg.cell[1] + 0.5 * chg.cell[2]
    length_z = float(np.linalg.norm(chg.cell[2]))
    q_sol = float(cfg["q_sol"])
    grid1 = Grid(chg.cell, (1, 1, nz))
    s_ion1 = plane_avg(s_ion3)
    cvhar1 = plane_avg(cvhar3)
    del s_ion3, cvhar3, a3_z, a3_scr, eps3, ez_scr3, grid3

    common = (cvhar1, s_ion1, grid1, params, q_sol, val_ion_dipole,
              center_abs, length_z, nz)
    print("== end-to-end (fine anchor, paired P_off): phi rmse / rhob rmse ==")
    vol3 = abs(np.linalg.det(np.asarray(chg.cell)))
    rhob_ref_z = (rhob_ref.values.reshape(rhob_ref.grid, order="F") / vol3).mean(axis=(0, 1))
    for name, A in (("A_form", A_form), ("A_scr", A_scr), ("A_ext", A_ext)):
        p_off = pz_ref - A * ez_ref
        phi_z, nb_z = solve_1d(A, p_off, *common)
        print(f"{name}: phi {rmse(phi_z, phi_ref_z):.6e}  rhob {rmse(nb_z, rhob_ref_z):.3e}")
        if name == "A_scr":
            np.savez(Path(os.environ.get("PRIOR_RUN_DIR", "pb_1d_test")) / "ascr_fine_anchor.npz",
                      phi_z=phi_z, nb_z=nb_z, rhob_ref_z=rhob_ref_z)


if __name__ == "__main__":
    main()
