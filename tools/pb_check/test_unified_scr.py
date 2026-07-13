"""Unified three-way comparison data for the closure_1d report figures:
DFT reference / naive 1-D / fully screened-vacuum-consistent 1-D
(A_scr = <a(|E_scr|)>, prior paired with A_scr). Also the naive solve's
effective A (response evaluated at its own converged 1-D mean field).

Outputs unified_scr.npz + printed cross-checks."""
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
from pure_python.solve_from_chgcar_newton import rmse, solve_nlpb_for_phi_sol
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


def a_from_emag(emag, s_diel, params):
    f_loc = local_field_factor(emag, params)
    y = float(params["PBETA"]) * emag * f_loc
    g = langevin_g(y) if bool(params["LNLDIEL"]) else np.ones_like(y)
    poe = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
    return f_loc * float(params["N_MOL"]) * s_diel * poe


def fixsol_loop(solver_step, grid1, cvhar1, val_ion_dipole, center_abs,
                length_z, nz):
    indmin_z = cdipol_indmin_from_center(nz, 0.5)
    mixer = EwaldDipoleMixer.fresh()
    qsol_cache = 0.0
    dsol_cache = np.zeros(3, dtype=float)
    phi_total = np.zeros(grid1.shape, dtype=float)
    n_b = np.zeros(grid1.shape, dtype=float)
    n_ion = np.zeros(grid1.shape, dtype=float)
    for step in range(FIXSOL_STEPS):
        dip_for_field = val_ion_dipole.copy()
        dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef_direct = mixer.ewald_dipol(dip_for_field, grid1.cell, 3)
        cvdip_z = cdipol_potential_1d(nz, length_z, ef_direct[2], indmin_z)
        phi_sol = cvhar1 + cvdip_z[None, None, :]
        phi_total, n_b, n_ion = solver_step(phi_total, phi_sol, step)
        qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, grid1.cell)
    return phi_total, n_b, n_ion


def main() -> None:
    cfg = load_config(str(REPO / "pure_python/configs/cal18.json"))
    chg = read_vasp_volumetric(str(DATA / "CHGCAR"))
    phi_ref = read_vasp_volumetric(str(DATA / "PHI"))

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

    # reference response + anchor
    a3_f, ez3_f = response_coefficient_3d(phi_ref3, s_diel3, grid3, params)
    del phi_ref3
    A_ext = plane_avg(a3_f)
    pz_ref, ez_ref = plane_avg(a3_f * ez3_f), plane_avg(ez3_f)
    del a3_f, ez3_f

    # fully screened-vacuum-consistent ingredients
    a3_z, _ = response_coefficient_3d(np.zeros_like(cvhar3), s_diel3, grid3, params)
    eps3 = 1.0 + EDEPS * a3_z
    del a3_z
    sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    w_b3 = normalized_gaussian_kernel_g(grid3, sigma_b)
    _, _, ez_v, emag_v = grid3.grad_from_recip(-np.conj(w_b3) * grid3.fft(cvhar3))
    del w_b3
    emag_scr = emag_v / eps3
    ez_scr3 = ez_v / eps3
    del emag_v, ez_v, eps3
    a3_scr = a_from_emag(emag_scr, s_diel3, params)
    del emag_scr
    A_scr = plane_avg(a3_scr)
    p_prior = plane_avg(a3_scr * ez_scr3) - A_scr * plane_avg(ez_scr3)
    del a3_scr, ez_scr3
    p_exact = pz_ref - A_scr * ez_ref
    print(f"A_scr vs ext: rms {rmse(A_scr, A_ext):.4f}")
    print(f"prior(scr-consistent) vs exact: rms {rmse(p_prior, p_exact):.3e}"
          f"  (peak |exact|={np.abs(p_exact).max():.3e})")

    # 1-D reductions
    val_ion_dipole = valence_ion_dipole_cart(valence_values, positions, zvals,
                                             counts, chg.cell)
    val_ion_dipole[0:2] = 0.0
    del valence_values
    center_abs = 0.5 * chg.cell[0] + 0.5 * chg.cell[1] + 0.5 * chg.cell[2]
    length_z = float(np.linalg.norm(chg.cell[2]))
    q_sol = float(cfg["q_sol"])
    grid1 = Grid(chg.cell, (1, 1, nz))
    s_ion1 = plane_avg(s_ion3)
    s_diel1 = plane_avg(s_diel3)
    cvhar1 = plane_avg(cvhar3)
    del s_ion3, s_diel3, cvhar3, grid3

    # screened-vacuum-consistent solve (frozen A_scr + paired prior)
    def scr_step(phi_total, phi_sol, step):
        phi_total, n_b, n_ion, _h = solve_frozen(
            phi_total, phi_sol, s_ion1, A_scr, grid1, params, q_sol,
            TOL, MAX_OUTER, CG_MAX_ITER, p_prior)
        return phi_total, n_b, n_ion

    phi_s, nb_s, nion_s = fixsol_loop(scr_step, grid1, cvhar1, val_ion_dipole,
                                      center_abs, length_z, nz)
    print(f"phi_z rmse (scr-consistent solve): {rmse(phi_s[0,0,:], phi_ref_z):.4f}")

    # naive solve (full nonlinear production solver on the 1-D grid)
    def naive_step(phi_total, phi_sol, step):
        phi_total, n_b, n_ion, _, _h = solve_nlpb_for_phi_sol(
            phi_total, phi_sol, s_ion1, s_diel1, grid1, params, q_sol,
            TOL, MAX_OUTER, CG_MAX_ITER, None, step)
        return phi_total, n_b, n_ion

    phi_n, nb_n, nion_n = fixsol_loop(naive_step, grid1, cvhar1, val_ion_dipole,
                                      center_abs, length_z, nz)
    print(f"phi_z rmse (naive solve, cross-check ~0.65): "
          f"{rmse(phi_n[0,0,:], phi_ref_z):.4f}")

    # naive's effective A: response at its own converged 1-D field
    a_nv, _ = response_coefficient_3d(phi_n, s_diel1, grid1, params)
    A_naive = a_nv[0, 0, :]

    z = np.arange(nz) * length_z / nz
    out = Path(os.environ.get("UNIFIED_SCR_OUT", "pb_1d_test/unified_scr.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             z=z, phi_ref_z=phi_ref_z,
             A_ext=A_ext, A_scr=A_scr, A_naive=A_naive,
             p_exact=p_exact, p_prior=p_prior,
             phi_scr=phi_s[0, 0, :],
             nb_scr=nb_s[0, 0, :] / grid1.volume,
             nion_scr=nion_s[0, 0, :] / grid1.volume,
             phi_naive=phi_n[0, 0, :],
             nb_naive=nb_n[0, 0, :] / grid1.volume,
             nion_naive=nion_n[0, 0, :] / grid1.volume)
    print(out)


if __name__ == "__main__":
    main()
