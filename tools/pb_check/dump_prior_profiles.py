"""Dump z-profiles for the closure_1d report figures: exact P_off vs the
screened-vacuum prior, and the 1-D solve (phi, rho_b, rho_ion) each produces.
Env overrides: CAL18_DATA (default data/case_cal18), PRIOR_PROFILES_OUT.

Reuses solve_1d_mixed_sources.py machinery on cal_18 but skips the coarse
3-D solve (not needed): fine-source fields come from the reference PHI,
prior fields from the vacuum solute potential (cvhar) screened by eps(r).
Outputs prior_profiles.npz in the run dir.
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
from pure_python.grid import Grid, EDEPS
from pure_python.pb import create_cavity, derived_params
from pure_python.potcar import read_potcar
from pure_python.solute_potential import solute_potential_g
from pure_python.solve_from_chgcar_newton import rmse
from solve_1d_frozen_response import plane_avg, response_coefficient_3d, solve_frozen

FIXSOL_STEPS = 5
TOL = 1.0e-3
MAX_OUTER = 12
CG_MAX_ITER = 40


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
    return (phi_total[0, 0, :], n_b[0, 0, :] / grid1.volume,
            n_ion[0, 0, :] / grid1.volume)


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
    del cvhar_g
    n_e_density = (valence_values + dencor) / grid3.volume
    del dencor
    params = derived_params(cfg["solvation"])
    s_ion3, s_diel3, _ = create_cavity(n_e_density, grid3, params, None)
    del n_e_density

    nz = grid3.shape[2]
    phi_ref3 = phi_ref.values.reshape(phi_ref.grid, order="F")
    phi_ref_z = phi_ref3.mean(axis=(0, 1))

    # fine-source response fields from the reference PHI
    a3_f, ez3_f = response_coefficient_3d(phi_ref3, s_diel3, grid3, params)
    del phi_ref3
    pz_fine, ez_fine = plane_avg(a3_f * ez3_f), plane_avg(ez3_f)
    del a3_f, ez3_f

    # formula A (zero-field) and the screened-vacuum prior fields
    a3_z, _ez_z = response_coefficient_3d(np.zeros_like(cvhar3), s_diel3, grid3, params)
    a3_v, ez3_v = response_coefficient_3d(cvhar3, s_diel3, grid3, params)
    del a3_v
    eps3 = 1.0 + EDEPS * a3_z
    ez3_scr = ez3_v / eps3
    del ez3_v, eps3
    pz_scr, ez_scr = plane_avg(a3_z * ez3_scr), plane_avg(ez3_scr)
    a_formula = plane_avg(a3_z)
    del a3_z, ez3_scr

    p_exact = pz_fine - a_formula * ez_fine
    p_prior = pz_scr - a_formula * ez_scr
    print(f"P_off[vacscr] vs exact: rmse={rmse(p_prior, p_exact):.3e}"
          f"  (peak |exact|={np.abs(p_exact).max():.3e})", flush=True)

    # 1-D solves: formula A + {fine, vacscr} anchor
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
    del s_ion3, s_diel3, cvhar3

    common = (cvhar1, s_ion1, grid1, params, q_sol, val_ion_dipole,
              center_abs, length_z, nz)
    phi_fine_z, nb_fine_z, nion_fine_z = solve_1d(a_formula, p_exact, *common)
    print(f"phi_z rmse (formula A + fine anchor):   {rmse(phi_fine_z, phi_ref_z):.6e}",
          flush=True)
    phi_prior_z, nb_prior_z, nion_prior_z = solve_1d(a_formula, p_prior, *common)
    print(f"phi_z rmse (formula A + vacscr prior):  {rmse(phi_prior_z, phi_ref_z):.6e}",
          flush=True)

    z = np.arange(nz) * length_z / nz
    out = Path(os.environ.get("PRIOR_PROFILES_OUT", "pb_1d_test/prior_profiles.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        z=z,
        p_exact=p_exact,
        p_prior=p_prior,
        phi_ref_z=phi_ref_z,
        phi_fine_z=phi_fine_z,
        phi_prior_z=phi_prior_z,
        nb_fine_z=nb_fine_z,
        nion_fine_z=nion_fine_z,
        nb_prior_z=nb_prior_z,
        nion_prior_z=nion_prior_z,
        rmse_poff=rmse(p_prior, p_exact),
        peak_poff=np.abs(p_exact).max(),
        rmse_phi_fine=rmse(phi_fine_z, phi_ref_z),
        rmse_phi_prior=rmse(phi_prior_z, phi_ref_z),
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
