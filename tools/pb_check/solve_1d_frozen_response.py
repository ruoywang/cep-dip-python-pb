"""1D PB with a frozen dielectric response profile extracted from a 3D field.

The planar-reduction failure mode is the nonlinear response: chi(E) saturates
at the strong local 3D interface fields, and averaging the FIELD first (then
applying chi) overshoots the bound charge by ~7x at the peak. Here the
nonlinearity is evaluated in 3D instead: the combined response coefficient

    A = f_loc * N_MOL * s_diel * (alpha0_rot * g(y) + alpha_pol) / EDEPS
    (P = A * E, with E the w_b-filtered raw gradient of phi)

is computed on the 3D grid from a supplied 3D potential, plane-averaged, and
frozen. The 1D solve then treats the bound charge as the LINEAR operator
-w_b d/dz (A(z) dphi_w/dz) with the exact Jacobian ("scalar" response =
A*EDEPS); the ion term stays fully nonlinear (it survived the plain-1D test).

Two averaging modes:
  --avg plain : A1(z) = <A>_xy
  --avg field : A1(z) = <A*Ez>_xy / <Ez>_xy where |<Ez>| is significant
                (exact for reproducing the reference plane-averaged P_z),
                blended to <A>_xy elsewhere.

The 3D potential source: --phi3d <file> (default: the DFT reference PHI —
the oracle test; a coarse 3D solve can be substituted later).
"""
from __future__ import annotations

import argparse
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
from pure_python.grid import EDEPS, Grid, fused_kernel, normalized_gaussian_kernel_g
from pure_python.pb import (
    create_cavity,
    derived_params,
    ion_density_values_from_phi,
    local_field_factor,
    minimize_l,
    residual_g,
)
from pure_python.potcar import read_potcar
from pure_python.solute_potential import solute_potential_g
from pure_python.solve_from_chgcar_newton import rmse


def plane_avg(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.mean(axis=(0, 1)).reshape(1, 1, -1))


def langevin_g(y: np.ndarray) -> np.ndarray:
    g = np.empty_like(y)
    small = y < 2.0e-4
    large = y > 100.0
    mid = ~(small | large)
    g[small] = 1.0
    g[large] = 3.0 * (1.0 - 1.0 / y[large]) / y[large]
    g[mid] = 3.0 * (1.0 / np.tanh(y[mid]) - 1.0 / y[mid]) / y[mid]
    return g


def response_coefficient_3d(phi3: np.ndarray, s_diel3: np.ndarray, grid3: Grid, params: dict):
    """A(r) and E_z(r) from a 3D potential: P = A * E (E = w_b-filtered gradient)."""
    w_b3 = normalized_gaussian_kernel_g(
        grid3, float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    )
    phi_g = grid3.fft(phi3)
    ex, ey, ez, emag = grid3.grad_from_recip(-np.conj(w_b3) * phi_g)
    f_loc = local_field_factor(emag, params)
    y = float(params["PBETA"]) * emag * f_loc
    if bool(params["LNLDIEL"]):
        g = langevin_g(y)
    else:
        g = np.ones_like(y)
    polar_over_eps = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
    a3 = f_loc * float(params["N_MOL"]) * s_diel3 * polar_over_eps
    return a3, ez


def ekappa2_1d(phi: np.ndarray, s_ion: np.ndarray, params: dict) -> np.ndarray | None:
    if not bool(params["LION"]):
        return None
    kern = fused_kernel("ekappa2_values")
    if kern is not None:
        return kern(
            np.ascontiguousarray(phi), np.ascontiguousarray(s_ion),
            float(params["ZBETA"]), float(params["theta_b"]), float(params["n_max"]),
            float(params["alpha0_ion"]), 1 if bool(params["LNLION"]) else 0,
        )
    x_ion = float(params["ZBETA"]) * phi
    theta = float(params["theta_b"])
    if bool(params["LNLION"]):
        ek = np.zeros_like(phi)
        not_large = np.abs(x_ion) <= 100.0
        x2 = np.empty_like(phi)
        small = np.abs(x_ion) < 2.0e-4
        x2[small] = 0.5 * x_ion[small] ** 2
        x2[~small] = np.cosh(np.clip(x_ion[~small], -100.0, 100.0)) - 1.0
        ek[not_large] = (1.0 + (1.0 - theta) * x2[not_large]) / (1.0 + theta * x2[not_large]) ** 2
    else:
        ek = np.ones_like(phi)
    return float(params["n_max"]) * float(params["alpha0_ion"]) * s_ion * ek


def frozen_densities(phi, s_ion, a1, grid, params, w_b):
    """n_b with frozen response A(z); n_ion fully nonlinear."""
    phi_g = grid.fft(np.ascontiguousarray(phi))
    ex, ey, ez, _ = grid.grad_from_recip(-np.conj(w_b) * phi_g)
    n_ion = ion_density_values_from_phi(phi, s_ion, grid, params)
    div_p = grid.div_real_vector(a1 * ex, a1 * ey, a1 * ez)
    n_b = grid.ifft_real(-w_b * div_p) * grid.volume
    return n_b, n_ion


def solve_frozen(phi_total, phi_sol, s_ion, a1, grid, params, q_sol, tol, max_outer, cg_max_iter):
    phi_solv_g = grid.fft(phi_total - phi_sol)
    w_b = normalized_gaussian_kernel_g(
        grid, float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    )
    response = ("scalar", np.ascontiguousarray(a1 * EDEPS))
    history = []
    n_b, n_ion = frozen_densities(phi_total, s_ion, a1, grid, params, w_b)
    for outer in range(max_outer + 1):
        resid, rms_val = residual_g(phi_solv_g, n_b, n_ion, q_sol, grid)
        if rms_val < tol and outer >= 1:
            history.append((outer, rms_val, 0, 0.0))
            break
        ekappa2 = ekappa2_1d(phi_total, s_ion, params)
        dphi_g, _, cg_iter = minimize_l(resid, response, ekappa2, w_b, grid,
                                        max(rms_val / 10.0, tol), cg_max_iter)
        dphi_real = grid.ifft_real(dphi_g)
        alpha = 1.0
        accepted_rms = float("inf")
        for _ in range(7):
            trial_phi = phi_total + alpha * dphi_real
            trial_solv_g = phi_solv_g + alpha * dphi_g
            tb, ti = frozen_densities(trial_phi, s_ion, a1, grid, params, w_b)
            _, trial_rms = residual_g(trial_solv_g, tb, ti, q_sol, grid)
            if trial_rms <= rms_val or alpha <= 1.0 / 64.0:
                phi_total = trial_phi
                phi_solv_g = trial_solv_g
                n_b, n_ion = tb, ti
                accepted_rms = trial_rms
                break
            alpha *= 0.5
        history.append((outer, rms_val, cg_iter, accepted_rms))
    return phi_total, n_b, n_ion, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--rhoion-ref", default="data/case_cal18/RHOION")
    parser.add_argument("--phi3d", default=None,
                        help="3D potential file for response extraction (default: --phi-ref)")
    parser.add_argument("--avg", choices=("plain", "field"), default="field")
    parser.add_argument("--out-dir", default="pb_1d_test/frozen")
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
    mark("solute_potential_3d")

    params = derived_params(cfg["solvation"])
    s_ion3, s_diel3, _ = create_cavity(n_e_density, grid3, params, None)
    mark("cavity_3d")

    phi3 = (read_vasp_volumetric(args.phi3d) if args.phi3d else phi_ref)
    phi3_values = phi3.values.reshape(phi3.grid, order="F")
    a3, ez3 = response_coefficient_3d(phi3_values, s_diel3, grid3, params)
    a_plain = plane_avg(a3)
    if args.avg == "field":
        pz = plane_avg(a3 * ez3)
        ez = plane_avg(ez3)
        a1 = a_plain.copy()
        scale = np.max(np.abs(ez))
        mask = np.abs(ez) > 0.02 * scale
        a1[mask] = pz[mask] / ez[mask]
        a1 = np.clip(a1, 0.0, None)
    else:
        a1 = a_plain
    mark("extract_response")

    nz = grid3.shape[2]
    grid1 = Grid(chg.cell, (1, 1, nz))
    s_ion1 = plane_avg(s_ion3)
    cvhar1 = plane_avg(cvhar3)

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
        phi_total, n_b, n_ion, history = solve_frozen(
            phi_total, phi_sol, s_ion1, a1, grid1, params, q_sol,
            args.tol, args.max_outer, args.cg_max_iter,
        )
        qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, chg.cell)
        dt = perf_counter() - t_fix
        last_outer, last_rms, _, _ = history[-1]
        lines.append(
            f"{step}\t{last_outer}\t{last_rms:.12e}\t{ef_direct[2]:.12e}\t{qsol_cache:.12e}\t{dsol_cache[2]:.12e}\t"
            f"{rmse(phi_total[0,0,:], phi_ref_z):.12e}\t{rmse(n_b[0,0,:]/grid1.volume, rhob_ref_z):.12e}\t"
            f"{rmse(n_ion[0,0,:]/grid1.volume, rhoion_ref_z):.12e}\t{dt:.6f}"
        )
        (out / "fixstep_summary.tsv").write_text("\n".join(lines) + "\n")
    mark("fixsteps_1d_frozen")

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
        },
    )
    print(out / "fixstep_summary.tsv")


if __name__ == "__main__":
    main()
