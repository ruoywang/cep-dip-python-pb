"""Component-level parity: torch_pb vs numpy solver on a synthetic slab.

Runs both backends full-spectrum (PB_RFFT=0), float64, CPU, and asserts
array-level agreement for FFT/grad/div, cavity, ion/bound density, the
Newton operator l_op, the residual, and a short full solve. Locates any
translation error to a single function.
"""

import os
import sys

os.environ["PB_RFFT"] = "0"
os.environ["PB_DISABLE_FUSED"] = "1"
os.environ["PB_DISABLE_FAST_LOCAL_FIELD"] = "1"

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pure_python import pb, grid as gmod
from pure_python.grid import Grid, normalized_gaussian_kernel_g
from pure_python import torch_pb as tp

torch.manual_seed(0)
np.random.seed(0)

CELL = np.array([[14.802, 0.0, 0.0], [-7.401, 12.818908, 0.0], [0.0, 0.0, 45.0]])
SHAPE = (36, 36, 108)  # small fft-friendly grid

SOLV = {
    "A_K": 0.125, "C_MOLAR": 1.0, "D_ION": -1.0, "D_STERN": 2.0, "EB_K": 78.4,
    "EPSILON_INF": 1.78, "I_NLOC_SOL": 1, "LAMBDA_D_K": 0.0, "LNLDIEL": True,
    "LNLION": True, "LNLTEST": False, "LVAC": True, "NC_K": 0.015, "N_MIN": 1e-4,
    "N_MOL": 0.0335, "P_MOL": 0.5, "R_B": 0.0, "R_CAV": 0.0, "R_DIEL": 1.0,
    "R_ION": 4.0, "R_SOLV": 1.4, "SIGMA_K": 0.6, "SOLTEMP": 298.0, "SOL_SIGMA": 0.8,
    "SOL_Z0": 7.0, "SOL_Z1": 30.0, "TAU": 0.009, "ZION": 1.0,
}


def _synth_density(g: Grid) -> np.ndarray:
    z = g._cartesian_z_mesh
    # a smooth slab-like electron density, positive
    return 0.3 * np.exp(-0.5 * ((z - 0.42 * CELL[2, 2]) / 4.0) ** 2) + 1e-3


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = max(np.max(np.abs(b)), 1e-30)
    return float(np.max(np.abs(a - b)) / denom)


def main() -> None:
    ng = Grid(CELL, SHAPE)
    tg = tp.TorchGrid(CELL, SHAPE, device="cpu", dtype=torch.float64)
    params = pb.derived_params(SOLV)

    def t2n(x):
        return x.detach().cpu().numpy()

    results = {}

    # 1. FFT round-trip amplitude convention
    field = _synth_density(ng)
    fn = ng.fft(field)
    ft = t2n(tg.fft(torch.as_tensor(field)))
    results["fft"] = _rel(ft, fn)
    results["ifft"] = _rel(t2n(tg.ifft_real(torch.as_tensor(fn))), ng.ifft_real(fn))

    # 2. grad / div
    phi = np.sin(2 * np.pi * ng._cartesian_z_mesh / CELL[2, 2]) * 0.1
    phi_g = ng.fft(phi)
    exn, eyn, ezn, magn = ng.grad_from_recip(phi_g)
    ext, eyt, ezt, magt = tg.grad_from_recip(tg.fft(torch.as_tensor(phi)))
    results["grad_ez"] = _rel(t2n(ezt), ezn)
    results["grad_mag"] = _rel(t2n(magt), magn)
    divn = ng.div_real_vector(exn, eyn, ezn)
    divt = tg.div_real_vector(ext, eyt, ezt)
    results["div"] = _rel(t2n(divt), divn)

    # 3. cavity
    ne = _synth_density(ng)
    s_ion_n, s_diel_n, s_cav_n = pb.create_cavity(ne, ng, params)
    s_ion_t, s_diel_t, s_cav_t = tp.create_cavity_torch(torch.as_tensor(ne), tg, params)
    results["s_ion"] = _rel(t2n(s_ion_t), s_ion_n)
    results["s_diel"] = _rel(t2n(s_diel_t), s_diel_n)

    # 4. ion density from a trial phi
    trial_phi = 0.2 * np.sin(2 * np.pi * ng._cartesian_z_mesh / CELL[2, 2] + 0.3)
    ni_n = pb.ion_density_values_from_phi(trial_phi, s_ion_n, ng, params)
    ni_t = tp.ion_density_values_from_phi_torch(torch.as_tensor(trial_phi), s_ion_t, tg, params)
    results["ion_density"] = _rel(t2n(ni_t), ni_n)

    # 5. bound density
    nb_n = pb.bound_density_values_from_phi(trial_phi, s_diel_n, ng, params)
    w_b_t = tp._normalized_gaussian_kernel_g(tg, params["A_K"])
    fq = tp._field_quantities(torch.as_tensor(trial_phi), s_ion_t, s_diel_t, tg, params, w_b_t)
    results["bound_density"] = _rel(t2n(fq["n_b"]), nb_n)

    # 6. l_op (Newton operator) on a trial dphi_g
    w_b_n = normalized_gaussian_kernel_g(ng, params["A_K"])
    fields_n = pb.nlpb_field_quantities(trial_phi, s_ion_n, s_diel_n, ng, params, w_b_n)
    resp_n, ek_n = pb.nlpb_response_from_fields(fields_n, s_ion_n, s_diel_n, ng, params)
    resp_t, ek_t = tp._response_from_fields(fq, s_ion_t, s_diel_t, tg, params)
    if ek_n is not None:
        results["ekappa2"] = _rel(t2n(ek_t), ek_n)
    dphi_g_n = ng.fft(0.05 * np.cos(2 * np.pi * ng._cartesian_z_mesh / CELL[2, 2]))
    lop_n = pb.l_op(dphi_g_n, resp_n, ek_n, w_b_n, ng)
    lop_t = tp._l_op(tg.fft(torch.as_tensor(0.05 * np.cos(2 * np.pi * ng._cartesian_z_mesh / CELL[2, 2]))), resp_t, ek_t, w_b_t, tg)
    results["l_op"] = _rel(t2n(lop_t), lop_n)

    # 7. short end-to-end solve (2 outer iters), compare phi
    q_sol = 1.0
    phi_sol = np.zeros(SHAPE)
    # a nonzero solute potential so the solve does something
    phi_sol = 0.5 * np.exp(-0.5 * ((ng._cartesian_z_mesh - 0.42 * CELL[2, 2]) / 3.0) ** 2)
    pt_n, nb2_n, ni2_n, _, hist_n = _solve_numpy(phi_sol, s_ion_n, s_diel_n, ng, params, q_sol)
    pt_t, nb2_t, ni2_t, _, hist_t = tp.solve_nlpb_for_phi_sol_torch(
        torch.zeros(SHAPE, dtype=torch.float64), torch.as_tensor(phi_sol),
        s_ion_t, s_diel_t, tg, params, q_sol, 1e-3, 20, 200,
    )
    results["solve_phi"] = _rel(t2n(pt_t), pt_n)
    results["solve_rhoion"] = _rel(t2n(ni2_t), ni2_n)
    print(f"numpy last rms={hist_n[-1][1]:.3e}  torch last rms={hist_t[-1][1]:.3e}")

    print("\n=== component max relative diff (torch vs numpy) ===")
    worst = 0.0
    for k, v in results.items():
        flag = "OK" if v < 1e-9 else ("~" if v < 1e-5 else "FAIL")
        worst = max(worst, v)
        print(f"  {k:16s} {v:.2e}  {flag}")
    ok = all(v < 1e-6 for v in results.values())
    print(f"\nworst = {worst:.2e}  ->  {'ALL PASS' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


def _solve_numpy(phi_sol, s_ion, s_diel, ng, params, q_sol):
    from pure_python.solve_from_chgcar_newton import solve_nlpb_for_phi_sol
    return solve_nlpb_for_phi_sol(
        np.zeros(SHAPE), phi_sol, s_ion, s_diel, ng, params, q_sol, 1e-3, 20, 200,
    )


if __name__ == "__main__":
    main()
