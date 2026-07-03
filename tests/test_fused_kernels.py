"""Numerical equivalence test: fused C kernels vs the NumPy reference path.

Run from the repo root:  PYTHONPATH=. python tests/test_fused_kernels.py
Toggles PB_DISABLE_FUSED per call, so one process checks both paths.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pure_python.grid import Grid, normalized_gaussian_kernel_g
from pure_python import pb


def with_fused(flag: bool):
    os.environ["PB_DISABLE_FUSED"] = "0" if flag else "1"


def main() -> None:
    rng = np.random.default_rng(7)
    cell = np.array([[9.3, 0.0, 0.0], [4.6, 8.1, 0.0], [0.0, 0.0, 17.0]])
    shape = (24, 20, 28)
    grid = Grid(cell, shape)

    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "pure_python", "configs", "cal18.json")))
    params = pb.derived_params(cfg["solvation"])

    # smooth-ish random fields
    def smooth(a):
        g = grid.fft(a)
        _, _, _, gsq = grid.reciprocal_mesh()
        return grid.ifft_real(g * np.exp(-0.15 * gsq))

    phi = smooth(rng.standard_normal(shape)) * 0.3
    s_ion = np.clip(smooth(rng.random(shape)), 0.0, 1.0)
    s_diel = np.clip(smooth(rng.random(shape)), 0.0, 1.0)
    w_b = normalized_gaussian_kernel_g(grid, float(params["A_K"]))

    failures = []

    def check(name, a, b, tol=1e-11):
        a = np.asarray(a)
        b = np.asarray(b)
        denom = np.maximum(np.max(np.abs(b)), 1e-300)
        err = np.max(np.abs(a - b)) / denom
        status = "OK " if err <= tol else "FAIL"
        print(f"[{status}] {name:34s} max rel err {err:.3e}")
        if err > tol:
            failures.append(name)

    # ---- nlpb_quantities: full outputs ----
    with_fused(True)
    nb1, ni1, resp1, ek1, _ = pb.nlpb_quantities(phi, s_ion, s_diel, grid, params, w_b=w_b)
    with_fused(False)
    nb0, ni0, resp0, ek0, _ = pb.nlpb_quantities(phi, s_ion, s_diel, grid, params, w_b=w_b)
    check("nlpb n_b", nb1, nb0)
    check("nlpb n_ion", ni1, ni0)
    check("nlpb ekappa2", ek1, ek0)
    assert resp1[0] == resp0[0] == "tensor_field"
    for k, nmk in ((1, "chi_perp"), (2, "chi_factor"), (3, "ex"), (4, "ey"), (5, "ez")):
        check(f"nlpb response {nmk}", resp1[k], resp0[k])

    # ---- l_op on a random complex vector ----
    dphi_g = grid.fft(smooth(rng.standard_normal(shape))) + 1j * grid.fft(smooth(rng.standard_normal(shape)))
    dphi_g = np.ascontiguousarray(dphi_g)
    with_fused(True)
    lp1 = pb.l_op(dphi_g, resp1, ek1, w_b, grid)
    with_fused(False)
    lp0 = pb.l_op(dphi_g, resp0, ek0, w_b, grid)
    check("l_op", lp1, lp0)
    with_fused(True)
    lp1n = pb.l_op(dphi_g, resp1, None, w_b, grid)
    with_fused(False)
    lp0n = pb.l_op(dphi_g, resp0, None, w_b, grid)
    check("l_op no-ekappa", lp1n, lp0n)

    # ---- minimize_l: full CG solve ----
    q_sol = float(cfg["q_sol"])
    phi_solv_g = grid.fft(phi)
    resid, rms = pb.residual_g(phi_solv_g, nb0, ni0, q_sol, grid)
    with_fused(True)
    d1, rms1, it1 = pb.minimize_l(resid, resp1, ek1, w_b, grid, rms / 10.0, 25)
    with_fused(False)
    d0, rms0, it0 = pb.minimize_l(resid, resp0, ek0, w_b, grid, rms / 10.0, 25)
    print(f"       CG iters fused/ref: {it1}/{it0}, rms {rms1:.6e}/{rms0:.6e}")
    check("minimize_l dphi", d1, d0, tol=1e-7)

    # ---- ion density direct ----
    with_fused(True)
    a = pb.ion_density_values_from_phi(phi, s_ion, grid, params)
    with_fused(False)
    b = pb.ion_density_values_from_phi(phi, s_ion, grid, params)
    check("ion_density_values_from_phi", a, b)

    # ---- grid ops ----
    with_fused(True)
    e1 = grid.grad_from_recip(np.ascontiguousarray(-np.conj(w_b) * grid.fft(phi)))
    with_fused(False)
    e0 = grid.grad_from_recip(-np.conj(w_b) * grid.fft(phi))
    for k, nmk in enumerate(("ex", "ey", "ez", "emag")):
        check(f"grad_from_recip {nmk}", e1[k], e0[k])
    vx, vy, vz = (smooth(rng.standard_normal(shape)) for _ in range(3))
    with_fused(True)
    dv1 = grid.div_real_vector(vx, vy, vz)
    with_fused(False)
    dv0 = grid.div_real_vector(vx, vy, vz)
    check("div_real_vector", dv1, dv0)

    with_fused(True)
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("ALL FUSED KERNELS MATCH")


if __name__ == "__main__":
    main()
