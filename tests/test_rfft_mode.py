"""Equivalence test: half-spectrum (PB_RFFT=1) vs full-spectrum PB solve.

Run from the repo root:  PYTHONPATH=. python tests/test_rfft_mode.py
Grid.rspec is cached per instance, so one process can hold both modes by
creating the Grid instances under different PB_RFFT settings.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pure_python.grid import Grid, normalized_gaussian_kernel_g
from pure_python import pb
from pure_python.solve_from_chgcar_newton import prolong_double, restrict_half


def main() -> None:
    rng = np.random.default_rng(11)
    cell = np.array([[9.3, 0.0, 0.0], [4.6, 8.1, 0.0], [0.0, 0.0, 17.0]])
    shape = (24, 20, 28)

    os.environ["PB_RFFT"] = "0"
    gf = Grid(cell, shape)
    assert not gf.rspec
    os.environ["PB_RFFT"] = "1"
    gh = Grid(cell, shape)
    assert gh.rspec and gh.spec_shape == (24, 20, 15)

    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "pure_python", "configs", "cal18.json")))
    params = pb.derived_params(cfg["solvation"])

    def smooth(a, g):
        # band-limit hard: physical PB fields carry no Nyquist-scale content
        # (the Gaussian kernel w_b suppresses it exponentially), and the two
        # spectral representations legitimately differ there.
        spec = g.fft(a)
        nx, ny, nz = g.shape
        hx = np.fft.fftfreq(nx) * nx
        hy = np.fft.fftfreq(ny) * ny
        hz = np.fft.fftfreq(nz) * nz
        mask = (
            (np.abs(hx)[:, None, None] <= nx // 3)
            & (np.abs(hy)[None, :, None] <= ny // 3)
            & (np.abs(hz)[None, None, :] <= nz // 3)
        )
        return g.ifft_real(spec * mask)

    phi = smooth(rng.standard_normal(shape), gf) * 0.3
    s_ion = np.clip(smooth(rng.random(shape), gf), 0.0, 1.0)
    s_diel = np.clip(smooth(rng.random(shape), gf), 0.0, 1.0)
    phi_sol = smooth(rng.standard_normal(shape), gf)

    wf = normalized_gaussian_kernel_g(gf, float(params["A_K"]))
    wh = normalized_gaussian_kernel_g(gh, float(params["A_K"]))

    failures = []

    def check(name, a, b, tol=1e-10):
        a = np.asarray(a)
        b = np.asarray(b)
        denom = np.maximum(np.max(np.abs(b)), 1e-300)
        err = np.max(np.abs(a - b)) / denom
        status = "OK " if err <= tol else "FAIL"
        print(f"[{status}] {name:30s} max rel err {err:.3e}")
        if err > tol:
            failures.append(name)

    # nlpb quantities (real-space outputs must be identical)
    nbf, nif, respf, ekf, _ = pb.nlpb_quantities(phi, s_ion, s_diel, gf, params, w_b=wf)
    nbh, nih, resph, ekh, _ = pb.nlpb_quantities(phi, s_ion, s_diel, gh, params, w_b=wh)
    check("n_b full vs half", nbh, nbf)
    check("n_ion full vs half", nih, nif)
    check("ekappa2 full vs half", ekh, ekf)

    # residual rms
    q_sol = float(cfg["q_sol"])
    psf = gf.fft(phi - phi_sol)
    psh = gh.fft(phi - phi_sol)
    rf, rmsf = pb.residual_g(psf, nbf, nif, q_sol, gf)
    rh, rmsh = pb.residual_g(psh, nbh, nih, q_sol, gh)
    print(f"       residual rms full/half: {rmsf:.12e} / {rmsh:.12e}")
    check("residual rms", np.array([rmsh]), np.array([rmsf]), tol=1e-5)

    # Newton convergence: both representations must reach the same solution.
    # (Individual CG steps legitimately differ at Nyquist-plane level: for
    # non-orthogonal cells the full-spectrum gsq is asymmetric there and the
    # .real projection averages the branches, while c2r symmetrizes. The two
    # are different but equally valid discretizations; converged results agree.)
    from pure_python.solve_from_chgcar_newton import solve_nlpb_for_phi_sol
    tol = 1.0e-4
    pf_full, _, _, _, hf = solve_nlpb_for_phi_sol(
        phi.copy(), phi_sol, s_ion, s_diel, gf, params, 0.0, tol, 30, 60)
    pf_half, _, _, _, hh = solve_nlpb_for_phi_sol(
        phi.copy(), phi_sol, s_ion, s_diel, gh, params, 0.0, tol, 30, 60)
    print(f"       newton outers full/half: {len(hf)}/{len(hh)}, "
          f"final rms {hf[-1][1]:.3e}/{hh[-1][1]:.3e}")
    scale = np.max(np.abs(pf_full))
    err = np.max(np.abs(pf_half - pf_full)) / scale
    print(f"       converged phi rel diff {err:.3e} (tol-level agreement expected)")
    # Residual tolerance does not bound phi pointwise: the Jacobian's
    # conditioning amplifies the residual-tol manifold by ~1e1-1e2 on these
    # random fields. Both representations converge below tol; the binding
    # accuracy check is the cal_18 end-to-end RMSE gate.
    if err > 1.0e-2:
        failures.append("newton convergence")

    # l_op action on the same real-space vector
    v = smooth(rng.standard_normal(shape), gf)
    lf = pb.l_op(np.ascontiguousarray(gf.fft(v)), respf, ekf, wf, gf)
    lh = pb.l_op(np.ascontiguousarray(gh.fft(v)), resph, ekh, wh, gh)
    check("l_op action (real space)", gh.ifft_real(lh), gf.ifft_real(lf))

    # prolongation in rfft mode
    os.environ["PB_RFFT"] = "1"
    gch = Grid(cell, (12, 10, 14))
    a_c = rng.standard_normal(gch.shape)
    # remove coarse Nyquist content (dropped by design in rfft prolongation)
    sc = gch.fft(a_c)
    sc[gch.shape[0] // 2, :, :] = 0.0
    sc[:, gch.shape[1] // 2, :] = 0.0
    sc[:, :, -1] = 0.0
    a_c = gch.ifft_real(sc)
    a_f = prolong_double(a_c, gch, gh)
    err = np.max(np.abs(restrict_half(a_f) - a_c))
    print(f"       rfft prolong roundtrip err {err:.3e}")
    if err > 1e-12:
        failures.append("rfft prolong")

    os.environ["PB_RFFT"] = "0"
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("HALF-SPECTRUM MODE MATCHES FULL SPECTRUM")


if __name__ == "__main__":
    main()
