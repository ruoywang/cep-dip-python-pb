"""Two diagnostics for the P_off/P_z discussion:
(1) absorption term (A_ext-A_scr)*<Ez>_ref and the pure covariance
    (extracted-A pairing) — extra curves for the P_off panel;
(2) decompose the scr-solve's P_z deviation (constitutive-law units):
    dP = A_scr*(E_solve - E_ref) + (prior - exact_paired)
    -> which part drives the P_z panel difference, the field or P_off?"""
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
from pure_python.grid import Grid, normalized_gaussian_kernel_g
from pure_python.pb import derived_params

HERE = Path(os.environ.get("PRIOR_RUN_DIR", "pb_1d_test"))
u = np.load(HERE / "unified_scr.npz")
a = np.load(HERE / "a_profiles.npz")

z = u["z"].ravel()
A_ext = u["A_ext"].ravel()
A_scr = u["A_scr"].ravel()
p_exact = u["p_exact"].ravel()   # pz_ref - A_scr*ez_ref  (cov + absorption)
p_prior = u["p_prior"].ravel()   # pure covariance of the prior (mean term = 0)
ez_ref = a["ez_ref"].ravel()

absorption = (A_ext - A_scr) * ez_ref
cov_exact = p_exact - absorption          # = pz_ref - A_ext*ez_ref

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

print("== P_off panel extras ==")
print(f"absorption: rms {rms(absorption):.3e}  peak {np.abs(absorption).max():.3e}")
print(f"cov_exact:  peak {np.abs(cov_exact).max():.3e}")
print(f"prior vs cov_exact:      rms {rms(p_prior - cov_exact):.3e}")
print(f"prior vs exact(paired):  rms {rms(p_prior - p_exact):.3e}")

# ---- (2) field vs P_off split of the P_z deviation ----
cfg = load_config(str(REPO / "pure_python/configs/cal18.json"))
chg = read_vasp_volumetric(str(DATA / "CHGCAR"))
params = derived_params(cfg["solvation"])
nz = len(z)
grid1 = Grid(chg.cell, (1, 1, nz))
sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
w_b1 = normalized_gaussian_kernel_g(grid1, sigma_b)

def wb_field(phi_z):
    phi3 = phi_z.reshape(1, 1, nz)
    _, _, ez, _ = grid1.grad_from_recip(-np.conj(w_b1) * grid1.fft(phi3))
    return ez.ravel()

ez_solve = wb_field(u["phi_scr"].ravel())
ez_ref1d = wb_field(u["phi_ref_z"].ravel())  # reference field via the SAME 1-D operator
term_field = A_scr * (ez_solve - ez_ref1d)
term_poff = p_prior - p_exact
total = term_field + term_poff

print("== P_z deviation split (constitutive units) ==")
print(f"field term  A*(E_solve-E_ref): rms {rms(term_field):.3e}  peak {np.abs(term_field).max():.3e}")
print(f"P_off term  (prior-exact):     rms {rms(term_poff):.3e}  peak {np.abs(term_poff).max():.3e}")
print(f"total:                         rms {rms(total):.3e}")
# note: ez_ref (3-D plane-avg) vs ez_ref1d (1-D operator on phi_ref_z) sanity
print(f"[sanity] ez_ref(3D avg) vs ez_ref(1D op): rms diff {rms(ez_ref - ez_ref1d):.3e} "
      f"(field scale rms {rms(ez_ref):.3e})")

np.savez(HERE / "poff_decomp.npz",
         z=z, absorption=absorption, cov_exact=cov_exact,
         term_field=term_field, term_poff=term_poff)
print("saved poff_decomp.npz")
