"""Grid-convergence study of the PB solve on cal_18 (rule-4 style).

Question: how coarse a grid still reproduces the reference solvent response,
for the MACE training use case that consumes the laterally averaged ionic
profile rho_ion(z) (and its moments)?

Method: spectrally restrict the cal_18 CHGCAR valence density to a series of
coarser FFT-friendly grids (target spacings given below), rebuild the solute
potential and cavity on each grid with the SAME validated pipeline (POTCAR
local pseudopotential + dencor evaluated directly at the coarse G-vectors),
run the full Newton solve (torch backend, rfft, float64, 5 dipole fixsteps,
tol 1e-3, coarse-init), and compare each grid's outputs term-by-term against
(a) the VASP reference fields and (b) the native-grid solution.

Reported per grid: rhoion_z / rhob_z / phi_z plane-profile RMSE vs VASP,
q_ion (integrated ionic charge), layer_mean of the ionic profile, the
solvent dipole potential term (what MACE's observable consumes), and the
warm solve wall time.

Usage (GPU node):
  PYTHONPATH=<repo> PB_RFFT=1 python tools/pb_check/grid_convergence.py \
      --chgcar data/CHGCAR --potcar data/POTCAR --rhoion-ref data/RHOION \
      --rhob-ref data/RHOB --phi-ref data/PHI --out grid_convergence.tsv \
      --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.vasp_volumetric import read_vasp_volumetric  # noqa: E402
from pure_python import pb  # noqa: E402
from pure_python import torch_pb as tp  # noqa: E402
from pure_python.config import load_config  # noqa: E402
from pure_python.dipole_correction import (  # noqa: E402
    EwaldDipoleMixer,
    cdipol_indmin_from_center,
    cdipol_potential_1d,
    solvent_moments,
    valence_ion_dipole_cart,
)
from pure_python.grid import Grid  # noqa: E402
from pure_python.potcar import read_potcar  # noqa: E402
from pure_python.solute_potential import solute_potential_g  # noqa: E402

POTENTIAL_SCALE = 4.0 * np.pi * 27.211386245988 / 1.8897261258369282  # eV*A/e /A^2


def fft_friendly_even(n: int) -> int:
    m = n + (n % 2)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 2


def restrict_values(values: np.ndarray, shape_c: tuple[int, int, int]) -> np.ndarray:
    """Fourier (band-limiting) restriction of a real field onto a coarser
    grid, in the amplitude-normalized convention (coefficients transfer
    unchanged). Coarse Nyquist planes are zeroed to keep the field real and
    the restriction unambiguous."""
    shape_f = values.shape
    spec_f = np.fft.fftn(values) / values.size
    spec_c = np.zeros(shape_c, dtype=complex)
    slices_f = []
    slices_c = []
    for nf, nc in zip(shape_f, shape_c):
        h = nc // 2
        slices_f.append((slice(0, h), slice(nf - (h - 1), nf) if h > 1 else None))
        slices_c.append((slice(0, h), slice(nc - (h - 1), nc) if h > 1 else None))
    for i, (cf, cc) in enumerate(zip(slices_f[0], slices_c[0])):
        if cf is None:
            continue
        for j, (kf, kc) in enumerate(zip(slices_f[1], slices_c[1])):
            if kf is None:
                continue
            for k, (lf, lc) in enumerate(zip(slices_f[2], slices_c[2])):
                if lf is None:
                    continue
                spec_c[cc, kc, lc] = spec_f[cf, kf, lf]
    out = np.fft.ifftn(spec_c * np.prod(shape_c)).real
    return np.ascontiguousarray(out)


def profile_metrics(rho_ion_z, z_c, ref_z, ref_vals, height):
    """Interpolate a coarse plane profile onto the reference z grid
    (periodic) and return the RMSE."""
    zs = np.concatenate([z_c, [z_c[0] + height]])
    rs = np.concatenate([rho_ion_z, [rho_ion_z[0]]])
    interp = np.interp(np.mod(ref_z, height), zs, rs)
    return float(np.sqrt(np.mean((interp - ref_vals) ** 2)))


def solve_on_grid(shape, cell, valence_f, entries, counts, positions, cfg,
                  params, device, tol, fixsol_steps, warm_repeat=False):
    """Full pipeline on one grid shape; returns metrics dict."""
    ngrid = Grid(cell, shape)  # numpy grid for setup (solute potential, dencor)
    valence_c = (
        valence_f if shape == valence_f.shape else restrict_values(valence_f, shape)
    )
    cvhar_g, dencor = solute_potential_g(ngrid, valence_c, entries, counts, positions)
    cvhar = ngrid.ifft_real_full(cvhar_g)
    n_e_density = (valence_c + dencor) / ngrid.volume

    tg = tp.TorchGrid(cell, shape, device=device, dtype=torch.float64, rspec=True)
    ne_t = tg.to_tensor(n_e_density)
    s_ion, s_diel, _ = tp.create_cavity_torch(ne_t, tg, params)

    zvals = [e.zval for e in entries]
    val_ion_dipole = valence_ion_dipole_cart(valence_c, positions, zvals, counts, cell)
    val_ion_dipole[0:2] = 0.0
    center_abs = 0.5 * (cell[0] + cell[1] + cell[2])
    nz = shape[2]
    indmin_z = cdipol_indmin_from_center(nz, 0.5)
    length_z = float(np.linalg.norm(cell[2]))
    mixer = EwaldDipoleMixer.fresh()

    def run_fixsteps():
        qsol_cache = 0.0
        dsol_cache = np.zeros(3)
        phi_total = torch.zeros(shape, dtype=torch.float64, device=tg.device)
        n_b = n_ion = None
        for step in range(fixsol_steps):
            dip = val_ion_dipole.copy()
            dip[2] += dsol_cache[2] - qsol_cache * center_abs[2]
            _, ef_direct = mixer.ewald_dipol(dip, cell, 3)
            cvdip_z = cdipol_potential_1d(nz, length_z, ef_direct[2], indmin_z)
            phi_sol = tg.to_tensor(cvhar + cvdip_z[None, None, :])
            if step == 0 and all(n % 2 == 0 for n in shape):
                tg_c = tp.TorchGrid(cell, tuple(n // 2 for n in shape),
                                    device=device, dtype=torch.float64, rspec=True)
                phi_c, _, _, _, _ = tp.solve_nlpb_for_phi_sol_torch(
                    torch.zeros(tg_c.shape, dtype=torch.float64, device=tg.device),
                    phi_sol[::2, ::2, ::2].contiguous(),
                    s_ion[::2, ::2, ::2].contiguous(),
                    s_diel[::2, ::2, ::2].contiguous(),
                    tg_c, params, cfg["q_sol"], tol, 12, 40)
                spec_c = tg_c.fft(phi_c)
                # Fourier zero-pad prolongation (drop coarse Nyquist)
                spec_fzp = torch.zeros(tg.spec_shape, dtype=tg.cdtype, device=tg.device)
                ncx, ncy, ncz = tg_c.shape
                hx, hy, hz = ncx // 2, ncy // 2, ncz // 2
                spec_fzp[:hx, :hy, :hz] = spec_c[:hx, :hy, :hz]
                spec_fzp[-(hx - 1):, :hy, :hz] = spec_c[-(hx - 1):, :hy, :hz]
                spec_fzp[:hx, -(hy - 1):, :hz] = spec_c[:hx, -(hy - 1):, :hz]
                spec_fzp[-(hx - 1):, -(hy - 1):, :hz] = spec_c[-(hx - 1):, -(hy - 1):, :hz]
                phi_total = tg.ifft_real(spec_fzp)
            phi_total, n_b, n_ion, _, hist = tp.solve_nlpb_for_phi_sol_torch(
                phi_total, phi_sol, s_ion, s_diel, tg, params,
                cfg["q_sol"], tol, 12, 40)
            qsol_cache, dsol_cache = solvent_moments(
                (n_b + n_ion).detach().cpu().numpy(), cell)
        return phi_total, n_b, n_ion, hist

    t0 = time.perf_counter()
    phi_total, n_b, n_ion, hist = run_fixsteps()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    vol = tg.volume
    area = vol / length_z
    dz = length_z / nz
    rho_ion_z = (n_ion / vol).mean(dim=(0, 1)).detach().cpu().numpy()
    rho_b_z = (n_b / vol).mean(dim=(0, 1)).detach().cpu().numpy()
    phi_z = phi_total.mean(dim=(0, 1)).detach().cpu().numpy()
    z_c = np.arange(nz) * dz
    q_ion = float(rho_ion_z.sum() * dz * area)              # electron-units
    denom = q_ion if abs(q_ion) > 1e-12 else 1e-12
    layer_mean = float((rho_ion_z * z_c).sum() * dz * area / denom)
    solvent_mu_phys = -q_ion * layer_mean                    # physics sign
    solvent_pot = POTENTIAL_SCALE * solvent_mu_phys / area   # V (unsigned conv.)
    return {
        "shape": shape, "wall_s": wall, "rms_last": hist[-1][1],
        "z": z_c, "rho_ion_z": rho_ion_z, "rho_b_z": rho_b_z, "phi_z": phi_z,
        "q_ion": q_ion, "layer_mean": layer_mean, "solvent_pot": solvent_pot,
        "height": length_z,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "pure_python/configs/cal18.json"))
    ap.add_argument("--chgcar", required=True)
    ap.add_argument("--potcar", required=True)
    ap.add_argument("--rhoion-ref", required=True)
    ap.add_argument("--rhob-ref", required=True)
    ap.add_argument("--phi-ref", required=True)
    ap.add_argument("--out", default="grid_convergence.tsv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1.0e-3)
    ap.add_argument("--fixsol-steps", type=int, default=5)
    ap.add_argument("--spacings", type=float, nargs="*",
                    default=[0.12, 0.15, 0.20, 0.25, 0.30, 0.35])
    args = ap.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    cell = chg.cell
    valence_f = chg.values.reshape(chg.grid, order="F")
    entries = read_potcar(args.potcar)
    positions = np.asarray(cfg["positions_direct"], dtype=float)
    counts = list(cfg["counts"])
    params = pb.derived_params(cfg["solvation"])

    rhoion_ref = read_vasp_volumetric(args.rhoion_ref)
    rhob_ref = read_vasp_volumetric(args.rhob_ref)
    phi_ref = read_vasp_volumetric(args.phi_ref)
    ref_z, rhoion_ref_z = rhoion_ref.plane_average_density()
    _, rhob_ref_z = rhob_ref.plane_average_density()
    _, phi_ref_z = phi_ref.plane_average_raw()

    lengths = np.linalg.norm(cell, axis=1)
    shapes = [tuple(chg.grid)]  # native first (anchor)
    for sp in args.spacings:
        s = tuple(fft_friendly_even(int(np.ceil(l / sp))) for l in lengths)
        if s not in shapes and all(a <= b for a, b in zip(s, chg.grid)):
            shapes.append(s)

    rows = ["shape\tnpts\tspacing_A\twall_s\trms_last\t"
            "rhoion_z_rmse\trhob_z_rmse\tphi_z_rmse\t"
            "q_ion\tlayer_mean_A\tsolvent_pot_V"]
    anchor = None
    for shape in shapes:
        m = solve_on_grid(shape, cell, valence_f, entries, counts, positions,
                          cfg, params, args.device, args.tol, args.fixsol_steps)
        sp_eff = float(np.mean([l / n for l, n in zip(lengths, shape)]))
        r_ion = profile_metrics(m["rho_ion_z"], m["z"], ref_z, rhoion_ref_z, m["height"])
        r_b = profile_metrics(m["rho_b_z"], m["z"], ref_z, rhob_ref_z, m["height"])
        r_phi = profile_metrics(m["phi_z"], m["z"], ref_z, phi_ref_z, m["height"])
        if anchor is None:
            anchor = m
        rows.append(
            f"{'x'.join(map(str, shape))}\t{int(np.prod(shape))}\t{sp_eff:.3f}\t"
            f"{m['wall_s']:.2f}\t{m['rms_last']:.2e}\t"
            f"{r_ion:.3e}\t{r_b:.3e}\t{r_phi:.3e}\t"
            f"{m['q_ion']:+.6f}\t{m['layer_mean']:.4f}\t{m['solvent_pot']:+.6f}")
        print(rows[-1], flush=True)
        np.savez(Path(args.out).with_suffix(f".{'x'.join(map(str, shape))}.npz"),
                 z=m["z"], rho_ion_z=m["rho_ion_z"], rho_b_z=m["rho_b_z"],
                 phi_z=m["phi_z"])
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"\nwrote {args.out}")
    print(f"anchor (native) layer_mean={anchor['layer_mean']:.4f} A, "
          f"q_ion={anchor['q_ion']:+.6f}, solvent_pot={anchor['solvent_pot']:+.6f} V")


if __name__ == "__main__":
    main()
