"""Grid self-convergence with a SMOOTH (MACE-like) input density.

Companion to grid_convergence.py, which showed that with real CHGCAR input
the ionic layer position drifts ~2 A between 0.09 and 0.25 A grids. That
test conflates (a) Fourier-truncation ringing of the sharp core density
(absent in the MACE use case, whose density is a band-limited Gaussian
model evaluated analytically on every grid) with (b) the solver's intrinsic
grid discretization (cavity kernels, FFT Poisson). This script isolates (b):

- geometry: real cal_18 atoms (positions/zvals from the config)
- electron density: analytic Gaussians — neutral-atom baseline (Z_val at
  sigma=0.5 A) plus a net Gaussian sheet carrying total_charge=-1 —
  exactly the polar-mace pb_solvent surrogate
- solute potential: Hartree of (electrons - Gaussian nuclei), same surrogate
- solve: torch rfft float64, fixed single dipole-free step (no CDIPOL loop,
  identical protocol at every grid), q_sol = +1
- metrics vs the finest grid (anchor): layer_mean drift, solvent-potential
  drift, q_ion, rho_ion(z) RMSE.

Usage: PB_RFFT=1 python grid_convergence_smooth.py [--device cuda]
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

from pure_python import pb  # noqa: E402
from pure_python import torch_pb as tp  # noqa: E402
from pure_python.config import load_config  # noqa: E402

POTENTIAL_SCALE = 4.0 * np.pi * 27.211386245988 / 1.8897261258369282
TOTAL_CHARGE = -1.0
NEUTRAL_SIGMA = 0.5
NUCLEAR_SIGMA = 0.4
NET_Z, NET_SIGMA = 15.0, 2.0


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


def gaussian_field_g(grid: tp.TorchGrid, pos_frac: np.ndarray, weights: np.ndarray,
                     sigma: float) -> torch.Tensor:
    """Reciprocal-space ('values'-amplitude) field of periodic Gaussians:
    sum_i w_i * exp(-g^2 sigma^2/2) * e^{-i g.tau_i}, grouped by weight."""
    g2 = grid.gsq * (2.0 * np.pi) ** 2
    damp = torch.exp(-0.5 * g2 * sigma * sigma)
    nx, ny, nz = grid.shape
    hx = torch.fft.fftfreq(nx, device=grid.device, dtype=grid.dtype) * nx
    hy = torch.fft.fftfreq(ny, device=grid.device, dtype=grid.dtype) * ny
    hz = torch.arange(nz // 2 + 1, device=grid.device, dtype=grid.dtype) \
        if grid.rspec else torch.fft.fftfreq(nz, device=grid.device, dtype=grid.dtype) * nz
    out = torch.zeros(grid.spec_shape, dtype=grid.cdtype, device=grid.device)
    tpi = 2.0 * np.pi
    for w in np.unique(weights):
        sel = np.where(weights == w)[0]
        tau = torch.as_tensor(pos_frac[sel], device=grid.device, dtype=grid.dtype)
        ex = torch.exp(-1j * tpi * tau[:, 0, None] * hx[None, :])
        ey = torch.exp(-1j * tpi * tau[:, 1, None] * hy[None, :])
        ez = torch.exp(-1j * tpi * tau[:, 2, None] * hz[None, :])
        sf = torch.einsum("ah,ak,al->hkl", ex, ey, ez)
        out = out + float(w) * sf
    return out * damp


def net_sheet_g(grid: tp.TorchGrid, q: float, z0: float, sigma: float) -> torch.Tensor:
    """Gaussian charge sheet at z0 (values-amplitude convention)."""
    length_z = float(np.linalg.norm(grid.cell[2].cpu().numpy()))
    nz = grid.shape[2]
    hz = torch.arange(nz // 2 + 1, device=grid.device, dtype=grid.dtype) \
        if grid.rspec else torch.fft.fftfreq(nz, device=grid.device, dtype=grid.dtype) * nz
    gz = 2.0 * np.pi * hz / length_z
    prof = q * torch.exp(-0.5 * gz * gz * sigma * sigma) * torch.exp(-1j * gz * z0)
    out = torch.zeros(grid.spec_shape, dtype=grid.cdtype, device=grid.device)
    out[0, 0, :] = prof
    return out


def solve_smooth(shape, cell, pos_frac, zvals, params, q_sol, device, tol):
    tg = tp.TorchGrid(cell, shape, device=device, dtype=torch.float64, rspec=True)
    # electron density (electron-positive): neutral Gaussians minus net sheet
    ne_g = gaussian_field_g(tg, pos_frac, zvals, NEUTRAL_SIGMA) \
        - net_sheet_g(tg, TOTAL_CHARGE, NET_Z, NET_SIGMA)
    n_e_values = tg.ifft_real(ne_g)
    n_e_density = torch.clamp(n_e_values / tg.volume, min=0.0)
    s_ion, s_diel, _ = tp.create_cavity_torch(n_e_density, tg, params)
    # solute potential: Hartree of electrons + Gaussian nuclei (-Z)
    nuc_g = -gaussian_field_g(tg, pos_frac, zvals, NUCLEAR_SIGMA)
    phi_sol = tg.ifft_real(tg.l0_inv_op(ne_g + nuc_g))

    t0 = time.perf_counter()
    phi_total, n_b, n_ion, _, hist = tp.solve_nlpb_for_phi_sol_torch(
        torch.zeros(shape, dtype=torch.float64, device=tg.device),
        phi_sol, s_ion, s_diel, tg, params, q_sol, tol, 20, 200)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    length_z = float(np.linalg.norm(cell[2]))
    nz = shape[2]
    dz = length_z / nz
    area = tg.volume / length_z
    rho = (n_ion / tg.volume).mean(dim=(0, 1)).detach().cpu().numpy()
    rho_b = (n_b / tg.volume).mean(dim=(0, 1)).detach().cpu().numpy()
    z = np.arange(nz) * dz
    q_ion = float(rho.sum() * dz * area)
    denom = q_ion if abs(q_ion) > 1e-12 else 1e-12
    layer_mean = float((rho * z).sum() * dz * area / denom)
    pot = POTENTIAL_SCALE * (-q_ion * layer_mean) / area
    # bound-charge dipole (electron-units z-moment; physics = -mu)
    mu_b = float((rho_b * z).sum() * dz * area)
    pot_b = POTENTIAL_SCALE * (-mu_b) / area
    # combined solvent term (what MACE consumes with include_bound)
    mu_tot = float(((rho + rho_b) * z).sum() * dz * area)
    pot_tot = POTENTIAL_SCALE * (-mu_tot) / area
    return {"z": z, "rho": rho, "rho_b": rho_b, "q_ion": q_ion,
            "layer_mean": layer_mean, "pot": pot, "mu_b": mu_b, "pot_b": pot_b,
            "pot_tot": pot_tot, "wall": wall, "rms": hist[-1][1],
            "height": length_z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "pure_python/configs/cal18.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1.0e-3)
    ap.add_argument("--spacings", type=float, nargs="*",
                    default=[0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--out", default="grid_convergence_smooth.tsv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cell = np.asarray(cfg["cell_A"], dtype=float)
    pos_frac = np.asarray(cfg["positions_direct"], dtype=float) % 1.0
    counts = list(cfg["counts"])
    zv_map = {k: float(v) for k, v in cfg["zval"].items()}
    zv_types = [zv_map[el] for el in cfg["elements"]]
    zvals = np.concatenate([[zv] * c for zv, c in zip(zv_types, counts)])
    params = pb.derived_params(cfg["solvation"])
    q_sol = -TOTAL_CHARGE
    lengths = np.linalg.norm(cell, axis=1)

    rows = ["shape\tnpts\tspacing_A\twall_s\trms\tq_ion\tlayer_mean_A\t"
            "dmean_A\tpot_ion_V\tdpot_ion_V\tpot_b_V\tdpot_b_V\t"
            "pot_tot_V\tdpot_tot_V\trho_ion_rmse\trho_b_rmse"]
    anchor = None
    for sp in args.spacings:
        shape = tuple(fft_friendly_even(int(np.ceil(l / sp))) for l in lengths)
        m = solve_smooth(shape, cell, pos_frac, zvals, params, q_sol,
                         args.device, args.tol)
        if anchor is None:
            anchor = m
            rr = rrb = 0.0
        else:
            def rmse_vs_anchor(vals):
                zs = np.concatenate([m["z"], [m["z"][0] + m["height"]]])
                rs = np.concatenate([vals, [vals[0]]])
                interp = np.interp(np.mod(anchor["z"], m["height"]), zs, rs)
                key = "rho" if vals is m["rho"] else "rho_b"
                return float(np.sqrt(np.mean((interp - anchor[key]) ** 2)))
            rr = rmse_vs_anchor(m["rho"])
            rrb = rmse_vs_anchor(m["rho_b"])
        rows.append(
            f"{'x'.join(map(str, shape))}\t{int(np.prod(shape))}\t{sp:.2f}\t"
            f"{m['wall']:.2f}\t{m['rms']:.1e}\t{m['q_ion']:+.6f}\t"
            f"{m['layer_mean']:.4f}\t{m['layer_mean']-anchor['layer_mean']:+.4f}\t"
            f"{m['pot']:+.4f}\t{m['pot']-anchor['pot']:+.4f}\t"
            f"{m['pot_b']:+.4f}\t{m['pot_b']-anchor['pot_b']:+.4f}\t"
            f"{m['pot_tot']:+.4f}\t{m['pot_tot']-anchor['pot_tot']:+.4f}\t"
            f"{rr:.3e}\t{rrb:.3e}")
        print(rows[-1], flush=True)
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
