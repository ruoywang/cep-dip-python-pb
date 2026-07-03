"""Warm-state single-solve timing of the torch PB solver at training grid
resolutions, on the current device. Mirrors the MACE use case: a NiN-family
cell, synthetic slab density, cal18 solvation block, cavity+solve on device.

Reports steady-state (warm) wall time per full solve at several grid
spacings, for comparison with the numpy CPU figures (6.0/2.4/2.2 s at
0.25/0.30/0.35 A on 16 CPU threads).

Usage: python time_torch_gpu_traingrid.py <solvation.json> [cpu|cuda]
"""

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])

from pure_python import pb
from pure_python import torch_pb as tp

CELL = np.array([[14.802, 0.0, 0.0], [-7.401, 12.818908, 0.0], [0.0, 0.0, 45.0]])
AREA = float(np.linalg.norm(np.cross(CELL[0], CELL[1])))
TOTAL_CHARGE = -1.0


def fft_even(n):
    m = n + (n % 2)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 2


def main():
    config_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    params = pb.derived_params(json.load(open(config_path))["solvation"])
    q_sol = -TOTAL_CHARGE
    print(f"device={device} dtype=float64")
    print(f"{'spacing':>8} {'grid':>16} {'npts':>10} {'warm_solve_s':>13} {'rms':>10}")
    for sp in (0.25, 0.30, 0.35):
        shape = tuple(fft_even(int(np.ceil(l / sp))) for l in np.linalg.norm(CELL, axis=1))
        grid = tp.TorchGrid(CELL, shape, device=device, dtype=torch.float64)
        z = grid.z_mesh
        ne = 0.3 * torch.exp(-0.5 * ((z - 0.42 * CELL[2, 2]) / 4.0) ** 2) + 1e-3
        phi_sol = 0.5 * torch.exp(-0.5 * ((z - 0.42 * CELL[2, 2]) / 3.0) ** 2)
        s_ion, s_diel, _ = tp.create_cavity_torch(ne, grid, params)

        def one_solve():
            pt, nb, ni, _, hist = tp.solve_nlpb_for_phi_sol_torch(
                torch.zeros(shape, dtype=torch.float64, device=grid.device),
                phi_sol, s_ion, s_diel, grid, params, q_sol, 1e-3, 20, 200,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return hist

        one_solve()  # warm up CUDA/cuFFT plans
        one_solve()
        t0 = time.perf_counter()
        hist = one_solve()
        dt = time.perf_counter() - t0
        npts = int(np.prod(shape))
        print(f"{sp:>8.2f} {str(shape):>16} {npts:>10} {dt:>13.2f} {hist[-1][1]:>10.2e}")


if __name__ == "__main__":
    main()
