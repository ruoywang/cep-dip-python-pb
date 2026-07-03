from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.vasp_volumetric import read_vasp_volumetric

from .config import load_config
from .grid import Grid
from .pb import update_from_total_phi


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--out", default="pure_python/results/ref_phi_diag/rhob_alignment.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chg = read_vasp_volumetric(args.chgcar)
    phi_ref = read_vasp_volumetric(args.phi_ref)
    rhob_ref = read_vasp_volumetric(args.rhob_ref)
    grid = Grid(chg.cell, chg.grid)

    timings: list[tuple[str, float]] = []
    state = update_from_total_phi(
        phi_ref.values.reshape(phi_ref.grid, order="F"),
        chg.values.reshape(chg.grid, order="F") / grid.volume,
        grid,
        cfg["solvation"],
        timings=timings,
    )
    py = state.rho_bound_values
    ref_f = rhob_ref.values.reshape(rhob_ref.grid, order="F")
    ref_c = rhob_ref.values.reshape(rhob_ref.grid, order="C")

    candidates: list[tuple[str, np.ndarray]] = [
        ("ref_order_F", ref_f),
        ("ref_order_C", ref_c),
        ("ref_F_swap_xy", ref_f.swapaxes(0, 1)),
        ("ref_C_swap_xy", ref_c.swapaxes(0, 1)),
        ("ref_F_flip_x", ref_f[::-1, :, :]),
        ("ref_F_flip_y", ref_f[:, ::-1, :]),
        ("ref_F_flip_xy", ref_f[::-1, ::-1, :]),
    ]
    lines = ["candidate\traw_rmse\tcorr_sample\tbest_scale\trmse_after_scale\tref_min\tref_max\tpy_min\tpy_max"]
    sample = (slice(None, None, 20), slice(None, None, 20), slice(None, None, 4))
    for name, arr in candidates:
        if arr.shape != py.shape:
            lines.append(f"{name}\tshape_mismatch\t{arr.shape}")
            continue
        corr = float(np.corrcoef(py[sample].ravel(), arr[sample].ravel())[0, 1])
        denom = float(np.sum(py * py))
        scale = float(np.sum(py * arr) / denom) if denom != 0.0 else float("nan")
        scaled_rmse = rmse(scale * py, arr)
        lines.append(
            f"{name}\t{rmse(py, arr):.12e}\t{corr:.12e}\t{scale:.12e}\t{scaled_rmse:.12e}"
            f"\t{arr.min():.12e}\t{arr.max():.12e}\t{py.min():.12e}\t{py.max():.12e}"
        )
    lines.extend(f"time_{label}_s\t{seconds:.6f}" for label, seconds in timings)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
