from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from tools.vasp_volumetric import read_vasp_volumetric, write_profile

from .config import load_config
from .dipole_correction import (
    EwaldDipoleMixer,
    cdipol_indmin_from_center,
    cdipol_potential_1d,
    solvent_moments,
    valence_ion_dipole_cart,
)
from .grid import Grid
from .grid import normalized_gaussian_kernel_g
from .pb import (
    create_cavity,
    derived_params,
    get_pb_timing,
    minimize_l,
    nlpb_field_quantities,
    nlpb_response_from_fields,
    reset_pb_timing,
    residual_g,
)
from .potcar import read_potcar
from .solute_potential import solute_potential_g


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def restrict_half(a: np.ndarray) -> np.ndarray:
    """Decimate a field onto the half-resolution grid (even shapes only)."""
    return np.ascontiguousarray(a[::2, ::2, ::2])


def prolong_double(a_coarse: np.ndarray, grid_c: Grid, grid_f: Grid) -> np.ndarray:
    """Fourier zero-padding prolongation from the half grid to the full grid.

    The normalized FFT convention stores amplitudes, so low-frequency
    coefficients transfer unchanged.
    """
    spec_c = grid_c.fft(a_coarse)
    spec_f = np.zeros(grid_f.spec_shape, dtype=complex)
    blocks = []
    for axis, (n_c, n_f) in enumerate(zip(grid_c.shape, grid_f.shape)):
        h = n_c // 2
        if grid_f.rspec:
            # Coarse Nyquist planes are self-conjugate composites; embedding
            # them as interior fine modes double-counts under c2r, so they
            # are dropped (their weight in smooth fields is negligible).
            if axis == 2:
                blocks.append(((slice(0, h), slice(0, h)),))
            else:
                blocks.append((
                    (slice(0, h), slice(0, h)),
                    (slice(h + 1, n_c), slice(n_f - (n_c - h - 1), n_f)),
                ))
        else:
            blocks.append((
                (slice(0, h), slice(0, h)),
                (slice(h, n_c), slice(n_f - (n_c - h), n_f)),
            ))
    for (cx, fx) in blocks[0]:
        for (cy, fy) in blocks[1]:
            for (cz, fz) in blocks[2]:
                spec_f[fx, fy, fz] = spec_c[cx, cy, cz]
    return grid_f.ifft_real(spec_f)


def solve_nlpb_for_phi_sol(
    phi_total: np.ndarray,
    phi_sol: np.ndarray,
    s_ion: np.ndarray,
    s_diel: np.ndarray,
    grid: Grid,
    params: dict,
    q_sol: float,
    tol: float,
    max_outer: int,
    cg_max_iter: int,
    progress_path: Path | None = None,
    fixstep: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, float, int, float]]]:
    phi_solv_g = grid.fft(phi_total - phi_sol)
    w_b = normalized_gaussian_kernel_g(grid, float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"]))
    history: list[tuple[int, float, int, float]] = []
    fields = None
    for outer in range(max_outer + 1):
        if fields is None:
            fields = nlpb_field_quantities(phi_total, s_ion, s_diel, grid, params, w_b)
        n_b = fields["n_b"]
        n_ion = fields["n_ion"]
        resid, rms = residual_g(phi_solv_g, n_b, n_ion, q_sol, grid)
        if rms < tol and outer >= 1:
            history.append((outer, rms, 0, 0.0))
            break
        response, ekappa2 = nlpb_response_from_fields(fields, s_ion, s_diel, grid, params)
        if response is None:
            raise RuntimeError("Newton solve requires dielectric response")
        dphi_g, cg_rms, cg_iter = minimize_l(resid, response, ekappa2, w_b, grid, max(rms / 10.0, tol), cg_max_iter)
        dphi_real = grid.ifft_real(dphi_g)
        alpha = 1.0
        accepted_rms = float("inf")
        for _ in range(7):
            trial_phi = phi_total + alpha * dphi_real
            trial_phi_solv_g = phi_solv_g + alpha * dphi_g
            trial_fields = nlpb_field_quantities(trial_phi, s_ion, s_diel, grid, params, w_b)
            _, trial_rms = residual_g(trial_phi_solv_g, trial_fields["n_b"], trial_fields["n_ion"], q_sol, grid)
            if trial_rms <= rms or alpha <= 1.0 / 64.0:
                phi_total = trial_phi
                phi_solv_g = trial_phi_solv_g
                fields = trial_fields
                accepted_rms = trial_rms
                break
            alpha *= 0.5
        history.append((outer, rms, cg_iter, accepted_rms))
        if progress_path is not None:
            with progress_path.open("a") as f:
                f.write(f"{fixstep}\t{outer}\t{rms:.12e}\t{cg_iter}\t{accepted_rms:.12e}\n")
    if fields is None:
        fields = nlpb_field_quantities(phi_total, s_ion, s_diel, grid, params, w_b)
    return phi_total, fields["n_b"], fields["n_ion"], phi_solv_g, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pure_python/configs/cal18.json")
    parser.add_argument("--chgcar", default="data/case_cal18/CHGCAR")
    parser.add_argument("--potcar", default="data/case_cal18/POTCAR")
    parser.add_argument("--phi-ref", default="data/case_cal18/PHI")
    parser.add_argument("--rhob-ref", default="data/case_cal18/RHOB")
    parser.add_argument("--rhoion-ref", default="data/case_cal18/RHOION")
    parser.add_argument("--out-dir", default="pure_python/results/from_chgcar_newton")
    parser.add_argument("--fixsol-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1.0e-3)
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--cg-max-iter", type=int, default=200)
    parser.add_argument("--coarse-init", type=int, default=1,
                        help="warm-start fixstep 0 from a half-resolution Newton solve")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy",
                        help="Newton-PCG solve backend (torch runs on --device)")
    parser.add_argument("--device", default="cpu",
                        help="torch device for --backend torch (cpu, cuda, cuda:0)")
    args = parser.parse_args()

    if args.backend == "torch":
        from .torch_pb import make_numpy_io_solver
        import torch as _torch

        _dtype = _torch.float64 if os.environ.get("PB_TORCH_DTYPE", "float64") == "float64" else _torch.float32
        _solve = make_numpy_io_solver(device=args.device, dtype=_dtype)
        globals()["solve_nlpb_for_phi_sol"] = _solve

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reset_pb_timing()
    stage_lines = ["stage\tseconds"]
    detail_lines = ["stage\tseconds"]
    (out / "stage_times.tsv").write_text(stage_lines[0] + "\n")
    (out / "stage_detail_times.tsv").write_text(detail_lines[0] + "\n")
    t_stage = perf_counter()
    t_detail = perf_counter()

    def mark_stage(label: str) -> None:
        nonlocal t_stage
        now = perf_counter()
        stage_lines.append(f"{label}\t{now - t_stage:.6f}")
        (out / "stage_times.tsv").write_text("\n".join(stage_lines) + "\n")
        t_stage = now

    def mark_detail(label: str) -> None:
        nonlocal t_detail
        now = perf_counter()
        detail_lines.append(f"{label}\t{now - t_detail:.6f}")
        (out / "stage_detail_times.tsv").write_text("\n".join(detail_lines) + "\n")
        t_detail = now

    def extend_detail(prefix: str, timings: list[tuple[str, float]]) -> None:
        nonlocal t_detail
        for label, seconds in timings:
            detail_lines.append(f"{prefix}:{label}\t{seconds:.6f}")
        (out / "stage_detail_times.tsv").write_text("\n".join(detail_lines) + "\n")
        t_detail = perf_counter()

    cfg = load_config(args.config)
    mark_detail("read_config")
    chg = read_vasp_volumetric(args.chgcar)
    mark_detail("read_chgcar")
    phi_ref = read_vasp_volumetric(args.phi_ref)
    mark_detail("read_phi_ref")
    rhob_ref = read_vasp_volumetric(args.rhob_ref)
    mark_detail("read_rhob_ref")
    rhoion_ref = read_vasp_volumetric(args.rhoion_ref)
    mark_detail("read_rhoion_ref")
    mark_stage("read_inputs")
    grid = Grid(chg.cell, chg.grid)
    valence_values = chg.values.reshape(chg.grid, order="F")
    mark_detail("build_grid_and_valence_array")
    entries = read_potcar(args.potcar)
    mark_detail("read_potcar")
    positions = np.asarray(cfg["positions_direct"], dtype=float)
    counts = list(cfg["counts"])
    zvals = [entry.zval for entry in entries]
    mark_detail("prepare_positions_counts_zvals")

    solute_timings: list[tuple[str, float]] = []
    cvhar_g, dencor = solute_potential_g(grid, valence_values, entries, counts, positions, solute_timings)
    extend_detail("solute_potential_and_dencor", solute_timings)
    cvhar = grid.ifft_real_full(cvhar_g)
    mark_detail("solute_ifft_cvhar")
    n_e_density = (valence_values + dencor) / grid.volume
    mark_detail("solute_build_electron_density")
    mark_stage("solute_potential_and_dencor")
    params = derived_params(cfg["solvation"])
    mark_detail("derive_solvation_params")
    cavity_timings: list[tuple[str, float]] = []
    s_ion, s_diel, _ = create_cavity(n_e_density, grid, params, cavity_timings)
    extend_detail("create_cavity", cavity_timings)
    mark_stage("create_cavity")

    phi_ref_values = phi_ref.values.reshape(phi_ref.grid, order="F")
    rhob_ref_values = rhob_ref.values.reshape(rhob_ref.grid, order="F")
    rhoion_ref_values = rhoion_ref.values.reshape(rhoion_ref.grid, order="F")

    val_ion_dipole = valence_ion_dipole_cart(valence_values, positions, zvals, counts, chg.cell)
    val_ion_dipole[0:2] = 0.0
    center_abs = 0.5 * chg.cell[0] + 0.5 * chg.cell[1] + 0.5 * chg.cell[2]
    mixer = EwaldDipoleMixer.fresh()
    qsol_cache = 0.0
    dsol_cache = np.zeros(3, dtype=float)
    phi_total = np.zeros(grid.shape, dtype=float)
    n_b = np.zeros(grid.shape, dtype=float)
    n_ion = np.zeros(grid.shape, dtype=float)
    indmin_z = cdipol_indmin_from_center(grid.shape[2], 0.5)
    length_z = float(np.linalg.norm(chg.cell[2]))

    lines = [
        "fixstep\touter_last\trms_last\tEFz_direct\tqsol\tdsol_z\tphi_rmse_eV\tphi_z_rmse_eV\trhob_z_rmse_e_A3\trhoion_z_rmse_e_A3"
    ]
    outer_lines = ["fixstep\touter\trms\tcg_iter\tpost_step_rms"]
    for step in range(args.fixsol_steps):
        t_fix = perf_counter()
        dip_for_field = val_ion_dipole.copy()
        dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
        _, ef_direct = mixer.ewald_dipol(dip_for_field, chg.cell, 3)
        cvdip_z = cdipol_potential_1d(grid.shape[2], length_z, ef_direct[2], indmin_z)
        phi_sol = cvhar + cvdip_z[None, None, :]
        progress_path = Path(args.out_dir) / "outer_history.tsv"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        if step == 0:
            progress_path.write_text("fixstep\touter\trms\tcg_iter\tpost_step_rms\n")
        if step == 0 and args.coarse_init and all(n % 2 == 0 for n in grid.shape):
            t_coarse = perf_counter()
            grid_c = Grid(chg.cell, tuple(n // 2 for n in grid.shape))
            phi_c, _, _, _, _ = solve_nlpb_for_phi_sol(
                np.zeros(grid_c.shape),
                restrict_half(phi_sol),
                restrict_half(s_ion),
                restrict_half(s_diel),
                grid_c,
                params,
                cfg["q_sol"],
                args.tol,
                args.max_outer,
                args.cg_max_iter,
                progress_path,
                -1,
            )
            phi_total = prolong_double(phi_c, grid_c, grid)
            stage_lines.append(f"coarse_init\t{perf_counter() - t_coarse:.6f}")
            (out / "stage_times.tsv").write_text("\n".join(stage_lines) + "\n")
        phi_total, n_b, n_ion, _, history = solve_nlpb_for_phi_sol(
            phi_total,
            phi_sol,
            s_ion,
            s_diel,
            grid,
            params,
            cfg["q_sol"],
            args.tol,
            args.max_outer,
            args.cg_max_iter,
            progress_path,
            step,
        )
        qsol_cache, dsol_cache = solvent_moments(n_b + n_ion, chg.cell)
        for outer, rms_val, cg_iter, cg_rms in history:
            outer_lines.append(f"{step}\t{outer}\t{rms_val:.12e}\t{cg_iter}\t{cg_rms:.12e}")
        phi_z = phi_total.mean(axis=(0, 1))
        phi_ref_z = phi_ref_values.mean(axis=(0, 1))
        rhob_z = (n_b / grid.volume).mean(axis=(0, 1))
        rhob_ref_z = (rhob_ref_values / grid.volume).mean(axis=(0, 1))
        rhoion_z = (n_ion / grid.volume).mean(axis=(0, 1))
        rhoion_ref_z = (rhoion_ref_values / grid.volume).mean(axis=(0, 1))
        last_outer, last_rms, _, _ = history[-1]
        lines.append(
            f"{step}\t{last_outer}\t{last_rms:.12e}\t{ef_direct[2]:.12e}\t{qsol_cache:.12e}\t{dsol_cache[2]:.12e}\t"
            f"{rmse(phi_total, phi_ref_values):.12e}\t{rmse(phi_z, phi_ref_z):.12e}\t"
            f"{rmse(rhob_z, rhob_ref_z):.12e}\t{rmse(rhoion_z, rhoion_ref_z):.12e}"
        )
        stage_lines.append(f"fixstep_{step}\t{perf_counter() - t_fix:.6f}")
        (out / "stage_times.tsv").write_text("\n".join(stage_lines) + "\n")

    (out / "fixstep_summary.tsv").write_text("\n".join(lines) + "\n")
    (out / "stage_times.tsv").write_text("\n".join(stage_lines) + "\n")
    timing = get_pb_timing()
    if timing:
        timing_lines = ["label\tseconds\tcount\tseconds_per_call"]
        for label, (seconds, count) in timing.items():
            timing_lines.append(f"{label}\t{seconds:.6f}\t{count}\t{seconds / max(count, 1):.6f}")
        (out / "pb_inner_timing.tsv").write_text("\n".join(timing_lines) + "\n")
    if not (out / "outer_history.tsv").exists():
        (out / "outer_history.tsv").write_text("\n".join(outer_lines) + "\n")
    z, phi_ref_z = phi_ref.plane_average_raw()
    _, rhob_ref_z = rhob_ref.plane_average_density()
    _, rhoion_ref_z = rhoion_ref.plane_average_density()
    write_profile(
        out / "profiles.tsv",
        {
            "z_A": z,
            "PHI_DFT_eV": phi_ref_z,
            "PHI_py_eV": phi_total.mean(axis=(0, 1)),
            "RHOB_DFT_e_A3": rhob_ref_z,
            "RHOB_py_e_A3": (n_b / grid.volume).mean(axis=(0, 1)),
            "RHOION_DFT_e_A3": rhoion_ref_z,
            "RHOION_py_e_A3": (n_ion / grid.volume).mean(axis=(0, 1)),
        },
    )
    print(out / "fixstep_summary.tsv")


if __name__ == "__main__":
    main()
