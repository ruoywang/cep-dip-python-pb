# torch (GPU) port of the nonlinear PB solver

Branch `torch-gpu`, 2026-07-03. Motivation: the PB solve is called inside a
GPU-resident MACE training forward; running it as a numpy/CPU op forces a
host round-trip every step. This ports the Newton-PCG hot path to torch so
the solve runs on the same device as the model.

## Scope

`pure_python/torch_pb.py` — line-for-line torch translation of the numpy
solver hot path: TorchGrid (FFT amplitude convention, reciprocal/deriv
meshes, grad/div, l0 operators), create_cavity, ion/bound density,
scalar + tensor-field dielectric response, l_op/lapl_tensor, minimize_l
(preconditioned CG with the ion G=0 special case), residual, and
solve_nlpb_for_phi_sol. `derived_params` reused from pb.py. Full-spectrum,
float64 default.

`solve_from_chgcar_newton.py` — `--backend {numpy,torch}` + `--device`
switch; numpy default and its code path unchanged. `make_numpy_io_solver`
is a drop-in numpy-signature solver with a per-shape TorchGrid cache.

Solute-potential setup (POTCAR Hartree/local-pseudopotential) and the
dipole-correction fixstep bookkeeping stay on the numpy side; in the MACE
integration the setup is the Gaussian surrogate instead.

## Validation (cal_18, compute node, full-spectrum)

Component parity (tests/test_torch_pb_components.py, CPU float64): every
operator agrees with numpy to <= 1.9e-12.

End-to-end cal_18 (168x168x384), last fixstep, vs VASP reference:

| backend        | PHI RMSE (eV) | RHOB_z RMSE  | RHOION_z RMSE | wall  |
|----------------|---------------|--------------|---------------|-------|
| numpy          | 3.0046326e-3  | 3.0814e-6    | 1.5876e-7     | 2:27  |
| torch GPU f64  | 3.0046326e-3  | 3.0814e-6    | 1.5876e-7     | 2:10  |
| torch GPU f32  | nan (diverges)| -            | -             | 3:09  |

torch f64 reproduces numpy to ~1e-9 relative in every reported quantity
and passes the PHI gate. (The 3.0046e-3 vs the 2.998e-3 gate value is the
full-spectrum vs rfft difference of the numpy path itself, not a torch
regression — both backends give the same 3.0046e-3 here.)

## Findings

- **Steady-state GPU speedup is real**: warm fixsteps (1-4) run 1.3-2.0 s
  on GPU f64 vs 10-26 s numpy (5-10x). The total wall is dominated by
  fixstep_0 + coarse_init, which carry one-time CUDA/cuFFT plan init; in
  training the solver is warm across steps so the per-step speedup is what
  matters. cal_18 (0.088 A, 10.8M pts) is also far finer than the training
  grid (~0.3 A, ~0.6M pts).
- **float32 diverges** on this ill-conditioned Poisson problem (CG cannot
  reach 1e-3 tol; outer loop hits the cap at nan). Production must use
  float64. A usable f32/mixed-precision path would need a better
  preconditioner or mixed-precision CG — deferred.

## Next

- rfft half-spectrum path (≈2x FFT, half memory) — currently full-spectrum.
- Put the cavity/solute setup on GPU too (in MACE this is the Gaussian
  surrogate, already GPU-friendly), removing the remaining host work.
- Wire the torch backend into polar-mace pb_solvent (device-resident, no
  host round-trip) and measure real training s/step at ~0.3 A.
