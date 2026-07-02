# Change Log

## 2026-06-30 PB solver optimization

- `pure_python/grid.py`
  - Added SciPy FFT backend with `PB_FFT_WORKERS` / `SLURM_CPUS_PER_TASK` controlled worker count.
  - Kept the same normalized FFT convention as the previous NumPy implementation.
  - Added optional `PB_FFT_BACKEND=pyfftw` backend with pyFFTW plan cache.
  - Added optional `PB_FFT_BACKEND=mkl` backend using `mkl_fft`.
- `pure_python/pb.py`
  - Added optional cached `w_b` argument to `nlpb_quantities`.
  - Added `need_response=False` mode so line-search trials can skip constructing `chi` and `ekappa2`.
  - Moved the `ekappa2` reciprocal transform in `minimize_l` outside the CG loop.
  - Replaced explicit `chi[..., 3, 3]` storage with an equivalent compact dielectric response.
    The tensor-vector contraction is evaluated directly inside `lapl_tensor`.
  - Added optional inner timing output controlled by `PB_PROFILE_INNER=1`.
- `pure_python/_pb_fast.c`, `pure_python/setup_pb_fast.py`
  - Added an OpenMP C implementation of the local-field fixed-point solver used by
    `local_field_factor`.
  - The Python implementation remains available with `PB_DISABLE_FAST_LOCAL_FIELD=1`.
- `pure_python/solve_from_chgcar_newton.py`
  - Computes the bound-charge Gaussian kernel once per PB solve and reuses it.
  - Computes the real-space Newton step once per line search and reuses it for trial step lengths.
  - Uses `need_response=False` during trial residual checks and final density evaluation.
  - Writes `stage_times.tsv` for read/input, solute potential, cavity, and each fixstep timing.
- `pure_python/solute_potential.py`
  - Reuses each element's structure factor between DENCOR and local pseudopotential construction.
  - Kept the same interpolation and reciprocal-space formulas.
  - Computes structure factors using separable 1D phase factors and `einsum` instead of per-atom
    full-grid phase arrays.

These changes are intended to reduce repeated work only. The PB equations, CDIPOL handling,
solute potential, cavity construction, and density definitions were not changed.
