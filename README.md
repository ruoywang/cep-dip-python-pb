# cep-dip-python-pb


**Results overview:** https://ruoywang.github.io/cep-dip-python-pb/
Reproduction of VASP/CEP-DIP implicit-solvent fields (PHI / RHOB / RHOION)
from a converged charge density (CHGCAR), validated in two independent routes:

1. **Patched VASP/CEP-DIP, fixed CHGCAR** — recompute the solvent fields from
   the original inputs plus the final CHGCAR, with the charge density held
   fixed. Reproduces the reference to PHI raw RMSE **1.7×10⁻⁷ eV**
   (CHGCAR byte-identical).
2. **Pure-Python nonlinear Poisson–Boltzmann solver** — from
   CHGCAR/POTCAR/config directly to RHOB/RHOION/PHI (Newton solve + CDIPOL
   dipole correction), with every input term aligned to patched-VASP debug
   output. Final PHI 3D RMSE **3.0×10⁻³ eV**, RHOB 1D RMSE 3×10⁻⁶ e/Å³.

Reference case: `cal_18`, a Ni₁N₁-doped graphene slab
(Ni₁N₁C₇₀H₈₉O₄₆, 168×168×384 grid, 35 Å c axis, VASPsol water, ISOL=2,
LDIPOL along z).

## Layout

| Path | Contents |
|------|----------|
| `pure_python/` | The Python PB solver package: `pb.py` (nonlinear PB, Newton), `solute_potential.py`, `dipole_correction.py` (CDIPOL/Ewald), `grid.py` (FFT backends), `_pb_fast.c` (OpenMP local-field solver, build with `setup_pb_fast.py`), entry point `solve_from_chgcar_newton.py`, diagnostic scripts, SLURM job files. |
| `patches/` | Diffs against the group CEP-DIP tree (no VASP source here — see `patches/README.md`). |
| `tools/` | `vasp_volumetric.py` (CHGCAR/PHI/RHOB/RHOION reader), `vasp_inputs.py`. |
| `tests/` | Field/debug-output comparison and audit scripts. |
| `results/` | Numerical summaries: accuracy checks (`checks/`), 1D profiles (`profiles/`), VASP-vs-Python timing (`comparison/`), final reproduction summaries. |
| `docs/` | Findings (including the negative result below), change logs, reproduction summaries, and `results_overview.html` — a self-contained results page (open in any browser). |
| `example/case_cal18/` | INCAR / POSCAR / KPOINTS / OUTCAR of the reference case, plus `CHECKSUMS.sha256` and `POTCAR_INFO.txt` for the files that stay out of the repo. |

## Files intentionally not in this repository

- **VASP / CEP-DIP source** — commercial license; only diffs are kept
  (`patches/`).
- **POTCAR** — VASP license. Rebuild from `example/case_cal18/POTCAR_INFO.txt`
  (PAW_PBE Ni 02Aug2007, O 08Apr2002, N 08Apr2002, C 08Apr2002, H 15Jun2001)
  and verify with `CHECKSUMS.sha256`.
- **Reference volumetric fields** (CHGCAR / PHI / RHOB / RHOION, 189 MB each) —
  archived on TACC `$WORK`; SHA-256 checksums in
  `example/case_cal18/CHECKSUMS.sha256`.

## Running the Python solver

```sh
cd pure_python
python setup_pb_fast.py build_ext --inplace   # optional OpenMP extension
python solve_from_chgcar_newton.py --help
```

Useful environment switches: `PB_FFT_BACKEND` (`scipy`/`pyfftw`/`mkl`),
`PB_FFT_WORKERS`, `PB_PROFILE_INNER=1`, `PB_DISABLE_FAST_LOCAL_FIELD=1`.
SLURM examples are under `pure_python/jobs/`.

## Key negative result

Planar-averaging the density first and then solving a 1D nonlinear PB does
**not** work: averaging does not commute with the nonlinear dielectric
response, and the bound charge can be wrong by orders of magnitude. Any
dimensional reduction must start from the validated 3D baseline in this
repository and be compared numerically against the reference fields
(`docs/findings.md`).

## Performance status

Same cal_18 fixed-CHGCAR solve: VASP 181 s vs pure Python 380 s (≈2.1×);
the bottleneck is PB step 1 (238 s vs 61 s). Stage-by-stage numbers in
`results/comparison/`.
