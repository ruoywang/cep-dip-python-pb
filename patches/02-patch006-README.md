# 2026-06-29 fixed-CHGCAR LDIPOL/solvation update

Source validated in:

`/scratch/08384/tg876840/tmp/a-band/bug_check/vasp_src/CEP-DIP`

Validation run:

`/scratch/08384/tg876840/tmp/a-band/bug_check/runs/case01_nscf3x3_patch006`

Reference SCF:

`/scratch/08384/tg876840/tmp/1-Au_single/1-32_GCE/cal_29`

Applied changes:

1. `src/pot.F`
   - Added `LDO_DIPOL_CORRECTION`.
   - For fixed-charge calculations (`INFO%ICHARG >= 10`), allow the first `POTLOK` call to include the dipole correction path.
   - Use the same condition for the `SOL_Vcorrection(..., CVDIP)` path.

2. `src/electron.F`
   - For fixed-charge calculations with `LDIPOL`, set `INFO%LPOTOK=.FALSE.` after each electronic step except the last.
   - This keeps the input charge density fixed but lets the solvation/dipole potential rebuild until it reaches the same fixed point as the SCF calculation.

Files intentionally not changed:

1. `src/dipol.F`
   - The validated fix keeps the original residual-dependent dipole mixing.

2. `src/force.F`
   - The `ASSOCIATED(DIP%FORCE)` guard was already present.

Validation summary from patch006:

- SCF reference:
  - `E-fermi = -3.1845 eV`
  - `alpha+bet = -3.6314 eV`
  - `Solvation Ecorr_sol = 2620.10021084 eV`
  - `dipolmoment_z = -3.958845 e A`

- Fixed-CHGCAR patch006:
  - `E-fermi = -3.1856 eV`
  - `alpha+bet = -3.6307 eV`
  - `Solvation Ecorr_sol = 2620.68987813 eV`
  - `dipolmoment_z = -3.957076 e A`

- Difference:
  - `dE-fermi = -0.0011 eV`
  - `d(alpha+bet) = +0.0007 eV`
  - `dEcorr_sol = +0.58966729 eV`
  - `ddipolmoment_z = +0.001769 e A`

Diff files in this directory:

- `pot.F.diff`
- `electron.F.diff`
- `force.F.diff`
- `dipol.F.diff`

Build/update result:

- `make std -j 8` successfully rebuilt `build/std/vasp`.
- The final `make` copy step failed because the old `bin/vasp_std` was in use and direct overwrite returned `Text file busy`.
- The new binary was installed at the original executable path: `/work/08384/tg876840/ls6/CEP-DIP/bin/vasp_std`.
- No alternate patch executable is required or kept in this record directory.
- `build/std/vasp` and `bin/vasp_std` have identical SHA256 hashes recorded in `binary_sha256.txt`.
