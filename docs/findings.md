# CEP-DIP PB Reproduction Notes

This directory now keeps the validated 3D VASP/CEP-DIP fixed-CHGCAR
reproduction path and removes the earlier standalone 1D PB approximation code.

## Current Useful Result

Using the VASP/CEP-DIP framework with fixed final `CHGCAR`, the workflow can
recompute solvent fields from the original VASP input plus `CHGCAR`.

The retained reference case is:

`data/case_cal18`

It contains:

- `INCAR`
- `POSCAR`
- `POTCAR`
- `KPOINTS`
- `CHGCAR`
- reference `PHI`, `RHOB`, `RHOION`, and `OUTCAR` for comparisons

The retained source tree is:

`reproduce3d/code/CEP-DIP`

The retained final summaries are:

- `reproduce3d/results/final_reproduction_summary.txt`
- `reproduce3d/results/compare_fast_solonly15_vs_dft_cal18.txt`
- `results/checks/cal18_dft_vs_fast_solonly15_summary.txt`
- `results/checks/cal18_step_scan_summary.tsv`
- `results/profiles/cal18_step_scan_profiles.tsv`
- `results/profiles/cal18_step05_rhob_rhoion_1d.tsv`

## Important Negative Result

> **2026-07-02 caveat:** the user suspects the standalone 1D code that produced
> this negative result was itself buggy — it predates the validated 3D
> reproduction below. Treat the conclusion as "unvalidated route with one failed
> (possibly buggy) attempt", not as settled physics. Any retry must be built on
> the validated 3D baseline and compared term-by-term against the retained
> reference fields.

The earlier standalone 1D PB approximation was removed from the active tree.
It was not a validated reproduction route. The reason is physical/numerical:
plane-averaging before applying the nonlinear dielectric response does not
commute with the original 3D cavity/dielectric calculation. The resulting bound
charge can be orders of magnitude wrong even when the ionic charge looks
reasonable.

For this reason, future work should start from the validated 3D CEP-DIP
fixed-CHGCAR reproduction and only then reduce/approximate pieces with explicit
numerical comparisons against the retained reference fields.
