# CLAUDE.md — project context and hard rules

## What this is

Validated reproduction of VASP/CEP-DIP implicit-solvent fields (PHI/RHOB/RHOION)
from a frozen CHGCAR, via (a) a patched VASP build and (b) a pure-Python
nonlinear PB solver. Reference case cal_18 (NiN-doped graphene slab).
Correctness status: **validated** — VASP route PHI RMSE 1.7e-7 eV, Python route
3.0e-3 eV. Performance: after the 2026-07 CPU round (fused OpenMP kernels,
coarse-grid warm start, half-spectrum FFT) the full solve is 81 s vs VASP's
181 s. Production defaults: `PB_RFFT=1` and `--coarse-init` on.
See README.md, docs/, and results/perf_cpu_round1/ for details.

## Hard rules (user-mandated — do not relax)

1. **Provenance is mandatory.** Before any compute job runs, code must be
   committed AND pushed; run directories must record the commit hash via
   `tools/record_provenance.sh <run-dir>`, which fails if the working tree is
   dirty. Never submit a job from a dirty tree.
2. **Validation gate.** Any change touching `pure_python/pb.py`,
   `solute_potential.py`, `dipole_correction.py`, `grid.py`, or `_pb_fast.c`
   goes on a branch and must rerun the cal_18 comparison before merging.
   RMSE must not regress: PHI 3D ≤ 2.998e-03 eV, PHI 1D ≤ 2.364e-04 eV,
   RHOB 1D ≤ 2.238e-06 e/Å³, RHOION 1D ≤ 6.848e-08 e/Å³ (best-known values,
   step3_rfft 2026-07-02; small tolerance-noise excursions must be judged
   against PHI first). Record the numbers in the commit message or docs
   change log. Performance changes also record stage timings.
3. **No VASP source in this repo, ever.** VASP/CEP-DIP changes are made in the
   tree at `$WORK/CEP-DIP`, then exported as whole-tree diffs into `patches/`
   (see patches/README.md for the layer stack). POTCAR and the large volumetric
   fields (CHGCAR/PHI/RHOB/RHOION) never enter the repo. Repo stays private.
4. **The 1D shortcut is unvalidated, not forbidden.** The original negative
   result (docs/findings.md) came from code the user now suspects was buggy —
   it predates the validated 3D reproduction. Revisiting a 1D/reduced model is
   a legitimate direction, but it MUST start from the validated 3D baseline
   and be compared term-by-term against the retained reference fields; never
   trust a reduced model that has not passed that comparison.

## External dependencies (not in this repo)

- Group CEP-DIP tree (patch baseline, licensed): `$WORK/CEP-DIP`
- Reference fields + POTCAR archive: `$WORK/pb_reference_fields/case_cal18/`
  (gzipped; verify against `example/case_cal18/CHECKSUMS.sha256`)
- Original DFT reference run: `$SCRATCH/tmp/4-NiN/2-codex/2-NiN/3-200_structures/cal_18`
  ($SCRATCH is purged periodically — do not rely on it)

## Conventions

- Compute runs live in run directories under the user's current working
  directory (they open sessions where they want to work) and reference repo
  code via `PYTHONPATH=$WORK/repos/cep-dip-python-pb`; do not copy solver
  code into run dirs, and do not create top-level directories elsewhere.
- Result review pages are self-contained English HTML
  (docs/index.html = docs/results_overview.html, served via GitHub Pages).
- Conversation with the user is in Chinese; repo content and HTML in English.
