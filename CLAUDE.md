# CLAUDE.md — project context and hard rules

## What this is

Validated reproduction of VASP/CEP-DIP implicit-solvent fields (PHI/RHOB/RHOION)
from a frozen CHGCAR, via (a) a patched VASP build and (b) a pure-Python
nonlinear PB solver. Reference case cal_18 (NiN-doped graphene slab).
Correctness status: **validated** — VASP route PHI RMSE 1.7e-7 eV, Python route
3.0e-3 eV. Current open front: performance (Python 2.1× slower than VASP;
bottleneck is PB step 1). See README.md and docs/ for details.

## Hard rules (user-mandated — do not relax)

1. **Provenance is mandatory.** Before any compute job runs, code must be
   committed AND pushed; run directories must record the commit hash via
   `tools/record_provenance.sh <run-dir>`, which fails if the working tree is
   dirty. Never submit a job from a dirty tree.
2. **Validation gate.** Any change touching `pure_python/pb.py`,
   `solute_potential.py`, `dipole_correction.py`, `grid.py`, or `_pb_fast.c`
   goes on a branch and must rerun the cal_18 comparison before merging.
   RMSE must not regress: PHI 3D ≤ 3.008e-03 eV, RHOB 1D ≤ 3.038e-06 e/Å³,
   RHOION 1D ≤ 1.553e-07 e/Å³. Record the numbers in the commit message or
   docs change log. Performance changes also record stage timings.
3. **No VASP source in this repo, ever.** VASP/CEP-DIP changes are made in the
   tree at `$WORK/CEP-DIP`, then exported as whole-tree diffs into `patches/`
   (see patches/README.md for the layer stack). POTCAR and the large volumetric
   fields (CHGCAR/PHI/RHOB/RHOION) never enter the repo. Repo stays private.
4. **Do not restart the 1D shortcut.** Planar-averaging before the nonlinear
   dielectric response is refuted (docs/findings.md). Any dimensional reduction
   starts from the 3D baseline with term-by-term numerical comparison.

## External dependencies (not in this repo)

- Group CEP-DIP tree (patch baseline, licensed): `$WORK/CEP-DIP`
- Reference fields + POTCAR archive: `$WORK/pb_reference_fields/case_cal18/`
  (gzipped; verify against `example/case_cal18/CHECKSUMS.sha256`)
- Original DFT reference run: `$SCRATCH/tmp/4-NiN/2-codex/2-NiN/3-200_structures/cal_18`
  ($SCRATCH is purged periodically — do not rely on it)

## Conventions

- Compute runs live on $SCRATCH and reference repo code via
  `PYTHONPATH=$WORK/repos/cep-dip-python-pb`; do not copy solver code into run dirs.
- Result review pages are self-contained English HTML
  (docs/index.html = docs/results_overview.html, served via GitHub Pages).
- Conversation with the user is in Chinese; repo content and HTML in English.
