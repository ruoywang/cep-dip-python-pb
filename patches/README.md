# CEP-DIP patches

Modifications made to the group's CEP-DIP distribution (VASP 6.3.2 + CEP-DIP)
for the fixed-CHGCAR solvent-field reproduction. **No VASP source is included
in this repository** — only these diffs. Apply them to a licensed CEP-DIP
source tree.

## Baseline

The baseline is the CEP-DIP tree at `$WORK/CEP-DIP` (VASP 6.3.2 with the
upstream CEP-DIP/vaspsol++ modifications; see `difflog` in that tree for the
upstream author's own diff against stock VASP release 6.3).

## Patch stack (apply in order)

| # | File | What it does |
|---|------|--------------|
| 1 | `01-forceguard-20260628.diff` | `force.F`: guard `DIP%FORCE` access with `ASSOCIATED()`; `dipol.F`: dipole-mixing robustness (direction-general `IDIR`, initialized `SAVE` variables). |
| 2 | `02a-patch006-pot.diff`, `02b-patch006-electron.diff` | Fixed-charge (`ICHARG>=10`) calculations: allow the dipole-correction path in the first `POTLOK` call and rebuild the solvation/dipole potential each electronic step (`LPOTOK=.FALSE.` except the last). Validated in patch006; see `02-patch006-README.md`. |
| 3 | `03-fixed-chgcar-repro.diff` | The fixed-CHGCAR reproduction changes: `fileio.F` (optional READCH flag to skip old/new-position charge correction), `main.F` (preserve and restore the fixed-density CHGCAR charge for final solvation/dipole output), `solvation.F` (debug output fields used for input-consistency verification), `dipol.F` (further dipole handling for the fixed-density path). |

Layers 1–2 are already present in the `$WORK/CEP-DIP` baseline; layer 3 is the
delta between that baseline and the validated reproduction tree
(`reproduce3d/code/CEP-DIP`, 2026-07-01).

## How to apply

```sh
cd /path/to/CEP-DIP        # licensed source tree
# if starting from a tree that already has layers 1-2 (e.g. $WORK/CEP-DIP):
patch -p1 < patches/03-fixed-chgcar-repro.diff
# if starting from a pristine upstream CEP-DIP tree, apply 01, 02a, 02b first.
```

Then rebuild `vasp_std` as usual (`make std`).
