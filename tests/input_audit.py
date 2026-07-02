from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vasp_inputs import incar_float, read_chgcar, read_incar, read_poscar, read_potcar_zvals, zval_total


CASE_DIR = ROOT.parent / "data" / "case_cal18"
OUT = ROOT / "results" / "input_audit_cal18.txt"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    incar = read_incar(CASE_DIR / "INCAR")
    poscar = read_poscar(CASE_DIR / "POSCAR")
    chgcar = read_chgcar(CASE_DIR / "CHGCAR")
    zvals = read_potcar_zvals(CASE_DIR / "POTCAR")

    nelect = incar_float(incar, "NELECT")
    ztot = zval_total(poscar, zvals)
    q_sol = nelect - ztot
    chg_int = chgcar.integrated_charge

    lines = [
        "input audit: cal18",
        f"case_dir: {CASE_DIR}",
        f"elements: {' '.join(poscar.elements)}",
        f"counts: {' '.join(str(x) for x in poscar.counts)}",
        f"zvals: {' '.join(f'{x:.8f}' for x in zvals)}",
        f"zval_total: {ztot:.12f}",
        f"nelect_incar: {nelect:.12f}",
        f"q_sol_nelect_minus_zval: {q_sol:.12f}",
        f"chgcar_grid: {chgcar.grid[0]} {chgcar.grid[1]} {chgcar.grid[2]}",
        f"chgcar_integrated_charge: {chg_int:.12f}",
        f"chgcar_minus_nelect: {chg_int - nelect:.12e}",
        f"cell_volume_A3: {chgcar.volume:.12f}",
    ]
    if abs(chg_int - nelect) > 1e-3:
        raise RuntimeError("CHGCAR integral does not match NELECT")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
