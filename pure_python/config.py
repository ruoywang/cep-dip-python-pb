from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.vasp_volumetric import read_vasp_volumetric


DEFAULTS: dict[str, Any] = {
    "LNLDIEL": True,
    "LNLION": True,
    "LNLTEST": False,
    "I_NLOC_SOL": 1,
    "NC_K": 0.015,
    "SIGMA_K": 0.6,
    "A_K": 0.125,
    "TAU": 8.79e-4,
    "R_SOLV": 1.4,
    "R_CAV": 0.0,
    "R_DIEL": 1.0,
    "R_ION": 0.0,
    "R_B": 0.0,
    "EB_K": 78.4,
    "LAMBDA_D_K": 0.0,
    "SOLTEMP": 298.0,
    "ZION": 1.0,
    "D_ION": -1.0,
    "C_MOLAR": 0.0,
    "N_MOL": 0.0335,
    "P_MOL": 0.50,
    "EPSILON_INF": 1.78,
    "LVAC": False,
    "SOL_Z0": 0.0,
    "SOL_Z1": 0.0,
    "SOL_SIGMA": 0.8,
    "D_STERN": 2.0,
    "N_MIN": 1.0e-4,
}


def _parse_bool(value: str) -> bool:
    text = value.strip().strip(".").upper()
    if text in {"T", "TRUE", "1"}:
        return True
    if text in {"F", "FALSE", "0"}:
        return False
    raise ValueError(f"not a VASP boolean: {value}")


def parse_incar(path: str | Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in Path(path).read_text(errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        token = value.replace(";", " ").split()[0]
        if not token:
            continue
        if token.strip().strip(".").upper() in {"T", "TRUE", "F", "FALSE"}:
            values[key] = _parse_bool(token)
        else:
            try:
                values[key] = int(token)
            except ValueError:
                try:
                    values[key] = float(token.replace("D", "E").replace("d", "e"))
                except ValueError:
                    continue
    return values


def parse_potcar_zvals(path: str | Path) -> list[float]:
    text = Path(path).read_text(errors="ignore")
    return [float(x) for x in re.findall(r"ZVAL\s*=\s*([-+0-9.EDed]+)", text)]


def build_config(case_dir: str | Path, output: str | Path) -> None:
    case_dir = Path(case_dir)
    chg = read_vasp_volumetric(case_dir / "CHGCAR")
    incar = parse_incar(case_dir / "INCAR")
    params = DEFAULTS.copy()
    # Fortran reader uses mixed spelling. Normalize likely INCAR keys.
    aliases = {
        "SOLTEMP": "SOLTEMP",
        "SOL_TEMP": "SOLTEMP",
        "EPSILON_INF": "EPSILON_INF",
        "I_NLOC_SOL": "I_NLOC_SOL",
    }
    for key, val in incar.items():
        canonical = aliases.get(key, key)
        if canonical in params:
            params[canonical] = val
    zvals = parse_potcar_zvals(case_dir / "POTCAR")
    if len(zvals) != len(chg.elements):
        raise ValueError(f"ZVAL count {len(zvals)} != element count {len(chg.elements)}")
    total_zval = float(sum(count * zval for count, zval in zip(chg.counts, zvals)))
    nelect = float(incar.get("NELECT", chg.integrated_charge()))
    config = {
        "format": "cep_dip_python_config_v1",
        "cell_A": chg.cell.tolist(),
        "grid": list(chg.grid),
        "elements": chg.elements,
        "counts": chg.counts,
        "positions_direct": chg.positions.tolist(),
        "zval": dict(zip(chg.elements, zvals)),
        "total_zval": total_zval,
        "nelect": nelect,
        "q_sol": nelect - total_zval,
        "solvation": params,
        "dencor": {"mode": "zero", "note": "first implementation; POTCAR DENCOR port not complete"},
        "solute_potential": {"mode": "poisson_point_ions", "note": "diagnostic starting point"},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def config_grid(config: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int, int]]:
    return np.asarray(config["cell_A"], dtype=float), tuple(int(x) for x in config["grid"])
