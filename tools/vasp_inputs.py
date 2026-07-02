from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class PoscarData:
    elements: list[str]
    counts: list[int]
    cell: np.ndarray


@dataclass(frozen=True)
class ChgcarData:
    poscar: PoscarData
    grid: tuple[int, int, int]
    values: np.ndarray

    @property
    def ngrid(self) -> int:
        nx, ny, nz = self.grid
        return nx * ny * nz

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.poscar.cell)))

    @property
    def integrated_charge(self) -> float:
        return float(np.sum(self.values) / self.ngrid)


def read_incar(path: str | Path) -> dict[str, str]:
    params: dict[str, str] = {}
    for raw in Path(path).read_text(errors="ignore").splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[key.strip().upper()] = value.strip()
    return params


def incar_float(params: dict[str, str], key: str, default: float | None = None) -> float:
    key = key.upper()
    if key not in params:
        if default is None:
            raise KeyError(key)
        return default
    return float(params[key].split()[0])


def read_poscar(path: str | Path) -> PoscarData:
    lines = Path(path).read_text(errors="ignore").splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float) * scale
    tokens5 = lines[5].split()
    if all(tok.lstrip("+-").isdigit() for tok in tokens5):
        elements = [f"X{i + 1}" for i in range(len(tokens5))]
        counts = [int(tok) for tok in tokens5]
    else:
        elements = tokens5
        counts = [int(tok) for tok in lines[6].split()]
    return PoscarData(elements=elements, counts=counts, cell=cell)


def read_chgcar(path: str | Path) -> ChgcarData:
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()
    poscar = read_poscar(path)
    idx = 7 if not all(tok.lstrip("+-").isdigit() for tok in lines[5].split()) else 6
    if lines[idx].strip().lower().startswith("s"):
        idx += 1
    idx += 1 + sum(poscar.counts)
    while idx < len(lines) and not lines[idx].split():
        idx += 1
    grid = tuple(int(x) for x in lines[idx].split()[:3])
    idx += 1
    ngrid = grid[0] * grid[1] * grid[2]
    values: list[float] = []
    for line in lines[idx:]:
        parts = line.split()
        if not parts:
            continue
        try:
            values.extend(float(x) for x in parts)
        except ValueError:
            break
        if len(values) >= ngrid:
            break
    if len(values) < ngrid:
        raise ValueError(f"{path}: expected {ngrid} grid values, got {len(values)}")
    return ChgcarData(poscar=poscar, grid=grid, values=np.asarray(values[:ngrid], dtype=float))


def read_potcar_zvals(path: str | Path) -> list[float]:
    text = Path(path).read_text(errors="ignore")
    return [float(x) for x in re.findall(r"ZVAL\s*=\s*([0-9.+\-Ee]+)", text)]


def zval_total(poscar: PoscarData, zvals: list[float]) -> float:
    if len(poscar.counts) != len(zvals):
        raise ValueError(f"element count {len(poscar.counts)} != ZVAL count {len(zvals)}")
    return float(sum(count * zval for count, zval in zip(poscar.counts, zvals)))
