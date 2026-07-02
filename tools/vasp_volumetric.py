from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VolumetricData:
    comment: str
    scale: float
    cell: np.ndarray
    elements: list[str]
    counts: list[int]
    coord_mode: str
    positions: np.ndarray
    grid: tuple[int, int, int]
    values: np.ndarray

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def area_xy(self) -> float:
        return float(np.linalg.norm(np.cross(self.cell[0], self.cell[1])))

    @property
    def length_z(self) -> float:
        return float(self.volume / self.area_xy)

    @property
    def ngrid(self) -> int:
        nx, ny, nz = self.grid
        return nx * ny * nz

    def density_e_per_a3(self) -> np.ndarray:
        return self.values / self.volume

    def plane_average_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Return z in Angstrom and xy-averaged density in e/A^3.

        VASP volumetric charge-like files are stored so that sum(values)/Ngrid
        gives the integrated charge. Therefore the physical density is
        values / cell_volume.
        """
        nx, ny, nz = self.grid
        rho = self.density_e_per_a3().reshape((nx, ny, nz), order="F")
        rho_z = rho.mean(axis=(0, 1))
        z = np.arange(nz, dtype=float) * self.length_z / nz
        return z, rho_z

    def plane_average_raw(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = self.grid
        raw = self.values.reshape((nx, ny, nz), order="F")
        raw_z = raw.mean(axis=(0, 1))
        z = np.arange(nz, dtype=float) * self.length_z / nz
        return z, raw_z

    def integrated_charge(self) -> float:
        return float(self.values.sum() / self.ngrid)


def _parse_counts_or_elements(line: str) -> list[str]:
    return line.split()


def read_vasp_volumetric(path: str | Path) -> VolumetricData:
    path = Path(path)
    with path.open("r", errors="ignore") as f:
        lines = f.readlines()

    comment = lines[0].rstrip("\n")
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
    cell *= scale

    tokens5 = _parse_counts_or_elements(lines[5])
    if all(tok.lstrip("+-").isdigit() for tok in tokens5):
        elements = [f"X{i+1}" for i in range(len(tokens5))]
        counts = [int(tok) for tok in tokens5]
        idx = 6
    else:
        elements = tokens5
        counts = [int(tok) for tok in lines[6].split()]
        idx = 7

    if lines[idx].strip().lower().startswith("s"):
        idx += 1
    coord_mode = lines[idx].strip()
    idx += 1

    natoms = sum(counts)
    positions = np.array([[float(x) for x in lines[idx + i].split()[:3]] for i in range(natoms)])
    idx += natoms

    while idx < len(lines) and not lines[idx].split():
        idx += 1
    grid = tuple(int(x) for x in lines[idx].split()[:3])
    idx += 1

    ngrid = grid[0] * grid[1] * grid[2]
    vals: list[float] = []
    for line in lines[idx:]:
        parts = line.split()
        if not parts:
            continue
        try:
            vals.extend(float(x) for x in parts)
        except ValueError:
            break
        if len(vals) >= ngrid:
            break
    if len(vals) < ngrid:
        raise ValueError(f"{path}: expected {ngrid} values, got {len(vals)}")

    return VolumetricData(
        comment=comment,
        scale=scale,
        cell=cell,
        elements=elements,
        counts=counts,
        coord_mode=coord_mode,
        positions=positions,
        grid=grid,
        values=np.asarray(vals[:ngrid], dtype=float),
    )


def write_profile(path: str | Path, columns: dict[str, np.ndarray]) -> None:
    path = Path(path)
    names = list(columns)
    n = len(columns[names[0]])
    for name in names:
        if len(columns[name]) != n:
            raise ValueError(f"column length mismatch for {name}")
    with path.open("w") as f:
        f.write("\t".join(names) + "\n")
        for i in range(n):
            f.write("\t".join(f"{columns[name][i]:.12e}" for name in names) + "\n")
