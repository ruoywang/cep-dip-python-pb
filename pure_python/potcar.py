from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

NPSPTS = 1000


@dataclass
class PotcarEntry:
    element: str
    zval: float
    psgmax: float
    psp_values: np.ndarray
    psp_spline: np.ndarray
    pspcor: np.ndarray | None
    psprho: np.ndarray


def _floats_from_lines(lines: list[str], start: int, count: int) -> tuple[np.ndarray, int]:
    vals: list[float] = []
    idx = start
    while idx < len(lines) and len(vals) < count:
        vals.extend(float(x) for x in lines[idx].replace("D", "E").split())
        idx += 1
    if len(vals) < count:
        raise ValueError(f"expected {count} floats, got {len(vals)}")
    return np.asarray(vals[:count], dtype=float), idx


def _spline_coefficients(x: np.ndarray, y: np.ndarray, y1p: float = 0.0) -> np.ndarray:
    n = len(x)
    p = np.zeros((n, 5), dtype=float)
    p[:, 0] = x
    p[:, 1] = y
    if y1p > 0.99e30:
        p[0, 3] = 0.0
        p[0, 2] = 0.0
    else:
        p[0, 3] = -0.5
        p[0, 2] = (3.0 / (p[1, 0] - p[0, 0])) * ((p[1, 1] - p[0, 1]) / (p[1, 0] - p[0, 0]) - y1p)
    for i in range(1, n - 1):
        s = (p[i, 0] - p[i - 1, 0]) / (p[i + 1, 0] - p[i - 1, 0])
        r = s * p[i - 1, 3] + 2.0
        p[i, 3] = (s - 1.0) / r
        p[i, 2] = (
            6.0
            * (
                (p[i + 1, 1] - p[i, 1]) / (p[i + 1, 0] - p[i, 0])
                - (p[i, 1] - p[i - 1, 1]) / (p[i, 0] - p[i - 1, 0])
            )
            / (p[i + 1, 0] - p[i - 1, 0])
            - s * p[i - 1, 2]
        ) / r
    p[n - 1, 3] = 0.0
    p[n - 1, 2] = 0.0
    for i in range(n - 2, -1, -1):
        p[i, 3] = p[i, 3] * p[i + 1, 3] + p[i, 2]
    for i in range(0, n - 1):
        s = p[i + 1, 0] - p[i, 0]
        r = (p[i + 1, 3] - p[i, 3]) / 6.0
        p[i, 4] = r / s
        p[i, 3] = p[i, 3] / 2.0
        p[i, 2] = (p[i + 1, 1] - p[i, 1]) / s - (p[i, 3] + r) * s
    return p


def read_potcar(path: str | Path) -> list[PotcarEntry]:
    lines = Path(path).read_text(errors="ignore").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("PAW_")]
    entries: list[PotcarEntry] = []
    for block_index, start in enumerate(starts):
        end = starts[block_index + 1] if block_index + 1 < len(starts) else len(lines)
        block = lines[start:end]
        element = ""
        zval = None
        for line in block[:80]:
            if "VRHFIN" in line:
                match = re.search(r"VRHFIN\s*=([A-Za-z]+)", line)
                if match:
                    element = match.group(1)
            if "ZVAL" in line:
                match = re.search(r"ZVAL\s*=\s*([0-9.+\-EeDd]+)", line)
                if match:
                    zval = float(match.group(1).replace("D", "E"))
        if not element or zval is None:
            raise ValueError(f"failed to parse POTCAR header at block {block_index}")

        local_idx = next(i for i, line in enumerate(block) if line.strip() == "local part")
        psgmax = float(block[local_idx + 1].split()[0].replace("D", "E"))
        psp_values, idx = _floats_from_lines(block, local_idx + 2, NPSPTS)
        psp_x = np.arange(NPSPTS, dtype=float) * (psgmax / NPSPTS)
        psp_spline = _spline_coefficients(psp_x, psp_values, 0.0)

        pspcor = None
        psprho = None
        while idx < len(block):
            label = block[idx].strip()
            idx += 1
            if label.startswith("gradient corrections"):
                idx += 1
                continue
            if label == "core charge-density (partial)":
                pspcor, idx = _floats_from_lines(block, idx, NPSPTS)
                continue
            if label.startswith("kinetic energy density"):
                _, idx = _floats_from_lines(block, idx, NPSPTS)
                continue
            if label == "atomic pseudo charge-density":
                psprho, idx = _floats_from_lines(block, idx, NPSPTS)
                break
        if psprho is None:
            raise ValueError(f"failed to parse atomic pseudo charge-density for {element}")
        entries.append(PotcarEntry(element, float(zval), psgmax, psp_values, psp_spline, pspcor, psprho))
    return entries
