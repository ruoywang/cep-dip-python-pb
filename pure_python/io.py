from __future__ import annotations

from pathlib import Path

import numpy as np

from src.vasp_volumetric import VolumetricData


def write_vasp_like(path: str | Path, template: VolumetricData, values: np.ndarray, label: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(values, dtype=float).reshape(template.grid, order="C").ravel(order="F")
    with path.open("w") as f:
        f.write(f"{label}\n")
        f.write(f"{template.scale:.16g}\n")
        for row in template.cell / template.scale:
            f.write("  " + " ".join(f"{x:20.12f}" for x in row) + "\n")
        f.write("  " + " ".join(template.elements) + "\n")
        f.write("  " + " ".join(str(x) for x in template.counts) + "\n")
        f.write(f"{template.coord_mode}\n")
        for pos in template.positions:
            f.write("  " + " ".join(f"{x:20.12f}" for x in pos) + "\n")
        f.write("\n")
        f.write("  " + " ".join(str(x) for x in template.grid) + "\n")
        for i in range(0, len(flat), 5):
            f.write(" ".join(f"{x:20.12E}" for x in flat[i : i + 5]) + "\n")

