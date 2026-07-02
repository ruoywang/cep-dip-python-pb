from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vasp_inputs import read_chgcar  # noqa: E402


DEFAULT_REFERENCE = Path("/scratch/08384/tg876840/tmp/4-NiN/2-codex/2-NiN/3-200_structures/cal_18")
DEFAULT_REBUILT = ROOT / "work" / "cal18_vasp_from_chgcar"
DEFAULT_RESULT = ROOT / "results" / "compare_vasp_from_chgcar_cal18.txt"

FIELDS = [
    "CHGCAR",
    "PHI",
    "PHISOLV",
    "RHOB",
    "RHOION",
    "SION",
    "SDIEL",
    "SSOLV",
    "SCAV",
]


def summarize_field(name: str, ref_dir: Path, new_dir: Path) -> list[str]:
    ref_path = ref_dir / name
    new_path = new_dir / name
    lines: list[str] = []
    lines.append(f"[{name}]")
    lines.append(f"  reference_exists: {ref_path.exists()}")
    lines.append(f"  rebuilt_exists: {new_path.exists()}")
    if not ref_path.exists() or not new_path.exists():
        return lines

    ref = read_chgcar(ref_path)
    new = read_chgcar(new_path)
    lines.append(f"  ref_grid: {ref.grid}")
    lines.append(f"  new_grid: {new.grid}")
    if ref.grid != new.grid:
        lines.append("  status: grid_mismatch")
        return lines

    delta = new.values - ref.values
    rmse = float(np.sqrt(np.mean(delta * delta)))
    mae = float(np.mean(np.abs(delta)))
    max_abs = float(np.max(np.abs(delta)))
    mean_delta = float(np.mean(delta))
    ref_int = ref.integrated_charge
    new_int = new.integrated_charge
    lines.append(f"  ref_integral_or_mean: {ref_int:.16e}")
    lines.append(f"  new_integral_or_mean: {new_int:.16e}")
    lines.append(f"  delta_integral_or_mean: {new_int - ref_int:.16e}")
    lines.append(f"  rmse_raw: {rmse:.16e}")
    lines.append(f"  mae_raw: {mae:.16e}")
    lines.append(f"  max_abs_raw: {max_abs:.16e}")
    lines.append(f"  mean_delta_raw: {mean_delta:.16e}")

    nx, ny, nz = ref.grid
    ref_z = ref.values.reshape((nx, ny, nz), order="F").mean(axis=(0, 1))
    new_z = new.values.reshape((nx, ny, nz), order="F").mean(axis=(0, 1))
    dz = new_z - ref_z
    lines.append(f"  rmse_zmean: {float(np.sqrt(np.mean(dz * dz))):.16e}")
    lines.append(f"  mae_zmean: {float(np.mean(np.abs(dz))):.16e}")
    lines.append(f"  max_abs_zmean: {float(np.max(np.abs(dz))):.16e}")
    lines.append("  status: compared")
    return lines


def main() -> None:
    out: list[str] = []
    ref_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    new_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REBUILT
    result = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RESULT

    out.append(f"reference: {ref_dir}")
    out.append(f"rebuilt: {new_dir}")
    for field in FIELDS:
        out.extend(summarize_field(field, ref_dir, new_dir))
    result.write_text("\n".join(out) + "\n")
    print(result)


if __name__ == "__main__":
    main()
