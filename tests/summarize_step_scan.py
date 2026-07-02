from __future__ import annotations

import re
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vasp_volumetric import read_vasp_volumetric, write_profile  # noqa: E402


REF_DIR = Path("/scratch/08384/tg876840/tmp/4-NiN/2-codex/2-NiN/3-200_structures/cal_18")
SCAN_DIR = ROOT / "step_scan"
OUT_DIR = ROOT / "results" / "checks"
PROFILE_DIR = ROOT / "results" / "profiles"


def read_elapsed(outcar: Path) -> float | None:
    if not outcar.exists():
        return None
    text = outcar.read_text(errors="ignore")
    m = re.search(r"Elapsed time \(sec\):\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def read_wall_log(run_dir: Path) -> float | None:
    logs = sorted(run_dir.glob("myjob.o*"))
    if not logs:
        return None
    text = logs[-1].read_text(errors="ignore")
    starts = re.findall(r"START .* ([0-9]+)", text)
    ends = re.findall(r"END .* ([0-9]+)", text)
    if starts and ends:
        return float(int(ends[-1]) - int(starts[-1]))
    return None


def parse_cdipol(outcar: Path) -> list[dict[str, str]]:
    if not outcar.exists():
        return []
    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in outcar.read_text(errors="ignore").splitlines():
        if "FIXSOL_STEP" in line:
            if current:
                rows.append(current)
            current = {"step_line": line.strip()}
        elif line.strip().startswith("FIXSOL_CDIPOL_"):
            key, val = line.split("=", 1)
            current[key.strip()] = val.strip()
    if current:
        rows.append(current)
    return rows


def field_metrics(ref_path: Path, new_path: Path) -> dict[str, float]:
    ref = read_vasp_volumetric(ref_path)
    new = read_vasp_volumetric(new_path)
    if ref.grid != new.grid:
        raise ValueError(f"grid mismatch: {ref_path} {ref.grid} vs {new_path} {new.grid}")
    raw_delta = new.values - ref.values
    dens_delta = new.density_e_per_a3() - ref.density_e_per_a3()
    _, ref_z = ref.plane_average_density()
    _, new_z = new.plane_average_density()
    z_delta = new_z - ref_z
    return {
        "raw_rmse": float(np.sqrt(np.mean(raw_delta * raw_delta))),
        "raw_mae": float(np.mean(np.abs(raw_delta))),
        "raw_max_abs": float(np.max(np.abs(raw_delta))),
        "density_rmse_e_A3": float(np.sqrt(np.mean(dens_delta * dens_delta))),
        "density_mae_e_A3": float(np.mean(np.abs(dens_delta))),
        "density_max_abs_e_A3": float(np.max(np.abs(dens_delta))),
        "profile_rmse_e_A3": float(np.sqrt(np.mean(z_delta * z_delta))),
        "profile_mae_e_A3": float(np.mean(np.abs(z_delta))),
        "profile_max_abs_e_A3": float(np.max(np.abs(z_delta))),
        "ref_integral_e": ref.integrated_charge(),
        "new_integral_e": new.integrated_charge(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    fields = ["PHI", "RHOB", "RHOION"]
    summary_lines = [
        "steps\telapsed_s\twall_s\tfield\traw_rmse\traw_mae\traw_max_abs\t"
        "density_rmse_e_A3\tdensity_mae_e_A3\tdensity_max_abs_e_A3\t"
        "profile_rmse_e_A3\tprofile_mae_e_A3\tprofile_max_abs_e_A3\t"
        "ref_integral_e\tnew_integral_e\tintegral_diff_e"
    ]
    diag_lines = ["run\tentry\tstep_line\tkey\tvalue"]

    profile_columns: dict[str, np.ndarray] = {}
    first_z: np.ndarray | None = None

    for steps in range(1, 6):
        run_dir = SCAN_DIR / f"step{steps:02d}"
        if not (run_dir / "OUTCAR").exists():
            continue
        elapsed = read_elapsed(run_dir / "OUTCAR")
        wall = read_wall_log(run_dir)
        for entry_i, row in enumerate(parse_cdipol(run_dir / "OUTCAR"), start=1):
            step_line = row.get("step_line", "")
            for key, value in row.items():
                if key == "step_line":
                    continue
                diag_lines.append(f"step{steps:02d}\t{entry_i}\t{step_line}\t{key}\t{value}")

        for field in fields:
            new_path = run_dir / field
            if not new_path.exists():
                continue
            metrics = field_metrics(REF_DIR / field, new_path)
            summary_lines.append(
                f"{steps}\t{elapsed if elapsed is not None else np.nan:.6f}\t"
                f"{wall if wall is not None else np.nan:.6f}\t{field}\t"
                f"{metrics['raw_rmse']:.16e}\t{metrics['raw_mae']:.16e}\t{metrics['raw_max_abs']:.16e}\t"
                f"{metrics['density_rmse_e_A3']:.16e}\t{metrics['density_mae_e_A3']:.16e}\t"
                f"{metrics['density_max_abs_e_A3']:.16e}\t{metrics['profile_rmse_e_A3']:.16e}\t"
                f"{metrics['profile_mae_e_A3']:.16e}\t{metrics['profile_max_abs_e_A3']:.16e}\t"
                f"{metrics['ref_integral_e']:.16e}\t{metrics['new_integral_e']:.16e}\t"
                f"{metrics['new_integral_e'] - metrics['ref_integral_e']:.16e}"
            )

            ref = read_vasp_volumetric(REF_DIR / field)
            new = read_vasp_volumetric(new_path)
            z, ref_prof = ref.plane_average_density()
            _, new_prof = new.plane_average_density()
            if first_z is None:
                first_z = z
                profile_columns["z_A"] = z
            if field not in profile_columns:
                profile_columns[f"{field}_DFT_e_A3"] = ref_prof
            profile_columns[f"{field}_step{steps:02d}_e_A3"] = new_prof

    summary_path = OUT_DIR / "cal18_step_scan_summary.tsv"
    diag_path = OUT_DIR / "cal18_step_scan_cdipol.tsv"
    profile_path = PROFILE_DIR / "cal18_step_scan_profiles.tsv"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    diag_path.write_text("\n".join(diag_lines) + "\n")
    if profile_columns:
        write_profile(profile_path, profile_columns)
    print(summary_path)
    print(diag_path)
    if profile_columns:
        print(profile_path)


if __name__ == "__main__":
    main()
