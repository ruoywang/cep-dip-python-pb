#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vasp_inputs import read_chgcar  # noqa: E402


def summarize(ref_path: Path, test_path: Path) -> str:
    ref = read_chgcar(ref_path)
    test = read_chgcar(test_path)
    if ref.grid != test.grid:
        raise ValueError(f"grid mismatch for {ref_path.name}: {ref.grid} vs {test.grid}")

    diff = test.values - ref.values
    nx, ny, nz = ref.grid
    zmean = diff.reshape((nx, ny, nz), order="F").mean(axis=(0, 1))
    return (
        f"{ref_path.name}\n"
        f"  integral_ref      {ref.integrated_charge:.12g}\n"
        f"  integral_test     {test.integrated_charge:.12g}\n"
        f"  integral_diff     {test.integrated_charge - ref.integrated_charge:.12g}\n"
        f"  raw_rmse          {np.sqrt(np.mean(diff * diff)):.12g}\n"
        f"  raw_mae           {np.mean(np.abs(diff)):.12g}\n"
        f"  raw_max_abs       {np.max(np.abs(diff)):.12g}\n"
        f"  raw_mean          {np.mean(diff):.12g}\n"
        f"  zmean_rmse        {np.sqrt(np.mean(zmean * zmean)):.12g}\n"
        f"  zmean_max_abs     {np.max(np.abs(zmean)):.12g}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("fields", nargs="+")
    args = parser.parse_args()

    for name in args.fields:
        print(summarize(args.reference / name, args.test / name))


if __name__ == "__main__":
    main()
