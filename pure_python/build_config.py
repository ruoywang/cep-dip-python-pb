from __future__ import annotations

import argparse

from .config import build_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default="data/case_cal18")
    parser.add_argument("--output", default="pure_python/configs/cal18.json")
    args = parser.parse_args()
    build_config(args.case_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

