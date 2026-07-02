#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path


TIMING_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z0-9_*<>.-]+):\s+cpu time\s+"
    r"(?P<cpu>[-+0-9.Ee]+):\s+real time\s+(?P<real>[-+0-9.Ee]+)"
)


def parse_outcar(path: Path):
    events = []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        match = TIMING_RE.match(line)
        if not match:
            continue
        label = match.group("label")
        events.append(
            {
                "event": len(events) + 1,
                "line": lineno,
                "label": label,
                "cpu_s": float(match.group("cpu")),
                "real_s": float(match.group("real")),
            }
        )
    return events


def write_events(events, path: Path):
    with path.open("w") as f:
        f.write("event\tline\tlabel\tcpu_s\treal_s\n")
        for event in events:
            f.write(
                f"{event['event']}\t{event['line']}\t{event['label']}\t"
                f"{event['cpu_s']:.9g}\t{event['real_s']:.9g}\n"
            )


def write_summary(events, path: Path):
    grouped = defaultdict(lambda: [0, 0.0, 0.0])
    for event in events:
        row = grouped[event["label"]]
        row[0] += 1
        row[1] += event["cpu_s"]
        row[2] += event["real_s"]
    with path.open("w") as f:
        f.write("label\tcount\tcpu_total_s\treal_total_s\tcpu_mean_s\treal_mean_s\n")
        for label, (count, cpu, real) in sorted(grouped.items(), key=lambda item: -item[1][2]):
            f.write(
                f"{label}\t{count}\t{cpu:.9g}\t{real:.9g}\t"
                f"{cpu / count:.9g}\t{real / count:.9g}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Parse VASP SOL timing labels from OUTCAR.")
    parser.add_argument("outcar", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    outdir = args.outdir or args.outcar.parent
    outdir.mkdir(parents=True, exist_ok=True)
    events = parse_outcar(args.outcar)
    write_events(events, outdir / "sol_timing_events.tsv")
    write_summary(events, outdir / "sol_timing_summary.tsv")
    print(f"parsed_events={len(events)}")
    print(f"events={outdir / 'sol_timing_events.tsv'}")
    print(f"summary={outdir / 'sol_timing_summary.tsv'}")


if __name__ == "__main__":
    main()
