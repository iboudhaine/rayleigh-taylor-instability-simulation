"""Run the benchmark sweep across Python, C, and CUDA runners and write a CSV.

Each runner emits a single JSON stats line on its final stdout line; this script
parses those lines into rows of a CSV. CUDA is included only if the binary
`src/cuda/run` exists (i.e. you've run `make -C src/cuda`).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_GRIDS = [(128, 256), (256, 512), (512, 1024)]
DEFAULT_IMPLS = ["python", "c", "cuda"]


def python_cmd() -> list[str]:
    venv_py = REPO / ".venv" / "bin" / "python"
    return [str(venv_py if venv_py.exists() else sys.executable)]


def impl_cmd(impl: str) -> list[str] | None:
    if impl == "python":
        return python_cmd() + [str(REPO / "src" / "python" / "run.py")]
    if impl == "c":
        bin_ = REPO / "src" / "c" / "run"
        return [str(bin_)] if bin_.exists() else None
    if impl == "cuda":
        bin_ = REPO / "src" / "cuda" / "run"
        return [str(bin_)] if bin_.exists() else None
    raise SystemExit(f"unknown impl: {impl}")


def parse_grid(s: str) -> tuple[int, int]:
    nx, ny = s.lower().split("x")
    return int(nx), int(ny)


def run_one(impl: str, base: list[str], Nx: int, Ny: int, common: list[str]) -> dict:
    cmd = base + ["--Nx", str(Nx), "--Ny", str(Ny)] + common
    print(f"  $ {' '.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"runner failed: {impl} {Nx}x{Ny}")
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--impls",
        default=",".join(DEFAULT_IMPLS),
        help="comma-separated subset of {python,c,cuda}",
    )
    p.add_argument(
        "--grids",
        default=",".join(f"{nx}x{ny}" for nx, ny in DEFAULT_GRIDS),
        help='comma-separated grids, e.g. "128x256,256x512"',
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=20, dest="warmup_steps")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheme", default="rk2", choices=["euler", "rk2"])
    p.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "bench.csv",
        help="output CSV path",
    )
    args = p.parse_args()

    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    grids = [parse_grid(s) for s in args.grids.split(",") if s.strip()]
    common = [
        "--steps",
        str(args.steps),
        "--scheme",
        args.scheme,
        "--warmup-steps",
        str(args.warmup_steps),
        "--repeats",
        str(args.repeats),
        "--no-output",
        "--seed",
        str(args.seed),
    ]

    rows = []
    for impl in impls:
        base = impl_cmd(impl)
        if base is None:
            print(f"skipping {impl}: binary not built", file=sys.stderr)
            continue
        for Nx, Ny in grids:
            print(f"[{impl}] {Nx}x{Ny}", file=sys.stderr, flush=True)
            stats = run_one(impl, base, Nx, Ny, common)
            # serialize the list-valued field so it fits in a CSV cell
            stats["wall_s_all"] = json.dumps(stats.get("wall_s_all", []))
            rows.append(stats)

    if not rows:
        raise SystemExit("no rows collected")

    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.output}  ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
