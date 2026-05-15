"""Shared CLI specification for all runners (Python, C, and CUDA)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Physics / problem setup
    Nx: int = 512
    Ny: int = 1024
    Lx: float = 1.0
    Ly: float = 2.0
    gamma: float = 1.4
    g: float = -10.0
    rho_heavy: float = 2.0
    rho_light: float = 1.0
    p0: float = 40.0
    perturbation_amp: float = 1e-3
    seed: int = 0

    # Numerics
    scheme: str = "rk2"  # "euler" | "rk2"
    cfl: float = 0.2
    k1_coef: float = 0.0125
    k2_coef: float = 0.125
    k3_coef: float = 0.0125

    # Run control (mutually exclusive: steps vs t_end; steps wins if both set)
    steps: Optional[int] = None
    t_end: Optional[float] = None
    save_interval: int = 0  # in simulation steps; 0 disables frame output
    output_dir: str = "results/run"
    no_output: bool = False
    overwrite: bool = False  # allow writing into an existing output_dir

    # Benchmark
    # warmup_steps: steps run before the timer starts (excludes startup costs)
    # repeats: re-runs of the timed section; median wall time is reported
    # Bench mode is implicit: triggered when warmup_steps > 0 OR repeats > 1
    warmup_steps: int = 0
    repeats: int = 1

    # Impl-specific (CUDA)
    block_x: int = 16
    block_y: int = 16


def add_shared_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("physics")
    g.add_argument("--Nx", type=int, default=512)
    g.add_argument("--Ny", type=int, default=1024)
    g.add_argument("--Lx", type=float, default=1.0)
    g.add_argument("--Ly", type=float, default=2.0)
    g.add_argument("--gamma", type=float, default=1.4)
    g.add_argument("--g", type=float, default=-10.0, dest="g")
    g.add_argument("--rho-heavy", type=float, default=2.0, dest="rho_heavy")
    g.add_argument("--rho-light", type=float, default=1.0, dest="rho_light")
    g.add_argument("--p0", type=float, default=40.0)
    g.add_argument(
        "--perturbation-amp", type=float, default=1e-3, dest="perturbation_amp"
    )
    g.add_argument("--seed", type=int, default=0)

    n = p.add_argument_group("numerics")
    n.add_argument("--scheme", choices=["euler", "rk2"], default="rk2")
    n.add_argument("--cfl", type=float, default=0.2)
    n.add_argument("--k1-coef", type=float, default=0.0125, dest="k1_coef")
    n.add_argument("--k2-coef", type=float, default=0.125, dest="k2_coef")
    n.add_argument("--k3-coef", type=float, default=0.0125, dest="k3_coef")

    r = p.add_argument_group("run control")
    r.add_argument(
        "--steps",
        type=int,
        default=None,
        help="fixed step count (preferred for benchmarking)",
    )
    r.add_argument(
        "--t-end",
        type=float,
        default=None,
        dest="t_end",
        help="physical end time (preferred for 'real' runs)",
    )
    r.add_argument(
        "--save-interval",
        type=int,
        default=0,
        dest="save_interval",
        help="steps between frame dumps; 0 disables",
    )
    r.add_argument("--output-dir", type=str, default="results/run", dest="output_dir")
    r.add_argument(
        "--no-output",
        action="store_true",
        dest="no_output",
        help="skip frame I/O entirely (pure compute benchmark)",
    )
    r.add_argument(
        "--overwrite", action="store_true", help="wipe output-dir if it already exists"
    )

    b = p.add_argument_group("benchmark")
    b.add_argument("--warmup-steps", type=int, default=0, dest="warmup_steps")
    b.add_argument("--repeats", type=int, default=1)


def config_from_args(args: argparse.Namespace) -> Config:
    fields = {f.name for f in Config.__dataclass_fields__.values()}
    return Config(**{k: v for k, v in vars(args).items() if k in fields})


def validate(cfg: Config) -> None:
    if cfg.steps is None and cfg.t_end is None:
        raise SystemExit("error: must specify either --steps or --t-end")
    if cfg.scheme not in ("euler", "rk2"):
        raise SystemExit(f"error: unknown scheme {cfg.scheme!r}")


def dump_config(cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2))
