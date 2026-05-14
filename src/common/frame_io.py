"""Binary frame format shared by all runners.

Layout (little-endian):
    int64   step
    float64 t
    int64   Nx
    int64   Ny
    float64[Nx*Ny]  r       (density)
    float64[Nx*Ny]  ru      (x-momentum)
    float64[Nx*Ny]  rv      (y-momentum)
    float64[Nx*Ny]  e       (total energy)

Arrays are row-major (i, j).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HEADER_FMT = "<qdqq"  # step:int64, t:float64, Nx:int64, Ny:int64
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def write_frame(path: Path, step: int, t: float, r, ru, rv, e) -> None:
    # Used only by the Python runner; the C and CUDA runners have their own
    # write_frame that produces byte-identical output.
    Nx, Ny = r.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack(HEADER_FMT, int(step), float(t), Nx, Ny))
        for arr in (r, ru, rv, e):
            np.ascontiguousarray(arr, dtype=np.float64).tofile(fh)


def read_frame(path: Path):
    with path.open("rb") as fh:
        step, t, Nx, Ny = struct.unpack(HEADER_FMT, fh.read(HEADER_SIZE))
        arrs = []
        for _ in range(4):
            arrs.append(
                np.fromfile(fh, dtype=np.float64, count=Nx * Ny).reshape(Nx, Ny)
            )
    r, ru, rv, e = arrs
    return {
        "step": step,
        "t": t,
        "Nx": Nx,
        "Ny": Ny,
        "r": r,
        "ru": ru,
        "rv": rv,
        "e": e,
    }
