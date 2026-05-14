"""Plot the density field of a single frame .bin as a PNG.

Usage:
    python scripts/plot_frame.py <frame.bin> [--out <path.png>] [--cmap viridis]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.frame_io import read_frame  # noqa: E402


def plot(frame_path: Path, out_path: Path, cmap: str) -> None:
    import matplotlib.pyplot as plt

    f = read_frame(frame_path)
    fig, ax = plt.subplots(figsize=(4, 8))
    # arrays are (Nx, Ny); transpose for the conventional "y up" image
    im = ax.imshow(f["r"].T, origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(f"step {f['step']}  t={f['t']:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="density")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot density field of one frame to a PNG.")
    p.add_argument("frame", type=Path, help="path to frame_NNNNNN.bin")
    p.add_argument("--out", type=Path, default=None, help="output .png (default: <frame>.png)")
    p.add_argument("--cmap", default="viridis")
    args = p.parse_args()
    out = args.out if args.out is not None else args.frame.with_suffix(".png")
    plot(args.frame, out, args.cmap)


if __name__ == "__main__":
    main()
