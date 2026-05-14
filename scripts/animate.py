"""Assemble all frame_*.bin in a results dir into an mp4 of the density field.

Usage:
    python scripts/animate.py <results_dir> [--out <out.mp4>] [--fps 15] [--cmap viridis]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.frame_io import read_frame  # noqa: E402


def animate(results_dir: Path, out_path: Path, fps: int, cmap: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    frames = sorted(results_dir.glob("frame_*.bin"))
    if not frames:
        raise SystemExit(f"no frame_*.bin found in {results_dir}")

    f0 = read_frame(frames[0])
    vmin, vmax = f0["r"].min(), f0["r"].max()

    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(f0["r"].T, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="density")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2000)
    with writer.saving(fig, str(out_path), dpi=120):
        for fp in frames:
            f = read_frame(fp)
            im.set_data(f["r"].T)
            title.set_text(f"step {f['step']}  t={f['t']:.4f}")
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out_path}  ({len(frames)} frames @ {fps} fps)")


def main() -> None:
    p = argparse.ArgumentParser(description="Animate a results directory into an mp4.")
    p.add_argument("results_dir", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="output mp4 (default: <results_dir>/density.mp4)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--cmap", default="viridis")
    args = p.parse_args()
    out = args.out if args.out is not None else args.results_dir / "density.mp4"
    animate(args.results_dir, out, args.fps, args.cmap)


if __name__ == "__main__":
    main()
