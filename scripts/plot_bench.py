"""Plot benchmark results from one or more bench CSVs.

Each CSV row is one (impl, grid, scheme) measurement produced by scripts/bench.py.
Passing multiple CSVs (e.g. a CPU run and a GPU run from different machines)
merges the rows; the `impl` column keeps them apart.

A row is keyed by (impl, Nx, Ny, scheme). A repeated key across the loaded
CSVs is a data error and aborts: the plots assume exactly one measurement per
key. All rows must share a single `scheme`, since it changes the work per step.

This script does NOT enforce that --steps, --seed, --warmup-steps or --repeats
match across the rows it merges. The reported metric is throughput
(cell-updates / s = Nx*Ny*steps / wall), normalized per cell per step, so the
step count divides out; --seed only picks the initial perturbation field, not
the amount of compute; and warmup/repeats only affect how the timer is sampled,
not the work. So rows that differ in those are still validly comparable here.
It is still good practice to keep them consistent: it makes a sweep
reproducible, keeps the raw (non-normalized) wall times comparable, and removes
any doubt that every impl solved the same problem -- but that is on you when you
choose what to run, not a precondition for plotting. `scheme` is the one
parameter that genuinely changes the work (rk2 does ~2x the flux evals of
euler), so it is enforced.

Error bars come from the per-repeat wall times in `wall_s_all`: each repeat is
turned into a throughput, and the bar spans min..max of those.

Usage:
    python scripts/plot_bench.py --csv results/bench_local.csv results/bench_cuda.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

IMPL_ORDER = ["python", "c", "cuda"]
IMPL_COLOR = {"python": "#3776ab", "c": "#a8b9cc", "cuda": "#76b900"}


def load_rows(csv_paths: list[Path]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for path in csv_paths:
        if not path.exists():
            raise SystemExit(f"no such CSV: {path}")
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                key = (r["impl"], int(r["Nx"]), int(r["Ny"]), r["scheme"])
                if key in rows:
                    raise SystemExit(f"duplicate measurement for {key} across CSVs")
                rows[key] = r
    schemes = {r["scheme"] for r in rows.values()}
    if len(schemes) > 1:
        raise SystemExit(f"rows mix schemes {sorted(schemes)}; plot one scheme at a time")
    return list(rows.values())


def cells(row: dict) -> int:
    return int(row["Nx"]) * int(row["Ny"])


def grid_label(row: dict) -> str:
    return f"{row['Nx']}x{row['Ny']}"


def throughputs(row: dict) -> list[float]:
    """Per-repeat throughput in M cell-updates/s, derived from wall_s_all."""
    work = cells(row) * int(row["timed_steps"])
    walls = json.loads(row["wall_s_all"])
    return [work / w / 1e6 for w in walls]


def throughput_median(row: dict) -> float:
    return float(row["cell_updates_per_s"]) / 1e6


def series_by_impl(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["impl"], []).append(r)
    for impl in out:
        out[impl].sort(key=cells)
    return out


def plot_throughput(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    by_impl = series_by_impl(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in IMPL_ORDER:
        if impl not in by_impl:
            continue
        s = by_impl[impl]
        x = [cells(r) for r in s]
        y = [throughput_median(r) for r in s]
        lo = [y[i] - min(throughputs(r)) for i, r in enumerate(s)]
        hi = [max(throughputs(r)) - y[i] for i, r in enumerate(s)]
        ax.errorbar(
            x, y, yerr=[lo, hi], fmt="o-", capsize=3,
            label=impl, color=IMPL_COLOR.get(impl),
        )
    ax.set_xscale("log")
    ax.set_xlabel("grid size (cells)")
    ax.set_ylabel("throughput (M cell-updates / s)")
    ax.set_title("Throughput vs grid size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_speedup(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    by_impl = series_by_impl(rows)
    if "python" not in by_impl:
        print("no python rows: skipping speedup plot")
        return

    baseline = {grid_label(r): throughput_median(r) for r in by_impl["python"]}
    grids = [grid_label(r) for r in by_impl["python"]]

    impls = [i for i in IMPL_ORDER if i in by_impl]
    width = 0.8 / len(impls)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, impl in enumerate(impls):
        rows_by_grid = {grid_label(r): r for r in by_impl[impl]}
        xs, ys, lo, hi = [], [], [], []
        for gi, g in enumerate(grids):
            if g not in rows_by_grid:
                continue
            r = rows_by_grid[g]
            med = throughput_median(r) / baseline[g]
            tps = [t / baseline[g] for t in throughputs(r)]
            xs.append(gi + k * width)
            ys.append(med)
            lo.append(med - min(tps))
            hi.append(max(tps) - med)
        bars = ax.bar(
            xs, ys, width, yerr=[lo, hi], capsize=3,
            label=impl, color=IMPL_COLOR.get(impl),
        )
        ax.bar_label(bars, fmt="%.1fx", padding=3, fontsize=8)
    ax.set_xticks([i + width * (len(impls) - 1) / 2 for i in range(len(grids))])
    ax.set_xticklabels(grids)
    ax.set_xlabel("grid")
    ax.set_ylabel("speedup vs Python")
    ax.set_title("Speedup relative to Python baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        help="one or more bench CSV files",
    )
    p.add_argument("--out-dir", type=Path, default=REPO / "assets")
    args = p.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("no rows loaded")

    plot_throughput(rows, args.out_dir / "bench_throughput.png")
    plot_speedup(rows, args.out_dir / "bench_speedup.png")


if __name__ == "__main__":
    main()
