"""Python reference implementation of the Rayleigh-Taylor instability.

Solves the 2D compressible Euler equations with artificial diffusion on a
uniform grid, using explicit Euler or RK2 time integration with periodic-x
and wall-y boundary conditions. CLI flags are defined in src/common/cli_spec.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cli_spec import (  # noqa: E402
    Config,
    add_shared_args,
    config_from_args,
    dump_config,
    validate,
)
from common.frame_io import write_frame  # noqa: E402


def initialize_grid(cfg: Config, rng: np.random.Generator):
    Nx, Ny = cfg.Nx, cfg.Ny
    dy = cfg.Ly / Ny

    r = np.zeros((Nx, Ny))
    j = np.arange(Ny)
    y = (j + 0.5) * dy  # shape (Ny,)
    heavy = y >= cfg.Ly / 2
    r[:, heavy] = cfg.rho_heavy
    r[:, ~heavy] = cfg.rho_light

    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))
    band = np.abs(y - cfg.Ly / 2) <= 0.05
    v[:, band] = cfg.perturbation_amp * (2 * rng.random((Nx, band.sum())) - 1)

    p = cfg.p0 + r * cfg.g * (y - cfg.Ly / 2)
    ru = r * u
    rv = r * v
    e = p / (cfg.gamma - 1) + 0.5 * r * (u * u + v * v)
    return r, ru, rv, e, p


def update_diffusion_coeffs(cfg: Config, dx: float, dt: float):
    s = (dx * dx) / (2 * dt)
    return cfg.k1_coef * s, cfg.k2_coef * s, cfg.k3_coef * s


def compute_dt(cfg: Config, r, ru, rv, p, dx, dy):
    nz = r > 1e-10
    u = np.zeros_like(r)
    v = np.zeros_like(r)
    u[nz] = ru[nz] / r[nz]
    v[nz] = rv[nz] / r[nz]
    c = np.sqrt(cfg.gamma * p / r)
    max_speed = np.max(np.maximum(np.maximum(np.abs(u), np.abs(v)), c))
    return cfg.cfl * min(dx, dy) / max_speed


def compute_derivatives(f, dx, dy):
    df_dx = np.zeros_like(f)
    df_dy = np.zeros_like(f)
    d2f_dx2 = np.zeros_like(f)
    d2f_dy2 = np.zeros_like(f)
    df_dx[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)
    df_dy[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dy)
    d2f_dx2[1:-1, 1:-1] = (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dx * dx)
    d2f_dy2[1:-1, 1:-1] = (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dy * dy)
    return df_dx, df_dy, d2f_dx2, d2f_dy2


def compute_rhs(cfg: Config, r, ru, rv, e, p, k1, k2, k3, dx, dy):
    nz = r > 1e-10
    u = np.zeros_like(r)
    v = np.zeros_like(r)
    u[nz] = ru[nz] / r[nz]
    v[nz] = rv[nz] / r[nz]

    dr_dx, dr_dy, d2r_dx2, d2r_dy2 = compute_derivatives(r, dx, dy)
    dru_dx, dru_dy, d2ru_dx2, _ = compute_derivatives(ru, dx, dy)
    drv_dx, drv_dy, _, d2rv_dy2 = compute_derivatives(rv, dx, dy)
    _, _, d2e_dx2, d2e_dy2 = compute_derivatives(e, dx, dy)
    dp_dx, dp_dy, _, _ = compute_derivatives(p, dx, dy)

    ruu = ru * u
    ruv = ru * v
    rvv = rv * v
    eu_p = u * (e + p)
    ev_p = v * (e + p)

    druu_dx, _, _, _ = compute_derivatives(ruu, dx, dy)
    druv_dx, druv_dy, _, _ = compute_derivatives(ruv, dx, dy)
    _, drvv_dy, _, _ = compute_derivatives(rvv, dx, dy)
    deup_dx, _, _, _ = compute_derivatives(eu_p, dx, dy)
    _, devp_dy, _, _ = compute_derivatives(ev_p, dx, dy)

    rhs_r = -(dru_dx + drv_dy) + k1 * (d2r_dx2 + d2r_dy2)
    rhs_ru = -(druu_dx + druv_dy) - dp_dx + k2 * d2ru_dx2
    rhs_rv = -(druv_dx + drvv_dy) - dp_dy + cfg.g * r + k2 * d2rv_dy2
    rhs_e = -(deup_dx + devp_dy) + cfg.g * r * v + k3 * (d2e_dx2 + d2e_dy2)
    # zero out boundaries (derivatives were only computed on interior)
    for a in (rhs_r, rhs_ru, rhs_rv, rhs_e):
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 0
    return rhs_r, rhs_ru, rhs_rv, rhs_e


def apply_bc(cfg: Config, r, ru, rv, e, p):
    Nx, Ny = cfg.Nx, cfg.Ny
    for f in (r, ru, rv, e, p):
        f[0, :] = f[Nx - 2, :]
        f[Nx - 1, :] = f[1, :]
    rv[:, 0] = 0
    rv[:, Ny - 1] = 0
    r[:, 0] = r[:, 1]
    ru[:, 0] = ru[:, 1]
    p[:, 0] = p[:, 1]
    e[:, 0] = (
        p[:, 0] / (cfg.gamma - 1) + 0.5 * (ru[:, 0] ** 2 + rv[:, 0] ** 2) / r[:, 0]
    )
    r[:, Ny - 1] = r[:, Ny - 2]
    ru[:, Ny - 1] = ru[:, Ny - 2]
    p[:, Ny - 1] = p[:, Ny - 2]
    e[:, Ny - 1] = (
        p[:, Ny - 1] / (cfg.gamma - 1)
        + 0.5 * (ru[:, Ny - 1] ** 2 + rv[:, Ny - 1] ** 2) / r[:, Ny - 1]
    )
    return r, ru, rv, e, p


def update_pressure(cfg: Config, r, ru, rv, e):
    nz = r > 1e-10
    u = np.zeros_like(r)
    v = np.zeros_like(r)
    u[nz] = ru[nz] / r[nz]
    v[nz] = rv[nz] / r[nz]
    p = (cfg.gamma - 1) * (e - 0.5 * r * (u * u + v * v))
    return np.maximum(p, 1e-10)


def step(cfg, r, ru, rv, e, p, dt, dx, dy):
    k1, k2, k3 = update_diffusion_coeffs(cfg, dx, dt)
    if cfg.scheme == "euler":
        rh_r, rh_ru, rh_rv, rh_e = compute_rhs(cfg, r, ru, rv, e, p, k1, k2, k3, dx, dy)
        r2 = r + dt * rh_r
        ru2 = ru + dt * rh_ru
        rv2 = rv + dt * rh_rv
        e2 = e + dt * rh_e
    else:  # rk2
        rh_r, rh_ru, rh_rv, rh_e = compute_rhs(cfg, r, ru, rv, e, p, k1, k2, k3, dx, dy)
        rs = r + dt * rh_r
        rus = ru + dt * rh_ru
        rvs = rv + dt * rh_rv
        es = e + dt * rh_e
        ps = update_pressure(cfg, rs, rus, rvs, es)
        rs, rus, rvs, es, ps = apply_bc(cfg, rs, rus, rvs, es, ps)
        rh_r2, rh_ru2, rh_rv2, rh_e2 = compute_rhs(
            cfg, rs, rus, rvs, es, ps, k1, k2, k3, dx, dy
        )
        r2 = 0.5 * r + 0.5 * (rs + dt * rh_r2)
        ru2 = 0.5 * ru + 0.5 * (rus + dt * rh_ru2)
        rv2 = 0.5 * rv + 0.5 * (rvs + dt * rh_rv2)
        e2 = 0.5 * e + 0.5 * (es + dt * rh_e2)
    p2 = update_pressure(cfg, r2, ru2, rv2, e2)
    return apply_bc(cfg, r2, ru2, rv2, e2, p2)


def prepare_output_dir(cfg: Config) -> Path:
    out_dir = Path(cfg.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not cfg.overwrite:
            raise SystemExit(
                f"error: output dir {out_dir} exists and is non-empty; "
                "pass --overwrite to wipe it, or choose a different --output-dir"
            )
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run(cfg: Config) -> None:
    validate(cfg)
    rng = np.random.default_rng(cfg.seed)
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny

    if not cfg.no_output:
        out_dir = prepare_output_dir(cfg)
        dump_config(cfg, out_dir / "config.json")
    else:
        out_dir = None

    r, ru, rv, e, p = initialize_grid(cfg, rng)
    r, ru, rv, e, p = apply_bc(cfg, r, ru, rv, e, p)

    t = 0.0
    step_idx = 0
    if cfg.save_interval and out_dir is not None:
        write_frame(out_dir / f"frame_{step_idx:06d}.bin", step_idx, t, r, ru, rv, e)

    wall_start = time.perf_counter()
    while True:
        if cfg.steps is not None and step_idx >= cfg.steps:
            break
        if cfg.t_end is not None and t >= cfg.t_end:
            break

        dt = compute_dt(cfg, r, ru, rv, p, dx, dy)
        if cfg.t_end is not None and t + dt > cfg.t_end:
            dt = cfg.t_end - t

        r, ru, rv, e, p = step(cfg, r, ru, rv, e, p, dt, dx, dy)
        t += dt
        step_idx += 1

        if (
            cfg.save_interval
            and out_dir is not None
            and step_idx % cfg.save_interval == 0
        ):
            write_frame(
                out_dir / f"frame_{step_idx:06d}.bin", step_idx, t, r, ru, rv, e
            )
            print(f"step {step_idx}  t={t:.4f}  dt={dt:.4e}", flush=True)

    wall = time.perf_counter() - wall_start
    print(
        f"done: {step_idx} steps in {wall:.2f}s "
        f"({step_idx / wall:.1f} steps/s, "
        f"{cfg.Nx * cfg.Ny * step_idx / wall:.2e} cell-updates/s)"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulate the Rayleigh-Taylor instability on a 2D uniform grid."
    )
    add_shared_args(p)
    args = p.parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
