/* CUDA/GPU implementation of the Rayleigh-Taylor instability.
 *
 * Solves the 2D compressible Euler equations with artificial diffusion on a
 * uniform grid, using explicit Euler or RK2 time integration with periodic-x
 * and wall-y boundary conditions. CLI flags match src/common/cli_spec.py
 * (kept in sync by hand).
 */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define cudaCheck(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                cudaGetErrorString(_e)); exit(1); } } while (0)

#define cudaCheckLast() do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); exit(1); } } while (0)

/* ---------- config ---------- */
typedef struct {
    int Nx, Ny;
    double Lx, Ly;
    double gamma, g;
    double rho_heavy, rho_light, p0, perturbation_amp;
    int seed;
    char scheme[8];
    double cfl;
    double k1_coef, k2_coef, k3_coef;
    long steps;
    double t_end;
    int save_interval;
    char output_dir[1024];
    int no_output, overwrite;
    int warmup_steps, repeats;
    int block_x, block_y;
} Config;

static void cfg_defaults(Config *c) {
    c->Nx = 512; c->Ny = 1024;
    c->Lx = 1.0; c->Ly = 2.0;
    c->gamma = 1.4; c->g = -10.0;
    c->rho_heavy = 2.0; c->rho_light = 1.0;
    c->p0 = 40.0; c->perturbation_amp = 1e-3;
    c->seed = 0;
    strcpy(c->scheme, "rk2");
    c->cfl = 0.2;
    c->k1_coef = 0.0125; c->k2_coef = 0.125; c->k3_coef = 0.0125;
    c->steps = -1;
    c->t_end = NAN;
    c->save_interval = 0;
    strcpy(c->output_dir, "results/run");
    c->no_output = 0; c->overwrite = 0;
    c->warmup_steps = 10; c->repeats = 1;
    c->block_x = 16; c->block_y = 16;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  physics:    --Nx --Ny --Lx --Ly --gamma --g --rho-heavy --rho-light\n"
        "              --p0 --perturbation-amp --seed\n"
        "  numerics:   --scheme {euler|rk2} --cfl --k1-coef --k2-coef --k3-coef\n"
        "  run:        --steps --t-end --save-interval --output-dir\n"
        "              --no-output --overwrite\n"
        "  bench:      --warmup-steps --repeats\n"
        "  cuda:       --block-x --block-y\n", prog);
}

static int parse_args(int argc, char **argv, Config *c) {
    enum {
        O_NX = 1000, O_NY, O_LX, O_LY, O_GAMMA, O_G, O_RHOH, O_RHOL, O_P0,
        O_PERT, O_SEED, O_SCHEME, O_CFL, O_K1, O_K2, O_K3,
        O_STEPS, O_TEND, O_SAVE, O_OUT, O_NOOUT, O_OVER, O_WARM, O_REP,
        O_BX, O_BY
    };
    static const struct option opts[] = {
        {"Nx", required_argument, 0, O_NX},
        {"Ny", required_argument, 0, O_NY},
        {"Lx", required_argument, 0, O_LX},
        {"Ly", required_argument, 0, O_LY},
        {"gamma", required_argument, 0, O_GAMMA},
        {"g", required_argument, 0, O_G},
        {"rho-heavy", required_argument, 0, O_RHOH},
        {"rho-light", required_argument, 0, O_RHOL},
        {"p0", required_argument, 0, O_P0},
        {"perturbation-amp", required_argument, 0, O_PERT},
        {"seed", required_argument, 0, O_SEED},
        {"scheme", required_argument, 0, O_SCHEME},
        {"cfl", required_argument, 0, O_CFL},
        {"k1-coef", required_argument, 0, O_K1},
        {"k2-coef", required_argument, 0, O_K2},
        {"k3-coef", required_argument, 0, O_K3},
        {"steps", required_argument, 0, O_STEPS},
        {"t-end", required_argument, 0, O_TEND},
        {"save-interval", required_argument, 0, O_SAVE},
        {"output-dir", required_argument, 0, O_OUT},
        {"no-output", no_argument, 0, O_NOOUT},
        {"overwrite", no_argument, 0, O_OVER},
        {"warmup-steps", required_argument, 0, O_WARM},
        {"repeats", required_argument, 0, O_REP},
        {"block-x", required_argument, 0, O_BX},
        {"block-y", required_argument, 0, O_BY},
        {"help", no_argument, 0, 'h'},
        {0,0,0,0}
    };
    int idx, opt;
    while ((opt = getopt_long(argc, argv, "h", opts, &idx)) != -1) {
        switch (opt) {
        case O_NX: c->Nx = atoi(optarg); break;
        case O_NY: c->Ny = atoi(optarg); break;
        case O_LX: c->Lx = atof(optarg); break;
        case O_LY: c->Ly = atof(optarg); break;
        case O_GAMMA: c->gamma = atof(optarg); break;
        case O_G: c->g = atof(optarg); break;
        case O_RHOH: c->rho_heavy = atof(optarg); break;
        case O_RHOL: c->rho_light = atof(optarg); break;
        case O_P0: c->p0 = atof(optarg); break;
        case O_PERT: c->perturbation_amp = atof(optarg); break;
        case O_SEED: c->seed = atoi(optarg); break;
        case O_SCHEME:
            if (strcmp(optarg, "euler") && strcmp(optarg, "rk2")) {
                fprintf(stderr, "error: --scheme must be euler or rk2\n"); return -1;
            }
            strncpy(c->scheme, optarg, sizeof(c->scheme) - 1); break;
        case O_CFL: c->cfl = atof(optarg); break;
        case O_K1: c->k1_coef = atof(optarg); break;
        case O_K2: c->k2_coef = atof(optarg); break;
        case O_K3: c->k3_coef = atof(optarg); break;
        case O_STEPS: c->steps = atol(optarg); break;
        case O_TEND: c->t_end = atof(optarg); break;
        case O_SAVE: c->save_interval = atoi(optarg); break;
        case O_OUT: strncpy(c->output_dir, optarg, sizeof(c->output_dir) - 1); break;
        case O_NOOUT: c->no_output = 1; break;
        case O_OVER: c->overwrite = 1; break;
        case O_WARM: c->warmup_steps = atoi(optarg); break;
        case O_REP: c->repeats = atoi(optarg); break;
        case O_BX: c->block_x = atoi(optarg); break;
        case O_BY: c->block_y = atoi(optarg); break;
        case 'h': usage(argv[0]); exit(0);
        default: usage(argv[0]); return -1;
        }
    }
    if (c->steps < 0 && isnan(c->t_end)) {
        fprintf(stderr, "error: must specify --steps or --t-end\n");
        return -1;
    }
    int bs = c->block_x * c->block_y;
    if (bs == 0 || (bs & (bs - 1)) != 0) {
        fprintf(stderr, "error: block_x*block_y must be a power of 2 "
                "(reduction kernel requires it); got %d*%d=%d\n",
                c->block_x, c->block_y, bs);
        return -1;
    }
    return 0;
}

/* ---------- kernels: device layout r[j*Nx + i], j outer, i contiguous ---------- */

__global__ void initialize_grid_kernel(double *r, double *ru, double *rv, double *e, double *p,
                                       int Nx, int Ny, double Ly, double dy,
                                       double rho_h, double rho_l, double p0, double g,
                                       double pert_amp, double gam, unsigned int user_seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    double y = (j + 0.5) * dy;
    double rho = (y >= Ly / 2) ? rho_h : rho_l;
    double v_pert = 0.0;
    if (fabs(y - Ly / 2) <= 0.05) {
        unsigned int s = i * 1723u + j * 93241u + user_seed * 2654435761u;
        s = (s * 196314165u) + 907633515u;
        double rv01 = (s % 10000u) / 5000.0 - 1.0;
        v_pert = pert_amp * rv01;
    }
    double u = 0.0, v = v_pert;
    size_t k = (size_t)j * Nx + i;
    r[k] = rho;
    p[k] = p0 + rho * g * (y - Ly / 2);
    ru[k] = rho * u;
    rv[k] = rho * v;
    e[k] = p[k] / (gam - 1) + 0.5 * rho * (u*u + v*v);
}

__global__ void compute_dt_kernel(const double *r, const double *ru, const double *rv,
                                  const double *p, double *block_max,
                                  int Nx, int Ny, double gam) {
    extern __shared__ double sm[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sm[tid] = 0.0;
    if (i < Nx && j < Ny) {
        size_t k = (size_t)j * Nx + i;
        double rho = r[k];
        double u = 0.0, v = 0.0;
        if (rho > 1e-10) { u = ru[k]/rho; v = rv[k]/rho; }
        double cs = sqrt(gam * p[k] / rho);
        sm[tid] = fmax(fmax(fabs(u), fabs(v)), cs);
    }
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] = fmax(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    if (tid == 0) block_max[blockIdx.y * gridDim.x + blockIdx.x] = sm[0];
}

__global__ void compute_velocities_kernel(const double *r, const double *ru, const double *rv,
                                          double *u, double *v, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    size_t k = (size_t)j * Nx + i;
    if (r[k] > 1e-10) { u[k] = ru[k]/r[k]; v[k] = rv[k]/r[k]; }
    else { u[k] = 0.0; v[k] = 0.0; }
}

__global__ void compute_fluxes_kernel(const double *ru, const double *rv,
                                      const double *e, const double *p,
                                      const double *u, const double *v,
                                      double *ruu, double *ruv, double *rvv,
                                      double *eu_p, double *ev_p, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    size_t k = (size_t)j * Nx + i;
    ruu[k] = ru[k] * u[k];
    ruv[k] = ru[k] * v[k];
    rvv[k] = rv[k] * v[k];
    eu_p[k] = u[k] * (e[k] + p[k]);
    ev_p[k] = v[k] * (e[k] + p[k]);
}

__global__ void compute_derivatives_kernel(const double *f, double *dfx, double *dfy,
                                           double *d2x, double *d2y,
                                           int Nx, int Ny, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= Nx - 1 || j >= Ny - 1) return;
    size_t k = (size_t)j * Nx + i;
    dfx[k] = (f[k + 1] - f[k - 1]) / (2 * dx);
    dfy[k] = (f[k + Nx] - f[k - Nx]) / (2 * dy);
    d2x[k] = (f[k + 1] - 2 * f[k] + f[k - 1]) / (dx * dx);
    d2y[k] = (f[k + Nx] - 2 * f[k] + f[k - Nx]) / (dy * dy);
}

__global__ void compute_rhs_kernel(const double *r, const double *v,
                                   const double *dru_dx, const double *drv_dy,
                                   const double *d2r_dx2, const double *d2r_dy2,
                                   const double *druu_dx, const double *druv_dy,
                                   const double *dp_dx, const double *d2ru_dx2,
                                   const double *druv_dx, const double *drvv_dy,
                                   const double *dp_dy, const double *d2rv_dy2,
                                   const double *deup_dx, const double *devp_dy,
                                   const double *d2e_dx2, const double *d2e_dy2,
                                   double *rhs_r, double *rhs_ru, double *rhs_rv, double *rhs_e,
                                   int Nx, int Ny, double g, double k1, double k2, double k3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= Nx - 1 || j >= Ny - 1) return;
    size_t k = (size_t)j * Nx + i;
    rhs_r[k]  = -(dru_dx[k] + drv_dy[k]) + k1 * (d2r_dx2[k] + d2r_dy2[k]);
    rhs_ru[k] = -(druu_dx[k] + druv_dy[k]) - dp_dx[k] + k2 * d2ru_dx2[k];
    rhs_rv[k] = -(druv_dx[k] + drvv_dy[k]) - dp_dy[k] + g * r[k] + k2 * d2rv_dy2[k];
    rhs_e[k]  = -(deup_dx[k] + devp_dy[k]) + g * r[k] * v[k] + k3 * (d2e_dx2[k] + d2e_dy2[k]);
}

__global__ void apply_bc_kernel(double *r, double *ru, double *rv, double *e, double *p,
                                int Nx, int Ny, double gam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    /* x boundaries (periodic) */
    if (j < Ny && (i == 0 || i == Nx - 1)) {
        size_t k = (size_t)j * Nx + i;
        size_t src = (i == 0) ? (size_t)j * Nx + (Nx - 2) : (size_t)j * Nx + 1;
        r[k]  = r[src];
        ru[k] = ru[src];
        rv[k] = rv[src];
        e[k]  = e[src];
        p[k]  = p[src];
    }
    /* y boundaries (wall) */
    if (i < Nx && (j == 0 || j == Ny - 1)) {
        size_t k = (size_t)j * Nx + i;
        size_t src = (j == 0) ? (size_t)1 * Nx + i : (size_t)(Ny - 2) * Nx + i;
        rv[k] = 0.0;
        r[k]  = r[src];
        ru[k] = ru[src];
        p[k]  = p[src];
        double u = 0.0, v = 0.0;
        if (r[k] > 1e-10) { u = ru[k]/r[k]; v = rv[k]/r[k]; }
        e[k] = p[k] / (gam - 1) + 0.5 * r[k] * (u*u + v*v);
    }
}

__global__ void update_pressure_kernel(const double *r, const double *ru, const double *rv,
                                       const double *e, double *p, int Nx, int Ny, double gam) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    size_t k = (size_t)j * Nx + i;
    double rho = r[k];
    double u = 0.0, v = 0.0;
    if (rho > 1e-10) { u = ru[k]/rho; v = rv[k]/rho; }
    double pv = (gam - 1) * (e[k] - 0.5 * rho * (u*u + v*v));
    p[k] = pv < 1e-10 ? 1e-10 : pv;
}

__global__ void axpy_kernel(double *out, const double *in, const double *rhs, double dt,
                            int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    size_t k = (size_t)j * Nx + i;
    out[k] = in[k] + dt * rhs[k];
}

__global__ void rk2_combine_kernel(double *out, const double *in, const double *star,
                                   const double *rhs_star, double dt, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    size_t k = (size_t)j * Nx + i;
    out[k] = 0.5 * in[k] + 0.5 * (star[k] + dt * rhs_star[k]);
}

/* ---------- host state ---------- */
typedef struct {
    /* primary */
    double *r, *ru, *rv, *e, *p;
    /* rhs scratch (persistent) */
    double *u, *v, *ruu, *ruv, *rvv, *eu_p, *ev_p;
    double *dr_dx, *dr_dy, *d2r_dx2, *d2r_dy2;
    double *dru_dx, *dru_dy, *d2ru_dx2, *d2ru_dy2;
    double *drv_dx, *drv_dy, *d2rv_dx2, *d2rv_dy2;
    double *de_dx, *de_dy, *d2e_dx2, *d2e_dy2;
    double *dp_dx, *dp_dy, *dummy1, *dummy2;
    double *druu_dx, *druv_dx, *druv_dy, *drvv_dy, *deup_dx, *devp_dy;
    double *rhs_r, *rhs_ru, *rhs_rv, *rhs_e;
    /* rk2 stages */
    double *r_s, *ru_s, *rv_s, *e_s, *p_s;
    double *rhs_r_s, *rhs_ru_s, *rhs_rv_s, *rhs_e_s;
    /* dt reduction buffer */
    double *block_max;
    int block_max_n;
} DState;

static void dstate_alloc(DState *s, const Config *c) {
    size_t n = (size_t)c->Nx * c->Ny;
    size_t sz = n * sizeof(double);
    double **ptrs[] = {
        &s->r, &s->ru, &s->rv, &s->e, &s->p,
        &s->u, &s->v, &s->ruu, &s->ruv, &s->rvv, &s->eu_p, &s->ev_p,
        &s->dr_dx, &s->dr_dy, &s->d2r_dx2, &s->d2r_dy2,
        &s->dru_dx, &s->dru_dy, &s->d2ru_dx2, &s->d2ru_dy2,
        &s->drv_dx, &s->drv_dy, &s->d2rv_dx2, &s->d2rv_dy2,
        &s->de_dx, &s->de_dy, &s->d2e_dx2, &s->d2e_dy2,
        &s->dp_dx, &s->dp_dy, &s->dummy1, &s->dummy2,
        &s->druu_dx, &s->druv_dx, &s->druv_dy, &s->drvv_dy,
        &s->deup_dx, &s->devp_dy,
        &s->rhs_r, &s->rhs_ru, &s->rhs_rv, &s->rhs_e,
        &s->r_s, &s->ru_s, &s->rv_s, &s->e_s, &s->p_s,
        &s->rhs_r_s, &s->rhs_ru_s, &s->rhs_rv_s, &s->rhs_e_s
    };
    for (size_t k = 0; k < sizeof(ptrs)/sizeof(ptrs[0]); k++) {
        cudaCheck(cudaMalloc(ptrs[k], sz));
        cudaCheck(cudaMemset(*ptrs[k], 0, sz));
    }
    int gx = (c->Nx + c->block_x - 1) / c->block_x;
    int gy = (c->Ny + c->block_y - 1) / c->block_y;
    s->block_max_n = gx * gy;
    cudaCheck(cudaMalloc(&s->block_max, s->block_max_n * sizeof(double)));
}

/* ---------- host driver ---------- */
static dim3 grid_full(const Config *c) {
    return dim3((c->Nx + c->block_x - 1) / c->block_x,
                (c->Ny + c->block_y - 1) / c->block_y);
}
static dim3 grid_interior(const Config *c) {
    return dim3((c->Nx - 2 + c->block_x - 1) / c->block_x,
                (c->Ny - 2 + c->block_y - 1) / c->block_y);
}
static dim3 block_dim(const Config *c) { return dim3(c->block_x, c->block_y); }

static double host_compute_dt(const Config *c, DState *s) {
    dim3 bd = block_dim(c), gd = grid_full(c);
    size_t shm = (size_t)c->block_x * c->block_y * sizeof(double);
    double dx = c->Lx / c->Nx, dy = c->Ly / c->Ny;
    compute_dt_kernel<<<gd, bd, shm>>>(s->r, s->ru, s->rv, s->p, s->block_max,
                                        c->Nx, c->Ny, c->gamma);
    cudaCheckLast();
    double *h = (double*)malloc(s->block_max_n * sizeof(double));
    cudaCheck(cudaMemcpy(h, s->block_max, s->block_max_n * sizeof(double),
                         cudaMemcpyDeviceToHost));
    double max_speed = 0.0;
    for (int k = 0; k < s->block_max_n; k++) if (h[k] > max_speed) max_speed = h[k];
    free(h);
    double safe = max_speed > 1e-10 ? max_speed : 1e-10;
    return c->cfl * fmin(dx, dy) / safe;
}

static void compute_rhs_host(const Config *c, DState *s,
                             double *r, double *ru, double *rv, double *e, double *p,
                             double k1, double k2, double k3,
                             double *rhs_r, double *rhs_ru, double *rhs_rv, double *rhs_e) {
    dim3 bd = block_dim(c), gd = grid_full(c), gi = grid_interior(c);
    double dx = c->Lx / c->Nx, dy = c->Ly / c->Ny;
    compute_velocities_kernel<<<gd, bd>>>(r, ru, rv, s->u, s->v, c->Nx, c->Ny);
    compute_fluxes_kernel<<<gd, bd>>>(ru, rv, e, p, s->u, s->v,
                                      s->ruu, s->ruv, s->rvv, s->eu_p, s->ev_p, c->Nx, c->Ny);

    compute_derivatives_kernel<<<gi, bd>>>(r, s->dr_dx, s->dr_dy, s->d2r_dx2, s->d2r_dy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(ru, s->dru_dx, s->dru_dy, s->d2ru_dx2, s->d2ru_dy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(rv, s->drv_dx, s->drv_dy, s->d2rv_dx2, s->d2rv_dy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(e, s->de_dx, s->de_dy, s->d2e_dx2, s->d2e_dy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(p, s->dp_dx, s->dp_dy, s->dummy1, s->dummy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(s->ruu, s->druu_dx, s->dummy1, s->dummy2, s->dummy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(s->ruv, s->druv_dx, s->druv_dy, s->dummy1, s->dummy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(s->rvv, s->dummy1, s->drvv_dy, s->dummy2, s->dummy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(s->eu_p, s->deup_dx, s->dummy1, s->dummy2, s->dummy2, c->Nx, c->Ny, dx, dy);
    compute_derivatives_kernel<<<gi, bd>>>(s->ev_p, s->dummy1, s->devp_dy, s->dummy2, s->dummy2, c->Nx, c->Ny, dx, dy);

    cudaCheck(cudaMemset(rhs_r, 0, (size_t)c->Nx * c->Ny * sizeof(double)));
    cudaCheck(cudaMemset(rhs_ru, 0, (size_t)c->Nx * c->Ny * sizeof(double)));
    cudaCheck(cudaMemset(rhs_rv, 0, (size_t)c->Nx * c->Ny * sizeof(double)));
    cudaCheck(cudaMemset(rhs_e, 0, (size_t)c->Nx * c->Ny * sizeof(double)));

    compute_rhs_kernel<<<gi, bd>>>(r, s->v,
        s->dru_dx, s->drv_dy, s->d2r_dx2, s->d2r_dy2,
        s->druu_dx, s->druv_dy, s->dp_dx, s->d2ru_dx2,
        s->druv_dx, s->drvv_dy, s->dp_dy, s->d2rv_dy2,
        s->deup_dx, s->devp_dy, s->d2e_dx2, s->d2e_dy2,
        rhs_r, rhs_ru, rhs_rv, rhs_e,
        c->Nx, c->Ny, c->g, k1, k2, k3);
    cudaCheckLast();
}

static void host_step(const Config *c, DState *s, double dt) {
    dim3 bd = block_dim(c), gd = grid_full(c);
    double dx = c->Lx / c->Nx;
    double dts = (dx * dx) / (2 * dt);
    double k1 = c->k1_coef * dts, k2 = c->k2_coef * dts, k3 = c->k3_coef * dts;

    compute_rhs_host(c, s, s->r, s->ru, s->rv, s->e, s->p, k1, k2, k3,
                     s->rhs_r, s->rhs_ru, s->rhs_rv, s->rhs_e);

    if (strcmp(c->scheme, "euler") == 0) {
        axpy_kernel<<<gd, bd>>>(s->r,  s->r,  s->rhs_r,  dt, c->Nx, c->Ny);
        axpy_kernel<<<gd, bd>>>(s->ru, s->ru, s->rhs_ru, dt, c->Nx, c->Ny);
        axpy_kernel<<<gd, bd>>>(s->rv, s->rv, s->rhs_rv, dt, c->Nx, c->Ny);
        axpy_kernel<<<gd, bd>>>(s->e,  s->e,  s->rhs_e,  dt, c->Nx, c->Ny);
    } else {
        axpy_kernel<<<gd, bd>>>(s->r_s,  s->r,  s->rhs_r,  dt, c->Nx, c->Ny);
        axpy_kernel<<<gd, bd>>>(s->ru_s, s->ru, s->rhs_ru, dt, c->Nx, c->Ny);
        axpy_kernel<<<gd, bd>>>(s->rv_s, s->rv, s->rhs_rv, dt, c->Nx, c->Ny);
        axpy_kernel<<<gd, bd>>>(s->e_s,  s->e,  s->rhs_e,  dt, c->Nx, c->Ny);
        update_pressure_kernel<<<gd, bd>>>(s->r_s, s->ru_s, s->rv_s, s->e_s, s->p_s, c->Nx, c->Ny, c->gamma);
        apply_bc_kernel<<<gd, bd>>>(s->r_s, s->ru_s, s->rv_s, s->e_s, s->p_s, c->Nx, c->Ny, c->gamma);
        compute_rhs_host(c, s, s->r_s, s->ru_s, s->rv_s, s->e_s, s->p_s, k1, k2, k3,
                         s->rhs_r_s, s->rhs_ru_s, s->rhs_rv_s, s->rhs_e_s);
        rk2_combine_kernel<<<gd, bd>>>(s->r,  s->r,  s->r_s,  s->rhs_r_s,  dt, c->Nx, c->Ny);
        rk2_combine_kernel<<<gd, bd>>>(s->ru, s->ru, s->ru_s, s->rhs_ru_s, dt, c->Nx, c->Ny);
        rk2_combine_kernel<<<gd, bd>>>(s->rv, s->rv, s->rv_s, s->rhs_rv_s, dt, c->Nx, c->Ny);
        rk2_combine_kernel<<<gd, bd>>>(s->e,  s->e,  s->e_s,  s->rhs_e_s,  dt, c->Nx, c->Ny);
    }
    update_pressure_kernel<<<gd, bd>>>(s->r, s->ru, s->rv, s->e, s->p, c->Nx, c->Ny, c->gamma);
    apply_bc_kernel<<<gd, bd>>>(s->r, s->ru, s->rv, s->e, s->p, c->Nx, c->Ny, c->gamma);
    cudaCheckLast();
}

/* ---------- I/O ---------- */
static int dir_exists(const char *p) {
    struct stat st; return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}
static int dir_nonempty(const char *p) {
    DIR *d = opendir(p); if (!d) return 0;
    struct dirent *ent; int n = 0;
    while ((ent = readdir(d))) {
        if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..")) { n = 1; break; }
    }
    closedir(d); return n;
}
static void mkdir_p(const char *path) {
    char buf[1024]; strncpy(buf, path, sizeof(buf)-1); buf[sizeof(buf)-1] = 0;
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(buf, 0755); *p = '/'; }
    }
    mkdir(buf, 0755);
}
static int prepare_output_dir(const Config *c) {
    if (dir_exists(c->output_dir) && dir_nonempty(c->output_dir)) {
        if (!c->overwrite) {
            fprintf(stderr, "error: output dir %s exists and is non-empty; "
                    "pass --overwrite to wipe it, or choose a different --output-dir\n",
                    c->output_dir);
            return -1;
        }
        char cmd[1100];
        snprintf(cmd, sizeof(cmd), "rm -rf '%s'", c->output_dir);
        if (system(cmd) != 0) return -1;
    }
    mkdir_p(c->output_dir);
    return 0;
}

static void dump_config(const Config *c) {
    char path[1100];
    snprintf(path, sizeof(path), "%s/config.json", c->output_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp,
        "{\n"
        "  \"Nx\": %d,\n  \"Ny\": %d,\n"
        "  \"Lx\": %.17g,\n  \"Ly\": %.17g,\n"
        "  \"gamma\": %.17g,\n  \"g\": %.17g,\n"
        "  \"rho_heavy\": %.17g,\n  \"rho_light\": %.17g,\n"
        "  \"p0\": %.17g,\n  \"perturbation_amp\": %.17g,\n"
        "  \"seed\": %d,\n"
        "  \"scheme\": \"%s\",\n  \"cfl\": %.17g,\n"
        "  \"k1_coef\": %.17g,\n  \"k2_coef\": %.17g,\n  \"k3_coef\": %.17g,\n",
        c->Nx, c->Ny, c->Lx, c->Ly, c->gamma, c->g,
        c->rho_heavy, c->rho_light, c->p0, c->perturbation_amp, c->seed,
        c->scheme, c->cfl, c->k1_coef, c->k2_coef, c->k3_coef);
    if (c->steps < 0) fprintf(fp, "  \"steps\": null,\n");
    else              fprintf(fp, "  \"steps\": %ld,\n", c->steps);
    if (isnan(c->t_end)) fprintf(fp, "  \"t_end\": null,\n");
    else                 fprintf(fp, "  \"t_end\": %.17g,\n", c->t_end);
    fprintf(fp,
        "  \"save_interval\": %d,\n"
        "  \"output_dir\": \"%s\",\n"
        "  \"no_output\": %s,\n  \"overwrite\": %s,\n"
        "  \"warmup_steps\": %d,\n  \"repeats\": %d,\n"
        "  \"block_x\": %d,\n  \"block_y\": %d\n"
        "}\n",
        c->save_interval, c->output_dir,
        c->no_output ? "true" : "false", c->overwrite ? "true" : "false",
        c->warmup_steps, c->repeats, c->block_x, c->block_y);
    fclose(fp);
}

/* device layout is (Ny, Nx) row-major; host frame format is (Nx, Ny) row-major.
 * Transpose during copy so on-disk bytes match Python/C. */
static void copy_transpose(const Config *c, const double *d_in, double *h_out) {
    static double *h_tmp = NULL;
    static size_t tmp_n = 0;
    size_t n = (size_t)c->Nx * c->Ny;
    if (tmp_n != n) {
        free(h_tmp); h_tmp = (double*)malloc(n * sizeof(double)); tmp_n = n;
    }
    cudaCheck(cudaMemcpy(h_tmp, d_in, n * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < c->Nx; i++)
        for (int j = 0; j < c->Ny; j++)
            h_out[(size_t)i * c->Ny + j] = h_tmp[(size_t)j * c->Nx + i];
}

static void write_frame(const Config *c, DState *s, int step_idx, double t) {
    char path[1100];
    snprintf(path, sizeof(path), "%s/frame_%06d.bin", c->output_dir, step_idx);
    FILE *fp = fopen(path, "wb"); if (!fp) return;
    int64_t step64 = step_idx, Nx64 = c->Nx, Ny64 = c->Ny;
    fwrite(&step64, sizeof(int64_t), 1, fp);
    fwrite(&t, sizeof(double), 1, fp);
    fwrite(&Nx64, sizeof(int64_t), 1, fp);
    fwrite(&Ny64, sizeof(int64_t), 1, fp);
    size_t n = (size_t)c->Nx * c->Ny;
    double *h = (double*)malloc(n * sizeof(double));
    copy_transpose(c, s->r, h);  fwrite(h, sizeof(double), n, fp);
    copy_transpose(c, s->ru, h); fwrite(h, sizeof(double), n, fp);
    copy_transpose(c, s->rv, h); fwrite(h, sizeof(double), n, fp);
    copy_transpose(c, s->e, h);  fwrite(h, sizeof(double), n, fp);
    free(h);
    fclose(fp);
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    Config c; cfg_defaults(&c);
    if (parse_args(argc, argv, &c) != 0) return 1;

    if (!c.no_output) {
        if (prepare_output_dir(&c) != 0) return 1;
        dump_config(&c);
    }

    DState s; dstate_alloc(&s, &c);

    dim3 bd = block_dim(&c), gd = grid_full(&c);
    double dy = c.Ly / c.Ny;
    initialize_grid_kernel<<<gd, bd>>>(s.r, s.ru, s.rv, s.e, s.p,
                                       c.Nx, c.Ny, c.Ly, dy,
                                       c.rho_heavy, c.rho_light, c.p0, c.g,
                                       c.perturbation_amp, c.gamma, (unsigned int)c.seed);
    cudaCheckLast();
    apply_bc_kernel<<<gd, bd>>>(s.r, s.ru, s.rv, s.e, s.p, c.Nx, c.Ny, c.gamma);
    cudaCheck(cudaDeviceSynchronize());

    double t = 0.0;
    long step_idx = 0;
    if (c.save_interval && !c.no_output) write_frame(&c, &s, step_idx, t);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        if (c.steps >= 0 && step_idx >= c.steps) break;
        if (!isnan(c.t_end) && t >= c.t_end) break;
        double dt = host_compute_dt(&c, &s);
        if (!isnan(c.t_end) && t + dt > c.t_end) dt = c.t_end - t;
        host_step(&c, &s, dt);
        t += dt;
        step_idx++;
        if (c.save_interval && !c.no_output && step_idx % c.save_interval == 0) {
            cudaCheck(cudaDeviceSynchronize());
            write_frame(&c, &s, (int)step_idx, t);
            printf("step %ld  t=%.4f  dt=%.4e\n", step_idx, t, dt);
            fflush(stdout);
        }
    }
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("done: %ld steps in %.2fs (%.1f steps/s, %.2e cell-updates/s)\n",
           step_idx, wall, step_idx / wall,
           (double)c.Nx * c.Ny * step_idx / wall);
    return 0;
}
