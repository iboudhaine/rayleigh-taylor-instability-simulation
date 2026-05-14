/* C/CPU implementation of the Rayleigh-Taylor instability.
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

/* ---------- config ---------- */
typedef struct {
    int Nx, Ny;
    double Lx, Ly;
    double gamma, g;
    double rho_heavy, rho_light, p0, perturbation_amp;
    int seed;
    char scheme[8];                /* "euler" | "rk2" */
    double cfl;
    double k1_coef, k2_coef, k3_coef;
    long steps;                    /* -1 = unset */
    double t_end;                  /* NaN = unset */
    int save_interval;
    char output_dir[1024];
    int no_output, overwrite;
    int warmup_steps, repeats;
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
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  physics:    --Nx --Ny --Lx --Ly --gamma --g --rho-heavy --rho-light\n"
        "              --p0 --perturbation-amp --seed\n"
        "  numerics:   --scheme {euler|rk2} --cfl --k1-coef --k2-coef --k3-coef\n"
        "  run:        --steps --t-end --save-interval --output-dir\n"
        "              --no-output --overwrite\n"
        "  bench:      --warmup-steps --repeats\n", prog);
}

static int parse_args(int argc, char **argv, Config *c) {
    enum {
        O_NX = 1000, O_NY, O_LX, O_LY, O_GAMMA, O_G, O_RHOH, O_RHOL, O_P0,
        O_PERT, O_SEED, O_SCHEME, O_CFL, O_K1, O_K2, O_K3,
        O_STEPS, O_TEND, O_SAVE, O_OUT, O_NOOUT, O_OVER, O_WARM, O_REP
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
        case O_OUT:
            strncpy(c->output_dir, optarg, sizeof(c->output_dir) - 1); break;
        case O_NOOUT: c->no_output = 1; break;
        case O_OVER: c->overwrite = 1; break;
        case O_WARM: c->warmup_steps = atoi(optarg); break;
        case O_REP: c->repeats = atoi(optarg); break;
        case 'h': usage(argv[0]); exit(0);
        default: usage(argv[0]); return -1;
        }
    }
    if (c->steps < 0 && isnan(c->t_end)) {
        fprintf(stderr, "error: must specify --steps or --t-end\n");
        return -1;
    }
    return 0;
}

/* ---------- state ---------- */
#define IX(i,j) ((i)*Ny + (j))

typedef struct {
    int Nx, Ny;
    /* primary */
    double *r, *ru, *rv, *e, *p;
    /* rhs scratch */
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
} State;

static double *xcalloc(size_t n) {
    double *p = calloc(n, sizeof(double));
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static void state_init(State *s, int Nx, int Ny) {
    s->Nx = Nx; s->Ny = Ny;
    size_t n = (size_t)Nx * Ny;
    double **slots[] = {
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
    for (size_t k = 0; k < sizeof(slots)/sizeof(slots[0]); k++)
        *slots[k] = xcalloc(n);
}

/* ---------- RNG (xorshift64) ---------- */
static uint64_t rng_state;
static void rng_seed(uint64_t seed) { rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL; }
static double rng_uniform(void) { /* [0,1) */
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return (x >> 11) * (1.0 / (double)(1ULL << 53));
}

/* ---------- physics ---------- */
static void initialize_grid(const Config *c, State *s) {
    int Nx = c->Nx, Ny = c->Ny;
    double dy = c->Ly / Ny;
    rng_seed((uint64_t)c->seed);

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double y = (j + 0.5) * dy;
            double rho = (y >= c->Ly / 2) ? c->rho_heavy : c->rho_light;
            double u = 0.0, v = 0.0;
            if (fabs(y - c->Ly / 2) <= 0.05) {
                v = c->perturbation_amp * (2.0 * rng_uniform() - 1.0);
            }
            s->r[IX(i,j)] = rho;
            s->ru[IX(i,j)] = rho * u;
            s->rv[IX(i,j)] = rho * v;
            s->p[IX(i,j)] = c->p0 + rho * c->g * (y - c->Ly / 2);
            s->e[IX(i,j)] = s->p[IX(i,j)] / (c->gamma - 1) + 0.5 * rho * (u*u + v*v);
        }
    }
}

static double compute_dt(const Config *c, const State *s) {
    int Nx = c->Nx, Ny = c->Ny;
    double dx = c->Lx / Nx, dy = c->Ly / Ny;
    double max_speed = 0.0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double rho = s->r[IX(i,j)];
            double u = 0.0, v = 0.0;
            if (rho > 1e-10) { u = s->ru[IX(i,j)] / rho; v = s->rv[IX(i,j)] / rho; }
            double cs = sqrt(c->gamma * s->p[IX(i,j)] / rho);
            double speed = fmax(fmax(fabs(u), fabs(v)), cs);
            if (speed > max_speed) max_speed = speed;
        }
    }
    return c->cfl * fmin(dx, dy) / max_speed;
}

static void compute_derivatives(const double *f, double *dfx, double *dfy,
                                double *d2x, double *d2y,
                                int Nx, int Ny, double dx, double dy) {
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            dfx[IX(i,j)] = (f[IX(i+1,j)] - f[IX(i-1,j)]) / (2 * dx);
            dfy[IX(i,j)] = (f[IX(i,j+1)] - f[IX(i,j-1)]) / (2 * dy);
            d2x[IX(i,j)] = (f[IX(i+1,j)] - 2*f[IX(i,j)] + f[IX(i-1,j)]) / (dx*dx);
            d2y[IX(i,j)] = (f[IX(i,j+1)] - 2*f[IX(i,j)] + f[IX(i,j-1)]) / (dy*dy);
        }
    }
}

static void compute_rhs(const Config *c, State *s,
                        const double *r, const double *ru, const double *rv,
                        const double *e, const double *p,
                        double k1, double k2, double k3,
                        double *rhs_r, double *rhs_ru, double *rhs_rv, double *rhs_e) {
    int Nx = c->Nx, Ny = c->Ny;
    double dx = c->Lx / Nx, dy = c->Ly / Ny;
    size_t n = (size_t)Nx * Ny;

    for (size_t k = 0; k < n; k++) {
        double rho = r[k];
        double u = 0.0, v = 0.0;
        if (rho > 1e-10) { u = ru[k] / rho; v = rv[k] / rho; }
        s->u[k] = u; s->v[k] = v;
        s->ruu[k] = ru[k] * u;
        s->ruv[k] = ru[k] * v;
        s->rvv[k] = rv[k] * v;
        s->eu_p[k] = u * (e[k] + p[k]);
        s->ev_p[k] = v * (e[k] + p[k]);
    }

    compute_derivatives(r,      s->dr_dx, s->dr_dy, s->d2r_dx2, s->d2r_dy2, Nx, Ny, dx, dy);
    compute_derivatives(ru,     s->dru_dx, s->dru_dy, s->d2ru_dx2, s->d2ru_dy2, Nx, Ny, dx, dy);
    compute_derivatives(rv,     s->drv_dx, s->drv_dy, s->d2rv_dx2, s->d2rv_dy2, Nx, Ny, dx, dy);
    compute_derivatives(e,      s->de_dx, s->de_dy, s->d2e_dx2, s->d2e_dy2, Nx, Ny, dx, dy);
    compute_derivatives(p,      s->dp_dx, s->dp_dy, s->dummy1, s->dummy2, Nx, Ny, dx, dy);
    compute_derivatives(s->ruu, s->druu_dx, s->dummy1, s->dummy2, s->dummy2, Nx, Ny, dx, dy);
    compute_derivatives(s->ruv, s->druv_dx, s->druv_dy, s->dummy1, s->dummy2, Nx, Ny, dx, dy);
    compute_derivatives(s->rvv, s->dummy1, s->drvv_dy, s->dummy2, s->dummy2, Nx, Ny, dx, dy);
    compute_derivatives(s->eu_p, s->deup_dx, s->dummy1, s->dummy2, s->dummy2, Nx, Ny, dx, dy);
    compute_derivatives(s->ev_p, s->dummy1, s->devp_dy, s->dummy2, s->dummy2, Nx, Ny, dx, dy);

    /* zero everything then fill interior — matches Python behavior */
    memset(rhs_r, 0, n * sizeof(double));
    memset(rhs_ru, 0, n * sizeof(double));
    memset(rhs_rv, 0, n * sizeof(double));
    memset(rhs_e, 0, n * sizeof(double));
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            size_t k = IX(i,j);
            rhs_r[k]  = -(s->dru_dx[k] + s->drv_dy[k]) + k1 * (s->d2r_dx2[k] + s->d2r_dy2[k]);
            rhs_ru[k] = -(s->druu_dx[k] + s->druv_dy[k]) - s->dp_dx[k] + k2 * s->d2ru_dx2[k];
            rhs_rv[k] = -(s->druv_dx[k] + s->drvv_dy[k]) - s->dp_dy[k] + c->g * r[k] + k2 * s->d2rv_dy2[k];
            rhs_e[k]  = -(s->deup_dx[k] + s->devp_dy[k]) + c->g * r[k] * s->v[k] + k3 * (s->d2e_dx2[k] + s->d2e_dy2[k]);
        }
    }
}

static void apply_bc(const Config *c,
                     double *r, double *ru, double *rv, double *e, double *p) {
    int Nx = c->Nx, Ny = c->Ny;
    for (int j = 0; j < Ny; j++) {
        r[IX(0,j)]      = r[IX(Nx-2,j)];
        r[IX(Nx-1,j)]   = r[IX(1,j)];
        ru[IX(0,j)]     = ru[IX(Nx-2,j)];
        ru[IX(Nx-1,j)]  = ru[IX(1,j)];
        rv[IX(0,j)]     = rv[IX(Nx-2,j)];
        rv[IX(Nx-1,j)]  = rv[IX(1,j)];
        e[IX(0,j)]      = e[IX(Nx-2,j)];
        e[IX(Nx-1,j)]   = e[IX(1,j)];
        p[IX(0,j)]      = p[IX(Nx-2,j)];
        p[IX(Nx-1,j)]   = p[IX(1,j)];
    }
    for (int i = 0; i < Nx; i++) {
        rv[IX(i,0)] = 0; rv[IX(i,Ny-1)] = 0;
        r[IX(i,0)] = r[IX(i,1)];
        ru[IX(i,0)] = ru[IX(i,1)];
        p[IX(i,0)] = p[IX(i,1)];
        e[IX(i,0)] = p[IX(i,0)] / (c->gamma - 1)
                   + 0.5 * (ru[IX(i,0)]*ru[IX(i,0)] + rv[IX(i,0)]*rv[IX(i,0)]) / r[IX(i,0)];
        r[IX(i,Ny-1)] = r[IX(i,Ny-2)];
        ru[IX(i,Ny-1)] = ru[IX(i,Ny-2)];
        p[IX(i,Ny-1)] = p[IX(i,Ny-2)];
        e[IX(i,Ny-1)] = p[IX(i,Ny-1)] / (c->gamma - 1)
                     + 0.5 * (ru[IX(i,Ny-1)]*ru[IX(i,Ny-1)] + rv[IX(i,Ny-1)]*rv[IX(i,Ny-1)]) / r[IX(i,Ny-1)];
    }
}

static void update_pressure(const Config *c, const double *r, const double *ru,
                            const double *rv, const double *e, double *p) {
    size_t n = (size_t)c->Nx * c->Ny;
    for (size_t k = 0; k < n; k++) {
        double rho = r[k];
        double u = 0.0, v = 0.0;
        if (rho > 1e-10) { u = ru[k]/rho; v = rv[k]/rho; }
        double pv = (c->gamma - 1) * (e[k] - 0.5 * rho * (u*u + v*v));
        p[k] = pv < 1e-10 ? 1e-10 : pv;
    }
}

static void update_diffusion_coeffs(const Config *c, double dt,
                                    double *k1, double *k2, double *k3) {
    double dx = c->Lx / c->Nx;
    double s = (dx*dx) / (2*dt);
    *k1 = c->k1_coef * s;
    *k2 = c->k2_coef * s;
    *k3 = c->k3_coef * s;
}

static void step(const Config *c, State *s, double dt) {
    int Nx = c->Nx, Ny = c->Ny;
    size_t n = (size_t)Nx * Ny;
    double k1, k2, k3;
    update_diffusion_coeffs(c, dt, &k1, &k2, &k3);

    if (strcmp(c->scheme, "euler") == 0) {
        compute_rhs(c, s, s->r, s->ru, s->rv, s->e, s->p, k1, k2, k3,
                    s->rhs_r, s->rhs_ru, s->rhs_rv, s->rhs_e);
        for (size_t k = 0; k < n; k++) {
            s->r[k]  += dt * s->rhs_r[k];
            s->ru[k] += dt * s->rhs_ru[k];
            s->rv[k] += dt * s->rhs_rv[k];
            s->e[k]  += dt * s->rhs_e[k];
        }
    } else { /* rk2 */
        compute_rhs(c, s, s->r, s->ru, s->rv, s->e, s->p, k1, k2, k3,
                    s->rhs_r, s->rhs_ru, s->rhs_rv, s->rhs_e);
        for (size_t k = 0; k < n; k++) {
            s->r_s[k]  = s->r[k]  + dt * s->rhs_r[k];
            s->ru_s[k] = s->ru[k] + dt * s->rhs_ru[k];
            s->rv_s[k] = s->rv[k] + dt * s->rhs_rv[k];
            s->e_s[k]  = s->e[k]  + dt * s->rhs_e[k];
        }
        update_pressure(c, s->r_s, s->ru_s, s->rv_s, s->e_s, s->p_s);
        apply_bc(c, s->r_s, s->ru_s, s->rv_s, s->e_s, s->p_s);
        compute_rhs(c, s, s->r_s, s->ru_s, s->rv_s, s->e_s, s->p_s, k1, k2, k3,
                    s->rhs_r_s, s->rhs_ru_s, s->rhs_rv_s, s->rhs_e_s);
        for (size_t k = 0; k < n; k++) {
            s->r[k]  = 0.5 * s->r[k]  + 0.5 * (s->r_s[k]  + dt * s->rhs_r_s[k]);
            s->ru[k] = 0.5 * s->ru[k] + 0.5 * (s->ru_s[k] + dt * s->rhs_ru_s[k]);
            s->rv[k] = 0.5 * s->rv[k] + 0.5 * (s->rv_s[k] + dt * s->rhs_rv_s[k]);
            s->e[k]  = 0.5 * s->e[k]  + 0.5 * (s->e_s[k]  + dt * s->rhs_e_s[k]);
        }
    }
    update_pressure(c, s->r, s->ru, s->rv, s->e, s->p);
    apply_bc(c, s->r, s->ru, s->rv, s->e, s->p);
}

/* ---------- I/O ---------- */
static int dir_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

static int dir_nonempty(const char *p) {
    DIR *d = opendir(p);
    if (!d) return 0;
    struct dirent *ent;
    int nonempty = 0;
    while ((ent = readdir(d))) {
        if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..")) { nonempty = 1; break; }
    }
    closedir(d);
    return nonempty;
}

static int mkdir_p(const char *path) {
    char buf[1024];
    strncpy(buf, path, sizeof(buf) - 1); buf[sizeof(buf)-1] = 0;
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(buf, 0755);
            *p = '/';
        }
    }
    return mkdir(buf, 0755);
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
        if (system(cmd) != 0) { fprintf(stderr, "error: failed to wipe %s\n", c->output_dir); return -1; }
    }
    mkdir_p(c->output_dir);
    return 0;
}

static void dump_config(const Config *c) {
    char path[1100];
    snprintf(path, sizeof(path), "%s/config.json", c->output_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "warn: cannot write %s\n", path); return; }
    fprintf(fp,
        "{\n"
        "  \"Nx\": %d,\n  \"Ny\": %d,\n"
        "  \"Lx\": %.17g,\n  \"Ly\": %.17g,\n"
        "  \"gamma\": %.17g,\n  \"g\": %.17g,\n"
        "  \"rho_heavy\": %.17g,\n  \"rho_light\": %.17g,\n"
        "  \"p0\": %.17g,\n  \"perturbation_amp\": %.17g,\n"
        "  \"seed\": %d,\n"
        "  \"scheme\": \"%s\",\n  \"cfl\": %.17g,\n"
        "  \"k1_coef\": %.17g,\n  \"k2_coef\": %.17g,\n  \"k3_coef\": %.17g,\n"
        "  \"steps\": %s,\n  \"t_end\": %s,\n"
        "  \"save_interval\": %d,\n"
        "  \"output_dir\": \"%s\",\n"
        "  \"no_output\": %s,\n  \"overwrite\": %s,\n"
        "  \"warmup_steps\": %d,\n  \"repeats\": %d\n"
        "}\n",
        c->Nx, c->Ny, c->Lx, c->Ly, c->gamma, c->g,
        c->rho_heavy, c->rho_light, c->p0, c->perturbation_amp, c->seed,
        c->scheme, c->cfl, c->k1_coef, c->k2_coef, c->k3_coef,
        (c->steps < 0 ? "null" : (snprintf(NULL,0,"%ld",c->steps), "STEPS")),
        (isnan(c->t_end) ? "null" : "TEND"),
        c->save_interval, c->output_dir,
        c->no_output ? "true" : "false", c->overwrite ? "true" : "false",
        c->warmup_steps, c->repeats);
    fclose(fp);
    /* re-render with actual numeric values for steps/t_end */
    fp = fopen(path, "w");
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
        "  \"warmup_steps\": %d,\n  \"repeats\": %d\n"
        "}\n",
        c->save_interval, c->output_dir,
        c->no_output ? "true" : "false", c->overwrite ? "true" : "false",
        c->warmup_steps, c->repeats);
    fclose(fp);
}

static void write_frame(const Config *c, int step_idx, double t, const State *s) {
    char path[1100];
    snprintf(path, sizeof(path), "%s/frame_%06d.bin", c->output_dir, step_idx);
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "warn: cannot write %s\n", path); return; }
    int64_t step64 = step_idx;
    int64_t Nx64 = c->Nx, Ny64 = c->Ny;
    fwrite(&step64, sizeof(int64_t), 1, fp);
    fwrite(&t, sizeof(double), 1, fp);
    fwrite(&Nx64, sizeof(int64_t), 1, fp);
    fwrite(&Ny64, sizeof(int64_t), 1, fp);
    size_t n = (size_t)c->Nx * c->Ny;
    fwrite(s->r, sizeof(double), n, fp);
    fwrite(s->ru, sizeof(double), n, fp);
    fwrite(s->rv, sizeof(double), n, fp);
    fwrite(s->e, sizeof(double), n, fp);
    fclose(fp);
}

/* ---------- main loop ---------- */
int main(int argc, char **argv) {
    Config c; cfg_defaults(&c);
    if (parse_args(argc, argv, &c) != 0) return 1;

    if (!c.no_output) {
        if (prepare_output_dir(&c) != 0) return 1;
        dump_config(&c);
    }

    State s;
    state_init(&s, c.Nx, c.Ny);
    initialize_grid(&c, &s);
    apply_bc(&c, s.r, s.ru, s.rv, s.e, s.p);

    double t = 0.0;
    long step_idx = 0;
    if (c.save_interval && !c.no_output)
        write_frame(&c, step_idx, t, &s);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        if (c.steps >= 0 && step_idx >= c.steps) break;
        if (!isnan(c.t_end) && t >= c.t_end) break;
        double dt = compute_dt(&c, &s);
        if (!isnan(c.t_end) && t + dt > c.t_end) dt = c.t_end - t;
        step(&c, &s, dt);
        t += dt;
        step_idx++;
        if (c.save_interval && !c.no_output && step_idx % c.save_interval == 0) {
            write_frame(&c, (int)step_idx, t, &s);
            printf("step %ld  t=%.4f  dt=%.4e\n", step_idx, t, dt);
            fflush(stdout);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("done: %ld steps in %.2fs (%.1f steps/s, %.2e cell-updates/s)\n",
           step_idx, wall, step_idx / wall,
           (double)c.Nx * c.Ny * step_idx / wall);
    return 0;
}
