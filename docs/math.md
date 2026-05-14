# Mathematical framework

## Conservative variables

The system models a compressible fluid with the following conservative variables:

- **Density**: $\rho$ (denoted as `r`)
- **Momentum**: $\rho u$ and $\rho v$ (denoted as `ru`, `rv`)
- **Total energy**: $e$

## Governing equations

The 2D conservation equations with artificial diffusion are:

### Continuity (mass conservation)

$$\frac{\partial \rho}{\partial t} + \frac{\partial(\rho u)}{\partial x} + \frac{\partial(\rho v)}{\partial y} = k_1\left(\frac{\partial^2 \rho}{\partial x^2} + \frac{\partial^2 \rho}{\partial y^2}\right)$$

### Momentum (x-direction)

$$\frac{\partial(\rho u)}{\partial t} + \frac{\partial(\rho u u)}{\partial x} + \frac{\partial(\rho u v)}{\partial y} + \frac{\partial p}{\partial x} = k_2\frac{\partial^2(\rho u)}{\partial x^2}$$

### Momentum (y-direction)

$$\frac{\partial(\rho v)}{\partial t} + \frac{\partial(\rho u v)}{\partial x} + \frac{\partial(\rho v v)}{\partial y} + \frac{\partial p}{\partial y} = -g\rho + k_2\frac{\partial^2(\rho v)}{\partial y^2}$$

### Total energy

$$\frac{\partial e}{\partial t} + \frac{\partial(u(e + p))}{\partial x} + \frac{\partial(v(e + p))}{\partial y} = -g\rho v + k_3\left(\frac{\partial^2 e}{\partial x^2} + \frac{\partial^2 e}{\partial y^2}\right)$$

with the equation of state

$$e = \frac{p}{\gamma - 1} + \tfrac{1}{2}\rho(u^2 + v^2),$$

where $\gamma = 1.4$ is the specific-heat ratio, $g = -10$ is gravitational acceleration, and $k_1, k_2, k_3$ are artificial-diffusion coefficients (see below).

## Initial and boundary conditions

### Initial conditions

The heavy/light interface is placed at $y = L_y/2$:

- **Density**: $\rho = 2$ for $y \geq L_y/2$, else $\rho = 1$
- **Velocity**: $u = 0$ everywhere; $v = 0$ except near the interface ($|y - L_y/2| \leq 0.05$), where $v$ is a small random perturbation in $[-10^{-3}, 10^{-3}]$
- **Pressure**: $p = p_0 + \rho g (y - L_y/2)$ (hydrostatic equilibrium, $p_0 = 40$)
- **Energy**: $e$ derived from $p$, $u$, $v$ via the EOS

### Boundary conditions

- **y-boundaries** (top/bottom): rigid walls, normal velocity $v = 0$, hydrostatic pressure
- **x-boundaries**: periodic

## Time integration

Two schemes are implemented; selectable via `--scheme`.

### Explicit Euler

$$q^{n+1} = q^n + \Delta t \cdot \mathrm{RHS}(q^n)$$

### Second-order Runge-Kutta (RK2)

$$q^\* = q^n + \Delta t \cdot \mathrm{RHS}(q^n)$$

$$q^{n+1} = \tfrac{1}{2} q^n + \tfrac{1}{2}\left(q^\* + \Delta t \cdot \mathrm{RHS}(q^\*)\right)$$

where $q = [\rho, \rho u, \rho v, e]$.

## CFL condition

The time step is chosen adaptively as

$$\Delta t = C \cdot \frac{\Delta x}{\max(|u|, |v|, c)}, \qquad c = \sqrt{\frac{\gamma p}{\rho}},$$

with $C = 0.2$ (configurable via `--cfl`).

## Artificial diffusion

For numerical stability the diffusion coefficients scale with grid spacing and time step:

$$k_1 = \alpha_1 \frac{\Delta x^2}{2\Delta t}, \quad k_2 = \alpha_2 \frac{\Delta x^2}{2\Delta t}, \quad k_3 = \alpha_3 \frac{\Delta x^2}{2\Delta t}$$

with defaults $\alpha_1 = 0.0125$, $\alpha_2 = 0.125$, $\alpha_3 = 0.0125$ (configurable via `--k1-coef`, `--k2-coef`, `--k3-coef`).
