import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.80  # Gravity
        self.alpha = 0.5  # viscosity
        self.rho_0 = 1000.0  # reference density
        self.m = self.ps.dx**self.ps.dim * self.rho_0
        self.CFL_v = 0.25  # CFL coefficient for velocity
        self.CFL_a = 0.05  # CFL coefficient for acceleration
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4

    @ti.func
    def cubic_kernel(self, r):
        h = self.ps.h
        # value of cubic spline smoothing kernel
        k = 10. / (7. * np.pi * h**self.ps.dim)
        q = r / h
        # assert q >= 0.0  # Metal backend is not happy with assert
        res = ti.cast(0.0, ti.f32)
        if q <= 1.0:
            res = k * (1 - 1.5 * q**2 + 0.75 * q**3)
        elif q < 2.0:
            res = k * 0.25 * (2 - q)**3
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.h
        # derivative of cubic spline smoothing kernel
        k = 10. / (7. * np.pi * h**self.ps.dim)
        q = r / h
        # assert q > 0.0
        res = ti.cast(0.0, ti.f32)
        if q < 1.0:
            res = (k / h) * (-3 * q + 2.25 * q**2)
        elif q < 2.0:
            res = -0.75 * (k / h) * (2 - q)**2
        return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r, r_mod):
        # Compute the viscosity force contribution, artificial viscosity
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        if v_xy < 0:
            # Artifical viscosity
            vmu = -2.0 * self.alpha * self.ps.dx * self.c_0 / (
                self.ps.density[p_i] +
                self.ps.density[p_j])
            res = -self.m * vmu * v_xy / (
                r_mod**2 + 0.01 * self.ps.dx**2) * self.cubic_kernel_derivative(
                    r_mod) * r / r_mod
        return res
    
    @ti.func
    def pressure_force(self, p_i, p_j, r, r_mod):
        # Compute the pressure force contribution, Symmetric Formula
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        res = -self.m * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                         + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
              * self.cubic_kernel_derivative(r_mod) * r / r_mod
        return res

    def substep(self):
        pass

    @ti.kernel
    def enforce_boundary(self):
        pass

    def step(self):
        self.ps.initialize_particle_system()
        self.substep()
        self.enforce_boundary()


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        self.gamma = 7.0
        self.c_0 = 200.0
        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.d_density = ti.field(float)

        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)
        particle_node.place(self.d_density)

    @ti.func
    def rho_derivative(self, p_i, p_j, r, r_mod):
        # density delta, i.e. divergence
        return self.m * self.cubic_kernel_derivative(r_mod) \
               * (self.ps.v[p_i] - self.ps.v[p_j]).dot(r / r_mod)

    @ti.func
    def pressure_update(self, rho, rho_0=1000.0, gamma=7.0, c_0=20.0):
        # Weakly compressible, tait function
        b = rho_0 * c_0**2 / gamma
        return b * ((rho / rho_0)**gamma - 1.0)

    @ti.kernel
    def compute_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            pass
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_rho = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]

                # Compute distance and its mod
                r = x_i - x_j
                r_mod = ti.max(r.norm(), 1e-5)

                # Compute Density change
                d_rho += self.rho_derivative(p_i, p_j, r, r_mod)

                if self.ps.material[p_i] == 1:
                    # Compute Viscosity force contribution
                    d_v += self.viscosity_force(p_i, p_j, r, r_mod)
                    # Compute Pressure force contribution
                    d_v += self.pressure_force(p_i, p_j, r, r_mod)

            # Add body force
            if self.ps.material[p_i] == 1:
                d_v += ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])
            self.d_velocity[p_i] = d_v
            self.d_density[p_i] = d_rho


    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == 1:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]
            self.ps.density[p_i] += self.dt[None] * self.d_density[p_i]
            self.ps.pressure[p_i] = self.pressure_update(
                self.ps.density[p_i], self.rho_0, self.gamma,
                self.c_0)


    def substep(self):
        self.compute_forces()
        self.advect()