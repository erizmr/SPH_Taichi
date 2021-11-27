import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.80  # Gravity
        self.viscosity = 0.1  # viscosity
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.CFL_v = 0.25  # CFL coefficient for velocity
        self.CFL_a = 0.05  # CFL coefficient for acceleration
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 8. / (np.pi * h**self.ps.dim)
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 48. / (np.pi * h**self.ps.dim)
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    @ti.func
    def pressure_force(self, p_i, p_j, r):
        # Compute the pressure force contribution, Symmetric Formula
        res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                         + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
              * self.cubic_kernel_derivative(r)
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
        self.exponent = 7.0
        self.stiffness = 50.0

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)

    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V * self.cubic_kernel((x_i - x_j).norm())
                # if  p_i == 25:
                #     print("p_i  x_i p_j x_j", p_i, x_i, p_j, x_j,  self.cubic_kernel((x_i - x_j).norm()), self.ps.particle_neighbors_num[p_i])
            self.ps.density[p_i] *= self.density_0

            # if p_i == 75 or p_i == 25:
            #     print("p_i ", p_i, x_i, self.ps.density[p_i], self.ps.particle_neighbors_num[p_i])

    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_i] == self.ps.material_fluid:
                    # Compute Pressure force contribution
                    d_v += self.pressure_force(p_i, p_j, x_i-x_j)
            self.d_velocity[p_i] += d_v



    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_i] == self.ps.material_fluid:
                    # Compute Viscosity force contribution
                    d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
                    pass

            # Add body force
            if self.ps.material[p_i] == self.ps.material_fluid:
                d_v += ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()
