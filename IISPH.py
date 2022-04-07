import taichi as ti
from sph_base import SPHBase


class IISPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)

        self.a_ii = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.density_deviation = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.last_pressure = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.avg_density_error = ti.field(dtype=float, shape=())

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.pressure_accel = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity, self.pressure_accel)

    @ti.kernel
    def predict_advection(self):
        # Compute a_ii
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            sum_neighbor = 0.0
            sum_neighbor_of_neighbor = 0.0
            m_Vi = self.ps.m_V[p_i]
            density_i = self.ps.density[p_i] / self.density_0
            density_i2 = density_i * density_i
            self.a_ii[p_i] = 0.0
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                sum_neighbor_inner = ti.Vector([0.0 for _ in range(self.ps.dim)])
                for k in range(self.ps.fluid_neighbors_num[p_i]):
                    density_k = self.ps.density[k] / self.density_0
                    density_k2 = density_k * density_k
                    p_k = self.ps.fluid_neighbors[p_i, j]
                    x_k = self.ps.x[p_k]
                    sum_neighbor_inner += self.ps.m_V[p_k] * self.cubic_kernel_derivative(x_i - x_k) / density_k2

                kernel_grad_ij = self.cubic_kernel_derivative(x_i - x_j)
                sum_neighbor -= (self.ps.m_V[p_j] * sum_neighbor_inner).dot(kernel_grad_ij)

                sum_neighbor_of_neighbor -= (self.ps.m_V[p_j] * kernel_grad_ij).dot(kernel_grad_ij)
            sum_neighbor_of_neighbor *= m_Vi / density_i2
            self.a_ii[p_i] += (sum_neighbor + sum_neighbor_of_neighbor) * self.dt[None] * self.dt[None]

            # Boundary neighbors
            ## Akinci2012
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                sum_neighbor_inner = ti.Vector([0.0 for _ in range(self.ps.dim)])
                for k in range(self.ps.fluid_neighbors_num[p_i]):
                    density_k = self.ps.density[k] / self.density_0
                    density_k2 = density_k * density_k
                    p_k = self.ps.boundary_neighbors[p_i, j]
                    x_k = self.ps.x[p_k]
                    sum_neighbor_inner += self.ps.m_V[p_k] * self.cubic_kernel_derivative(x_i - x_k) / density_k2

                kernel_grad_ij = self.cubic_kernel_derivative(x_i - x_j)
                sum_neighbor -= (self.ps.m_V[p_j] * sum_neighbor_inner).dot(kernel_grad_ij)

                sum_neighbor_of_neighbor -= (self.ps.m_V[p_j] * kernel_grad_ij).dot(kernel_grad_ij)
            sum_neighbor_of_neighbor *= m_Vi / density_i2
            self.a_ii[p_i] += (sum_neighbor + sum_neighbor_of_neighbor) * self.dt[None] * self.dt[None]

        # Compute source term (i.e., density deviation)
        # Compute the predicted v^star
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]

        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            density_i = self.ps.density[p_i] / self.density_0
            divergence = 0.0
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                divergence += self.ps.m_V[p_j] * (self.ps.v[p_i] - self.ps.v[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))

            # Boundary neighbors
            ## Akinci2012
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                divergence += self.ps.m_V[p_j] * (self.ps.v[p_i] - self.ps.v[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))

            self.density_deviation[p_i] = self.density_0 / self.density_0 - density_i - self.dt[None] * divergence

    def pressure_solve(self):
        for i in range(20):
            self.pressure_solve_iteration()
            # print(self.ps.pressure[0])

    @ti.kernel
    def pressure_solve_iteration(self):
        omega = 0.5
        # Compute pressure acceleration
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                self.pressure_accel[p_i].fill(0)
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])

            dpi = self.last_pressure[p_i] / self.ps.density[p_i] ** 2
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                dpj = self.last_pressure[p_j] / self.ps.density[p_j] ** 2
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)

            # Boundary neighbors
            dpj = self.last_pressure[p_i] / self.density_0 ** 2
            ## Akinci2012
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)
            self.pressure_accel[p_i] += d_v

        # Compute Ap and compute new pressure
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            Ap = 0.0
            dt2 = self.dt[None] * self.dt[None]
            accel_p_i = self.pressure_accel[p_i]
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                Ap += self.ps.m_V[p_j] * (accel_p_i - self.pressure_accel[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))
            # Boundary neighbors
            ## Akinci2012
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                Ap += self.ps.m_V[p_j] * (accel_p_i - self.pressure_accel[p_j]).dot(self.cubic_kernel_derivative(x_i - x_j))
            Ap *= dt2

            # Relaxed Jacobi
            # self.ps.pressure[p_i] = ti.max(self.last_pressure[p_i] + omega * (self.density_deviation[p_i] - Ap) / self.a_ii[p_i], 0.0)
            self.ps.pressure[p_i] = self.last_pressure[p_i] + omega * (self.density_deviation[p_i] - Ap) / self.a_ii[p_i]

        for p_i in range(self.ps.particle_num[None]):
            # Update the pressure
            self.last_pressure[p_i] = self.ps.pressure[p_i]



    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
            # Boundary neighbors
            ## Akinci2012
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
            self.ps.density[p_i] *= self.density_0

    @ti.kernel
    def compute_pressure_forces(self):
        # for p_i in range(self.ps.particle_num[None]):
        #     if self.ps.material[p_i] != self.ps.material_fluid:
        #         continue
        #     self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
        #     self.ps.pressure[p_i] = self.stiffness * (
        #                 ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                self.d_velocity[p_i].fill(0)
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])

            dpi = self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                dpj = self.ps.pressure[p_j] / self.ps.density[p_j] ** 2
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)

            # Boundary neighbors
            dpj = self.ps.pressure[p_i] / self.density_0 ** 2
            ## Akinci2012
            for j in range(self.ps.boundary_neighbors_num[p_i]):
                p_j = self.ps.boundary_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                       * self.cubic_kernel_derivative(x_i - x_j)

            self.d_velocity[p_i] += d_v

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                self.d_velocity[p_i].fill(0)
                continue
            x_i = self.ps.x[p_i]
            # Add body force
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[1] = self.g
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
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

        self.predict_advection()
        self.pressure_solve()

        self.compute_pressure_forces()
        self.advect()
