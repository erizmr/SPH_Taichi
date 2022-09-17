import numpy as np
import time
from functools import reduce
import taichi as ti


@ti.data_oriented
class SPHSolver:
    method_WCSPH = 0
    method_PCISPH = 1
    method_DFSPH = 2
    methods = {
        'WCSPH': method_WCSPH,
        'PCISPH': method_PCISPH,
        'DFSPH': method_DFSPH
    }
    material_fluid = 1
    material_bound = 0
    materials = {'fluid': material_fluid, 'bound': material_bound}

    def __init__(self,
                 res,
                 screen_to_world_ratio,
                 bound,
                 alpha=0.5,
                 dx=0.2,
                 max_num_particles=2**20,
                 padding=12,
                 max_time=5.0,
                 max_steps=50000,
                 dynamic_allocate=False,
                 adaptive_time_step=True,
                 method=0):
        self.method = method
        self.adaptive_time_step = adaptive_time_step
        self.dim = len(res)
        self.res = res
        self.screen_to_world_ratio = screen_to_world_ratio
        self.dynamic_allocate = dynamic_allocate
        # self.padding = padding  / screen_to_world_ratio
        self.padding = 2 * dx
        self.max_time = max_time
        self.max_steps = max_steps
        self.max_num_particles = max_num_particles

        self.g = -9.80  # Gravity
        self.alpha = alpha  # viscosity
        self.rho_0 = 1000.0  # reference density
        self.CFL_v = 0.25  # CFL coefficient for velocity
        self.CFL_a = 0.05  # CFL coefficient for acceleration

        self.df_fac = 1.3
        self.dx = dx  # Particle radius
        self.dh = self.dx * self.df_fac  # Smooth length
        # Declare dt as ti var for adaptive time step
        self.dt = ti.field(ti.f32, shape=())
        # Particle parameters
        self.m = self.dx**self.dim * self.rho_0

        self.grid_size = 2 * self.dh
        self.grid_pos = np.ceil(
            np.array(res) / self.screen_to_world_ratio /
            self.grid_size).astype(int)

        self.top_bound = bound[0]  # top_bound
        self.bottom_bound = bound[1]  # bottom_bound
        self.left_bound = bound[2]  # left_bound
        self.right_bound = bound[3]  # right_bound

        # ----------- WCSPH parameters-----------------
        # Pressure state function parameters(WCSPH)
        self.gamma = 7.0
        self.c_0 = 200.0

        # ----------- PCISPH parameters-----------------
        # Scaling factor for PCISPH
        self.s_f = ti.field(ti.f32, shape=())
        # Max iteration steps for pressure correction
        self.it = 0
        self.max_it = 0
        self.sub_max_iteration = 3
        self.rho_err = ti.Vector.field(1, dtype=ti.f32, shape=())
        self.max_rho_err = ti.Vector.field(1, dtype=ti.f32, shape=())

        # ----------- DFSPH parameters-----------------
        # Summing up the rho for all particles to compute the average rho
        self.sum_rho_err = ti.field(ti.f32, shape=())
        self.sum_drho = ti.field(ti.f32, shape=())

        # Dynamic Fill particles use
        self.source_bound = ti.Vector.field(self.dim, dtype=ti.f32, shape=2)
        self.source_velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        self.source_pressure = ti.Vector.field(1, dtype=ti.f32, shape=())
        self.source_density = ti.Vector.field(1, dtype=ti.f32, shape=())

        self.particle_num = ti.field(ti.i32, shape=())
        self.particle_positions = ti.Vector.field(self.dim, dtype=ti.f32)
        self.particle_velocity = ti.Vector.field(self.dim, dtype=ti.f32)
        self.particle_positions_new = ti.Vector.field(
            self.dim, dtype=ti.f32)  # Prediction values for P-C scheme use
        self.particle_velocity_new = ti.Vector.field(
            self.dim, dtype=ti.f32)  # Prediction values for P-C scheme use
        self.particle_pressure = ti.Vector.field(1, dtype=ti.f32)
        self.particle_pressure_acc = ti.Vector.field(
            self.dim, dtype=ti.f32)  # pressure force for PCISPH use
        self.particle_density = ti.Vector.field(1, dtype=ti.f32)
        self.particle_density_new = ti.Vector.field(
            1, dtype=ti.f32)  # Prediction values for P-C scheme use
        self.particle_alpha = ti.Vector.field(1, dtype=ti.f32)  # For DFSPH use
        self.particle_stiff = ti.Vector.field(1, dtype=ti.f32)  # For DFSPH use

        self.color = ti.field(dtype=ti.f32)
        self.material = ti.field(dtype=ti.f32)

        self.d_velocity = ti.Vector.field(self.dim, dtype=ti.f32)
        self.d_density = ti.Vector.field(1, dtype=ti.f32)

        self.grid_num_particles = ti.field(ti.i32)
        self.grid2particles = ti.field(ti.i32)
        self.particle_num_neighbors = ti.field(ti.i32)
        self.particle_neighbors = ti.field(ti.i32)

        self.max_num_particles_per_cell = 100
        self.max_num_neighbors = 100

        self.max_v = 0.0
        self.max_a = 0.0
        self.max_rho = 0.0
        self.max_pressure = 0.0

        if dynamic_allocate:
            ti.root.dynamic(ti.i, max_num_particles, 2**18).place(
                self.particle_positions, self.particle_velocity,
                self.particle_pressure, self.particle_density,
                self.particle_density_new, self.d_velocity, self.d_density,
                self.material, self.color, self.particle_positions_new,
                self.particle_velocity_new, self.particle_pressure_acc,
                self.particle_alpha, self.particle_stiff)
        else:
            # Allocate enough memory
            ti.root.dense(ti.i, 2**18).place(
                self.particle_positions, self.particle_velocity,
                self.particle_pressure, self.particle_density,
                self.particle_density_new, self.d_velocity, self.d_density,
                self.material, self.color, self.particle_positions_new,
                self.particle_velocity_new, self.particle_pressure_acc,
                self.particle_alpha, self.particle_stiff)

        if self.dim == 2:
            grid_snode = ti.root.dense(ti.ij, self.grid_pos)
            grid_snode.place(self.grid_num_particles)
            grid_snode.dense(ti.k, self.max_num_particles_per_cell).place(
                self.grid2particles)
        else:
            grid_snode = ti.root.dense(ti.ijk, self.grid_pos)
            grid_snode.place(self.grid_num_particles)
            grid_snode.dense(ti.l, self.max_num_particles_per_cell).place(
                self.grid2particles)

        nb_node = ti.root.dynamic(ti.i, max_num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j,
                      self.max_num_neighbors).place(self.particle_neighbors)

        # Initialize dt
        if method == SPHSolver.method_WCSPH:
            self.dt.from_numpy(
                np.array(0.1 * self.dh / self.c_0, dtype=np.float32))
            self.CFL_v = 0.20  # CFL coefficient for velocity
            self.CFL_a = 0.20  # CFL coefficient for acceleration

        if method == SPHSolver.method_PCISPH:
            self.s_f.from_numpy(np.array(1.0, dtype=np.float32))
            if self.adaptive_time_step:
                self.dt.from_numpy(np.array(0.0015, dtype=np.float32))
            else:
                self.dt.from_numpy(np.array(0.00015, dtype=np.float32))

        if method == SPHSolver.method_DFSPH:
            self.dt.from_numpy(
                np.array(1.0 * self.dh / self.c_0, dtype=np.float32))
            self.CFL_v = 0.30  # CFL coefficient for velocity
            self.CFL_a = 0.05  # CFL coefficient for acceleration

    @ti.func
    def compute_grid_index(self, pos):
        return (pos / (2 * self.dh)).cast(int)

    @ti.kernel
    def allocate_particles(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        # allocate particles to grid
        for p_i in range(self.particle_num[None]):
            # Compute the grid index
            cell = self.compute_grid_index(self.particle_positions[p_i])
            offs = self.grid_num_particles[cell].atomic_add(1)
            self.grid2particles[cell, offs] = p_i

    @ti.func
    def is_in_grid(self, c):
        res = 1
        for i in ti.static(range(self.dim)):
            res = ti.atomic_and(res, (0 <= c[i] < self.grid_pos[i]))
        return res

    @ti.func
    def is_fluid(self, p):
        # check fluid particle or bound particle
        return self.material[p]

    @ti.kernel
    def search_neighbors(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            nb_i = 0
            if self.is_fluid(p_i) == 1 or self.is_fluid(p_i) == 0:
                # Compute the grid index on the fly
                cell = self.compute_grid_index(self.particle_positions[p_i])
                for offs in ti.static(
                        ti.grouped(ti.ndrange(*((-1, 2), ) * self.dim))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check) == 1:
                        for j in range(self.grid_num_particles[cell_to_check]):
                            p_j = self.grid2particles[cell_to_check, j]
                            if nb_i < self.max_num_neighbors and p_j != p_i and (
                                    pos_i - self.particle_positions[p_j]
                            ).norm() < self.dh * 2.00:
                                self.particle_neighbors[p_i, nb_i] = p_j
                                nb_i.atomic_add(1)
            self.particle_num_neighbors[p_i] = nb_i

    @ti.func
    def cubic_kernel(self, r, h):
        # value of cubic spline smoothing kernel
        k = 10. / (7. * np.pi * h**self.dim)
        q = r / h
        # assert q >= 0.0  # Metal backend is not happy with assert
        res = ti.cast(0.0, ti.f32)
        if q <= 1.0:
            res = k * (1 - 1.5 * q**2 + 0.75 * q**3)
        elif q < 2.0:
            res = k * 0.25 * (2 - q)**3
        return res

    @ti.func
    def cubic_kernel_derivative(self, r, h):
        # derivative of cubic spline smoothing kernel
        k = 10. / (7. * np.pi * h**self.dim)
        q = r / h
        # assert q > 0.0
        res = ti.cast(0.0, ti.f32)
        if q < 1.0:
            res = (k / h) * (-3 * q + 2.25 * q**2)
        elif q < 2.0:
            res = -0.75 * (k / h) * (2 - q)**2
        return res

    @ti.func
    def rho_derivative(self, ptc_i, ptc_j, r, r_mod):
        # density delta, i.e. divergence
        return self.m * self.cubic_kernel_derivative(r_mod, self.dh) \
               * (self.particle_velocity[ptc_i] - self.particle_velocity[ptc_j]).dot(r / r_mod)

    @ti.func
    def p_update(self, rho, rho_0=1000.0, gamma=7.0, c_0=20.0):
        # Weakly compressible, tait function
        b = rho_0 * c_0**2 / gamma
        return b * ((rho / rho_0)**gamma - 1.0)

    @ti.func
    def pressure_force(self, ptc_i, ptc_j, r, r_mod):
        # Compute the pressure force contribution, Symmetric Formula
        res = ti.Vector([0.0 for _ in range(self.dim)])
        res = -self.m * (self.particle_pressure[ptc_i][0] / self.particle_density[ptc_i][0] ** 2
                         + self.particle_pressure[ptc_j][0] / self.particle_density[ptc_j][0] ** 2) \
              * self.cubic_kernel_derivative(r_mod, self.dh) * r / r_mod
        return res

    @ti.func
    def viscosity_force(self, ptc_i, ptc_j, r, r_mod):
        # Compute the viscosity force contribution, artificial viscosity
        res = ti.Vector([0.0 for _ in range(self.dim)])
        v_xy = (self.particle_velocity[ptc_i] -
                self.particle_velocity[ptc_j]).dot(r)
        if v_xy < 0:
            # Artifical viscosity
            vmu = -2.0 * self.alpha * self.dx * self.c_0 / (
                self.particle_density[ptc_i][0] +
                self.particle_density[ptc_j][0])
            res = -self.m * vmu * v_xy / (
                r_mod**2 + 0.01 * self.dx**2) * self.cubic_kernel_derivative(
                    r_mod, self.dh) * r / r_mod
        return res

    @ti.func
    def simulate_collisions(self, ptc_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.3
        self.particle_positions[ptc_i] += vec * d
        self.particle_velocity[ptc_i] -= (
            1.0 + c_f) * self.particle_velocity[ptc_i].dot(vec) * vec
        if self.method == SPHSolver.method_DFSPH:
            self.particle_velocity_new[ptc_i] -= (
                1.0 + c_f) * self.particle_velocity_new[ptc_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary(self):
        # TODO: only handle 2D case currently
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                pos = self.particle_positions[p_i]
                if pos[0] < self.left_bound + 0.5 * self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([1.0, 0.0]),
                        self.left_bound + 0.5 * self.padding - pos[0])
                if pos[0] > self.right_bound - 0.5 * self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([-1.0, 0.0]),
                        pos[0] - self.right_bound + 0.5 * self.padding)
                if pos[1] > self.top_bound - self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, -1.0]),
                        pos[1] - self.top_bound + self.padding)
                if pos[1] < self.bottom_bound + self.padding:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 1.0]),
                        self.bottom_bound + self.padding - pos[1])

    @ti.kernel
    def wc_compute_deltas(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.dim)])
            d_rho = 0.0
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]

                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = ti.max(r.norm(), 1e-5)

                # Compute Density change
                d_rho += self.rho_derivative(p_i, p_j, r, r_mod)

                if self.is_fluid(p_i) == 1:
                    # Compute Viscosity force contribution
                    d_v += self.viscosity_force(p_i, p_j, r, r_mod)

                    # Compute Pressure force contribution
                    d_v += self.pressure_force(p_i, p_j, r, r_mod)

            # Add body force
            if self.is_fluid(p_i) == 1:
                val = [0.0 for _ in range(self.dim - 1)]
                val.extend([self.g])
                d_v += ti.Vector(val)
            self.d_velocity[p_i] = d_v
            self.d_density[p_i][0] = d_rho

    @ti.kernel
    def wc_update_time_step(self):
        # Simple Forward Euler currently
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                self.particle_velocity[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.particle_positions[
                    p_i] += self.dt[None] * self.particle_velocity[p_i]
            self.particle_density[p_i][0] += self.dt[None] * self.d_density[p_i][0]
            self.particle_pressure[p_i][0] = self.p_update(
                self.particle_density[p_i][0], self.rho_0, self.gamma,
                self.c_0)

    @ti.kernel
    def pci_scaling_factor(self):
        grad_sum = ti.Vector([0.0 for _ in range(self.dim)])
        grad_dot_sum = 0.0
        range_num = ti.cast(self.dh * 2.0 / self.dx, ti.i32)
        half_range = ti.cast(0.5 * range_num, ti.i32)
        # TODO: only handle 2D case currently
        for x in range(-half_range, half_range):
            for y in range(-half_range, half_range):
                r = ti.Vector([-x * self.dx, -y * self.dx])
                r_mod = r.norm()
                if 2.0 * self.dh > r_mod > 1e-5:
                    grad = self.cubic_kernel_derivative(r_mod,
                                                        self.dh) * r / r_mod
                    grad_sum += grad
                    grad_dot_sum += grad.dot(grad)

        beta = 2 * (self.dt[None] * self.m / self.rho_0)**2
        self.s_f[None] = 1.0 / ti.max(
            beta * (grad_sum.dot(grad_sum) + grad_dot_sum), 1e-6)

    @ti.kernel
    def pci_pos_vel_prediction(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                self.particle_velocity_new[
                    p_i] = self.particle_velocity[p_i] + self.dt[None] * (
                        self.d_velocity[p_i] + self.particle_pressure_acc[p_i])
                self.particle_positions_new[p_i] = self.particle_positions[
                    p_i] + self.dt[None] * self.particle_velocity_new[p_i]
        # Initialize the max_rho_err
        self.max_rho_err[None][0] = 0.0

    @ti.kernel
    def pci_update_pressure(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions_new[p_i]
            d_rho = 0.0
            curr_rho = 0.0
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions_new[p_j]

                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()
                if r_mod > 1e-5:
                    # Compute Density change
                    d_rho += self.cubic_kernel_derivative(r_mod, self.dh) \
                    * (self.particle_velocity_new[p_i] - self.particle_velocity_new[p_j]).dot(r / r_mod)

            self.d_density[p_i][0] = d_rho
            # Avoid negative density variation
            self.rho_err[None][0] = self.particle_density[p_i][
                0] + self.dt[None] * d_rho - self.rho_0
            self.max_rho_err[None][0] = max(abs(self.rho_err[None][0]),
                                            self.max_rho_err[None][0])
            self.particle_pressure[p_i][
                0] += self.s_f[None] * self.rho_err[None][0]

    @ti.kernel
    def pci_update_pressure_force(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions_new[p_i]
            d_vp = ti.Vector([0.0 for _ in range(self.dim)])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions_new[p_j]
                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()
                if r_mod > 1e-5:
                    # Compute Pressure force contribution
                    d_vp += self.pressure_force(p_i, p_j, r, r_mod)
            self.particle_pressure_acc[p_i] = d_vp

    def pci_pc_iteration(self):
        self.pci_pos_vel_prediction()
        self.pci_update_pressure()
        self.pci_update_pressure_force()

    @ti.kernel
    def pci_compute_deltas(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.dim)])

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]

                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()

                if r_mod > 1e-5 and self.is_fluid(p_i) == 1:
                    # Compute Viscosity force contribution
                    d_v += self.viscosity_force(p_i, p_j, r, r_mod)

            # Add body force
            if self.is_fluid(p_i) == 1:
                val = [0.0 for _ in range(self.dim - 1)]
                val.extend([self.g])
                d_v += ti.Vector(val)
            self.d_velocity[p_i] = d_v
            # Initialize the pressure
            self.particle_pressure[p_i][0] = 0.0
            self.particle_pressure_acc[p_i] = ti.Vector(
                [0.0 for _ in range(self.dim)])

    @ti.kernel
    def pci_update_time_step(self):
        # Final position and velocity update
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                self.particle_velocity[p_i] += self.dt[None] * (
                    self.d_velocity[p_i] + self.particle_pressure_acc[p_i])
                self.particle_positions[
                    p_i] += self.dt[None] * self.particle_velocity[p_i]
            # Update density
            self.particle_density[p_i][0] += self.dt[None] * self.d_density[p_i][0]

    @ti.kernel
    def df_compute_deltas(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.dim)])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]

                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()

                if r_mod > 1e-4 and self.is_fluid(p_i) == 1:
                    # Compute Viscosity force contribution
                    d_v += self.viscosity_force(p_i, p_j, r, r_mod)

            # Add body force
            if self.is_fluid(p_i) == 1:
                val = [0.0 for _ in range(self.dim - 1)]
                val.extend([self.g])
                d_v += ti.Vector(val)
            self.d_velocity[p_i] = d_v

    @ti.kernel
    def df_predict_velocities(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                self.particle_velocity_new[p_i] = self.particle_velocity[
                    p_i] + self.dt[None] * self.d_velocity[p_i]

    @ti.kernel
    def df_correct_density_predict(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            d_rho = 0.0
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]

                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = ti.max(r.norm(), 1e-5)

                # Compute Density change
                if self.is_fluid(p_j) == 1:
                    d_rho += self.m * self.cubic_kernel_derivative(r_mod, self.dh) \
                             * (self.particle_velocity_new[p_i] - self.particle_velocity_new[p_j]).dot(r / r_mod)
                elif self.is_fluid(p_j) == 0:
                    d_rho += self.m * self.cubic_kernel_derivative(r_mod, self.dh) \
                             * self.particle_velocity_new[p_i].dot(r / r_mod)

            # Compute the predicted density rho star
            self.particle_density_new[p_i][
                0] = self.particle_density[p_i][0] + self.dt[None] * d_rho

            # Only consider compressed
            err = ti.max(0.0, self.particle_density_new[p_i][0] - self.rho_0)
            self.particle_stiff[p_i][0] = err * self.particle_alpha[p_i][0]

            # Compute the density error sum for average use
            self.sum_rho_err[None] += err

    @ti.kernel
    def df_correct_density_adapt_vel(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.dim)])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]
                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()

                if r_mod > 1e-4:
                    if self.is_fluid(p_j) == 1:
                        d_v += self.m * (self.particle_stiff[p_i][0] +
                                         self.particle_stiff[p_j][0]
                                         ) * self.cubic_kernel_derivative(
                                             r_mod, self.dh) * r / r_mod
                    elif self.is_fluid(p_j) == 0:
                        d_v += self.m * self.particle_stiff[
                            p_i][0] * self.cubic_kernel_derivative(
                                r_mod, self.dh) * r / r_mod

            # Predict velocity using pressure contribution
            self.particle_velocity_new[p_i] += d_v / ti.max(self.dt[None], 1e-5)
            # Store the pressure contribution to acceleration
            self.particle_pressure_acc[p_i] = d_v / ti.max(
                self.dt[None] * self.dt[None], 1e-8)

    @ti.kernel
    def df_update_positions(self):
        for p_i in range(self.particle_num[None]):
            # Update the positions
            if self.is_fluid(p_i) == 1:
                self.particle_positions[
                    p_i] += self.dt[None] * self.particle_velocity_new[p_i]

    @ti.kernel
    def df_compute_density_alpha(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            grad_sum = ti.Vector([0.0 for _ in range(self.dim)])
            grad_square_sum = 0.0
            curr_rho = 0.0
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]
                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()
                if r_mod > 1e-4:
                    # Compute the grad sum and grad square sum for denominator alpha
                    grad_val = self.m * self.cubic_kernel_derivative(
                        r_mod, self.dh) * r / r_mod
                    grad_sum += grad_val

                    if self.is_fluid(p_j) == 1:
                        grad_square_sum += grad_val.dot(grad_val)
                    # Compute the density
                    curr_rho += self.m * self.cubic_kernel(r_mod, self.dh)
            # Update the density
            self.particle_density[p_i][0] = curr_rho
            # Set a threshold of 10^-6 to avoid instability
            self.particle_alpha[p_i][0] = -1.0 / ti.max(
                grad_sum.dot(grad_sum) + grad_square_sum, 1e-6)

    @ti.kernel
    def df_correct_divergence_compute_drho(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            d_rho = 0.0
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]
                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()

                if r_mod > 1e-4:
                    if self.is_fluid(p_j) == 1:
                        d_rho += self.m * (
                            self.particle_velocity_new[p_i] -
                            self.particle_velocity_new[p_j]).dot(
                                r / r_mod) * self.cubic_kernel_derivative(
                                    r_mod, self.dh)
                    # Boundary particles have no contributions to pressure force
                    elif self.is_fluid(p_j) == 0:
                        d_rho += self.m * self.particle_velocity_new[p_i].dot(
                            r / r_mod) * self.cubic_kernel_derivative(
                                r_mod, self.dh)

            self.d_density[p_i][0] = ti.max(d_rho, 0.0)

            # if the density is less than the rest density, skip update in this iteration
            if self.particle_density[p_i][0] + self.dt[None] * self.d_density[p_i][
                    0] < self.rho_0 and self.particle_density[p_i][
                        0] < self.rho_0:
                self.d_density[p_i][0] = 0.0
            self.particle_stiff[p_i][
                0] = self.d_density[p_i][0] * self.particle_alpha[p_i][0]

            # Compute the predicted density rho star
            self.sum_drho[None] += self.d_density[p_i][0]

    @ti.kernel
    def df_correct_divergence_adapt_vel(self):
        for p_i in range(self.particle_num[None]):
            pos_i = self.particle_positions[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.dim)])

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]
                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()

                if r_mod > 1e-5:
                    if self.is_fluid(p_j) == 1:
                        d_v += self.m * (self.particle_stiff[p_i][0] +
                                         self.particle_stiff[p_j][0]
                                         ) * self.cubic_kernel_derivative(
                                             r_mod, self.dh) * r / r_mod
                    elif self.is_fluid(p_j) == 0:
                        d_v += self.m * self.particle_stiff[
                            p_i][0] * self.cubic_kernel_derivative(
                                r_mod, self.dh) * r / r_mod

            # Predict velocity using pressure contribution, dt has been cancelled
            self.particle_velocity_new[p_i] += d_v
            # Store the pressure contribution to acceleration
            self.particle_pressure_acc[p_i] = d_v / self.dt[None]

    @ti.kernel
    def df_update_velocities(self):
        for p_i in range(self.particle_num[None]):
            if self.is_fluid(p_i) == 1:
                # Update the velocities from prediction values to next step
                self.particle_velocity[p_i] = self.particle_velocity_new[p_i]

    def sim_info(self, output=False):
        print("Time step: ", self.dt[None])
        print(
            "Domain: (%s, %s, %s, %s)" %
            (self.x_min, self.x_max, self.y_min, self.y_max), )
        print("Fluid area: (%s, %s, %s, %s)" %
              (self.left_bound, self.right_bound, self.bottom_bound,
               self.top_bound))
        print("Grid: ", self.grid_pos)

    def sim_info_realtime(self, frame, t, curr_start, curr_end, total_start):
        print(
            "Step: %d, physics time: %s, progress: %s %%, time used: %s, total time used: %s"
            % (frame, t,
               100 * np.max([t / self.max_time, frame / self.max_steps]),
               curr_end - curr_start, curr_end - total_start))
        print(
            "Max velocity: %s, Max acceleration: %s, Max density: %s, Max pressure: %s"
            % (self.max_v, self.max_a, self.max_rho, self.max_pressure))
        if self.method == SPHSolver.methods['PCISPH']:
            print("Max iter: %d, Max density variation: %s" %
                  (self.max_it, self.max_rho_err[None][0]))
        if self.method == SPHSolver.methods['DFSPH']:
            print("Max iter: %d, Max density variation: %s" %
                  (self.it, self.sum_rho_err[None] / self.particle_num[None]))
            print("Max iter: %d, Max divergence variation: %s" %
                  (self.it, self.sum_drho[None] / self.particle_num[None]))
        print("Adaptive time step: ", self.dt[None])

    def adaptive_step(self):
        total_num = self.particle_num[None]
        self.max_v = np.max(
            np.linalg.norm(self.particle_velocity.to_numpy()[:total_num],
                           2,
                           axis=1))
        # CFL analysis, constrained by v_max
        dt_cfl = self.CFL_v * self.dh / self.max_v

        self.max_a = np.max(
            np.linalg.norm((self.d_velocity.to_numpy() +
                            self.particle_pressure_acc.to_numpy())[:total_num],
                           2,
                           axis=1))
        # Constrained by a_max
        dt_f = self.CFL_a * np.sqrt(self.dh / self.max_a)

        if self.adaptive_time_step and self.method == SPHSolver.method_DFSPH:
            self.dt[None] = np.min([dt_cfl, dt_f])
            return

        self.max_rho = np.max(self.particle_density.to_numpy()[:total_num])
        self.max_pressure = np.max(
            self.particle_pressure.to_numpy()[:total_num])
        dt_a = 0.20 * self.dh / (self.c_0 * np.sqrt(
            (self.max_rho / self.rho_0)**self.gamma))

        if self.adaptive_time_step and self.method == SPHSolver.method_WCSPH:
            self.dt[None] = np.min([dt_cfl, dt_f, dt_a])
        if self.adaptive_time_step and self.method == SPHSolver.method_PCISPH:
            self.dt[None] = np.min([dt_cfl, dt_f])

    def step(self, frame, t, total_start):
        curr_start = time.process_time()

        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles()
        self.search_neighbors()

        if self.method == SPHSolver.methods['WCSPH']:
            # Compute deltas
            self.wc_compute_deltas()
            # timestep Update
            self.wc_update_time_step()
        elif self.method == SPHSolver.methods['PCISPH']:
            # Compute viscosity and gravity force
            self.pci_compute_deltas()
            # Compute the scaling factor for pressure update
            self.pci_scaling_factor()
            # Start density prediction-correction iteration process
            self.it = 0
            self.max_it = 0
            # 1% rho_0
            while self.max_rho_err[None][
                    0] >= 0.01 * self.rho_0 or self.it < self.sub_max_iteration:
                self.pci_pc_iteration()
                self.it += 1
                self.max_it += 1
                self.max_it = max(self.it, self.max_it)
                if self.it > 1000:
                    print(
                        "Warning: PCISPH density does not converge, iterated %d steps"
                        % self.it)
                    break
            # Compute new velocity, position and density
            self.pci_update_time_step()
        elif self.method == SPHSolver.methods['DFSPH']:
            # The step order changed a bit to fit the step function
            # update rho and alpha
            self.df_compute_density_alpha()
            # Correct divergence error
            self.it = 0
            self.sum_drho[None] = 0.0
            # 1% rho_0
            while self.sum_drho[None] >= 0.01 * self.particle_num[
                    None] * self.rho_0 or self.it < 1:
                self.sum_drho[None] = 0.0
                self.df_correct_divergence_compute_drho()
                self.df_correct_divergence_adapt_vel()
                self.it += 1
                if self.it > 1000:
                    print(
                        "Warning: DFSPH divergence does not converge, iterated %d steps"
                        % self.it)
                    break
            # Update velocities v
            self.df_update_velocities()
            # Compute non-pressure forces
            self.df_compute_deltas()
            # Adapt time step
            self.adaptive_step()
            # Predict velocities v_star
            self.df_predict_velocities()
            # Correct density error
            self.it = 0
            self.sum_rho_err[None] = 0.0
            # 1% rho_0
            while self.sum_rho_err[None] >= 0.01 * self.particle_num[
                    None] * self.rho_0 or self.it < 2:
                self.sum_rho_err[None] = 0.0
                self.df_correct_density_predict()
                self.df_correct_density_adapt_vel()
                self.it += 1
                if self.it > 1000:
                    print(
                        "Warning: DFSPH density does not converge, iterated %d steps"
                        % self.it)
                    break
            self.df_update_positions()

        # Handle potential leak particles
        self.enforce_boundary()
        if self.method != SPHSolver.methods['DFSPH']:
            self.adaptive_step()

        curr_end = time.process_time()

        if frame % 10 == 0:
            self.sim_info_realtime(frame, t, curr_start, curr_end, total_start)
        return self.dt[None]

    @ti.func
    def fill_particle(self, i, x, material, color, velocity, pressure,
                      density):
        self.particle_positions[i] = x
        self.particle_positions_new[i] = x
        self.particle_velocity[i] = velocity
        self.particle_velocity_new[i] = velocity
        self.d_velocity[i] = ti.Vector([0.0 for _ in range(self.dim)])
        self.particle_pressure[i] = pressure
        self.particle_pressure_acc[i] = ti.Vector([0.0 for _ in range(self.dim)])
        self.particle_density[i] = density
        self.particle_density_new[i] = density
        self.d_density[i][0] = 0.0
        self.particle_alpha[i][0] = 0.0
        self.particle_stiff[i][0] = 0.0
        self.color[i] = color
        self.material[i] = material

    @ti.kernel
    def fill(self, new_particles: ti.i32, new_positions: ti.types.ndarray(),
             new_material: ti.i32, color: ti.i32):
        for i in range(self.particle_num[None],
                       self.particle_num[None] + new_particles):
            self.material[i] = new_material
            x = ti.Vector.zero(ti.f32, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = new_positions[k, i - self.particle_num[None]]
            self.fill_particle(i, x, new_material, color,
                               self.source_velocity[None],
                               self.source_pressure[None],
                               self.source_density[None])

    def set_source_velocity(self, velocity):
        if velocity is not None:
            velocity = list(velocity)
            assert len(velocity) == self.dim
            self.source_velocity[None] = velocity
        else:
            for i in range(self.dim):
                self.source_velocity[None][i] = 0

    def set_source_pressure(self, pressure):
        if pressure is not None:
            self.source_pressure[None] = pressure
        else:
            self.source_pressure[None][0] = 0.0

    def set_source_density(self, density):
        if density is not None:
            self.source_density[None] = density
        else:
            self.source_density[None][0] = 0.0

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.dx))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        assert self.particle_num[
            None] + num_new_particles <= self.max_num_particles

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(
            -1, reduce(lambda x, y: x * y, list(new_positions.shape[1:])))
        print(new_positions.shape)

        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            self.source_bound[1][i] = cube_size[i]

        self.set_source_velocity(velocity=velocity)
        self.set_source_pressure(pressure=pressure)
        self.set_source_density(density=density)

        self.fill(num_new_particles, new_positions, material, color)
        # Add to current particles count
        self.particle_num[None] += num_new_particles

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in range(self.particle_num[None]):
            np_x[i] = input_x[i]

    def particle_info(self):
        np_x = np.ndarray((self.particle_num[None], self.dim),
                          dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.particle_positions)
        np_v = np.ndarray((self.particle_num[None], self.dim),
                          dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.particle_velocity)
        np_material = np.ndarray((self.particle_num[None], ), dtype=np.int32)
        self.copy_dynamic(np_material, self.material)
        np_color = np.ndarray((self.particle_num[None], ), dtype=np.int32)
        self.copy_dynamic(np_color, self.color)
        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }
