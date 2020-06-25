# SPH taichi implementation by mzhang
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import count
from functools import reduce
import taichi as ti

ti.init(arch=ti.cpu)


@ti.data_oriented
class SPHSolver:
    method_WCSPH = 0
    method_PCISPH = 1
    methods = {'WCSPH': method_WCSPH, 'PCISPH': method_WCSPH}
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
                 max_time=10000,
                 max_steps=1000,
                 dynamic_allocate=False):
        self.dim = len(res)
        self.res = res
        self.screen_to_world_ratio = screen_to_world_ratio
        self.dynamic_allocate = dynamic_allocate
        # self.padding = padding  / screen_to_world_ratio
        self.padding = 2 * dx
        # Solver parameters
        self.max_time = max_time
        self.max_steps = max_steps

        self.g = -9.80  # Gravity
        self.alpha = alpha  # viscosity
        self.rho_0 = 1000.0  # reference density
        self.CFL = 0.20  # CFL coefficient

        self.df_fac = 1.3
        self.dx = dx
        self.dh = self.dx * self.df_fac
        self.kernel_norm = 10. / (7. * np.pi * self.dh**2)

        # Pressure state function parameters(WCSPH)
        self.gamma = 7.0
        self.c_0 = 100.0

        # Compute dt, a naive initial test value
        self.dt = 0.1 * self.dh / self.c_0

        # Particle parameters
        self.m = self.dx**2 * self.rho_0
        self.max_num_particles = max_num_particles

        self.grid_size = 2 * self.dh
        self.grid_pos = np.ceil(
            np.array(res) / self.screen_to_world_ratio /
            self.grid_size).astype(int)

        self.top_bound = bound[0]  # top_bound
        self.bottom_bound = bound[1]  # bottom_bound
        self.left_bound = bound[2]  # left_bound
        self.right_bound = bound[3]  # right_bound

        # Dynamic Fill particles use
        self.source_bound = ti.Vector(self.dim, dt=ti.f32, shape=2)
        self.source_velocity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.source_pressure = ti.Vector(1, dt=ti.f32, shape=())
        self.source_density = ti.Vector(1, dt=ti.f32, shape=())

        self.particle_num = ti.var(ti.i32, shape=())
        self.particle_positions = ti.Vector(self.dim, dt=ti.f32)
        self.particle_velocity = ti.Vector(self.dim, dt=ti.f32)
        self.particle_pressure = ti.Vector(1, dt=ti.f32)
        self.particle_density = ti.Vector(1, dt=ti.f32)

        self.color = ti.var(dt=ti.f32)
        self.material = ti.var(dt=ti.f32)

        self.d_velocity = ti.Vector(self.dim, dt=ti.f32)
        self.d_density = ti.Vector(1, dt=ti.f32)

        self.grid_num_particles = ti.var(ti.i32)
        self.grid2particles = ti.var(ti.i32)
        self.particle_num_neighbors = ti.var(ti.i32)
        self.particle_neighbors = ti.var(ti.i32)

        self.max_num_particles_per_cell = 100
        self.max_num_neighbors = 100

        self.max_v = 0.0
        self.max_a = 0.0
        self.max_rho = 0.0
        self.max_pressure = 0.0

        if dynamic_allocate:
            ti.root.dynamic(ti.i, max_num_particles, 2**18).place(
                self.particle_positions, self.particle_velocity,
                self.particle_pressure, self.particle_density, self.d_velocity,
                self.d_density, self.material, self.color)
        else:
            ti.root.dense(ti.i, 3890).place(self.particle_positions,
                                            self.particle_velocity,
                                            self.particle_pressure,
                                            self.particle_density,
                                            self.d_velocity, self.d_density,
                                            self.material, self.color)

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

    @ti.func
    def compute_grid_index(self, pos):
        return (pos / (2 * self.dh)).cast(int)

    @ti.kernel
    def allocate_particles(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        # allocate particles to grid
        for p_i in self.particle_positions:
            # Compute the grid index
            cell = self.compute_grid_index(self.particle_positions[p_i])
            offs = self.grid_num_particles[cell].atomic_add(1)
            self.grid2particles[cell, offs] = p_i

    @ti.func
    def is_in_grid(self, c):
        res = 1
        for i in ti.static(range(self.dim)):
            res = ti.atomic_and(res, (0 <= c[i] and c[i] < self.grid_pos[i]))
        return res

    @ti.func
    def is_fluid(self, p):
        # check fluid particle or bound particle
        return self.material[p]

    def stencil_range(self):
        # for item in ti.ndrange(*((-1, 2)*self.dim)):
        return ti.ndrange(*(((-1, 2), ) * self.dim))

    @ti.kernel
    def search_neighbors(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        for p_i in self.particle_positions:
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
        k = 10. / (7. * np.pi * h**2)
        q = r / h
        assert q >= 0.0
        res = ti.cast(0.0, ti.f32)
        if q <= 1.0:
            res = k * (1 - 1.5 * q**2 + 0.75 * q**3)
        elif q < 2.0:
            res = k * 0.25 * (2 - q)**3
        return res

    @ti.func
    def cubic_kernel_derivative(self, r, h):
        # derivative of cubcic spline smoothing kernel
        k = 10. / (7. * np.pi * h**2)
        q = r / h
        assert q > 0.0
        res = ti.cast(0.0, ti.f32)
        if q < 1.0:
            res = (k / h) * (-3 * q + 2.25 * q**2)
        elif q < 2.0:
            res = -0.75 * (k / h) * (2 - q)**2
        return res

    @ti.func
    def rho_derivative(self, ptc_i, ptc_j, r, r_mod):
        # density delta
        return self.m * self.cubic_kernel_derivative(r_mod, self.dh) \
               * (self.particle_velocity[ptc_i] - self.particle_velocity[ptc_j]).dot(r / r_mod)

    @ti.func
    def p_update(self, rho, rho_0=1000, gamma=7.0, c_0=20.0):
        # Weakly compressible, tait function
        b = rho_0 * c_0**2 / gamma
        return b * ((rho / rho_0)**gamma - 1.0)

    @ti.func
    def pressure_force(self, ptc_i, ptc_j, r, r_mod, mirror_pressure=0):
        # Compute the pressure force contribution, Symmetric Formula
        res = ti.Vector([0.0, 0.0])
        res = -self.m * (self.particle_pressure[ptc_i][0] / self.particle_density[ptc_i][0] ** 2
                         + self.particle_pressure[ptc_j][0] / self.particle_density[ptc_j][0] ** 2) \
              * self.cubic_kernel_derivative(r_mod, self.dh) * r / r_mod
        return res

    @ti.func
    def viscosity_force(self, ptc_i, ptc_j, r, r_mod):
        # Compute the viscosity force contribution, artificial viscosity
        res = ti.Vector([0.0, 0.0])
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
    def simualte_collisions(self, ptc_i, vec, d):
        # Collision factor, assume roughly 50% velocity loss after collision, i.e. m_f /(m_f + m_b)
        c_f = 0.5
        self.particle_positions[ptc_i] += vec * d
        self.particle_velocity[ptc_i] -= (
            1.0 + c_f) * self.particle_velocity[ptc_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary(self):
        for p_i in self.particle_positions:
            if self.is_fluid(p_i) == 1:
                pos = self.particle_positions[p_i]
                if pos[0] < self.left_bound + 0.5 * self.padding:
                    self.simualte_collisions(
                        p_i, ti.Vector([1.0, 0.0]),
                        self.left_bound + 0.5 * self.padding - pos[0])
                if pos[0] > self.right_bound - 0.5 * self.padding:
                    self.simualte_collisions(
                        p_i, ti.Vector([-1.0, 0.0]),
                        pos[0] - self.right_bound + 0.5 * self.padding)
                if pos[1] > self.top_bound - self.padding:
                    self.simualte_collisions(
                        p_i, ti.Vector([0.0, -1.0]),
                        pos[1] - self.top_bound + self.padding)
                if pos[1] < self.bottom_bound + self.padding:
                    self.simualte_collisions(
                        p_i, ti.Vector([0.0, 1.0]),
                        self.bottom_bound + self.padding - pos[1])

    @ti.kernel
    def compute_deltas(self):
        for p_i in self.particle_positions:
            pos_i = self.particle_positions[p_i]
            d_v = ti.Vector([0.0, 0.0])
            d_rho = 0.0
            # if self.is_fluid(p_i) == 1:
            #     d_v = ti.Vector([0.0, -9.8])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]

                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()

                # Compute Density change
                d_rho += self.rho_derivative(p_i, p_j, r, r_mod)

                if self.is_fluid(p_i) == 1:
                    # Compute Viscosity force contribution
                    d_v += self.viscosity_force(p_i, p_j, r, r_mod)

                    # Compute Pressure force contribution
                    d_v += self.pressure_force(p_i, p_j, r, r_mod)

            # Add body force
            if self.is_fluid(p_i) == 1:
                d_v += ti.Vector([0.0, self.g])
            self.d_velocity[p_i] = d_v
            self.d_density[p_i][0] = d_rho

    @ti.kernel
    def update_time_step(self):
        # Simple Forward Euler currently
        for p_i in self.particle_positions:
            if self.is_fluid(p_i) == 1:
                self.particle_velocity[p_i] += self.dt * self.d_velocity[p_i]
                self.particle_positions[
                    p_i] += self.dt * self.particle_velocity[p_i]
            self.particle_density[p_i][0] += self.dt * self.d_density[p_i][0]
            self.particle_pressure[p_i][0] = self.p_update(
                self.particle_density[p_i][0], self.rho_0, self.gamma,
                self.c_0)

    def sim_info(self, output=False):
        print("Time step: ", self.dt)
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
        print("Adaptive time step: ", self.dt)

    def adaptive_step(self):
        self.max_v = np.max(
            np.linalg.norm(self.particle_velocity.to_numpy(), 2, axis=1))
        self.max_a = np.max(
            np.linalg.norm(self.d_velocity.to_numpy(), 2, axis=1))
        self.max_rho = np.max(self.particle_density.to_numpy())
        self.max_pressure = np.max(self.particle_pressure.to_numpy())

        # CFL analysis, adaptive dt
        dt_cfl = self.dh / self.max_v
        dt_f = np.sqrt(self.dh / self.max_a)
        dt_a = self.dh / (self.c_0 * np.sqrt(
            (self.max_rho / self.rho_0)**self.gamma))
        self.dt = self.CFL * np.min([dt_cfl, dt_f, dt_a])

    def step(self, frame, t, total_start):
        curr_start = time.process_time()

        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles()
        self.search_neighbors()
        # Compute deltas
        self.compute_deltas()
        # timestep Update
        self.update_time_step()
        # Handle potential leak particles
        self.enforce_boundary()
        self.adaptive_step()

        curr_end = time.process_time()

        if frame % 10 == 0:
            self.sim_info_realtime(frame, t, curr_start, curr_end, total_start)
        return self.dt

    @ti.func
    def fill_particle(self, i, x, material, color, velocity, pressure,
                      density):
        self.particle_positions[i] = x
        self.particle_velocity[i] = velocity
        self.particle_pressure[i] = pressure
        self.particle_density[i] = density
        self.color[i] = color
        self.material[i] = material

    @ti.kernel
    def fill(self, new_particles: ti.i32, new_positions: ti.ext_arr(),
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
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.particle_positions:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.particle_positions:
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


def main():

    res = (400, 400)
    screen_to_world_ratio = 35
    dynamic_allocate = True
    save_frames = True

    gui = ti.GUI('WCSPH2D', res, background_color=0x112F41)
    dx = 0.1
    u, b, l, r = np.array([res[1], 0, 0, res[0]]) / screen_to_world_ratio
    sph = SPHSolver(res,
                    screen_to_world_ratio, [u, b, l, r],
                    alpha=0.30,
                    dx=dx,
                    max_steps=50000,
                    dynamic_allocate=dynamic_allocate)

    # Add fluid particles
    sph.add_cube(lower_corner=[res[0] / 2 / screen_to_world_ratio - 3, 4 * dx],
                 cube_size=[6, 6],
                 velocity=[0.0, -5.0],
                 density=[1000],
                 material=SPHSolver.material_fluid)

    # Add bottom boundary
    sph.add_cube(lower_corner=[0.0, 0.0],
                 cube_size=[res[0] / screen_to_world_ratio, 2 * dx],
                 velocity=[0.0, 0.0],
                 density=[1000],
                 material=SPHSolver.material_bound)

    colors = np.array([0xED553B, 0x068587, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)

    t = 0.0
    frame = 0
    total_start = time.process_time()
    while frame < 50000 and t < 30:
        dt = sph.step(frame, t, total_start)
        particles = sph.particle_info()

        # if dynamic_allocate and frame < 50 and frame % 10 == 0:
        #     sph.add_cube(lower_corner=[3, 3],
        #                  cube_size=[0.4, 0.4],
        #                  velocity=[-5.0, 0.0],
        #                  density=[1000],
        #                  material=SPHSolver.material_fluid)

        for pos in particles['position']:
            for j in range(len(res)):
                pos[j] *= screen_to_world_ratio / res[j]

        gui.circles(particles['position'],
                    radius=1.5,
                    color=colors[particles['material']])
        if frame % 50 == 0:
            gui.show(f'{frame:06d}.png' if save_frames else None)

        frame += 1
        t += dt

    print('done')


if __name__ == '__main__':
    main()
