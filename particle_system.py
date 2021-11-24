import taichi as ti
import numpy as np


@ti.data_oriented
class ParticleSystem:
    def __init__(self, res):
        self.dim = len(res)
        assert self.dim > 1

        # Material
        self.boundary = 0
        self.fluid = 1

        self.h = 1.0  # support radius
        self.particle_max_num = 2**18
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100
        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.h
        self.grid_num = np.ceil(np.array(res) / self.grid_size).astype(int)
        self.grid_particles_num = ti.field(int)
        self.grid_particles = ti.field(int)

        # Particle related properties
        self.x = ti.Vector.field(self.dim, dtype=float)
        self.v = ti.Vector.field(self.dim, dtype=float)
        self.density = ti.Vector.field(self.dim, dtype=float)
        self.material = ti.Vector.field(self.dim, dtype=float)
        self.color = ti.Vector.field(self.dim, dtype=float)
        self.particle_neighbors = ti.field(int)

        particles_node = ti.root.dense(ti.i, self.particle_max_num)
        particles_node.place(self.x, self.v, self.density, self.material, self.color)
        particle_node = particles_node.dense(ti.j, self.particle_max_num_neighbor)
        particle_node.place(self.particle_neighbors)

        index = ti.ij if self.dim == 2 else ti.ijk
        grid_node = ti.root.dense(index, self.grid_num)
        grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)
        cell_node.place(self.grid_particles)

    @ti.func
    def add_particle(self, p, x, v, density, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.ext_arr(),
                      new_particles_velocity: ti.ext_arr(),
                      new_particles_density: ti.ext_arr(),
                      new_particles_material: ti.ext_arr(),
                      new_particles_color: ti.ext_arr()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            self.add_particle(p,
                              ti.Vector(new_particles_positions[p]),
                              ti.Vector(new_particles_velocity[p]),
                              new_particles_density[p],
                              new_particles_material[p],
                              new_particles_color[p])
    
    @ti.func
    def pos_to_index(self, pos):
        return int(pos / self.grid_size)

    @ti.func
    def is_valid_cell(self, cell):
        # Check whether the cell is in the grid
        flag = True
        for d in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[d] < self.grid_num[d])

    @ti.kernel
    def allocate_particles_to_grid(self):
        for p in range(self.particle_max_num[None]):
            cell = self.pos_to_index(self.x[p])
            offset = self.grid_particles_num[None]
            self.grid_particles[cell, offset] = p
            self.grid_particles_num[None] += 1

    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_max_num[None]):
            # Skip boundary particles
            if self.material[p_i] == self.boundary:
                continue
            center_cell = self.pos_to_index(self.x[p_i])
            cnt = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2), ) * self.dim)):
                if cnt >= self.particle_max_num_neighbor:
                    break
                cell = center_cell + offset
                if not is_valid_cell(cell):
                    break
                for p_j in range(self.grid_particles[cell]):
                    distance = (self.x[p] - self.x[p_j]).norm()
                    if p_i != p_j and distance < self.h:
                        self.particle_neighbors[p, cnt] = p_j
                        cnt += 1
            self.grid_particles_num[None] = cnt
