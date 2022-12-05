import taichi as ti
import numpy as np
from particle_system import ParticleSystem

@ti.data_oriented
class NSearchBase():
    def __init__(self,ps:ParticleSystem):
        self.ps = ps
        self.max_num_neighbors = 50

        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.ps.particle_max_num)

        # Grid related properties
        self.grid_size = self.ps.support_radius
        self.grid_num = np.ceil(self.ps.domain_size / self.grid_size).astype(int)
        self.padding = self.grid_size

        # Particle num of each grid
        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.particle_neighbors = ti.field(int, shape=((self.ps.num_particles,) + (self.max_num_neighbors,)))

    
    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(self.ps.x):
            grid_index = self.get_flatten_grid_index(self.ps.x[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)

    def initialize_particle_system(self):
        self.update_grid_id()

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.ps.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
                    task(p_i, p_j, ret)