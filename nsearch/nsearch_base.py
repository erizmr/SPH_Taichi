import taichi as ti
import numpy as np
from particle_system import ParticleSystem

@ti.data_oriented
class NSearchBase():
    def __init__(self,ps:ParticleSystem, support_radius:float, max_num_neighbors:int = 50) -> None:
        self.ps = ps
        self.support_radius = support_radius
        self.max_num_neighbors = max_num_neighbors
        self.grid_size = self.ps.support_radius
        self.grid_num = np.ceil(self.ps.domain_size / self.grid_size).astype(int)

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
        pass

    def initialize_particle_system(self):
        pass
    
    def neighborhood_search(self):
        pass