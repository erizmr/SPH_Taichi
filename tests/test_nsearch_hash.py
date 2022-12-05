'''
这次换一下数据结构，为普通的taichi field
'''
import taichi as ti
import numpy as np
import sys
sys.path.append(".")

ti.init()

@ti.data_oriented
class NeighborhoodSearchHash():
    def __init__(self, pos) -> None:
        # common paramters
        dim = 3
        self.num_particles = 100
        particle_radius = 3.0
        h = 1.1

        # read positions from the test data input
        self.positions = ti.Vector.field(dim, float, self.num_particles)
        self.positions.from_numpy(pos)

        # nsearch parameters
        self.grid_size = (16,16,16)
        cell_size = 2.51
        self.cell_recpr = 1.0 / cell_size
        self.neighbor_radius = h * 1.05
        self.max_num_particles_per_cell = 20
        self.max_num_neighbors = 20

        # nsearch fields new
        self.grid_size_hash = 2 * self.num_particles
        self.grid_num_particles_hash = ti.field(int,self.grid_size_hash)
        self.grid2particles_hash = ti.field(int, ((self.grid_size_hash,) + (self.max_num_particles_per_cell,)))
        self.particle_num_neighbors = ti.field(int,self.num_particles)
        self.particle_neighbors = ti.field(int, shape=((self.num_particles,) + (self.max_num_neighbors,)))

    
    @ti.func
    def cell2hash(self,cell):
        res =   ( (73856093 * cell[0]) ^ (19349663 * cell[1]) ^ (83492791*cell[2]))  % (self.grid_size_hash)
        return int(res)

    @ti.func
    def get_cell(self,pos):
        return int(pos * self.cell_recpr)

    @ti.func
    def is_in_grid(self,c):
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[
            1] < self.grid_size[1]

    @ti.func
    def is_unique(self, p_i, pj):
        flag = True
        for nb_i in range(self.particle_num_neighbors[p_i]):
            if self.particle_neighbors[p_i, nb_i] == pj:
                flag = False
        return flag

    @ti.kernel
    def neighborhood_search(self):
        # clear neighbor lookup table
        for I in ti.grouped(self.grid_num_particles_hash):
            self.grid_num_particles_hash[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        # update grid
        for p_i in self.positions:
            cell = self.get_cell(self.positions[p_i])
            hash = self.cell2hash(cell)
            offs = ti.atomic_add(self.grid_num_particles_hash[hash], 1)
            self.grid2particles_hash[hash, offs] = p_i
        # find particle neighbors
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            cell = self.get_cell(pos_i)
            hash = self.cell2hash(cell)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2),(-1, 2)))):
                cell_to_check = cell + offs
                hash_to_check = self.cell2hash(cell_to_check)
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles_hash[hash_to_check]):
                        p_j = self.grid2particles_hash[hash_to_check, j]
                        if nb_i < self.max_num_neighbors and p_j != p_i and (
                                pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                            if self.is_unique(p_i, p_j):
                                self.particle_neighbors[p_i, nb_i] = p_j
                                nb_i += 1
                                self.particle_num_neighbors[p_i] = nb_i


def test():
    pos = np.loadtxt('./tests/data/input_nsearch.csv',dtype=float)
    search = NeighborhoodSearchHash(pos)
    search.neighborhood_search()
    np.savetxt('./tests/data/actual_nsearch_hash.csv', search.particle_neighbors.to_numpy(), "%d")
    np.savetxt('./tests/data/actual_neigbor_num.csv', search.particle_num_neighbors.to_numpy(), "%d")

if __name__ == '__main__':
    test()
