import taichi as ti
from particle_system import ParticleSystem

@ti.data_oriented
class NeighborhoodSearchHash():
    def __init__(self, positions:ti.template(), support_radius) -> None:
        # common paramters
        self.support_radius = support_radius
        self.num_particles = positions.shape[0]

        # # read positions from the test data input
        # self.positions = ti.Vector.field(dim, float, self.num_particles)
        # self.positions.from_numpy(pos)

        # # nsearch parameters
        self.cell_size = 2 * self.support_radius #2 is heuristic, you can tune it
        self.cell_recpr = 1.0 / self.cell_size
        self.neighbor_radius = self.support_radius * 1.05
        self.max_num_particles_per_cell = 50
        self.max_num_neighbors = 50

        # nsearch fields new
        self.grid_size = 2 * self.num_particles
        self.grid_particles_num = ti.field(int,self.grid_size)
        self.grid2particle = ti.field(int, ((self.grid_size,) + (self.max_num_particles_per_cell,)))
        self.particle_num_neighbors = ti.field(int,self.num_particles)
        self.particle_neighbors = ti.field(int, shape=((self.num_particles,) + (self.max_num_neighbors,)))

    
    @ti.func
    def cell2hash(self,cell):
        res =   ( (73856093 * cell[0]) ^ (19349663 * cell[1]) ^ (83492791*cell[2]))  % (self.grid_size)
        return int(res)

    @ti.func
    def pos_to_index(self,pos):
        return int(pos * self.cell_recpr)

    @ti.func
    def is_in_grid(self,c):
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[
            1] < self.grid_size[1]

    @ti.kernel
    def neighborhood_search(self):
        # clear neighbor lookup table
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        # update grid
        for p_i in self.positions:
            cell = self.pos_to_index(self.positions[p_i])
            hash = self.cell2hash(cell)
            offs = ti.atomic_add(self.grid_particles_num[hash], 1)
            self.grid2particle[hash, offs] = p_i
        # find particle neighbors
        for p_i in self.positions:
            pos_i = self.positions[p_i]
            cell = self.pos_to_index(pos_i)
            hash = self.cell2hash(cell)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2),(-1, 2)))):
                cell_to_check = cell + offs
                hash_to_check = self.cell2hash(cell_to_check)
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_particles_num[hash_to_check]):
                        p_j = self.grid2particle[hash_to_check, j]
                        if nb_i < self.max_num_neighbors and p_j != p_i and (
                                pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i