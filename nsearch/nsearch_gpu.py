import taichi as ti
import numpy as np
@ti.data_oriented
class NSearchGpu():
    def __init__(self,ps):
        self.ps = ps
        self.particle_max_num = ps.particle_max_num

        # Buffer for sort
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(self.ps.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.ps.dim, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.ps.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(self.ps.dim, dtype=float, shape=self.particle_max_num)
        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.color_buffer = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        # if ti.static(self.ps.simulation_method == 4):
        self.dfsph_factor_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_adv_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        
        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        # Grid related properties
        self.grid_size = self.ps.support_radius
        self.grid_num = np.ceil(self.ps.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        # Particle num of each grid
        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])



    
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
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]
    
    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.particle_max_num):
            I = self.particle_max_num - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I]-1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for I in ti.grouped(self.grid_ids):
            new_index = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index] = self.grid_ids[I]
            self.object_id_buffer[new_index] = self.ps.object_id[I]
            self.x_0_buffer[new_index] = self.ps.x_0[I]
            self.x_buffer[new_index] = self.ps.x[I]
            self.v_buffer[new_index] = self.ps.v[I]
            self.acceleration_buffer[new_index] = self.ps.acceleration[I]
            self.m_V_buffer[new_index] = self.ps.m_V[I]
            self.m_buffer[new_index] = self.ps.m[I]
            self.density_buffer[new_index] = self.ps.density[I]
            self.pressure_buffer[new_index] = self.ps.pressure[I]
            self.material_buffer[new_index] = self.ps.material[I]
            self.color_buffer[new_index] = self.ps.color[I]
            self.is_dynamic_buffer[new_index] = self.ps.is_dynamic[I]

            if ti.static(self.ps.simulation_method == 4):
                self.dfsph_factor_buffer[new_index] = self.ps.dfsph_factor[I]
                self.density_adv_buffer[new_index] = self.ps.density_adv[I]
        
        for I in ti.grouped(self.ps.x):
            self.grid_ids[I] = self.grid_ids_buffer[I]
            self.ps.object_id[I] = self.object_id_buffer[I]
            self.ps.x_0[I] = self.x_0_buffer[I]
            self.ps.x[I] = self.x_buffer[I]
            self.ps.v[I] = self.v_buffer[I]
            self.ps.acceleration[I] = self.acceleration_buffer[I]
            self.ps.m_V[I] = self.m_V_buffer[I]
            self.ps.m[I] = self.m_buffer[I]
            self.ps.density[I] = self.density_buffer[I]
            self.ps.pressure[I] = self.pressure_buffer[I]
            self.ps.material[I] = self.material_buffer[I]
            self.ps.color[I] = self.color_buffer[I]
            self.ps.is_dynamic[I] = self.is_dynamic_buffer[I]


    

    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()
    

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.ps.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
                    task(p_i, p_j, ret)