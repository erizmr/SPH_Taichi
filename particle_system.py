import taichi as ti
import numpy as np
from functools import reduce


@ti.data_oriented
class ParticleSystem:
    def __init__(self, domain_size, GGUI=False):
        self.GGUI = GGUI
        self.domain_size = domain_size
        self.dim = len(domain_size)
        assert self.dim > 1
        self.screen_to_world_ratio = 50
        self.bound = self.domain_size # self.screen_to_world_ratio
        # All objects id and its particle num
        self.object_id_collection = dict()

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim
        self.particle_max_num = 2 ** 18
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100
        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(np.array(self.bound) / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.grid_particles_num = ti.field(int)
        self.grid_particles = ti.field(int)
        self.padding = self.grid_size

        # Particle related properties
        self.object_id = ti.field(dtype=int)
        self.x = ti.Vector.field(self.dim, dtype=float)
        self.x_0 = ti.Vector.field(self.dim, dtype=float)
        self.rigid_rest_cm = ti.Vector.field(self.dim, dtype=float, shape=())

        self.v = ti.Vector.field(self.dim, dtype=float)
        self.acceleration = ti.Vector.field(self.dim, dtype=float)
        self.m_V = ti.field(dtype=float)
        self.m = ti.field(dtype=float)
        self.density = ti.field(dtype=float)
        self.pressure = ti.field(dtype=float)
        self.material = ti.field(dtype=int)
        self.color = ti.Vector.field(3, dtype=int)
        self.is_dynamic = ti.field(dtype=int)

        # Neighbors information
        self.fluid_neighbors = ti.field(int)
        self.fluid_neighbors_num = ti.field(int)

        self.solid_neighbors = ti.field(int)
        self.solid_neighbors_num = ti.field(int)

        # Allocate memory
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        self.particles_node.place(self.object_id, self.x, self.x_0, self.v, self.acceleration, self.density, self.m_V, self.m, self.pressure, self.material, self.color, self.is_dynamic)
        self.particles_node.place(self.fluid_neighbors_num, self.solid_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbor)
        self.particle_node.place(self.fluid_neighbors, self.solid_neighbors)

        index = ti.ij if self.dim == 2 else ti.ijk
        if self.dim == 2:
            self.grid_node = ti.root.dense(index, self.grid_num)
            self.grid_node.place(self.grid_particles_num)
        elif self.dim == 3:
            # self.grid_node = ti.root.pointer(index, self.grid_num)
            self.grid_node = ti.root.dense(index, self.grid_num)
            self.grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l
        if self.dim == 2:
            cell_node = self.grid_node.dense(cell_index, self.particle_max_num_per_cell)
            cell_node.place(self.grid_particles)
        elif self.dim == 3:
            self.grid_node.dense(cell_index, self.particle_max_num_per_cell).place(self.grid_particles)
            # self.grid_node.dynamic(cell_index, self.particle_max_num_per_cell).place(self.grid_particles)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float)
            self.particles_node.place(self.x_vis_buffer, self.color_vis_buffer)

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.object_id[p] = obj_id
        self.x[p] = x
        self.x_0[p] = x
        self.v[p] = v
        self.density[p] = density
        self.m_V[p] = self.m_V0
        self.m[p] = self.m_V0 * density
        self.pressure[p] = pressure
        self.material[p] = material
        self.is_dynamic[p] = is_dynamic
        self.color[p] = color
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):
        if object_id in self.object_id_collection:
            self.object_id_collection[object_id] += new_particles_num
        else:
            self.object_id_collection[object_id] = new_particles_num
        
        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      )

    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def is_valid_cell(self, cell):
        # Check whether the cell is in the grid
        flag = True
        for d in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[d] < self.grid_num[d])
        return flag
    

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])


    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]


    @ti.kernel
    def allocate_particles_to_grid(self):
        for p in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[p])
            offset = ti.atomic_add(self.grid_particles_num[cell], 1)
            self.grid_particles[cell, offset] = p

    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            center_cell = self.pos_to_index(self.x[p_i])
            cnt_fluid = 0
            cnt_boundary = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                if cnt_fluid + cnt_boundary >= self.particle_max_num_neighbor:
                    break
                cell = center_cell + offset
                for j in range(self.grid_particles_num[cell]):
                    p_j = self.grid_particles[cell, j]
                    distance = (self.x[p_i] - self.x[p_j]).norm()
                    if p_i != p_j and distance < self.support_radius:
                        if self.material[p_j] == self.material_fluid:
                            self.fluid_neighbors[p_i, cnt_fluid] = p_j
                            cnt_fluid += 1
                        elif self.material[p_j] == self.material_solid:
                            self.solid_neighbors[p_i, cnt_boundary] = p_j
                            cnt_boundary += 1
            self.fluid_neighbors_num[p_i] = cnt_fluid
            self.solid_neighbors_num[p_i] = cnt_boundary

    def initialize_particle_system(self):
        self.grid_particles_num.fill(0)
        self.fluid_neighbors.fill(-1)
        self.solid_neighbors.fill(-1)
        self.allocate_particles_to_grid()
        self.search_neighbors()

    @ti.kernel
    def copy_to_numpy_nd(self, obj_id: int, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            if self.object_id[i] == obj_id:
                for j in ti.static(range(self.dim)):
                    np_arr[i, j] = src_arr[i][j]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, invisible_objects=[]):
        for obj_id in self.object_id_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        for i in range(self.particle_num[None]):
            if self.object_id[i] == obj_id:
                self.x_vis_buffer[i] = self.x[i]
                self.color_vis_buffer[i] = self.color[i] / 255.0

    # def dump(self):
    #     np_x = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
    #     self.copy_to_numpy_nd(np_x, self.x)

    #     np_v = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
    #     self.copy_to_numpy_nd(np_v, self.v)

    #     np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
    #     self.copy_to_numpy(np_material, self.material)

    #     np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
    #     self.copy_to_numpy(np_color, self.color)

    #     return {
    #         'position': np_x,
    #         'velocity': np_v,
    #         'material': np_material,
    #         'color': np_color
    #     }
    
    def dump(self, obj_id):
        particle_num = self.object_id_collection[obj_id]
        np_x = np.ndarray((particle_num, self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(obj_id, np_x, self.x)

        np_v = np.ndarray((particle_num, self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(obj_id, np_v, self.v)

        np_material = np.ndarray((particle_num,), dtype=np.int32)
        self.copy_to_numpy(obj_id, np_material, self.material)

        np_color = np.ndarray((particle_num,), dtype=np.int32)
        self.copy_to_numpy(obj_id, np_color, self.color)

        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 is_dynamic,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        print('particle num ', num_new_particles)
        assert self.particle_num[
                   None] + num_new_particles <= self.particle_max_num

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)