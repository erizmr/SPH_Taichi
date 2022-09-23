import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig
from WCSPH import WCSPHSolver
from scan_single_buffer import parallel_prefix_sum_inclusive_inplace

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        assert self.dim > 1


        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim

        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        #========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        fluid_blocks = self.cfg.get_fluid_blocks()
        fluid_particle_num = 0
        for fluid in fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        #### Process Rigid Blocks ####
        rigid_blocks = self.cfg.get_rigid_blocks()
        rigid_particle_num = 0
        for rigid in rigid_blocks:
            particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num
        
        #### Process Rigid Bodies ####
        rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += voxelized_points_np.shape[0]
        
        self.particle_max_num = fluid_particle_num + rigid_particle_num

        #### TODO: Handle the Particle Emitter ####
        # self.particle_max_num += emitted particles
        print(f"Current particle num: {self.particle_num[None]}, Particle max num: {self.particle_max_num}")

        #========== Allocate memory ==========#
        # Particle num of each grid
        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))   

        # Particle related properties
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.rigid_rest_cm = ti.Vector.field(self.dim, dtype=float, shape=len(rigid_blocks)+len(rigid_bodies))

        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)

        # Buffer for sort
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.color_buffer = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)


        #========== Initialize particles ==========#

        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=1, # enforce fluid dynamic
                          color=color,
                          material=1) # 1 indicates fluid
        
        # TODO: Handle rigid block

        # Rigid bodies
        for rigid_body in rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            velocity = np.array(rigid_body["velocity"], dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body["color"], dtype=np.int32)
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.array([0 for _ in range(num_particles_obj)], dtype=np.int32), # material is solid
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32), # is_dynamic
                               np.stack([color for _ in range(num_particles_obj)])) # color
    

    def build_solver(self):
        solver_type = self.cfg.get_cfg("simulationMethod")
        if solver_type == 0:
            return WCSPHSolver(self)
        else:
            raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")

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
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))
    

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])


    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]
    

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(self.x):
            grid_index = self.get_flatten_grid_index(self.x[I])
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
            self.object_id_buffer[new_index] = self.object_id[I]
            self.x_0_buffer[new_index] = self.x_0[I]
            self.x_buffer[new_index] = self.x[I]
            self.v_buffer[new_index] = self.v[I]
            self.acceleration_buffer[new_index] = self.acceleration[I]
            self.m_V_buffer[new_index] = self.m_V[I]
            self.m_buffer[new_index] = self.m[I]
            self.density_buffer[new_index] = self.density[I]
            self.pressure_buffer[new_index] = self.pressure[I]
            self.material_buffer[new_index] = self.material[I]
            self.color_buffer[new_index] = self.color[I]
            self.is_dynamic_buffer[new_index] = self.is_dynamic[I]
        
        for I in ti.grouped(self.x):
            self.grid_ids[I] = self.grid_ids_buffer[I]
            self.object_id[I] = self.object_id_buffer[I]
            self.x_0[I] = self.x_0_buffer[I]
            self.x[I] = self.x_buffer[I]
            self.v[I] = self.v_buffer[I]
            self.acceleration[I] = self.acceleration_buffer[I]
            self.m_V[I] = self.m_V_buffer[I]
            self.m[I] = self.m_buffer[I]
            self.density[I] = self.density_buffer[I]
            self.pressure[I] = self.pressure_buffer[I]
            self.material[I] = self.material_buffer[I]
            self.color[I] = self.color_buffer[I]
            self.is_dynamic[I] = self.is_dynamic_buffer[I]
    

    def initialize_particle_system(self):
        self.update_grid_id()
        # FIXME: change to taichi built-in prefix_sum_inclusive_inplace after next Taichi release i.e., 1.1.4
        parallel_prefix_sum_inclusive_inplace(self.grid_particles_num, self.grid_particles_num.shape[0])
        self.counting_sort()
    

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_radius:
                    task(p_i, p_j, ret)


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
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        # FIXME: make it equal to actual particle num
        for i in range(self.particle_max_num):
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
    
    # def dump(self, obj_id):
    #     particle_num = self.object_collection[obj_id]
    #     np_x = np.ndarray((particle_num, self.dim), dtype=np.float32)
    #     self.copy_to_numpy_nd(obj_id, np_x, self.x)

    #     np_v = np.ndarray((particle_num, self.dim), dtype=np.float32)
    #     self.copy_to_numpy_nd(obj_id, np_v, self.v)

    #     np_material = np.ndarray((particle_num,), dtype=np.int32)
    #     self.copy_to_numpy(obj_id, np_material, self.material)

    #     np_color = np.ndarray((particle_num,), dtype=np.int32)
    #     self.copy_to_numpy(obj_id, np_color, self.color)

    #     return {
    #         'position': np_x,
    #         'velocity': np_v,
    #         'material': np_material,
    #         'color': np_color
    #     }

    def dump(self, obj_id):
        particle_num = self.object_collection[obj_id]["particleNum"]
        np_x = np.ndarray((particle_num, self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(obj_id, np_x, self.x)

        np_v = np.ndarray((particle_num, self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(obj_id, np_v, self.v)

        return {
            'position': np_x,
            'velocity': np_v
        }
    

    def load_rigid_body(self, rigid_body):
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        
        is_dynamic = rigid_body["isDynamic"]
        if is_dynamic:
            # Backup the original mesh for exporting obj
            mesh_backup = mesh.copy()
            mesh_backup.vertices += offset
            rigid_body["mesh"] = mesh_backup
            rigid_body["restPosition"] = mesh_backup.vertices
            rigid_body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
            is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=ps.particle_diameter).hollow()
        # voxelized_mesh.show()
        voxelized_points_np = voxelized_mesh.points + offset
        print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")
        
        return voxelized_points_np


    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.particle_diameter))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])

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

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)