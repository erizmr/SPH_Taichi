import numpy as np
import trimesh as tm
import taichi as ti
from functools import reduce
@ti.data_oriented
class FluidLoader():
    def __init__(self,ps):
        self.ps = ps
        self.cfg = ps.cfg

        self.fluid_blocks = self.cfg.get_fluid_blocks()
        self.fluid_bodies = self.cfg.get_fluid_bodies()
        self.rigid_blocks = self.cfg.get_rigid_blocks()
        self.rigid_bodies = self.cfg.get_rigid_bodies()

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()
        self.object_id_fluid_body = set()

        self.fluid_particle_num = 0
        self.solid_particle_num = 0
        self.rigid_bodies_particle_num = 0
        self.rigid_blocks_particle_num = 0
        self.fluid_blocks_particle_num = 0
        self.fluid_bodies_particle_num = 0


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
        voxelized_mesh = mesh.voxelized(pitch=self.ps.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=self.ps.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        # mesh.show()
        voxelized_points_np = voxelized_mesh.points + offset
        print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")
        return voxelized_points_np

    def load_fluid_body(self, fluid_body):
        obj_id = fluid_body["objectId"]
        mesh = tm.load(fluid_body["geometryFile"])
        mesh.apply_scale(fluid_body["scale"])
        offset = np.array(fluid_body["translation"])

        angle = fluid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = fluid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)

        is_dynamic = 1
        if is_dynamic:
            # Backup the original mesh for exporting obj
            mesh_backup = mesh.copy()
            mesh_backup.vertices += offset
            fluid_body["mesh"] = mesh_backup
            fluid_body["restPosition"] = mesh_backup.vertices
            fluid_body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
            is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)
        voxelized_mesh = mesh.voxelized(pitch=self.ps.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=self.ps.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        # mesh.show()
        voxelized_points_np = voxelized_mesh.points + offset
        print(f"fluid body {obj_id} num: {voxelized_points_np.shape[0]}")

        return voxelized_points_np
    

    def add_blocks_bodies(self):
        # Fluid block
        for fluid in self.fluid_blocks:
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
        # Rigid block
        for rigid in self.rigid_blocks:
            obj_id = rigid["objectId"]
            offset = np.array(rigid["translation"])
            start = np.array(rigid["start"]) + offset
            end = np.array(rigid["end"]) + offset
            scale = np.array(rigid["scale"])
            velocity = rigid["velocity"]
            density = rigid["density"]
            color = rigid["color"]
            is_dynamic = rigid["isDynamic"]
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=is_dynamic,
                          color=color,
                          material=0) # 1 indicates solid

        # Rigid bodies
        for rigid_body in self.rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.ps.dim)], dtype=np.float32)
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

        # Fluid bodies
        for fluid_body in self.fluid_bodies:
            obj_id = fluid_body["objectId"]
            self.object_id_fluid_body.add(obj_id)
            num_particles_obj = fluid_body["particleNum"]
            voxelized_points_np = fluid_body["voxelizedPoints"]
            is_dynamic = 1
            if is_dynamic:
                velocity = np.array(fluid_body["velocity"], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = fluid_body["density"]
            color = np.array(fluid_body["color"], dtype=np.int32)
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32),  # position
                               np.stack([velocity for _ in range(num_particles_obj)]),  # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32),  # density
                               np.zeros(num_particles_obj, dtype=np.float32),  # pressure
                               np.array([1 for _ in range(num_particles_obj)], dtype=np.int32),  # material is fluid
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32),  # is_dynamic
                               np.stack([color for _ in range(num_particles_obj)]))  # color
                               
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
        for i in range(self.ps.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.ps.particle_diameter))
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

    
    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.ps.object_id[p] = obj_id
        self.ps.x[p] = x
        self.ps.x_0[p] = x
        self.ps.v[p] = v
        self.ps.density[p] = density
        self.ps.m_V[p] = self.ps.m_V0
        self.ps.m[p] = self.ps.m_V0 * density
        self.ps.pressure[p] = pressure
        self.ps.material[p] = material
        self.ps.is_dynamic[p] = is_dynamic
        self.ps.color[p] = color
    
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
        for p in range(self.ps.particle_num[None], self.ps.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.ps.dim)
            x = ti.Vector.zero(float, self.ps.dim)
            for d in ti.static(range(self.ps.dim)):
                v[d] = new_particles_velocity[p - self.ps.particle_num[None], d]
                x[d] = new_particles_positions[p - self.ps.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.ps.particle_num[None]],
                              new_particle_pressure[p - self.ps.particle_num[None]],
                              new_particles_material[p - self.ps.particle_num[None]],
                              new_particles_is_dynamic[p - self.ps.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.ps.particle_num[None], i] for i in range(3)])
                              )
        self.ps.particle_num[None] += new_particles_num


    def fluid_blocks_pnum(self):
        fluid_particle_num = 0
        for fluid in self.fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num
        return fluid_particle_num

    #### Process Fluid Bodies ####
    def fluid_bodies_pnum(self):
        fluid_particle_num = 0
        for fluid_body in self.fluid_bodies:
            voxelized_points_np_fluid = self.load_fluid_body(fluid_body)
            fluid_body["particleNum"] = voxelized_points_np_fluid.shape[0]
            fluid_body["voxelizedPoints"] = voxelized_points_np_fluid
            self.object_collection[fluid_body["objectId"]] = fluid_body
            fluid_particle_num += voxelized_points_np_fluid.shape[0]
        return fluid_particle_num

    #### Process Rigid Blocks ####
    def rigid_blocks_pnum(self):
        rigid_particle_num = 0
        for rigid in self.rigid_blocks:
            particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            self.rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num
        return rigid_particle_num
    
    #### Process Rigid Bodies ####
    def rigid_bodies_pnum(self):
        rigid_particle_num = 0
        for rigid_body in self.rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += voxelized_points_np.shape[0]
        return rigid_particle_num

    def compute_particle_max_num(self):
        fluid_blocks_particle_num = self.fluid_blocks_pnum()
        fluid_bodies_particle_num = self.fluid_bodies_pnum()
        self.fluid_particle_num = fluid_blocks_particle_num + fluid_bodies_particle_num

        rigid_blocks_particle_num = self.rigid_blocks_pnum()
        rigid_bodies_particle_num = self.rigid_bodies_pnum()
        self.solid_particle_num = rigid_blocks_particle_num + rigid_bodies_particle_num

        self.particle_max_num = self.fluid_particle_num + self.rigid_bodies_particle_num
        self.num_rigid_bodies = len(self.rigid_blocks)+len(self.rigid_bodies)
        self.num_fluid_bodies = len(self.fluid_blocks) + len(self.fluid_bodies)
        return self.particle_max_num

    
    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.ps.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.ps.particle_diameter))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])