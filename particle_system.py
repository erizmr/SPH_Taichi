import taichi as ti
import numpy as np
import trimesh as tm
from config_builder import SimConfig
from WCSPH import WCSPHSolver
from DFSPH import DFSPHSolver
from scan_single_buffer import parallel_prefix_sum_inclusive_inplace
from readwrite.read_ply import read_ply

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
        # Simulation method
        self.simulation_method = self.cfg.get_cfg("simulationMethod")

        # paramters
        self.material_solid = 0
        self.material_fluid = 1
        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim

        self.particle_num = ti.field(int, shape=())


        # 实例化一个loader 计算粒子的最大数目
        from loader import FluidLoader
        self.loader = FluidLoader(self) 
        self.particle_max_num = self.loader.compute_particle_max_num()

        #### TODO: Handle the Particle Emitter ####
        # self.particle_max_num += emitted particles
        print(f"Current particle num: {self.particle_num[None]}, Particle max num: {self.particle_max_num}")

        #========== Allocate memory ==========#
        # Rigid body properties
        if self.loader.num_rigid_bodies > 0:
            self.rigid_rest_cm = ti.Vector.field(self.dim, dtype=float, shape=self.loader.num_rigid_bodies)
        # Rigid body properties
        if self.loader.num_fluid_bodies > 0:
            self.fluid_rest_cm = ti.Vector.field(self.dim, dtype=float, shape=self.loader.num_fluid_bodies)

        # Particle related properties
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv = ti.field(dtype=float, shape=self.particle_max_num)
        
        self.loader.add_blocks_bodies()
        #为了便利 为常用变量赋别名
        self.fluid_particle_num = self.loader.fluid_particle_num 
        self.object_collection = self.loader.object_collection


        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

        # 实例化邻域搜索ns
        from nsearch_gpu import NSearchGpu 
        self.ns = NSearchGpu(self)
        self.padding = self.ns.padding #为了便利

        # 输入外界数据
        self.input_data()
        #========== Initialize particles ==========#
        
    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]

    def build_solver(self):
        solver_type = self.cfg.get_cfg("simulationMethod")
        if solver_type == 0:
            return WCSPHSolver(self)
        elif solver_type == 4:
            return DFSPHSolver(self)
        else:
            raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")


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
        if len(invisible_objects) != 0:
            self.x_vis_buffer.fill(0.0)
            self.color_vis_buffer.fill(0.0)
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
    

    # MYADD
    def update_data(self, outside_data):
        self.pts.from_numpy(outside_data)

    def input_data(self):
        # MYADD ======================
        self.cnt = ti.field(int, ()) # 当前帧号
        self.g_id = ti.field(dtype=int, shape=self.particle_max_num) # 外界点的网格编号
        self.num_pts = 1 # 外界点的总数

        # 每帧外界读入的数据存入pts, 使用update_pts更新
        # read ply
        if self.cfg.get_cfg("readPly") == True :
            ply_path = self.cfg.get_cfg("plyPath")
            ply_range = self.cfg.get_cfg("plyRange")
            self.plys = read_ply(ply_path, ply_range[0], ply_range[1])
            
            self.num_pts = self.plys[0].shape[0] #外界粒子数目

        # read vdb
        if self.cfg.get_cfg("readVdb") == True :
            from readwrite.read_vdb import read_vdb

            vdb_path = self.cfg.get_cfg("vdbPath")
            grid_name = self.cfg.get_cfg("vdbObjName")
            vdb_range = self.cfg.get_cfg("vdbRange")
            self.vdbs = read_vdb(vdb_path, grid_name,  vdb_range[0], vdb_range[1])
            # num_pts = self.vdbs[0].shape[0] #外界粒子数目

        self.pts = ti.Vector.field(3,dtype=float, shape=self.num_pts)
        # END of MYADD ======================