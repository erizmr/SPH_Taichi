import taichi as ti
import numpy as np

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.ps.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        self.g = np.array(self.ps.cfg.get_cfg("gravitation"))

        self.viscosity = 0.01  # viscosity

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.ps.cfg.get_cfg("density0")

        # self.dt = ti.field(float, shape=())
        self.ps.dt[None] = 1e-4

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r_norm):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        # k = 1.0
        # if self.ps.dim == 1:
        #     k = 4 / 3
        # elif self.ps.dim == 2:
        #     k = 40 / 7 / np.pi
        # elif self.ps.dim == 3:
        #     k = 8 / np.pi

        k = 6. * 8 / np.pi / h ** self.ps.dim
        # r_norm = r.norm(1e-5)
        q = r_norm / h
        # res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        res = ti.cast(0.0, ti.f32)
        grad_q = 1 / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        elif q <= 1.0:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
        return res

    # @ti.func
    # def cubic_kernel_derivative(self, r):
    #     h = self.ps.support_radius
    #     # derivative of cubic spline smoothing kernel
    #     k = 1.0
    #     if self.ps.dim == 1:
    #         k = 4 / 3
    #     elif self.ps.dim == 2:
    #         k = 40 / 7 / np.pi
    #     elif self.ps.dim == 3:
    #         k = 8 / np.pi
    #     k = 6. * k / h ** self.ps.dim
    #     r_norm = r.norm()
    #     q = r_norm / h
    #     res = ti.Vector([0.0 for _ in range(self.ps.dim)])
    #     if r_norm > 1e-5 and q <= 1.0:
    #         grad_q = r / (r_norm * h)
    #         if q <= 0.5:
    #             res = k * q * (3.0 * q - 2.0) * grad_q
    #         else:
    #             factor = 1.0 - q
    #             res = k * (-factor * factor) * grad_q
    #     return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    def initialize(self):
        self.ps.initialize_particle_system()
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        # self.compute_static_boundary_volume() # TODO: uncomment this if there are static rigid
        self.compute_moving_boundary_volume()

    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)

    # @ti.kernel
    # def compute_static_boundary_volume(self):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.is_static_rigid_body(p_i):
    #             delta = self.cubic_kernel(0.0)
    #             self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
    #             self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    # @ti.func
    # def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
    #     if self.ps.material[p_j] == self.ps.material_solid:
    #         delta += self.cubic_kernel((self.ps.x[p_i] - self.ps.x[p_j]).norm())


    # @ti.kernel
    # def compute_moving_boundary_volume(self):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.is_dynamic_rigid_body(p_i):
    #             delta = self.cubic_kernel(0.0)
    #             self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
    #             self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                for _ in range(1):
                    self.ps.m_V[p_i] = self.cubic_kernel(0.0)
                for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                    center_cell = self.ps.pos_to_index(self.ps.x[p_i]) #TODO: no grad version
                    grid_index = self.ps.flatten_grid_index(center_cell + offset)
                    for p_j in range(self.ps.grid_particles_num[ti.min(ti.max(0, grid_index-1), self.ps.grid_num_total-1)], self.ps.grid_particles_num[ti.min(ti.max(0, grid_index), self.ps.grid_num_total-1)]):
                        if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
                            self.compute_boundary_volume_task(p_i, p_j)
                for _ in range(1):
                    self.ps.m_V[p_i] = 1.0 / self.ps.m_V[p_i] * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly


    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j):
        if self.ps.material[p_j] == self.ps.material_solid:
            self.ps.m_V[p_i] += self.cubic_kernel((self.ps.x[p_i] - self.ps.x[p_j]).norm())

    

    @ti.kernel
    def init_mV(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic_rigid_body(p_i):
                self.ps.m_V[p_i] = 8 / 3.1415926 / self.ps.support_radius**self.ps.dim  # self.cubic_kernel(0.0)
    @ti.kernel
    def scale_mV(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic_rigid_body(p_i):
                self.ps.m_V[p_i] = 1.0 / self.ps.m_V[p_i] * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly
    
    @ti.kernel
    def compute_moving_boundary_volume_kernel(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic_rigid_body(p_i):
                for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                    center_cell = self.ps.pos_to_index(self.ps.x_val_no_grad[p_i])
                    grid_index = self.ps.flatten_grid_index(center_cell + offset)
                    for p_j in range(self.ps.grid_particles_num[ti.min(ti.max(0, grid_index-1), self.ps.grid_num_total-1)], self.ps.grid_particles_num[ti.min(ti.max(0, grid_index), self.ps.grid_num_total-1)]):
                        if p_i[0] != p_j and (self.ps.x_val_no_grad[p_i] - self.ps.x_val_no_grad[ti.min(ti.max(p_j, 0), self.ps.particle_max_num)]).norm() < self.ps.support_radius:
                            self.compute_boundary_volume_task(p_i, ti.min(ti.max(p_j, 0), self.ps.particle_max_num))
    

    def compute_moving_boundary_volume(self):
        self.ps.copy_x_to_val_no_grad()
        self.init_mV()
        self.compute_moving_boundary_volume_kernel()
        self.scale_mV()


    # @ti.kernel
    # def compute_moving_boundary_volume(self):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.is_dynamic_rigid_body(p_i):
    #             self.ps.m_V[p_i] = self.cubic_kernel(0.0)
    #             for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
    #                 center_cell = self.ps.pos_to_index(self.ps.x[p_i])
    #                 grid_index = self.ps.flatten_grid_index(center_cell + offset)
    #                 for p_j in range(self.ps.grid_particles_num[ti.max(0, grid_index-1)], self.ps.grid_particles_num[ti.max(0, grid_index)]):
    #                     if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
    #                         self.compute_boundary_volume_task(p_i, p_j)
    #             self.ps.m_V[p_i] = 1.0 / self.ps.m_V[p_i] * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly



    def substep(self):
        pass

    # @ti.func
    # def simulate_collisions(self, p_i, vec):
    #     # Collision factor, assume roughly (1-c_f)*velocity loss after collision
    #     c_f = 0.5
    #     self.ps.v[p_i] -= (
    #         1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec
    

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v_new[p_i] = self.ps.v[p_i] - (
            1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]: 
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)

    # @ti.kernel
    # def enforce_boundary_3D(self, particle_type:int):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
    #             pos = self.ps.x[p_i]
    #             collision_normal = ti.Vector([0.0, 0.0, 0.0])
    #             if pos[0] > self.ps.domain_size[0] - self.ps.padding:
    #                 collision_normal[0] += 1.0
    #                 self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
    #             if pos[0] <= self.ps.padding:
    #                 collision_normal[0] += -1.0
    #                 self.ps.x[p_i][0] = self.ps.padding

    #             if pos[1] > self.ps.domain_size[1] - self.ps.padding:
    #                 collision_normal[1] += 1.0
    #                 self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
    #             if pos[1] <= self.ps.padding:
    #                 collision_normal[1] += -1.0
    #                 self.ps.x[p_i][1] = self.ps.padding

    #             if pos[2] > self.ps.domain_size[2] - self.ps.padding:
    #                 collision_normal[2] += 1.0
    #                 self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
    #             if pos[2] <= self.ps.padding:
    #                 collision_normal[2] += -1.0
    #                 self.ps.x[p_i][2] = self.ps.padding

    #             collision_normal_length = collision_normal.norm()
    #             if collision_normal_length > 1e-6:
    #                 self.simulate_collisions(
    #                         p_i, collision_normal / collision_normal_length)


    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x_new[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x_new[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x_new[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x_new[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x_new[p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x_new[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x_new[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)


    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
                cm += mass * self.ps.x[p_i]
                sum_m += mass
        cm /= sum_m
        return cm
    

    # @ti.kernel
    # def compute_com_kernel(self, object_id: int)->ti.types.vector(3, float):
    #     return self.compute_com(object_id)
    
    # @ti.kernel
    # def compute_com_kernel(self, object_id: int):
    #     for _ in range(1):
    #         self.ps.sum_ret[None] = 0.0
    #         self.ps.cm_ret[None] = ti.Vector([0.0, 0.0, 0.0])
    #     for p_i in range(self.ps.particle_num[None]):
    #         if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
    #             mass = self.ps.m_V0 * self.ps.density[p_i]
    #             self.ps.cm_ret[None] += mass * self.ps.x[p_i]
    #             self.ps.sum_ret[None] += mass
    #     for _ in range(1):
    #         self.ps.cm_ret[None] /= self.ps.sum_ret[None]

    @ti.kernel
    def clean_ret(self):
        self.ps.sum_ret[None] = 0.0
        self.ps.cm_ret[None] = ti.Vector([0.0, 0.0, 0.0])
        self.ps.cm_ret_new[None] = ti.Vector([0.0, 0.0, 0.0])
        self.ps.A_ret[None] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    @ti.kernel
    def cm_normlize(self):
        self.ps.cm_ret_new[None] = self.ps.cm_ret[None] / self.ps.sum_ret[None]


    def compute_com_rigid(self, object_id):
        self.clean_ret()
        self.compute_com_kernel(object_id)
        self.cm_normlize()

    @ti.kernel
    def compute_com_kernel(self, object_id: int):
        # for _ in range(1):
        #     self.ps.sum_ret[None] = 0.0
        #     self.ps.cm_ret[None] = ti.Vector([0.0, 0.0, 0.0])
        #     self.ps.cm_ret_new[None] = ti.Vector([0.0, 0.0, 0.0])
        #     self.ps.A_ret[None] = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
                self.ps.cm_ret[None] += mass * self.ps.x_new[p_i]
                self.ps.sum_ret[None] += mass
        # for _ in range(1):
        #     # self.ps.cm_ret[None] /= self.ps.sum_ret[None]
        #     self.ps.cm_ret_new[None] = self.ps.cm_ret[None] / self.ps.sum_ret[None]


    @ti.kernel
    def compute_A(self, object_id: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id]
                p = self.ps.x_new[p_i] - self.ps.cm_ret_new[None]
                self.ps.A_ret[None] += self.ps.m_V0 * self.ps.density[p_i] * p.outer_product(q)
    
    @ti.func
    def ssvd(self, F):
        U, sig, V = ti.svd(F)
        if U.determinant() < 0:
            for i in ti.static(range(3)):
                U[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        if V.determinant() < 0:
            for i in ti.static(range(3)):
                V[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        return U, sig, V

    @ti.kernel
    def polar_decompose(self):
        for _ in range(1):
            U, sig, V = self.ssvd(self.ps.A_ret[None])
            R = U @ V.transpose()
            if all(abs(R) < 1e-6):
                R = ti.Matrix.identity(ti.f32, 3)
            self.ps.R_ret[None] = R

    
    # @ti.kernel
    # def polar_decompose(self):
    #     for _ in range(1):
    #         A = self.ps.A_ret[None]
    #         R, S = ti.polar_decompose(A)
    #         if all(abs(R) < 1e-6):
    #             R = ti.Matrix.identity(ti.f32, 3)
    #         self.ps.R_ret[None] = R


    @ti.kernel
    def update_matched_pos(self, object_id: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                goal = self.ps.cm_ret_new[None] + self.ps.R_ret[None] @ (self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id])
                corr = (goal - self.ps.x_new[p_i]) * 1.0
                self.ps.x[p_i] = self.ps.x_new[p_i] + corr
                self.ps.x_new[p_i] = self.ps.x[p_i]


    # @ti.kernel
    def solve_constraints(self, object_id: int): #-> ti.types.matrix(3, 3, float):
        # compute center of mass
        # cm = self.compute_com(object_id)

        self.compute_com_rigid(object_id)
        self.compute_A(object_id)
        self.polar_decompose()
        self.update_matched_pos(object_id)

        # return R

    def solve_rigid_body(self):
        for i in range(1):
            for r_obj_id in self.ps.object_id_rigid_body:
                # R = self.solve_constraints(r_obj_id)
                self.solve_constraints(r_obj_id)

                if self.ps.cfg.get_cfg("exportObj"):
                    R = self.ps.R_ret[None]
                    # For output obj only: update the mesh
                    cm = self.compute_com_kernel(r_obj_id)
                    ret = R.to_numpy() @ (self.ps.object_collection[r_obj_id]["restPosition"] - self.ps.object_collection[r_obj_id]["restCenterOfMass"]).T
                    self.ps.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T

                self.enforce_boundary_3D(self.ps.material_solid)


    @ti.kernel
    def copy_fields(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] = self.ps.v_new[p_i]
                self.ps.x[p_i] = self.ps.x_new[p_i]

    def step(self):
        self.ps.initialize_particle_system()
        # self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        elif self.ps.dim == 3:
            self.enforce_boundary_3D(self.ps.material_fluid)
        self.copy_fields()
