import taichi as ti
from sph_base import SPHBase


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        self.exponent = 7.0
        self.exponent = self.ps.cfg.get_cfg("exponent")

        self.stiffness = 50000.0
        self.stiffness = self.ps.cfg.get_cfg("stiffness")
        
        self.surface_tension = 0.01
        self.ps.dt[None] = self.ps.cfg.get_cfg("timeStepSize")
    

    # @ti.func
    # def compute_densities_task(self, p_i, p_j, ret: ti.template()):
    #     x_i = self.ps.x[p_i]
    #     if self.ps.material[p_j] == self.ps.material_fluid:
    #         # Fluid neighbors
    #         x_j = self.ps.x[p_j]
    #         ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
    #     elif self.ps.material[p_j] == self.ps.material_solid:
    #         # Boundary neighbors
    #         ## Akinci2012
    #         x_j = self.ps.x[p_j]
    #         ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())


    # @ti.kernel
    # def compute_densities(self):
    #     # for p_i in range(self.ps.particle_num[None]):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.material[p_i] == self.ps.material_fluid:
    #             self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
    #             den = 0.0
    #             self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
    #             self.ps.density[p_i] += den
    #             self.ps.density[p_i] *= self.density_0

    @ti.func
    def compute_densities_task(self, p_i, p_j):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())


    @ti.kernel
    def compute_densities(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == self.ps.material_fluid:
                for _ in range(1):
                    self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
                # den = 0.0
                for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                    center_cell = self.ps.pos_to_index(self.ps.x[p_i])
                    grid_index = self.ps.flatten_grid_index(center_cell + offset)
                    for p_j in range(self.ps.grid_particles_num[ti.max(0, grid_index-1)], self.ps.grid_particles_num[grid_index]):
                        if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
                            self.compute_densities_task(p_i, p_j)
                # self.ps.density[p_i] += den
                for _ in range(1):
                    self.ps.density[p_i] *= self.density_0
    

    # @ti.func
    # def compute_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
    #     x_i = self.ps.x[p_i]
    #     dpi = self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
    #     # Fluid neighbors
    #     if self.ps.material[p_j] == self.ps.material_fluid:
    #         x_j = self.ps.x[p_j]
    #         density_j = self.ps.density[p_j] * self.density_0 / self.density_0  # TODO: The density_0 of the neighbor may be different when the fluid density is different
    #         dpj = self.ps.pressure[p_j] / (density_j * density_j)
    #         # Compute the pressure force contribution, Symmetric Formula
    #         ret += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
    #             * self.cubic_kernel_derivative(x_i-x_j)
    #     elif self.ps.material[p_j] == self.ps.material_solid:
    #         # Boundary neighbors
    #         dpj = self.ps.pressure[p_i] / self.density_0 ** 2
    #         ## Akinci2012
    #         x_j = self.ps.x[p_j]
    #         # Compute the pressure force contribution, Symmetric Formula
    #         f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
    #             * self.cubic_kernel_derivative(x_i-x_j)
    #         ret += f_p
    #         if self.ps.is_dynamic_rigid_body(p_j):
    #             self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]
    
    # @ti.kernel
    # def compute_pressure_forces(self):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.material[p_i] == self.ps.material_fluid:
    #             self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
    #             self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.is_static_rigid_body(p_i):
    #             self.ps.acceleration[p_i].fill(0)
    #         elif self.ps.is_dynamic_rigid_body(p_i):
    #             pass
    #         else:
    #             dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
    #             self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dv)
    #             self.ps.acceleration[p_i] += dv
    

    @ti.func
    def compute_pressure_forces_task(self, p_i, p_j):
        x_i = self.ps.x[p_i]
        dpi = self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
        # Fluid neighbors
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            density_j = self.ps.density[p_j] * self.density_0 / self.density_0  # TODO: The density_0 of the neighbor may be different when the fluid density is different
            dpj = self.ps.pressure[p_j] / (density_j * density_j)
            # Compute the pressure force contribution, Symmetric Formula
            self.ps.acceleration[p_i] += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                * self.cubic_kernel_derivative(x_i-x_j)
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            dpj = self.ps.pressure[p_i] / self.density_0 ** 2
            ## Akinci2012
            x_j = self.ps.x[p_j]
            # Compute the pressure force contribution, Symmetric Formula
            f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                * self.cubic_kernel_derivative(x_i-x_j)
            self.ps.acceleration[p_i] += f_p
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]


    # @ti.kernel
    # def compute_pressure_forces(self):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.material[p_i] == self.ps.material_fluid:
    #             self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
    #             self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.is_static_rigid_body(p_i):
    #             self.ps.acceleration[p_i].fill(0)
    #         elif self.ps.is_dynamic_rigid_body(p_i):
    #             pass
    #         else:
    #             for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
    #                 center_cell = self.ps.pos_to_index(self.ps.x[p_i])
    #                 grid_index = self.ps.flatten_grid_index(center_cell + offset)
    #                 for p_j in range(self.ps.grid_particles_num[ti.max(0, grid_index-1)], self.ps.grid_particles_num[grid_index]):
    #                     if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
    #                         self.compute_pressure_forces_task(p_i, p_j)

    @ti.kernel
    def compute_pressure(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
                self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)


    @ti.kernel
    def compute_pressure_forces_kernel(self):
        # for p_i in ti.grouped(self.ps.x):
        for p_i in range(self.ps.particle_max_num):
            # if self.ps.is_static_rigid_body(p_i):
            #     self.ps.acceleration[p_i].fill(0)
            # elif self.ps.is_dynamic_rigid_body(p_i):
            #     pass
            # else:
            if self.ps.material[p_i] == self.ps.material_fluid:
                for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                    center_cell = self.ps.pos_to_index(self.ps.x[p_i])
                    grid_index = self.ps.flatten_grid_index(center_cell + offset)
                    x_i = self.ps.x[p_i]
                    dpi = self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
                    for p_j in range(self.ps.grid_particles_num[ti.max(0, grid_index-1)], self.ps.grid_particles_num[grid_index]):
                        if p_i != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
                            # Fluid neighbors
                            if self.ps.material[p_j] == self.ps.material_fluid:
                                x_j = self.ps.x[p_j]
                                density_j = self.ps.density[p_j] * self.density_0 / self.density_0  # TODO: The density_0 of the neighbor may be different when the fluid density is different
                                dpj = self.ps.pressure[p_j] / (density_j * density_j)
                                # Compute the pressure force contribution, Symmetric Formula
                                self.ps.acceleration[p_i] += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                                    * self.cubic_kernel_derivative(x_i-x_j)
                            elif self.ps.material[p_j] == self.ps.material_solid:
                                # Boundary neighbors
                                dpj = self.ps.pressure[p_i] / self.density_0 ** 2
                                ## Akinci2012
                                x_j = self.ps.x[p_j]
                                # Compute the pressure force contribution, Symmetric Formula
                                f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                                    * self.cubic_kernel_derivative(x_i-x_j)
                                self.ps.acceleration[p_i] += f_p

                                if self.ps.is_dynamic_rigid_body(p_j):
                                    self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]
            elif self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)

    def compute_pressure_forces(self):
        self.compute_pressure()
        self.compute_pressure_forces_kernel()


    # @ti.func
    # def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
    #     x_i = self.ps.x[p_i]
        
    #     ############## Surface Tension ###############
    #     if self.ps.material[p_j] == self.ps.material_fluid:
    #         # Fluid neighbors
    #         diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
    #         x_j = self.ps.x[p_j]
    #         r = x_i - x_j
    #         r2 = r.dot(r)
    #         if r2 > diameter2:
    #             ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
    #         else:
    #             ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
            
        
    #     ############### Viscosoty Force ###############
    #     d = 2 * (self.ps.dim + 2)
    #     x_j = self.ps.x[p_j]
    #     # Compute the viscosity force contribution
    #     r = x_i - x_j
    #     v_xy = (self.ps.v[p_i] -
    #             self.ps.v[p_j]).dot(r)
        
    #     if self.ps.material[p_j] == self.ps.material_fluid:
    #         f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
    #             r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
    #         ret += f_v
    #     elif self.ps.material[p_j] == self.ps.material_solid:
    #         boundary_viscosity = 0.0
    #         # Boundary neighbors
    #         ## Akinci2012
    #         f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
    #             r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
    #         ret += f_v
    #         if self.ps.is_dynamic_rigid_body(p_j):
    #             self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]


    # @ti.kernel
    # def compute_non_pressure_forces(self):
    #     for p_i in ti.grouped(self.ps.x):
    #         if self.ps.is_static_rigid_body(p_i):
    #             self.ps.acceleration[p_i].fill(0.0)
    #         else:
    #             ############## Body force ###############
    #             # Add body force
    #             d_v = ti.Vector(self.g)
    #             self.ps.acceleration[p_i] = d_v
    #             if self.ps.material[p_i] == self.ps.material_fluid:
    #                 self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
    #                 self.ps.acceleration[p_i] = d_v


    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j):
        x_i = self.ps.x[p_i]
        
        ############## Surface Tension ###############
        # if self.ps.material[p_j] == self.ps.material_fluid:
        #     # Fluid neighbors
        #     diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
        #     x_j = self.ps.x[p_j]
        #     r = x_i - x_j
        #     r2 = r.dot(r)
        #     if r2 > diameter2:
        #         self.ps.acceleration[p_i] -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
        #     else:
        #         self.ps.acceleration[p_i] -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
            
        
        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        
        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            self.ps.acceleration[p_i] += f_v
        # elif self.ps.material[p_j] == self.ps.material_solid:
        #     boundary_viscosity = 0.0
        #     # Boundary neighbors
        #     ## Akinci2012
        #     f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
        #         r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
        #     self.ps.acceleration[p_i] += f_v
        #     if self.ps.is_dynamic_rigid_body(p_j):
        #         self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]


    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
            else:
                ############## Body force ###############
                # Add body force
                # d_v = ti.Vector(self.g)
                # self.ps.acceleration[p_i] = d_v
                if self.ps.material[p_i] == self.ps.material_fluid:
                    for _ in range(1):
                        self.ps.acceleration[p_i] = ti.Vector(self.g)
                    for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                        center_cell = self.ps.pos_to_index(self.ps.x[p_i])
                        grid_index = self.ps.flatten_grid_index(center_cell + offset)
                        for p_j in range(self.ps.grid_particles_num[ti.max(0, grid_index-1)], self.ps.grid_particles_num[grid_index]):
                            if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
                                self.compute_non_pressure_forces_task(p_i, p_j)
                else:
                    self.ps.acceleration[p_i] = ti.Vector(self.g)


    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.ps.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.ps.dt[None] * self.ps.v[p_i]


    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()


    @ti.kernel
    def compute_avg_pos(self, object_id: int, object_particle_num: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.object_id[p_i] == object_id:
                self.ps.objects_center[object_id] += self.ps.x[p_i] / object_particle_num


    @ti.kernel
    def compute_loss(self, object_id: int):
        dist = (self.ps.objects_center[object_id] - ti.Vector(self.ps.target_position))**2
        self.ps.loss[None] += dist[0] + dist[1] + dist[2]


    @ti.kernel
    def set_object_density(self, object_id: int):
        for i in self.ps.density:
            if self.ps.object_id[i] == object_id:
                self.ps.density[i] = self.ps.object_density[None]
