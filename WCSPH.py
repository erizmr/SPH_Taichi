import taichi as ti
from sph_base import SPHBase


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        self.exponent = 7.0
        self.stiffness = 50000.0
        # self.dt[None] = 3e-4
        self.dt[None] = 3e-4


    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
            # Boundary neighbors
            ## Akinci2012
            for j in range(self.ps.solid_neighbors_num[p_i]):
                p_j = self.ps.solid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
            self.ps.density[p_i] *= self.density_0


    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue

            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])

            dpi = self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
            # Fluid neighbors
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]   
                x_j = self.ps.x[p_j]
                density_j = self.ps.density[p_j] * self.density_0 / self.density_0  # TODO: The density_0 of the neighbor may be different when the fluid density is different
                dpj = self.ps.pressure[p_j] / (density_j * density_j)
                # Compute the pressure force contribution, Symmetric Formula
                d_v += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                    * self.cubic_kernel_derivative(x_i-x_j)
            
            # Boundary neighbors
            dpj = self.ps.pressure[p_i] / self.density_0 ** 2
            ## Akinci2012
            for j in range(self.ps.solid_neighbors_num[p_i]):
                p_j = self.ps.solid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                # Compute the pressure force contribution, Symmetric Formula
                f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
                    * self.cubic_kernel_derivative(x_i-x_j)
                d_v += f_p
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[p_j] += -f_p

            self.ps.acceleration[p_i] += d_v


    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            x_i = self.ps.x[p_i]

            ############## Body force ###############
            # Add body force
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            d_v[1] = self.g
            self.ps.acceleration[p_i] = d_v

            if self.ps.material[p_i] == self.ps.material_fluid:
                
                ############## Surface Tension ###############
                surface_tension = 0.01
                diameter2 = self.ps.particle_diameter * self.ps.particle_diameter

                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    r = x_i - x_j
                    r2 = r.dot(r)
                    if r2 > diameter2:
                        d_v -= surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
                    else:
                        d_v -= surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
                    

                ############### Viscosoty Force ###############
                d = 2 * (self.ps.dim + 2)
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    # Compute the viscosity force contribution
                    r = x_i - x_j
                    v_xy = (self.ps.v[p_i] -
                            self.ps.v[p_j]).dot(r)
                    f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
                        r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
                    d_v += f_v
                
                boundary_viscosity = 0.0
                # Boundary neighbors
                ## Akinci2012
                for j in range(self.ps.solid_neighbors_num[p_i]):
                    p_j = self.ps.solid_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]

                    r = x_i - x_j
                    v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
                    f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
                        r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)

                    d_v += f_v
                    if self.ps.is_dynamic_rigid_body(p_j):
                        self.ps.acceleration[p_j] += -f_v
            
                self.ps.acceleration[p_i] = d_v


    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]


    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()
