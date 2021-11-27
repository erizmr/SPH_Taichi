import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from sph_base import WCSPHSolver

ti.init(arch=ti.gpu, device_memory_GB=3, packed=True)

if __name__ == "__main__":
    ps = ParticleSystem((512, 512))

    # ps.add_cube(lower_corner=[3, 1],
    #             cube_size=[1.0, 1.0],
    #             velocity=[0.0, 12.0],
    #             density=1000.0,
    #             color=0xFFFFF0,
    #             material=1)

    ps.add_cube(lower_corner=[6, 2],
                cube_size=[1.0, 1.0],
                velocity=[-10.0, -0.0],
                density=1000.0,
                color=0xFFFFF0,
                material=1)

    ps.add_cube(lower_corner=[3, 2],
                cube_size=[2.0, 6.0],
                velocity=[0.0, -20.0],
                density=1000.0,
                color=0xEEEEE0,
                material=1)

    wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        for i in range(5):
            wcsph_solver.step()

        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=0x956333)
        gui.show()
