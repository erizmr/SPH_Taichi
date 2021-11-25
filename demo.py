import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from sph_base import WCSPHSolver
ti.init(arch=ti.gpu)

if __name__ == "__main__":
    ps = ParticleSystem((512, 512))
    
    ps.add_cube(lower_corner=[3, 4],
                         cube_size=[2.0, 2.0],
                         velocity=[0.0, -10.0],
                         density=[1000.0],
                         color=0xEEEEF0,
                         material=1)
    wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI()
    while gui.running:
        for i in range(5):
            wcsph_solver.step()

        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.dx / 2  * ps.screen_to_world_ratio,
                    color=0x956333)
        gui.show()