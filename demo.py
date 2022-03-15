import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from wcsph import WCSPHSolver

# ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.cuda, device_memory_GB=3, packed=True)



if __name__ == "__main__":
    ps = ParticleSystem((512, 512), GGUI=True)

    ps.add_cube(lower_corner=[6, 2],
                cube_size=[3.0, 5.0],
                velocity=[-5.0, -10.0],
                density=1000.0,
                color=0x956333,
                material=1)

    ps.add_cube(lower_corner=[3, 1],
                cube_size=[2.0, 6.0],
                velocity=[0.0, -20.0],
                density=1000.0,
                color=0x956333,
                material=1)

    wcsph_solver = WCSPHSolver(ps)
    # gui = ti.GUI(background_color=0xFFFFFF)
    # while gui.running:
    #     for i in range(5):
    #         wcsph_solver.step()
    #     particle_info = ps.dump()
    #     gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
    #                 radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
    #                 color=0x956333)
    #     gui.show()

    window = ti.ui.Window('Window Title', (1024, 1024))
    canvas = window.get_canvas()
    radius = 0.003
    background_color = (1, 1, 1)  # 0xFFFFFF
    particle_color = (149 / 255, 99 / 255, 51 / 255)  # 0x956333

    while window.running:
        for i in range(5):
            wcsph_solver.step()
        ps.copy_to_vis_buffer()
        canvas.set_background_color(background_color)
        canvas.circles(ps.x_vis_buffer, radius=radius, color=particle_color)
        window.show()
