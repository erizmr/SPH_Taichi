import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from wcsph import WCSPHSolver

# ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.cuda, device_memory_fraction=0.8, packed=True) #, log_level=ti.TRACE)



if __name__ == "__main__":
    domain_size = 5
    dim = 2
    ps = ParticleSystem((domain_size,)*dim, GGUI=True)

    if dim == 2:
        ps.add_cube(lower_corner=[1, 1],
                cube_size=[0.5, 0.5],
                velocity=[0.0, -5.0],
                density=1000.0,
                # color=0x956333,
                material=1)
    elif dim == 3:
        ps.add_cube(lower_corner=[1, 1, 1],
                    cube_size=[0.2, 0.2, 0.2],
                    velocity=[0.0, 0.0, 0.0],
                    density=1000.0,
                    # color=0x956333,
                    material=1)

        # ps.add_cube(lower_corner=[1, 1.2, 1],
        #             cube_size=[0.1, 0.1, 0.1],
        #             velocity=[0.0, 2.0, 0.0],
        #             density=1000.0,
        #             # color=0x956333,
        #             material=1)

    # ps.add_cube(lower_corner=[3, 1, 1],
    #             cube_size=[2.0, 6.0, 1.0],
    #             velocity=[0.0, -20.0, 0.0],
    #             density=1000.0,
    #             # color=0x956333,
    #             material=1)

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

    window = ti.ui.Window('Window Title', (1024, 1024), show_window = True)

    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.6, 0.5, 0.6)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(0.2, 0.2, 0.2)
    camera.fov(60)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.001
    movement_speed = 0.02
    background_color = (1, 1, 1)  # 0xFFFFFF
    particle_color = (149 / 255, 99 / 255, 51 / 255)  # 0x956333

    while window.running:
        for i in range(5):
            # ps.grid_node.deactivate_all()
            wcsph_solver.step()
        ps.copy_to_vis_buffer()
        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=radius, color=particle_color)
        elif ps.dim == 3:
            # user controlling of camera
            position_change = ti.Vector([0.0, 0.0, 0.0])
            up = ti.Vector([0.0, 1.0, 0.0])
            # move camera up and down
            if window.is_pressed("e"):
                position_change = up * movement_speed
            if window.is_pressed("q"):
                position_change = -up * movement_speed
            camera.position(*(camera.curr_position + position_change))
            camera.lookat(*(camera.curr_lookat + position_change))
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            scene.particles(ps.x_vis_buffer, radius=radius, color=particle_color)
            canvas.scene(scene)
        window.show()
