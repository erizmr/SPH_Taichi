import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from WCSPH import WCSPHSolver
from IISPH import IISPHSolver

# ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
ti.init(arch=ti.vulkan, device_memory_fraction=0.8, packed=True) #, log_level=ti.TRACE)


if __name__ == "__main__":
    domain_size = 2
    dim = 3
    substeps = 5
    output_frames = False
    # solver_type = "WCSPH"
    solver_type = "IISPH"
    ps = ParticleSystem((domain_size,)*dim, GGUI=True)

    if dim == 2:
        ps.add_cube(lower_corner=[0.8, 0.8],
                    cube_size=[0.8, 0.3],
                    velocity=[0.0, -5.0],
                    density=1000.0,
                    color=(177,213,200),
                    material=1)

        ps.add_cube(lower_corner=[0.8, 0.1],
                    cube_size=[0.4, 0.3],
                    velocity=[0.0, 5.0],
                    density=1000.0,
                    color=(177,213,200),
                    material=1)

        # Boundaries
        # Bottom
        ps.add_cube(lower_corner=[0.0, 0.0],
                    cube_size=[2.0, 0.025],
                    velocity=[0.0, 0.0],
                    density=1000.0,
                    color=(255,255,255),
                    material=0)

        # Left
        ps.add_cube(lower_corner=[0.0, 0.025],
                    cube_size=[0.025, 2.0],
                    velocity=[0.0, 0.0],
                    density=1000.0,
                    color=(255,255,255),
                    material=0)

        # Top
        ps.add_cube(lower_corner=[0.025, 2.0 - 0.025],
                    cube_size=[2.0, 0.025],
                    velocity=[0.0, 0.0],
                    density=1000.0,
                    color=(255,255,255),
                    material=0)

        # Right
        ps.add_cube(lower_corner=[2.0 - 0.025, 0.05],
                    cube_size=[0.025, 2.0 - 0.025],
                    velocity=[0.0, 0.0],
                    density=1000.0,
                    color=(255,255,255),
                    material=0)

    elif dim == 3:
        # Fluid -1 
        ps.add_cube(lower_corner=[0.6, 0.1, 0.6],
                    cube_size=[0.6, 1.8, 0.6],
                    velocity=[0.0, -1.0, 0.0],
                    density=1000.0,
                    color=(177,213,200),
                    material=1)

        # Fluid -2 
        # ps.add_cube(lower_corner=[1.2, 1.0, 1.2],
        #             cube_size=[0.4, 0.6, 0.4],
        #             velocity=[0.0, -1.0, 0.0],
        #             density=1000.0,
        #             color=(255,177,27),
        #             material=1)
        # Boundary -1
        # ps.add_cube(lower_corner=[0.6, 0.025, 0.6],
        #             cube_size=[0.4, 0.2, 0.4],
        #             velocity=[0.0, 0.0, 0.0],
        #             density=1000.0,
        #             color=(255,255,255),
        #             material=0)

        # Bottom boundary
        ps.add_cube(lower_corner=[0.0, 0.0, 0.0],
                    cube_size=[2.0, ps.particle_diameter, 2.0],
                    velocity=[0.0, 0.0, 0.0],
                    density=1000.0,
                    color=(255,255,255),
                    material=0)

    if solver_type == "WCSPH":
        solver = WCSPHSolver(ps)
    elif solver_type == "IISPH":
        solver = IISPHSolver(ps)


    window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=True)

    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.5, 1.0, 2.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(60)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    cnt = 0
    solver.initialize_solver()
    while window.running:
        for i in range(substeps):
            solver.step()
        ps.copy_to_vis_buffer()
        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius / 5, color=particle_color)
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
            scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)
            canvas.scene(scene)

        if output_frames:
            if cnt % 20 == 0:
                window.write_image(f"img_output/{cnt:04}.png")
        cnt += 1
        window.show()
