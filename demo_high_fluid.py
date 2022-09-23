import taichi as ti
import numpy as np
import trimesh as tm
from particle_system import ParticleSystem
from WCSPH import WCSPHSolver
from IISPH import IISPHSolver

# ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
ti.init(arch=ti.cuda,  device_memory_GB=2, kernel_profiler=True)


if __name__ == "__main__":
    x_max = 2.0
    y_max = 6.0
    z_max = 2.0

    domain_size = np.array([x_max, y_max, z_max])

    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
    box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
    box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
    box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
    box_anchors[3] = ti.Vector([x_max, y_max, 0.0])

    box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
    box_anchors[5] = ti.Vector([0.0, y_max, z_max])
    box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
    box_anchors[7] = ti.Vector([x_max, y_max, z_max])

    box_lines_indices = ti.field(int, shape=(2 * 12))

    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val

    dim = 3
    substeps = 1
    output_frames = False
    output_ply = False
    solver_type = "WCSPH"
    # solver_type = "IISPH"
    ps = ParticleSystem(domain_size, GGUI=True)

    x_offset = 0.2
    y_offset = 0.2
    z_offset = 0.2

    # mesh = tm.load("./data/Dragon_50k.obj")
    # # mesh = tm.load("./data/bunny_sparse.obj")
    # # mesh = tm.load("./data/bunny.stl")
    # mesh_scale = 0.8
    # mesh.apply_scale(mesh_scale)
    # offset = np.array([1.5, 0.0 + y_offset, 1.0])
    # is_success = tm.repair.fill_holes(mesh)
    # print("Is the mesh successfully repaired? ", is_success)
    # voxelized_mesh = mesh.voxelized(pitch=ps.particle_diameter).fill()
    # # voxelized_mesh = mesh.voxelized(pitch=ps.particle_diameter).hollow()
    # # voxelized_mesh.show()
    # voxelized_points_np = voxelized_mesh.points + offset
    # num_particles_obj = voxelized_points_np.shape[0]
    # voxelized_points = ti.Vector.field(3, ti.f32, num_particles_obj)
    # voxelized_points.from_numpy(voxelized_points_np)

    # print("Rigid body, num of particles: ", num_particles_obj)

    # ps.add_particles(2,
    #                  num_particles_obj,
    #                  voxelized_points_np, # position
    #                  0.0 * np.ones((num_particles_obj, 3)), # velocity
    #                  10 * np.ones(num_particles_obj), # density
    #                  np.zeros(num_particles_obj), # pressure
    #                  np.array([0 for _ in range(num_particles_obj)], dtype=int), # material
    #                  1 * np.ones(num_particles_obj), # is_dynamic
    #                  255 * np.ones((num_particles_obj, 3))) # color

    # Fluid -1 
    ps.add_cube(object_id=0,
                lower_corner=[0.1+x_offset, 0.1 + y_offset, 0.5+z_offset],
                cube_size=[0.6, 5.4, 0.6],
                velocity=[0.0, -0.0, 0.0],
                density=1000.0,
                is_dynamic=1,
                color=(50,100,200),
                material=1)

    # # Bottom boundary
    # ps.add_cube(object_id=1,
    #             lower_corner=[0.0+x_offset, 0.0 + y_offset, 0.0+z_offset],
    #             cube_size=[x_max-x_offset*2, ps.particle_diameter-0.001, z_max-z_offset*2],
    #             velocity=[0.0, 0.0, 0.0],
    #             density=1000.0,
    #             is_dynamic=0,
    #             color=(255,255,255),
    #             material=0)
    
    # # left boundary
    # ps.add_cube(object_id=1, 
    #             lower_corner=[0.0+x_offset, 0.0 + y_offset, 0.0+z_offset],
    #             cube_size=[ps.particle_diameter-0.001, y_max-y_offset*2, z_max-z_offset*2],
    #             velocity=[0.0, 0.0, 0.0],
    #             density=1000.0,
    #             is_dynamic=0,
    #             color=(255,255,255),
    #             material=0)
    
    # # back
    # ps.add_cube(object_id=1,
    #             lower_corner=[0.0+x_offset, 0.0 + y_offset, 0.0+z_offset],
    #             cube_size=[x_max-x_offset*2, y_max-y_offset*2, ps.particle_diameter-0.001],
    #             velocity=[0.0, 0.0, 0.0],
    #             density=1000.0,
    #             is_dynamic=0,
    #             color=(255,255,255),
    #             material=0)
    
    # # front
    # ps.add_cube(object_id=1,
    #             lower_corner=[0.0+x_offset, 0.0 + y_offset, z_max - z_offset],
    #             cube_size=[x_max-x_offset*2, y_max-y_offset*2, ps.particle_diameter-0.001],
    #             velocity=[0.0, 0.0, 0.0],
    #             density=1000.0,
    #             is_dynamic=0,
    #             color=(255,255,255),
    #             material=0)
    
    # # right
    # ps.add_cube(object_id=1,
    #             lower_corner=[x_max-x_offset, 0.0 + y_offset, 0.0+z_offset],
    #             cube_size=[ps.particle_diameter-0.001, y_max-y_offset*2, z_max-z_offset*2],
    #             velocity=[0.0, 0.0, 0.0],
    #             density=1000.0,
    #             is_dynamic=0,
    #             color=(255,255,255),
    #             material=0)
        
    if solver_type == "WCSPH":
        solver = WCSPHSolver(ps)
    elif solver_type == "IISPH":
        solver = IISPHSolver(ps)


    window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)

    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.0, 3.0, 5.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(60)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    cnt = 0
    cnt_ply = 0
    solver.initialize_solver()
    series_prefix = "output/object_{}_demo_test.ply"


    # ti.profiler.clear_kernel_profiler_info()
    while window.running:
        for i in range(substeps):
            solver.step()
        # ps.copy_to_vis_buffer(invisible_objects=[1])
        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius / 5, color=particle_color)
        elif ps.dim == 3:
            # # user controlling of camera
            # position_change = ti.Vector([0.0, 0.0, 0.0])
            # up = ti.Vector([0.0, 1.0, 0.0])
            # # move camera up and down
            # if window.is_pressed("e"):
            #     position_change = up * movement_speed
            # if window.is_pressed("q"):
            #     position_change = -up * movement_speed
            # camera.position(*(camera.curr_position + position_change))
            # camera.lookat(*(camera.curr_lookat + position_change))
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            # scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)
            scene.particles(ps.x, radius=ps.particle_radius, color=(50/255,100/255,200/255))

            scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
            canvas.scene(scene)
    
        if output_frames:
            if cnt % 2 == 0:
                window.write_image(f"img_high_fluid_output/{cnt:04}.png")
        if output_ply:
            if cnt % 20 == 0:
                obj_id = 0
                obj_data = ps.dump(obj_id=obj_id)
                np_pos = obj_data["position"]
                writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id])
                writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
                cnt_ply += 1
        cnt += 1
        window.show()
    ti.profiler.print_kernel_profiler_info()
