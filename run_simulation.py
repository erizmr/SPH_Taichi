import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system import ParticleSystem

ti.init(arch=ti.gpu, device_memory_fraction=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]

    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_frames = config.get_cfg("exportFrame")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"{scene_name}_output", exist_ok=True)


    ps = ParticleSystem(config, GGUI=True)
    solver = ps.build_solver()
    solver.initialize()

    window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5.5, 2.5, 4.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, 0.0)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []

    # Draw the lines for domain
    x_max, y_max, z_max = config.get_cfg("domainEnd")
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

    cnt = 0
    cnt_ply = 0

    while window.running:
        for i in range(substeps):
            solver.step()
        ps.copy_to_vis_buffer(invisible_objects=invisible_objects)
        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius, color=particle_color)
        elif ps.dim == 3:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)

            scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
            canvas.scene(scene)
    
        if output_frames:
            if cnt % output_interval == 0:
                window.write_image(f"{scene_name}_output_img/{cnt:06}.png")
        
        if cnt % output_interval == 0:
            if output_ply:
                obj_id = 0
                obj_data = ps.dump(obj_id=obj_id)
                np_pos = obj_data["position"]
                writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id]["particleNum"])
                writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
            if output_obj:
                for r_body_id in ps.object_id_rigid_body:
                    with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
                        e = ps.object_collection[r_body_id]["mesh"].export(file_type='obj')
                        f.write(e)
            cnt_ply += 1

        cnt += 1
        # if cnt > 6000:
        #     break
        window.show()
