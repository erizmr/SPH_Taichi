import trimesh
import numpy as np
import time

plys=[]
def read_ply(ply_path_no_ext, start=1, stop=1000):
    pts=[]
    for i in range(start, stop):
        ply_path = ply_path_no_ext + f".{i:}.ply"
        print("Reading ", ply_path)
        mesh = trimesh.load(ply_path)
        v = mesh.vertices
        # mesh.show()
        pts.append(np.array(v))
    return pts


ply_path = "D:\CG\meshSequence\sphere_points\sphere_points"
plys = read_ply(ply_path, start=1, stop=1000)


import taichi as ti

ti.init()

N=6121
particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)

window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(1,1,2)
camera.lookat(0, 0, 0)
frame_num = 0
while window.running:
    frame_num += 1
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    particles_pos.from_numpy(plys[frame_num])

    scene.particles(particles_pos, color = (0.6, 0.26, 0.19), radius = 0.01)
    canvas.scene(scene)

    time.sleep(1.0/24)

    window.show()