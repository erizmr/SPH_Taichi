import trimesh
import numpy as np
import time
pts=[]

def test():
    for i in range(1,1000):
        PlyPath = f"D:\CG\meshSequence\sphere_points\sphere_points.{i:}.ply"
        print("Reading ", PlyPath)
        mesh = trimesh.load(PlyPath)
        v = mesh.vertices
        # mesh.show()
        pts.append(np.array(v))

test()

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
frameNum = 0
while window.running:
    frameNum += 1
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    particles_pos.from_numpy(pts[frameNum])

    scene.particles(particles_pos, color = (0.68, 0.26, 0.19), radius = 0.01)
    canvas.scene(scene)

    time.sleep(1.0/24)

    window.show()