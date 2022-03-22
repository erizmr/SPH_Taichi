import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# ti.init(arch=ti.vulkan, log_level=ti.TRACE)
ti.init(arch=ti.cpu, device_memory_fraction=0.5, print_ir=False, ad_stack_size=128)
screen_res = (1000, 1000)

boundary_box_np = np.array([[0, 0, 0], [1, 1, 1]])
spawn_box_np = np.array([[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]])

particle_radius = 0.01
particle_diameter = particle_radius * 2
N_np = ((spawn_box_np[1] - spawn_box_np[0]) / particle_diameter + 1).astype(int)

h = 4.0 * particle_radius

particle_num = N_np[0] * N_np[1] * N_np[2]
print(particle_num)
steps = 1024

pos = ti.Vector.field(3, float)
vel = ti.Vector.field(3, float)
acc = ti.Vector.field(3, float)
control_force = ti.Vector.field(3, float)
external_force = ti.Vector.field(3, float, shape=())

col = ti.field(float)
den = ti.field(float)
pre = ti.field(float)
loss = ti.field(float)

pos_vis_buffer = ti.Vector.field(3, float, shape=particle_num)

pos_output_buffer = ti.Vector.field(3, float)

ti.root.dense(ti.i, steps).dense(ti.j, int(particle_num)).place(pos, vel, acc, col, den, pre, control_force, pos_output_buffer)
ti.root.place(loss)
ti.root.lazy_grad()


boundary_box = ti.Vector.field(3, float, shape=2)
spawn_box = ti.Vector.field(3, float, shape=2)
N = ti.Vector([N_np[0], N_np[1], N_np[2]])

gravity = ti.Vector([0, -9.8, 0])

boundary_box.from_numpy(boundary_box_np)
spawn_box.from_numpy(spawn_box_np)

rest_density = 1000.0
mass = rest_density * particle_diameter * particle_diameter * particle_diameter * 0.8
pressure_scale = 10000.0
viscosity_scale = 0.1 * 3
tension_scale = 0.005
gamma = 1.0
substeps = 100
dt = 0.016 / substeps
# dt = 1.0/60
eps = 1e-6
damping = 0.5
pi = math.pi


@ti.func
def W_poly6(R, h):
    r = R.norm(eps)
    res = 0.0
    if r <= h:
        h2 = h * h
        h4 = h2 * h2
        h9 = h4 * h4 * h
        h2_r2 = h2 - r * r
        res = 315.0 / (64 * pi * h9) * h2_r2 * h2_r2 * h2_r2
    else:
        res = 0.0
    return res


@ti.func
def W_spiky_gradient(R, h):
    r = R.norm(eps)
    res = ti.Vector([0.0, 0.0, 0.0])
    if r == 0.0:
        res = ti.Vector([0.0, 0.0, 0.0])
    elif r <= h:
        h3 = h * h * h
        h6 = h3 * h3
        h_r = h - r
        res = -45.0 / (pi * h6) * h_r * h_r * (R / r)
    else:
        res = ti.Vector([0.0, 0.0, 0.0])
    return res


W = W_poly6
W_gradient = W_spiky_gradient


@ti.kernel
def initialize_particle(t: ti.int32, pos: ti.template(), N: ti.template()):
    for i in range(particle_num):
        pos[t, i] = (
            ti.Vector(
                [int(i % N[0]), int(i / N[0]) % N[1], int(i / N[0] / N[1] % N[2])]
            )
            * particle_diameter
            + spawn_box[0]
        )


@ti.kernel
def update_density(t: ti.int32, pos: ti.template(), den: ti.template(), pre: ti.template()):
    for i in range(particle_num):
        den[t, i] = 0
        for j in range(particle_num):
            R = pos[t, i] - pos[t, j]
            den[t, i] += mass * W(R, h)
        pre[t, i] = pressure_scale * max(pow(den[t, i] / rest_density, gamma) - 1, 0)


@ti.kernel
def update_force(t: ti.int32,
    pos: ti.template(), vel: ti.template(), den: ti.template(), pre: ti.template()
):
    for i in range(particle_num):
        # acc[t, i] = gravity
        # for k in ti.static(range(3)):
        #     acc[t, i][k] = control_force[t, i][k]
        acc[t, i] = external_force[None]
        for j in range(particle_num):
            R = pos[t, i] - pos[t, j]

            acc[t, i] += (
                -mass
                * (pre[t, i] / (den[t, i] * den[t, i]) + pre[t, j] / (den[t, j] * den[t, j]))
                * W_gradient(R, h)
            )

            acc[t, i] += (
                viscosity_scale
                * mass
                * (vel[t, i] - vel[t, j]).dot(R)
                / (R.norm(eps) + 0.01 * h * h)
                / den[t, j]
                * W_gradient(R, h)
            )
            # TODO: RuntimeError: [verify.cpp:visit@75] block(0x7790290)->parent($12441) != current_container_stmt($11940)
            # R2 = R.dot(R)
            # D2 = particle_diameter * particle_diameter
            # if R2 > D2:
            #     acc[t, i] += -tension_scale * R * W(R, h)
            # else:
            #     acc[t, i] += (
            #         -tension_scale
            #         * R
            #         * W(ti.Vector([0.0, 1.0, 0.0]) * particle_diameter, h)
            #     )


@ti.kernel
def advance(t:ti.int32, pos: ti.template(), vel: ti.template(), acc: ti.template()):
    for i in range(particle_num):
        vel[t, i] += acc[t, i] * dt
        pos[t, i] += vel[t, i] * dt
        col[t, i] = 0xFFFFFF
    # col[i] = ti.Vector([den[i] / rest_density, 0.0, 0.0])
    # col[i] = pre[i] / rest_density * 255


@ti.kernel
def boundary_handle(t:ti.int32,
    pos: ti.template(), vel: ti.template(), boundary_box: ti.template()
):
    for i in range(particle_num):
        collision_normal = ti.Vector([0.0, 0.0, 0.0])
        for j in ti.static(range(3)):
            if pos[t, i][j] < boundary_box[0][j]:
                pos[t, i][j] = boundary_box[0][j]
                collision_normal[j] += -1.0
        for j in ti.static(range(3)):
            if pos[t, i][j] > boundary_box[1][j]:
                pos[t, i][j] = boundary_box[1][j]
                collision_normal[j] += 1.0
        collision_normal_length = collision_normal.norm()
        if collision_normal_length > eps:
            collision_normal /= collision_normal_length
            vel[t, i] -= (1.0 + damping) * collision_normal.dot(vel[t, i]) * collision_normal

        vel[t+1, i] = vel[t, i]
        pos[t+1, i] = pos[t, i]


@ti.kernel
def compute_loss(t: ti.int32):
    for i in range(particle_num):
        loss[None] += ti.sqrt((pos[t, i][1] - (0.7 + 0.001 * t)) ** 2)

@ti.kernel
def copy_to_vis(t: ti.int32):
    for i in range(particle_num):
        for j in ti.static(range(3)):
            pos_vis_buffer[i][j] = pos[t, i][j]


@ti.kernel
def copy_to_output_buffer(t:ti.int32):
    for i in range(particle_num):
        for j in ti.static(range(3)):
            pos_output_buffer[t, i][j] = pos[t, i][j]


@ti.kernel
def copy_from_output_to_vis(t: ti.int32):
    for i in range(particle_num):
        for j in ti.static(range(3)):
            pos_vis_buffer[i][j] = pos_output_buffer[t, i][j]


@ti.kernel
def optimize_control_forces():
    for i in range(substeps):
        for j in range(particle_num):
            # control_force[i, j] -= learning_rate * acc.grad[i, j]
            external_force[None] -= learning_rate * acc.grad[i, j]


initialize_particle(0, pos, N)

window = ti.ui.Window("SPH", screen_res, vsync=True, show_window=False)
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 2.0)
camera.up(0.0, 1.0, 0.0)
camera.lookat(0.5, 0.5, 0.5)
camera.fov(70)
scene.set_camera(camera)
canvas = window.get_canvas()
movement_speed = 0.02

# while window.running:

# while 1:
    # with ti.Tape(loss=loss):
    #     # loss.grad[None] = 0.0
    #     for i in range(substeps):
    #         update_density(i, pos, den, pre)
    #         update_force(i, pos, vel, den, pre)
    #         advance(i, pos, vel, acc)
    #         boundary_handle(i, pos, vel, boundary_box)
    #     compute_loss(substeps - 1)
    #     print('loss ', loss[None], 'grad ', pos.grad[substeps - 1, 0])
learning_rate = 1.0
losses = []
opt_iters = 90
for opt_iter in range(opt_iters):
    with ti.Tape(loss=loss):
        for i in range(substeps+1):
            update_density(i, pos, den, pre)
            update_force(i, pos, vel, den, pre)
            advance(i, pos, vel, acc)
            boundary_handle(i, pos, vel, boundary_box)
            # print(f"{i} step finished.")
            copy_to_output_buffer(i)
            compute_loss(i)
    optimize_control_forces()

    print('loss ', loss[None])
    losses.append(loss[None])
    copy_to_vis(substeps-1)
    p_n = 100

    # for i in range(substeps):
    #     print(' Opt Iter ', opt_iter,  ' Step ', i, ' pos', pos[i, p_n], ' grad pos ', pos.grad[i, p_n],
    #           ' grad acc ', acc.grad[i, p_n],
    #           ' control_force ', control_force[i, p_n])
    print('opt iter ', opt_iter, ' loss ', loss[None], 'external force', external_force[None], ' control force [0, 0] ', control_force[0, 0], ' control force [-1, 5]', control_force[substeps-1,5])

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
    # scene.set_camera(camera)
    # scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
    # scene.particles(pos_vis_buffer, radius=particle_radius, color=(0.4, 0.7, 1.0))
    # canvas.scene(scene)
    # window.show()
    if opt_iter % 10 == 0:
        os.makedirs(f"output_img/{opt_iter}", exist_ok=True)
        for i in range(substeps + 1):
            copy_from_output_to_vis(i)
            scene.set_camera(camera)
            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            scene.particles(pos_vis_buffer, radius=particle_radius, color=(0.4, 0.7, 1.0))
            canvas.scene(scene)
            window.write_image(f'output_img/{opt_iter}/{i:04}.png')
print(losses)
plt.plot([i for i in range(len(losses))], losses)
plt.show()
