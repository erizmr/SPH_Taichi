# SPH taichi implementation by mzhang
import taichi as ti
from engine.sph_solver import *
import argparse

# Default run on CPU
# cuda performance has not been tested
ti.init(arch=ti.cpu)


def main(opt):
    dynamic_allocate = opt.dynamic_allocate
    save_frames = opt.save
    adaptive_time_step = opt.adaptive
    method_name = opt.method
    sim_physical_time = 5.0
    max_frame = 50000

    res = (400, 400)
    screen_to_world_ratio = 35
    dx = 0.1
    u, b, l, r = np.array([res[1], 0, 0, res[0]]) / screen_to_world_ratio

    gui = ti.GUI('SPH', res, background_color=0x112F41)
    sph = SPHSolver(res,
                    screen_to_world_ratio, [u, b, l, r],
                    alpha=0.30,
                    dx=dx,
                    max_time=sim_physical_time,
                    max_steps=max_frame,
                    dynamic_allocate=dynamic_allocate,
                    adaptive_time_step=adaptive_time_step,
                    method=SPHSolver.methods[method_name])

    print("Method use: %s" % method_name)
    # Add fluid particles
    sph.add_cube(lower_corner=[res[0] / 2 / screen_to_world_ratio - 3, 4 * dx],
                 cube_size=[6, 6],
                 velocity=[0.0, -5.0],
                 density=[1000],
                 color=0x068587,
                 material=SPHSolver.material_fluid)

    colors = np.array([0xED553B, 0x068587, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    add_cnt = 0.0
    add = True
    save_cnt = 0.0
    output_fps = 60
    save_point = 1.0 / output_fps
    t = 0.0
    frame = 0
    total_start = time.process_time()
    while frame < max_frame and t < sim_physical_time:
        dt = sph.step(frame, t, total_start)
        particles = sph.particle_info()

        # if frame == 1000:
        if add and add_cnt > 0.40:
            sph.add_cube(lower_corner=[6, 6],
                         cube_size=[2.0, 2.0],
                         velocity=[0.0, -5.0],
                         density=[1000.0],
                         color=0xED553B,
                         material=SPHSolver.material_fluid)

        # if frame == 1000:
        if add and add_cnt > 0.40:
            sph.add_cube(lower_corner=[3, 8],
                         cube_size=[1.0, 1.0],
                         velocity=[0.0, -10.0],
                         density=[1000.0],
                         color=0xEEEEF0,
                         material=SPHSolver.material_fluid)
            add = False

        for pos in particles['position']:
            for j in range(len(res)):
                pos[j] *= screen_to_world_ratio / res[j]

        gui.circles(particles['position'],
                    radius=1.5,
                    color=particles['color'])

        # Save in fixed frame interval, for fixed time step
        if not adaptive_time_step:
            if frame % 50 == 0:
                gui.show(f'{frame:06d}.png' if save_frames else None)
            else:
                gui.show()

        # Save in fixed frame per second, for adaptive time step
        if adaptive_time_step:
            if save_cnt >= save_point:
                gui.show(f'{frame:06d}.png' if save_frames else None)
            else:
                gui.show()
            save_cnt = 0.0

        frame += 1
        t += dt
        save_cnt += dt
        add_cnt += dt

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                        type=str,
                        default="PCISPH",
                        help="SPH methods: WCSPH, PCISPH, DFSPH")
    parser.add_argument("--save",
                        action='store_true',
                        help="save frames")
    parser.add_argument("--adaptive",
                        action='store_true',
                        help="whether apply adaptive step size")
    parser.add_argument("--dynamic-allocate",
                        action='store_true',
                        help="whether apply dynamic allocation")
    opt = parser.parse_args()
    main(opt)
