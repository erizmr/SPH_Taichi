# SPH taichi implementation by mzhang
import taichi as ti
from engine.sph_solver import *

# Default run on CPU
# cuda performance has not been tested
ti.init(arch=ti.cpu)


def main():
    dynamic_allocate = True
    save_frames = True
    # method_name = 'WCSPH'
    method_name = 'PCISPH'
    # method_name = 'DFSPH'

    res = (400, 400)
    screen_to_world_ratio = 35
    dx = 0.1
    u, b, l, r = np.array([res[1], 0, 0, res[0]]) / screen_to_world_ratio

    gui = ti.GUI('SPH', res, background_color=0x112F41)
    sph = SPHSolver(res,
                    screen_to_world_ratio, [u, b, l, r],
                    alpha=0.30,
                    dx=dx,
                    max_time=5.0,
                    max_steps=50000,
                    dynamic_allocate=dynamic_allocate,
                    method=SPHSolver.methods[method_name])

    print("Method use: %s" % method_name)
    # Add fluid particles
    sph.add_cube(lower_corner=[res[0] / 2 / screen_to_world_ratio - 3, 4 * dx],
                 cube_size=[6, 6],
                 velocity=[0.0, -5.0],
                 density=[1000],
                 color=0x068587,
                 material=SPHSolver.material_fluid)

    # Add bottom boundary
    # sph.add_cube(lower_corner=[0.0, 0.0],
    #              cube_size=[res[0] / screen_to_world_ratio, 2 * dx],
    #              velocity=[0.0, 0.0],
    #              density=[1000],
    #              material=SPHSolver.material_bound)

    colors = np.array([0xED553B, 0x068587, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)

    t = 0.0
    frame = 0
    total_start = time.process_time()
    while frame < 50000 and t < 30:
        dt = sph.step(frame, t, total_start)
        particles = sph.particle_info()

        if dynamic_allocate and frame == 1000:
            sph.add_cube(lower_corner=[6, 6],
                         cube_size=[2.0, 2.0],
                         velocity=[0.0, -5.0],
                         density=[1000],
                         color=0xED553B,
                         material=SPHSolver.material_fluid)

        if dynamic_allocate and frame == 1000:
            sph.add_cube(lower_corner=[3, 8],
                         cube_size=[1.0, 1.0],
                         velocity=[0.0, -10.0],
                         density=[1000],
                         color=0xEEEEF0,
                         material=SPHSolver.material_fluid)

        for pos in particles['position']:
            for j in range(len(res)):
                pos[j] *= screen_to_world_ratio / res[j]

        gui.circles(particles['position'],
                    radius=1.5,
                    color=particles['color'])
        if frame % 50 == 0:
            gui.show(f'{frame:06d}.png' if save_frames else None)

        frame += 1
        t += dt

    print('done')


if __name__ == '__main__':
    main()
