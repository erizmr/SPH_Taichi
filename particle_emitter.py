import taichi as ti
import numpy as np


def transform_by_T(pos, T):
    assert pos.shape[-1] == 3
    assert T.ndim == 2 # TODO: handle batched T

    if pos.ndim == 2:
        new_pos = np.hstack([pos, np.ones_like(pos[:, :1])]).T
        new_pos = (T @ new_pos).T
        new_pos = new_pos[:, :3]
    elif pos.ndim == 1:
        new_pos = np.append(pos, np.array(1, dtype=pos.dtype))
        new_pos = T @ new_pos
        new_pos = new_pos[:3]
    else:
        assert False

    return new_pos


def trans_R_to_T(trans, R):
    T = np.eye(4, dtype=np.result_type(trans, R))

    if trans.ndim == 1:
        T[:3, 3] = trans
        T[:3, :3] = R
    elif trans.ndim == 2:
        assert R.ndim == 3
        T = np.tile(T, [trans.shape[0], 1, 1])
        T[:, :3, 3] = trans
        T[:, :3, :3] = R
    else:
        raise Exception()
        
    return T


def normalize(vec):
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        raise Exception('Cannot normalize zero-length vector.')
    return vec / vec_norm


def z_up_to_R(z, up=np.array([0, 0, 1])):
    z = normalize(z)
    up = normalize(up)
    x = np.cross(up, z)
    if np.linalg.norm(x) == 0:
        R = np.eye(3)
    else:
        x = normalize(x)
        y = normalize(np.cross(z, x))
        R = np.vstack([x, y, z]).T
    return R


def z_to_R(z):
    if np.linalg.norm(z) == 0:
        raise Exception('Cannot convert zero-length vector to rotation matrix.')
        
    z = np.array(z)
    if np.allclose(z[:2], np.array([0, 0])):
        up = np.array([1, 0, 0])
    else:
        up = np.array([0, 0, 1])
    return z_up_to_R(z, up)


def n_particles_1D(p_size=0.01, length=1.0):
    return max(1, round(length / p_size))


def n_particles_3D(p_size=0.01, size=(1.0, 1.0, 1.0)):
    return (
        max(1, round(size[0] / p_size))
        * max(1, round(size[1] / p_size))
        * max(1, round(size[2] / p_size))
    )


def _box_to_particles(p_size, pos, size):
    '''
    Private function to sample particles from a box. This function only supports `random` and `regular` samplers.
    This is a private function that does not consider additional mesh offset or scale.
    '''
    size = np.array(size)
    pos = np.array(pos)
    lower = pos - size / 2
    upper = pos + size / 2

    n_particles = n_particles_3D(p_size, size)
    positions = np.random.uniform(low=lower, high=upper, size=(n_particles, 3))


    n_x = n_particles_1D(p_size, size[0])
    n_y = n_particles_1D(p_size, size[1])
    n_z = n_particles_1D(p_size, size[2])
    p_lower = lower + 0.5 * p_size
    p_upper = upper - 0.5 * p_size
    x = np.linspace(p_lower[0], p_upper[0], n_x)
    y = np.linspace(p_lower[1], p_upper[1], n_y)
    z = np.linspace(p_lower[2], p_upper[2], n_z)
    positions = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1).reshape((-1, 3))

    return positions


def sphere_to_particles(p_size=0.01, pos=(0, 0, 0), radius=0.5):
    # Sample a cube
    size = np.array([2 * radius, 2 * radius, 2 * radius])
    positions = _box_to_particles(
        p_size  = p_size,
        pos     = pos,
        size    = size
    )
    # Discard out-of-boundary particles
    positions_r = np.linalg.norm(positions - np.array(pos), axis=1)
    positions = positions[positions_r <= radius]

    return positions


def box_to_particles(p_size=0.01, pos=(0, 0, 0), size=(1, 1, 1)):
    positions = _box_to_particles(
        p_size  = p_size,
        pos     = pos,
        size    = size
    )
    return positions


@ti.data_oriented
class Emitter():
    def __init__(self, particle_system, max_particles):
        self.ps = particle_system
        self.cfg = self.ps.cfg
        self.particle_size = 0.01
        self.particle_size = self.cfg.get_cfg("particleRadius")
        self._entity = None
        self._max_particles  = max_particles
        print(f"Creating emitter, max_particles: {max_particles}.")

    def set_entity(self, entity):
        self._entity = entity
        self._sim = entity.sim
        self._solver = entity.solver
        self._next_particle = 0

    def reset(self):
        self._next_particle = 0

    # def emit(self, droplet_shape, droplet_size, droplet_length=None, pos=(0.5, 0.5, 1.0), direction=(0, 0, -1), speed=1.0):
    #     assert self._entity is not None

    #     if droplet_shape in ['sphere', 'square']:
    #         assert isinstance(droplet_size, (int, float))
    #     else:
    #         raise Exception(f'Unsupported shape {droplet_shape}.')

    #     if np.linalg.norm(direction) < 1e-5:
    #         raise Exception('Zero-length direction.')
    #     else:
    #         direction = np.array(direction) / np.linalg.norm(direction)

    #     pos = np.array(pos)
    #     if droplet_length is None:
    #         # Use the speed to determine the length of the droplet in the emitting direction
    #         droplet_length = max(speed * self._solver.substep_dt * self._sim.substeps, self._solver.particle_size)


    #     if droplet_shape == 'sphere': # sphere droplet ignores droplet_length
    #         positions = sphere_to_particles(
    #             p_size  = self.particle_size,
    #             radius  = droplet_size / 2,
    #         )
    #     elif droplet_shape == 'square':
    #         positions = box_to_particles(
    #             p_size  = self.particle_size,
    #             size    = np.array([droplet_size, droplet_size, droplet_length]),
    #         )
    #     else:
    #         raise Exception(f'Unsupported shape {droplet_shape}')

    #     positions = transform_by_T(
    #         positions,
    #         trans_R_to_T(
    #             pos,
    #             z_to_R(direction)
    #         )
    #     ).astype(np.float)
            
    #     n_particles = len(positions)

    #     vels = np.tile(direction * speed, (n_particles, 1)).astype(np.float)

    #     self._solver._kernel_set_particles_pos(
    #         self._entity.particle_start + self._next_particle,
    #         n_particles,
    #         positions,
    #     )
    #     self._solver._kernel_set_particles_vel(
    #         self._entity.particle_start + self._next_particle,
    #         n_particles,
    #         vels,
    #     )
    #     self._solver._kernel_set_particles_active(
    #         self._entity.particle_start + self._next_particle,
    #         n_particles
    #     )

    #     self._next_particle += n_particles

    #     # recycle particles
    #     if self._next_particle + n_particles > self._entity.n_particles:
    #         self._next_particle = 0

    #     print(f'Emitted {n_particles} particles. Next particle index: {self._next_particle}.')

    def emit(self, fluid):
        if self._next_particle < self.max_particles:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            num_new_particles = self.ps.add_cube(object_id=obj_id,
                                    lower_corner=start,
                                    cube_size=(end-start)*scale,
                                    velocity=velocity,
                                    density=density, 
                                    is_dynamic=1, # enforce fluid dynamic
                                    color=color,
                                    material=1) # 1 indicates fluid
            # num_new_particles = 125
            self._next_particle += num_new_particles
            print(f"num new particles {num_new_particles} next particle {self._next_particle} max particles {self.max_particles}")
    @property
    def id(self):
        return self._id

    @property
    def entity(self):
        return self._entity
    
    @property
    def max_particles(self):
        return self._max_particles
    
    @property
    def solver(self):
        return self._solver
    
    @property
    def next_particle(self):
        return self._next_particle
    