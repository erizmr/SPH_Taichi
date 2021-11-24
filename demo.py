import taichi as ti
from particle_system import ParticleSystem
ti.init(arch=ti.gpu)

if __name__ == "__main__":
    ps = ParticleSystem((512, 512))