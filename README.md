# SPH_Taichi
A [Taichi](https://github.com/taichi-dev/taichi) implementation of Weakly Compressible Smooth Particle Hydrodynamics (WCSPH). Taichi is a productive & portable programming language for high-performance, sparse & differentiable computing. The 2D case below can be run and rendered efficiently on a laptop thanks to Taichi.

## Example
- Demo1 (1k particles)

run ```python test_sample.py```
<p align="center">
  <img src="https://github.com/erizmr/SPH_Taichi/blob/master/img/sph_hv.gif" width="50%" height="50%" />
</p>

- Demo2 (4k particles)

Run ```python scene.py```
<p align="center">
  <img src="https://github.com/erizmr/SPH_Taichi/blob/master/img/wcsph_alpha030.gif" width="50%" height="50%" />
</p>

## Reference
M. Becker and M. Teschner (2007). “Weakly compressible SPH for free surface flows”. In:Proceedings of the 2007 ACM SIGGRAPH/Eurographics symposium on Computer animation. Eurographics Association, pp. 209–217.
