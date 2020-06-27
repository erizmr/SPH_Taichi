# SPH_Taichi
A [Taichi](https://github.com/taichi-dev/taichi) implementation of Smooth Particle Hydrodynamics (SPH) simulator. Taichi is a productive & portable programming language for high-performance, sparse & differentiable computing. The 2D case below can be run and rendered efficiently on a laptop thanks to Taichi.

## Features
Currently, the following features have been implemented:
- Weakly Compressible SPH (WCSPH)[1]
- Predictive-Corrective Incompressible SPH (PCISPH)[2]

## Example
- Demo1 (WCSPH, 1k particles)

run ```python test_sample.py```
<p align="center">
  <img src="https://github.com/erizmr/SPH_Taichi/blob/master/img/sph_hv.gif" width="50%" height="50%" />
</p>

- Demo2 (WCSPH, 4k particles)

Run ```python scene.py```
<p align="center">
  <img src="https://github.com/erizmr/SPH_Taichi/blob/master/img/wcsph_alpha030.gif" width="50%" height="50%" />
</p>

## Reference
1. M. Becker and M. Teschner (2007). "Weakly compressible SPH for free surface flows". In:Proceedings of the 2007 ACM SIGGRAPH/Eurographics symposium on Computer animation. Eurographics Association, pp. 209–217.
2. B. Solenthaler and R. Pajarola (2009). “Predictive-corrective incompressible SPH”. In: ACM SIGGRAPH 2009 papers, pp. 1–6.
 
