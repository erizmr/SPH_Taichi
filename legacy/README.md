# SPH_Taichi
A [Taichi](https://github.com/taichi-dev/taichi) implementation of Smooth Particle Hydrodynamics (SPH) simulator. Taichi is a productive & portable programming language for high-performance, sparse & differentiable computing. The 2D case below can be run and rendered efficiently on a laptop thanks to Taichi.

## Features
Currently, the following features have been implemented:
- Weakly Compressible SPH (WCSPH)[1]
- Predictive-Corrective Incompressible SPH (PCISPH)[2]
- Divergence free SPH (DFSPH)[3]

### Note: Updates on November 6, 2021
The code is now compatible with Taichi `v0.8.3`.

## Example

- Demo (PCISPH, 4.5k particles)

Run ```python scene.py --method PCISPH```
<p align="center">
  <img src="https://github.com/erizmr/SPH_Taichi/blob/master/img/PCISPH.gif" width="50%" height="50%" />
</p>

Demos for the other two methods: ```python scene.py --method WCSPH``` or ```python scene.py --method DFSPH```

## Reference
1. M. Becker and M. Teschner (2007). "Weakly compressible SPH for free surface flows". In:Proceedings of the 2007 ACM SIGGRAPH/Eurographics symposium on Computer animation. Eurographics Association, pp. 209–217.
2. B. Solenthaler and R. Pajarola (2009). “Predictive-corrective incompressible SPH”. In: ACM SIGGRAPH 2009 papers, pp. 1–6.
3. J. Bender, D. Koschier (2015) Divergence-free smoothed particle hydrodynamics[C]//Proceedings of the 14th ACM SIGGRAPH/Eurographics symposium on computer animation. ACM, 2015: 147-155.
 
