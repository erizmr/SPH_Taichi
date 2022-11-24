# SPH Taichi

A high-performance implementation of Smooth Particle Hydrodynamics (SPH) simulator in [Taichi](https://github.com/taichi-dev/taichi). (working in progress)

## Examples

- Dragon Bath (~420 K particles, ~280 FPS on RTX 3090 GPU, with timestep 4e-4)

<p align="center">
  <img src="https://github.com/erizmr/SPH_Taichi/blob/master/data/gif/dragon_bath_large.gif" width="50%" height="50%" />
</p>

- Armadillo Bath (~1.74 M particles, ~80 FPS on RTX 3090 GPU, with timestep 4e-4)

<p align="center">
  <img src="https://github.com/erizmr/SPH_Taichi/blob/master/data/gif/armadillo_bath.gif" width="50%" height="50%" />
</p>

## Features

Currently, the following features have been implemented:
- Cross-platform: Windows, Linux
- Support massively parallel GPU computing
- Weakly Compressible SPH (WCSPH)[1]
- One-way/two-way fluid-solid coupling[2]
- Shape-matching based rigid-body simulator[3]
- Neighborhood search accelerated by GPU parallel prefix sum + counting sort

### Note
The GPU parallel prefix sum is only supported by cuda/vulkan backend currently. 

## Install

```
python -m pip install -r requirements.txt
```

To reproduce the demos show above:

```
python run_simulation.py --scene_file ./data/scenes/dragon_bath.json
```

```
python run_simulation.py --scene_file ./data/scenes/armadillo_bath_dynamic.json
```


## Reference
1. M. Becker and M. Teschner (2007). "Weakly compressible SPH for free surface flows". In:Proceedings of the 2007 ACM SIGGRAPH/Eurographics symposium on Computer animation. Eurographics Association, pp. 209–217.
2. N. Akinci, M. Ihmsen, G. Akinci, B. Solenthaler, and M. Teschner. 2012. Versatile
rigid-fluid coupling for incompressible SPH. ACM Transactions on Graphics 31, 4 (2012), 62:1–62:8.
3. Miles Macklin, Matthias Müller, Nuttapong Chentanez, and Tae-Yong Kim. 2014. Unified particle physics for real-time applications. ACM Trans. Graph. 33, 4, Article 153 (July 2014), 12 pages.


## Acknowledgement
Implementation is largely inspired by [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH).
 