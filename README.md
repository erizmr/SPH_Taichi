# Intro
This repo forks from mrzhang's SPH Taichi. It is a SPH solver written in Taichi。

本项目是fork自[erizmr](https://github.com/erizmr)的[SPH_Taichi](https://github.com/erizmr/SPH_Taichi)。本项目实现了一个基于taichi的SPH的流体求解
器。目前处于开发状态。
s
# How to run

Insall
```
python -m pip install -r requirements.txt
```

Run
```
python run_simulation.py --scene_file data/scenes/dragon_bath.json
```

You can replace the scene file by .json files in data/scenes, or implement your own scene file.

# Structure
run_simulation: The enter point.

particle_system: Store all particles data. Instantiate most other instances.

utils/loader: Load fluid blocks, fluid bodies(obj mesh), rigid bodies and rigid blocks

utils/config_builder: Get the parameters from the .json file

sph_base: Solver base class

WCSPH/IISPH/DFSPH: The implementations of different SPH solvers

nsearch_gpu/nsearch_hash: The gpu and cpu neighborhood search, respectively.

readwrite/*: ply sequence andd vdb sequence reader

data/models/*: Mesh files

data/*.json: Scence files.

tests/*: Test files.

# Difference between original repo
- 对particle_system进行了重构。流体刚体的导入分离到单独loader模块。邻域搜索分离为单独模块。
- 实现了新的cpu的hash邻域搜索。参考taichi官方的pbf邻域搜索实现的3维hash版本。
- ply和vdb序列的导入器。注意：pyopenvdb的安装较为复杂，编译c++版本的vdb后，请参考tests/test_vdb_read.py 导入相应的dll文件后使用。
- (WIP) dance_impulse。音乐节奏冲量。在外来输入数据指定网格上施加冲量。配合新增的sphere_dance.json使用。后续会重构，BUG较多。
- (WIP) sph_solver: 除了run_simulation以外的另一个入口。单纯作为一个引擎使用。分为init和update两步。原本用于houdini中的python模块。
