import os
os.add_dll_directory("E:/codes/vcpkg/installed/x64-windows/bin")
os.add_dll_directory("E:/codes/openvdb/install/bin")  
import sys
sys.path.append("E:/codes/openvdb/install/lib/python3.10/site-packages")
import pyopenvdb as vdb 
# ===========================
cube = vdb.FloatGrid()
cube.fill(min=(100, 100, 100), max=(199, 199, 199), value=1.0)
cube.name = 'cube'
sphere = vdb.createLevelSetSphere(radius=50, center=(1.5, 2, 3))
sphere['radius'] = 11.5
sphere.transform = vdb.createLinearTransform(voxelSize=0.5)
sphere.name = 'sphere'
vdb.write('./tests/test_data/mygrids.vdb', grids=[cube, sphere])