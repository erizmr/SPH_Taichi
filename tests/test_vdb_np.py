import os
os.add_dll_directory("E:/codes/vcpkg/installed/x64-windows/bin")
os.add_dll_directory("E:/codes/openvdb/install/bin")  
import sys
sys.path.append("E:/codes/openvdb/install/lib/python3.10/site-packages")
import pyopenvdb as vdb 
import numpy as np
# ===========================
def test_vdb_to_np():
    grid = vdb.createLevelSetSphere(radius=10.0)
    array = np.ndarray((40, 40, 40), int)
    array.fill(0)
    # Copy values from a grid of floats into
    # a three-dimensional array of ints.
    grid.copy(array, ijk=(-15, -20, -35))
    print(array[15, 20])

def test_np_to_vdb():
    array = np.random.rand(200, 200, 200)
    grid = vdb.FloatGrid()
    grid.copyFromArray(array)
    pass
    # if grid.activeVoxelCount() == array.size:
    #     print("true")

test_np_to_vdb()