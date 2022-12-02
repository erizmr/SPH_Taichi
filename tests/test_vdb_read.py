import os
os.add_dll_directory("E:/codes/vcpkg/installed/x64-windows/bin")
os.add_dll_directory("E:/codes/openvdb/install/bin")  
import sys
sys.path.append("E:/codes/openvdb/install/lib/python3.10/site-packages")
import pyopenvdb as vdb 
# ===========================
def test1():
    filename = './tests/test_data/mygrids.vdb'
    sphere = vdb.read(filename, 'sphere')
    print(sphere.metadata)


def test2():
    filename = './tests/test_data/collision_v1.vdb'
    surface = vdb.read(filename, 'surface')
    print(surface.metadata)

test2()




# grids = vdb.readAllGridMetadata(filename)

# sphere = None
# for grid in grids:
#     if (grid.gridClass == vdb.GridClass.LEVEL_SET and 'radius' in grid and grid['radius'] > 10.0):
#         sphere = vdb.read(filename, grid.name)
#     else:
#         print ('skipping grid', grid.name)

# if sphere:
#     outside = sphere.background
#     width = 2.0 * outside

#     for iter in sphere.iterOnValues():
#         dist = iter.value
#         iter.value = (outside - dist) / width

#     for iter in sphere.iterOffValues():
#         if iter.value < 0.0:
#             iter.value = 1.0
#             iter.active = False

#     sphere.background = 0.0

#     sphere.gridClass = vdb.GridClass.FOG_VOLUME