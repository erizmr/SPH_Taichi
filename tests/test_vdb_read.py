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
    filename = 'D:\CG\meshSequence\collision_v1.vdb'
    surface = vdb.read(filename, 'surface')
    print(surface.metadata)

test2()

