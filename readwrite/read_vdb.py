import os
os.add_dll_directory("E:/codes/vcpkg/installed/x64-windows/bin")
os.add_dll_directory("E:/codes/openvdb/install/bin")  
import sys
sys.path.append("E:/codes/openvdb/install/lib/python3.10/site-packages")
import pyopenvdb as vdb 
import numpy as np
# ===========================
'''
读取vdb数据序列。返回的数据会被转化为numpy.
'''
def read_vdb(vdb_path, obj_name='surface', start=1, stop=1000):
    vdbs=[]
    for i in range(start, stop):
        path = vdb_path + f".{i:}.vdb"
        print("Reading ", path)

        vdb_data = vdb.read(path, obj_name)

        if i == start:
            print(vdb_data.metadata)

        print("the range of voxels")
        bbox_min = vdb_data.metadata['file_bbox_min']
        bbox_max = vdb_data.metadata['file_bbox_max']
        print(bbox_min) #注意每帧数据都是变的，这里只是打印下第一帧
        print(bbox_max)

        bbox = (bbox_max[0] - bbox_min[0],
                bbox_max[1] - bbox_min[1],
                bbox_max[2] - bbox_min[2]
            )
        np_array = np.ndarray(bbox, float)
        np_array.fill(0)

        # dump_vdb_to_np(vdb_data, np_array)
        # vdb_data.copyToArray(array, ijk=bbox_min)  
        vdbs.append(vdb_data)
    return vdbs

# def dump_vdb_to_np(vdb_data, np_array):
#     pass

# ===========================
def test_read_vdb():
    vdb_path = 'D:/CG/meshSequence/sphere_points_v2_vdb/sphere_points_v2_vdb'
    obj_name = 'v'
    read_vdb(vdb_path, obj_name, 1, 10)

if __name__ == "__main__":
    test_read_vdb()