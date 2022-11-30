import trimesh
import numpy as np

'''
给定ply文件的绝对路径。不要加后缀的数字编号和.ply拓展名。
给定文件序列的起始和结束编号。
返回的是一个python list，list的每一项都是一个numpy数组。
'''
def read_ply(ply_path, start=1, stop=1000):
    pts=[]
    for i in range(start, stop):
        ply_path = ply_path + f"{i:}.ply"
        print("Reading ", ply_path)
        mesh = trimesh.load(ply_path)
        v = mesh.vertices
        # mesh.show()
        pts.append(np.array(v))
    return pts