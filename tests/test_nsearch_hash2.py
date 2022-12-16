'''
测试哈希的邻域搜索
'''
import taichi as ti
import numpy as np

import sys
sys.path.append(".")

ti.init()

from nsearch.nsearch_hash import NeighborhoodSearchHash

def test():
    pos_np = np.loadtxt('E:/Dev/SPH_Taichi/tests/test_data_input_nsearch.csv',dtype=float)
    pos = ti.Vector.field(3, dtype=ti.f32, shape=pos_np.shape)
    search = NeighborhoodSearchHash(pos, 1.1)
    search.neighborhood_search()
    np.savetxt('test_data_actual_output_nsearch_hash.csv', search.particle_neighbors.to_numpy(), "%d")

if __name__ == '__main__':
    test()
