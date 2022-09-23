import taichi as ti


@ti.func
def warp_inclusive_add_cuda(val: ti.template()):
    global_tid = ti.global_thread_idx()
    lane_id = global_tid % 32
    # Intra-warp scan, manually unroll
    offset_j = 1
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 2
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 4
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 8
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    offset_j = 16
    n = ti.simt.warp.shfl_up_i32(ti.simt.warp.active_mask(), val, offset_j)
    if (lane_id >= offset_j):
        val += n
    return val


# target = ti.cfg.cuda
target = ti.cuda
if target == ti.cuda:
    inclusive_add = warp_inclusive_add_cuda
    barrier = ti.simt.block.sync
elif target == ti.vulkan:
    inclusive_add = ti.simt.subgroup.inclusive_add
    barrier = ti.simt.subgroup.barrier
else:
    raise RuntimeError(f"Arch {target} not supported for parallel scan.")


@ti.kernel
def shfl_scan(arr_in: ti.template(), in_beg: ti.i32, in_end: ti.i32,
              sum_smem: ti.template(), single_block: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(in_beg, in_end):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int((i - in_beg) // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ

        val = inclusive_add(val)
        barrier()

        # Put warp scan results to smem
        if (thread_id % WARP_SZ == WARP_SZ - 1):
            sum_smem[block_id, warp_id] = val
        barrier()

        # Inter-warp scan, use the first thread in the first warp
        if (warp_id == 0 and lane_id == 0):
            for k in range(1, BLOCK_SZ / WARP_SZ):
                sum_smem[block_id, k] += sum_smem[block_id, k - 1]
        barrier()

        # Update data with warp sums
        warp_sum = 0
        if (warp_id > 0):
            warp_sum = sum_smem[block_id, warp_id - 1]
        val += warp_sum
        arr_in[i] = val

        # Update partial sums
        if not single_block and (thread_id == BLOCK_SZ - 1):
            arr_in[in_end + block_id] = val


@ti.kernel
def uniform_add(arr_in: ti.template(), in_beg: ti.i32, in_end: ti.i32):
    ti.loop_config(block_dim=BLOCK_SZ)
    for i in range(in_beg + BLOCK_SZ, in_end):
        block_id = int((i - in_beg) // BLOCK_SZ)
        arr_in[i] += arr_in[in_end + block_id - 1]


WARP_SZ = 32
BLOCK_SZ = 128

@ti.kernel
def blit_from_field_to_field(
    dst: ti.template(), src: ti.template(), offset: ti.i32, size: ti.i32
):
    for i in range(size):
        dst[i + offset] = src[i]


smem = None
arrs = [0]
ele_nums = [0]
ele_nums_pos = [0]
large_arr = None


def parallel_prefix_sum_inclusive_inplace(input_arr, length):

    global smem
    global arrs
    global ele_nums
    global large_arr
    
    GRID_SZ = int((length + BLOCK_SZ - 1) / BLOCK_SZ)
    if smem is None:

        # Declare input array and all partial sums
        ele_num = length

        # Get starting position and length
        ele_nums[0] = ele_num
        start_pos = 0
        ele_nums_pos[0] = start_pos

        while (ele_num > 1):
            ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
            ele_nums.append(ele_num)
            start_pos += BLOCK_SZ * ele_num
            ele_nums_pos.append(start_pos)
        
        large_arr = ti.field(ti.i32, shape = start_pos)
        smem = ti.field(ti.i32, shape=(int(GRID_SZ), 64))

    blit_from_field_to_field(large_arr, input_arr, 0, length)

    for i in range(len(ele_nums) - 1):
        if i == len(ele_nums) - 2:
            shfl_scan(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1], smem, True)
        else:
            shfl_scan(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1], smem, False)

    for i in range(len(ele_nums) - 3, -1, -1):
        uniform_add(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1])
    
    blit_from_field_to_field(input_arr, large_arr, 0, length)
