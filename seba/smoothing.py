import numpy as np
import cupy as cp
from numba import njit,prange


@njit(parallel=True)
def smooth_diag_numba(sim, iter_num=40):
    m, n = sim.shape
    out = np.empty_like(sim)

    for _ in range(iter_num):
        for i in prange(m):
            for j in range(n):
                s = sim[i, j]
                c = 1.0

                if i > 0 and j > 0:
                    s += sim[i-1, j-1]
                    c += 1.0
                if i < m-1 and j < n-1:
                    s += sim[i+1, j+1]
                    c += 1.0

                out[i, j] = s / c

        sim, out = out, sim

    return sim
def smooth_diag_numpy(similarity_matrix: np.ndarray, iter_num: int = 40) -> np.ndarray:
    sim = np.asarray(similarity_matrix)
    m, n = sim.shape

    # 退化情况：一行或一列时，原代码每次迭代都“原样拷贝”，结果不变
    if m <= 1 or n <= 1 or iter_num <= 0:
        return sim.copy()

    # 为了“结果基本一致”，建议保持和原来一样的 dtype（通常 float64）
    sim = sim.astype(sim.dtype, copy=True)

    out = np.empty_like(sim)

    # 预计算每个位置参与平均的个数：内部=3，边界=2，右上/左下=1
    cnt = np.ones((m, n), dtype=np.int8)
    cnt[1:, 1:] += 1        # 有 (i-1,j-1)
    cnt[:-1, :-1] += 1      # 有 (i+1,j+1)
    inv_cnt = 1.0 / cnt.astype(sim.dtype, copy=False)

    for _ in range(iter_num):
        # out = sim（拷贝一份）
        out[...] = sim

        # 加上左上邻居：out[i,j] += sim[i-1,j-1]
        out[1:, 1:] += sim[:-1, :-1]

        # 加上右下邻居：out[i,j] += sim[i+1,j+1]
        out[:-1, :-1] += sim[1:, 1:]

        # 除以参与平均的数量（用乘法更快）
        out *= inv_cnt

        # 交换缓冲区，复用内存
        sim, out = out, sim

    return sim

def smooth_diag_cupy(similarity_matrix, iter_num: int = 40, dtype=None):
    # similarity_matrix 可以是 numpy 或 cupy 数组
    sim = cp.asarray(similarity_matrix)
    if dtype is not None:
        sim = sim.astype(dtype, copy=False)

    m, n = sim.shape
    if m <= 1 or n <= 1 or iter_num <= 0:
        return sim.copy()

    out = cp.empty_like(sim)

    cnt = cp.ones((m, n), dtype=cp.int8)
    cnt[1:, 1:] += 1
    cnt[:-1, :-1] += 1
    inv_cnt = (1.0 / cnt).astype(sim.dtype)

    for _ in range(iter_num):
        out[...] = sim
        out[1:, 1:] += sim[:-1, :-1]
        out[:-1, :-1] += sim[1:, 1:]
        out *= inv_cnt
        sim, out = out, sim
    sim=cp.asnumpy(sim)
    return sim