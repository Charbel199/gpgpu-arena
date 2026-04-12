#!/usr/bin/env python3
import cupy as cp
import cuda.tile as ct
from cutile_base import main

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

M, K, N = 1024, 1024, 1024


@ct.kernel(occupancy=2)
def matmul_kernel(
    A: ct.Array, B: ct.Array, C: ct.Array,
    K_DIM: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)

    for k in range(ct.cdiv(K_DIM, BLOCK_K)):
        a_tile = ct.load(A, (pid_m, k), (BLOCK_M, BLOCK_K)).astype(ct.float16)
        b_tile = ct.load(B, (k, pid_n), (BLOCK_K, BLOCK_N)).astype(ct.float16)
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(C, (pid_m, pid_n), acc)


dummy_A = cp.ones((M, K), dtype=cp.float32)
dummy_B = cp.ones((K, N), dtype=cp.float32)
dummy_C = cp.zeros((M, N), dtype=cp.float32)

main(
    kernel_fn=matmul_kernel,
    dummy_args=(dummy_A, dummy_B, dummy_C, K, BLOCK_M, BLOCK_N, BLOCK_K),
    constants={"K_DIM": K, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
)
