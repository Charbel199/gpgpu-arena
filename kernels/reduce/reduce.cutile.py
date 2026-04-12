#!/usr/bin/env python3
import cupy as cp
import cuda.tile as ct
from cutile_base import main

TILE_SIZE = 256

@ct.kernel(occupancy=2)
def reduce_sum(input_ptr, output_ptr, TILE_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(input_ptr, index=(pid,), shape=(TILE_SIZE,))
    block_sum = ct.sum(tile)
    ct.atomic_add(output_ptr, (0,), block_sum)


# dummy args for compilation (shapes/dtypes must match runtime usage)
dummy_input = cp.ones(1024, dtype=cp.float32)
dummy_output = cp.zeros(1, dtype=cp.float32)

main(
    kernel_fn=reduce_sum,
    dummy_args=(dummy_input, dummy_output, TILE_SIZE),
    constants={"TILE_SIZE": TILE_SIZE},
)
