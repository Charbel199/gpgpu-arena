#!/usr/bin/env python3
from warp_base import wp, main

BLOCK_SIZE = 256


@wp.kernel
def reduce_sum(
    input: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
    n: wp.int32,
):
    tid = wp.tid()
    if tid < n:
        wp.atomic_add(output, 0, input[tid])


main(
    kernel_fn=reduce_sum,
    constants={"BLOCK_SIZE": BLOCK_SIZE},
)
