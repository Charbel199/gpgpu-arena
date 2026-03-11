#!/usr/bin/env python3
import triton
import triton.language as tl
from triton_base import main


@triton.jit
def reduce_sum(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    stride = BLOCK_SIZE * NUM_BLOCKS

    thread_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    num_iters = tl.cdiv(n, stride)
    for i in range(0, num_iters):
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + i * stride
        mask = offset < n
        thread_sum += tl.load(input_ptr + offset, mask=mask, other=0.0)

    block_sum = tl.sum(thread_sum)
    tl.atomic_add(output_ptr, block_sum)


main(
    fn=reduce_sum,
    signature={"input_ptr": "*fp32", "output_ptr": "*fp32", "n": "i32"},
    constants={"BLOCK_SIZE": 128, "NUM_BLOCKS": 2560},
)
