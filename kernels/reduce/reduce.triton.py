#!/usr/bin/env python3
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import triton.language as tl
import triton
from triton_base import main


@triton.jit
def reduce_sum(
    input_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(data)
    tl.atomic_add(output_ptr, block_sum)


main(
    fn=reduce_sum,
    signature={0: "*fp32", 1: "*fp32", 2: "i32"},
    constants={"BLOCK_SIZE": 256},
)
