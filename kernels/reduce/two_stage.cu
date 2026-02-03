// SOL6: Two-Stage Reduction: helps eliminate global atomic contention
// Stage 1: each block reduces to a single value (no global atomics)
// Stage 2: small kernel reduces all block results


/*
THIS BREAKS if you have millions of blocks in stage 1:

100,000 blocks → Each thread handles 390 values (still okay)
1,000,000 blocks → Each thread handles 3,900 values (getting slow!)

This is why CUB recursively reduces until it can do the final reduction in one block
*/
#include <cuda/atomic>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

// Stage 1: Block-level reduction, output per-block results
extern "C" __global__ void reduce_sum_blocks(
    const float* __restrict__ input,
    float* __restrict__ block_results,
    int n
) {
    __shared__ float warp_sums[33];  // padded to avoid bank conflicts
    /* 
    GPU shared memory is divided into 32 banks (one per thread in a warp). When threads access different banks, they're parallel.
    When multiple threads access different addresses in the same bank, it's serialized (conflict!)
    */
    float thread_sum = 0.0f;
    int stride = blockDim.x * gridDim.x;

    // grid-stride loop
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
        thread_sum += input[i];
    }

    float sum = warp_reduce_sum(thread_sum);

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // store warp sums (no atomics), getting ready for another warp reduction
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // second warp reduction
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);

        // write block result (no atomic!)
        if (lane_id == 0) {
            block_results[blockIdx.x] = sum;
        }
    }
}

// Stage 2: Final reduction of block results
extern "C" __global__ void reduce_sum_final(
    const float* __restrict__ block_results,
    float* __restrict__ output,
    int num_blocks
) {
    __shared__ float warp_sums[33];  // padded

    float thread_sum = 0.0f;

    // each thread sums multiple block results if needed (stride is the blockDim.x)
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        thread_sum += block_results[i];
    }

    float sum = warp_reduce_sum(thread_sum);

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane_id == 0) {
            *output = sum;  // Single write, no atomic needed!
        }
    }
}
