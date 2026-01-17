// SOL4: Grid-Stride + Warp Shuffle + Shared Memory Array
// eliminates ALL shared memory atomics
#include <cuda/atomic>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

extern "C" __global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float warp_sums[32];
    
    float thread_sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    
    // grid stride loop
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
        sum = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f; // if the thread is the first in the warp, add the warp sum, otherwise add 0
        sum = warp_reduce_sum(sum);
        
        if (lane_id == 0) {
            cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
            output_ref.fetch_add(sum);
        }
    }
}
