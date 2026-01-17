// SOL3: Grid-Stride Loop + Warp Shuffle + Shared Memory
// Each thread processes multiple elements via grid-stride loop
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
    __shared__ float block_shared_sum;
    
    if (threadIdx.x == 0) {
        block_shared_sum = 0.0f;
    }
    __syncthreads();
    
    float thread_sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride) {
        thread_sum += input[i];
    }
    
    const float warp_sum = warp_reduce_sum(thread_sum);
    
    const int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_block> block_sum_ref(block_shared_sum);
        block_sum_ref.fetch_add(warp_sum);
    }

    // waiting for the threads of the sublist to ADD their elements in the block_shared_sum
    __syncthreads();
    
    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
        output_ref.fetch_add(block_shared_sum);
    }
}
