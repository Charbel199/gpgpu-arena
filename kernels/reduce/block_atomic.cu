// SOL1: block-Level Atomic Reduction
// each block reduces to shared memory, then one global atomic per block
#include <cuda/atomic>
extern "C" __global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float block_sum;
    
    if (threadIdx.x == 0) {
        block_sum = 0.0f;
    }
    __syncthreads(); //or else we could get another thread (not the #0 adding to a jibberish block_sum)
    
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id < n) {
        cuda::atomic_ref<float, cuda::thread_scope_block> block_sum_ref(block_sum);
        block_sum_ref.fetch_add(input[id]);
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
        output_ref.fetch_add(block_sum);
    }
}
