// naive row-wise softmax: one block per row
// each block finds max, computes exp(x - max), sums, and normalizes
extern "C" __global__ void softmax_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // pass 1: find row max (for numerical stability)
    float local_max = -1e38f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = row_in[i];
        if (val > local_max) local_max = val;
    }
    smem[tid] = local_max;
    __syncthreads();

    // tree reduce for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) {
            smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // pass 2: compute exp(x - max) and local sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(row_in[i] - row_max);
        row_out[i] = val;
        local_sum += val;
    }
    smem[tid] = local_sum;
    __syncthreads();

    // tree reduce for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    // pass 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}
