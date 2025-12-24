/**
 * Level I: The Recruit - Naive Matrix Multiplication
 * 
 * What you learn:
 * - Global memory access patterns
 * - Memory latency bottlenecks
 * - Why naive implementations are slow
 * 
 * This kernel reads directly from global memory for every operation,
 * resulting in severe memory bandwidth bottlenecks.
 */

extern "C" __global__ void matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Dot product of row of A and column of B
        for (int k = 0; k < K; ++k) {
            // These are uncoalesced, strided accesses - very slow!
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

