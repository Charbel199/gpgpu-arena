// Classic shared-memory tiled matmul: each block computes one 16x16 tile of
// C, one output per thread, stepping through K in 16-wide slabs of A and B.
// The textbook first step past naive and a starting point for more.
//
// The descriptor reads the launch shape from the directive below: grid is
// ceil(N / BLOCK_N) x ceil(M / BLOCK_M) blocks of THREADS threads, with SHMEM
// bytes of dynamic shared memory (0 here; static arrays need none).
//
// arena: BLOCK_M=16 BLOCK_N=16 THREADS=256

#define TILE 16

extern "C" __global__ void matmul_tiled(
    const float* __restrict__ A,   // M x K, row-major
    const float* __restrict__ B,   // K x N, row-major
    float* __restrict__ C,         // M x N, row-major
    int M, int K, int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x % TILE;
    const int ty = threadIdx.x / TILE;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}
