// Naive Matrix Multiplication
// Each thread computes one element of C, reading from global memory

extern "C" __global__ void matmul_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // M: How many rows in matrix A
    // K: How many columns in matrix A / how many rows in matrix B
    // N: How many columns in matrix B

    // small tip: on a flat matrix
    // to reach Matrix(i,j) -> FlatMatrix[i * num of columns + j]


    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // this thread is responsible for one specific row with a specific column
    if (row < M && col < N){
        float sum = 0.0f;
        for (int k = 0; k < K; k++){
            sum+=A[row*K + k] * B[col+k*N];
        }
        C[row*N+col] = sum;
    }
}
