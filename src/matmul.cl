__kernel void matmul(
    __global const int* A,
    __global const int* B,
    __global int* C,
    const int M,
    const int N,
    const int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}